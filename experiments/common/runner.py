from __future__ import annotations

from typing import Any, Dict, Optional

import os
import time

import numpy as np
import tensorflow as tf

import threading

from src.benchmark import MemorySampler
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.kalman import KalmanFilter
from src.filters.pf_bootstrap import BootstrapParticleFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.flows.edh import EDHFlow
from src.flows.ledh import LEDHFlow
from src.flows.kernel_embedded import KernelParticleFlow
from src.flows.stochastic_pf import StochasticParticleFlow
from src.flows.beta_schedule import BetaScheduleConfig


def _normalize_y(y: tf.Tensor, obs_dim: int) -> tf.Tensor:
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    if y.shape.rank == 2:
        y = y[tf.newaxis, ...]
    if y.shape.rank is not None:
        y = tf.ensure_shape(y, [None, None, obs_dim])
    return y


def _uniform_weights(batch: tf.Tensor, T: tf.Tensor, n: int) -> tf.Tensor:
    n_f = tf.cast(n, tf.float32)
    shape = tf.concat([batch, tf.stack([T, tf.cast(n, tf.int32)])], axis=0)
    return tf.ones(shape, dtype=tf.float32) / n_f


def _flow_kind(method: str) -> Optional[str]:
    method = str(method).lower()
    if method.startswith("edh"):
        return "edh"
    if method.startswith("ledh"):
        return "ledh"
    if method.startswith("stochastic_pf") or method.startswith("stochastic-pf") or method == "spf":
        return "stochastic_pf"
    return None


def _flow_reweight_default(method: str, fallback: str) -> str:
    if "pfpf" in str(method).lower():
        return "always"
    return fallback


def _flow_resample_default(method: str, fallback: str) -> str:
    if "pfpf" in str(method).lower():
        return "auto"
    return fallback


def _beta_schedule_from_cfg(value: Any) -> BetaScheduleConfig | None:
    if value is None:
        return None
    if isinstance(value, BetaScheduleConfig):
        return value
    if isinstance(value, str):
        return BetaScheduleConfig(mode=value)
    if isinstance(value, dict):
        if "beta" in value or "beta_dot" in value:
            raise ValueError("explicit beta arrays are not supported; use mode 'linear' or 'optimal'")
        solver_steps = value.get("solver_steps")
        max_bisect = value.get("max_bisect", 50)
        max_bracket = value.get("max_bracket", 30)
        tol = value.get("tol", 1e-6)
        if solver_steps is not None:
            solver_steps = int(solver_steps)
        guard = value.get("guard", value.get("beta_guard"))
        if guard is not None and not isinstance(guard, bool):
            raise ValueError("beta_schedule.guard must be a boolean (true/false) or None")
        return BetaScheduleConfig(
            mode=value.get("mode", "linear"),
            mu=float(value.get("mu", 0.2)),
            guard=guard,
            solver_steps=solver_steps,
            max_bisect=int(max_bisect),
            max_bracket=int(max_bracket),
            tol=float(tol),
        )
    raise TypeError("beta_schedule must be None, BetaScheduleConfig, dict, or str")


def _particles_to_stats(ssm, x: tf.Tensor, w: Optional[tf.Tensor]) -> Dict[str, tf.Tensor]:
    if w is None:
        w = tf.ones(tf.shape(x)[:-1], dtype=tf.float32)
        w = w / tf.cast(tf.shape(x)[-2], tf.float32)
    mean = ssm.state_mean(x, w)
    cov = ssm.state_cov(x, w)
    return {"mean": mean, "cov": cov}


def _parse_diffusion_matrix(value: Any, state_dim: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.eye(state_dim, dtype=np.float32) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != state_dim:
            raise ValueError(
                f"stochastic_pf.diffusion length {arr.shape[0]} must match state_dim={state_dim}"
            )
        return np.diag(arr)
    return arr


def _runtime_and_memory(
    diagnostics: Dict[str, Any],
    wall_time_s: Optional[float] = None,
    num_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if wall_time_s is not None:
        total = float(wall_time_s)
        runtime = {"total": total}
        if batch_size and batch_size > 0:
            runtime["per_batch"] = float(total / float(batch_size))
        out["runtime"] = runtime
    rss = diagnostics.get("memory_rss")
    rss_baseline = diagnostics.get("memory_rss_baseline")
    if rss is not None:
        rss_arr = np.asarray(rss, dtype=np.float64)
        if rss_arr.size > 0:
            peak = float(np.max(rss_arr))
            baseline = float(rss_baseline) if rss_baseline is not None else float(rss_arr[0])
            incr_bytes = max(0.0, peak - baseline)
            incr_mb = incr_bytes / (1024.0 * 1024.0)
            peak_mb = peak / (1024.0 * 1024.0)
            # Use peak RSS as primary metric so values are comparable across methods
            # when running multiple methods in sequence (baseline is per-run and can be inflated).
            memory = {
                "peak_mb": peak_mb,
                "incremental_mb": incr_mb,
                "total": peak_mb,
            }
            if batch_size and batch_size > 0:
                memory["per_batch"] = float(peak_mb / float(batch_size))
            out["memory"] = memory
    return out


def _add_pred_stats(out: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    m_pred = diagnostics.get("m_pred")
    if m_pred is not None:
        out["m_pred"] = m_pred
    P_pred = diagnostics.get("P_pred")
    if P_pred is not None:
        out["P_pred"] = P_pred


def run_filter(ssm, y_obs: tf.Tensor, method: str, **cfg) -> Dict[str, Any]:
    method = str(method).lower()
    y_obs = _normalize_y(y_obs, ssm.obs_dim)
    track_memory = bool(cfg.get("track_memory", True))
    profile = bool(cfg.get("track_profile", False))
    profile_root = cfg.get("profile_dir", "results/tf_profiler")
    mem_interval = float(cfg.get("memory_sample_interval_s", 0.01))
    sample_gpu = bool(cfg.get("sample_gpu", False))
    init_seed = cfg.get("init_seed")
    if init_seed is not None:
        init_seed = tf.convert_to_tensor(init_seed, dtype=tf.int32)
    warmup_batch = tf.shape(y_obs)[0]
    warmup_y = tf.zeros([warmup_batch, 2, int(ssm.obs_dim)], dtype=tf.float32)

    def _run_profiled(fn):
        import gc
        gc.collect()

        t0 = time.perf_counter()
        mem_rss: list[int] = []
        rss_baseline: Optional[int] = None
        stop_event = threading.Event()
        sample_thread = None

        if track_memory:
            sampler = MemorySampler(sample_gpu=sample_gpu)
            rss0, _ = sampler.sample()
            rss_baseline = rss0
            mem_rss.append(rss0)

            def _sample_loop():
                while not stop_event.is_set():
                    rss, _ = sampler.sample()
                    mem_rss.append(rss)
                    time.sleep(mem_interval)

            sample_thread = threading.Thread(target=_sample_loop, daemon=True)
            sample_thread.start()

        if profile:
            logdir = os.path.join(profile_root, f"{method}_{int(time.time())}")
            tf.profiler.experimental.start(logdir)
        try:
            result = fn()
        finally:
            if hasattr(tf.experimental, "async_wait"):
                try:
                    tf.experimental.async_wait()
                except Exception:
                    pass
            if profile:
                tf.profiler.experimental.stop()
            if track_memory:
                stop_event.set()
                if sample_thread is not None:
                    sample_thread.join()
                rss1, _ = sampler.sample()
                mem_rss.append(rss1)
        return result, time.perf_counter() - t0, mem_rss, rss_baseline

    m0 = cfg.get("m0")
    P0 = cfg.get("P0")
    init_particles = cfg.get("init_particles")

    # --- Gaussian filters (KF, EKF, UKF) ---
    def _run_gaussian_filter(filt, name):
        print(f"Running {name}...")
        filt.warmup(y=warmup_y)
        res, wall_time_s, mem_rss, rss_baseline = _run_profiled(
            lambda: filt.filter(y_obs, m0=m0, P0=P0)
        )
        mean = res["m_filt"]
        cov = res["P_filt"]
        batch = tf.shape(mean)[:-2]
        T = tf.shape(mean)[-2]
        x_particles = mean[..., tf.newaxis, :]
        w = _uniform_weights(batch, T, 1)
        diagnostics = {k: v for k, v in res.items() if k not in ("m_filt", "P_filt")}
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
            diagnostics["memory_rss_baseline"] = rss_baseline
        out = {
            "x_particles": x_particles,
            "w": w,
            "mean": mean,
            "cov": cov,
            "diagnostics": diagnostics,
            "m_pred": res.get("m_pred"),
            "P_pred": res.get("P_pred"),
            "is_gaussian": True,
        }
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    if method in ("kf", "kalman"):
        return _run_gaussian_filter(KalmanFilter(ssm), "Kalman filter")

    if method == "ekf":
        return _run_gaussian_filter(ExtendedKalmanFilter(ssm, joseph=True), "EKF")

    if method == "ukf":
        filt = UnscentedKalmanFilter(
            ssm,
            alpha=float(cfg.get("alpha", 1e-3)),
            beta=float(cfg.get("beta", 2.0)),
            kappa=float(cfg.get("kappa", 0.0)),
            joseph=True,
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        return _run_gaussian_filter(filt, "UKF")

    # --- Particle filter helper ---
    def _run_particle_filter(filt, name, filter_fn, warmup_kwargs=None, extra_out=None):
        """Run a particle filter and build output dict."""
        print(f"Running {name}...")
        warmup_kwargs = warmup_kwargs or {}
        filt.warmup(y=warmup_y, **warmup_kwargs)
        (x, w, diagnostics, parents), wall_time_s, mem_rss, rss_baseline = _run_profiled(filter_fn)
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
        }
        if extra_out:
            out.update(extra_out)
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
            diagnostics["memory_rss_baseline"] = rss_baseline
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    if method in ("pf", "bootstrap"):
        resample_mode = cfg.get("reweight", "auto")
        filt = BootstrapParticleFilter(
            ssm,
            resample=resample_mode,
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
        )
        return _run_particle_filter(
            filt, "bootstrap particle filter",
            lambda: filt.filter(
                y_obs,
                resample=resample_mode,
                init_dist=cfg.get("init_dist"),
                init_seed=init_seed,
                init_particles=init_particles,
            ),
            warmup_kwargs={"resample": resample_mode},
        )

    flow_kind = _flow_kind(method)

    # --- EDH/LEDH flows (share common structure) ---
    if flow_kind in ("edh", "ledh"):
        reweight = cfg.get("reweight")
        if reweight is None:
            reweight = _flow_reweight_default(method, "never")
        resample = cfg.get("resample")
        if resample is None:
            resample = _flow_resample_default(method, "never")
        flow_cls = EDHFlow if flow_kind == "edh" else LEDHFlow
        flow_name = "EDH flow" if flow_kind == "edh" else "LEDH flow"
        filt = flow_cls(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=reweight,
            beta_schedule=_beta_schedule_from_cfg(cfg.get("beta_schedule")),
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        return _run_particle_filter(
            filt, flow_name,
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                resample=resample,
                init_seed=init_seed,
                init_particles=init_particles,
            ),
            warmup_kwargs={"reweight": reweight, "resample": resample},
        )

    if flow_kind == "stochastic_pf":
        reweight = cfg.get("reweight", "never")
        resample = cfg.get("resample", "never")
        diffusion = _parse_diffusion_matrix(cfg.get("diffusion", None), int(ssm.state_dim))
        filt = StochasticParticleFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=reweight,
            diffusion=diffusion,
            beta_schedule=_beta_schedule_from_cfg(cfg.get("beta_schedule")),
            jitter=float(cfg.get("jitter", 1e-6)),
            debug=bool(cfg.get("debug", False)),
        )
        return _run_particle_filter(
            filt, "stochastic particle flow",
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                resample=resample,
                init_seed=init_seed,
                init_particles=init_particles,
            ),
            warmup_kwargs={"reweight": reweight, "resample": resample},
        )

    if method.startswith("kflow") or method.startswith("kernel"):
        if "diag" in method:
            kernel_type = "diag"
        elif "scalar" in method:
            kernel_type = "scalar"
        else:
            kernel_type = str(cfg.get("kernel_type", "diag")).lower()
        reweight = cfg.get("reweight", "never")
        filt = KernelParticleFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            alpha=cfg.get("alpha", 1.0),
            alpha_update_every=cfg.get("alpha_update_every", 1),
            kernel_type=kernel_type,
            ll_grad_mode=cfg.get("ll_grad_mode") or "linearized",
            localization_radius=cfg.get("localization_radius", None),
            ds_init=cfg.get("ds_init", 0.05),
            optimizer=cfg.get("optimizer", None),
            optimizer_eps=cfg.get("optimizer_eps", None),
            optimizer_beta_1=cfg.get("optimizer_beta_1", None),
            optimizer_beta_2=cfg.get("optimizer_beta_2", None),
            max_flow_norm=cfg.get("max_flow_norm", 10.0),
            debug=bool(cfg.get("debug", False)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=reweight,
        )
        return _run_particle_filter(
            filt, "kernel flow",
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                init_seed=init_seed,
                init_particles=init_particles,
            ),
            warmup_kwargs={"reweight": reweight},
            extra_out={"kernel_type": kernel_type},
        )

    raise ValueError(f"Unknown method '{method}'")
