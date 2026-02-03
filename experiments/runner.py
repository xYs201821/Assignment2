from __future__ import annotations

from typing import Any, Dict, Optional

import os
import time

import numpy as np
import tensorflow as tf

import threading

from src.benchmark import MemorySampler, summarize_gpu, summarize_rss, summarize_step_times
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
        return BetaScheduleConfig(
            mode=value.get("mode", "linear"),
            mu=float(value.get("mu", 0.2)),
            guard=value.get("guard", value.get("beta_guard")),
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
    step_times = diagnostics.get("step_time_s")
    if step_times is not None:
        out["runtime"] = summarize_step_times(np.asarray(step_times))
    if wall_time_s is not None:
        out.setdefault("runtime", {})
        runtime = out["runtime"]
        runtime["total_s"] = float(wall_time_s)
        if num_steps and num_steps > 0:
            runtime["mean_s"] = float(wall_time_s / float(num_steps))
            if runtime.get("p95_s", 0.0) == 0.0:
                runtime["p95_s"] = runtime["mean_s"]
        if batch_size and batch_size > 0:
            runtime["per_batch_s"] = float(wall_time_s / float(batch_size))
            if num_steps and num_steps > 0:
                runtime["per_step_per_batch_s"] = float(
                    wall_time_s / float(num_steps * batch_size)
                )
    rss = diagnostics.get("memory_rss")
    if rss is not None:
        out["memory"] = summarize_rss(np.asarray(rss))
    gpu = diagnostics.get("memory_gpu")
    if gpu is not None:
        out.setdefault("memory", {}).update(summarize_gpu(np.asarray(gpu)))
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
    profile = bool(cfg.get("track_profile", track_memory))
    profile_root = cfg.get("profile_dir", "results/tf_profiler")
    mem_interval = float(cfg.get("memory_sample_interval_s", 0.01))
    sample_gpu = bool(cfg.get("sample_gpu", False))
    init_seed = cfg.get("init_seed")
    if init_seed is not None:
        init_seed = tf.convert_to_tensor(init_seed, dtype=tf.int32)

    def _run_profiled(fn):
        t0 = time.perf_counter()
        mem_rss: list[int] = []
        mem_gpu: list[int] = []
        stop_event = threading.Event()
        sample_thread = None

        if track_memory:
            sampler = MemorySampler(sample_gpu=sample_gpu)
            rss0, gpu0 = sampler.sample()
            mem_rss.append(rss0)
            if gpu0 is not None:
                mem_gpu.append(gpu0)

            def _sample_loop():
                while not stop_event.is_set():
                    rss, gpu = sampler.sample()
                    mem_rss.append(rss)
                    if gpu is not None:
                        mem_gpu.append(gpu)
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
                rss1, gpu1 = sampler.sample()
                mem_rss.append(rss1)
                if gpu1 is not None:
                    mem_gpu.append(gpu1)
        return result, time.perf_counter() - t0, mem_rss, mem_gpu

    m0 = cfg.get("m0")
    P0 = cfg.get("P0")
    init_particles = cfg.get("init_particles")

    if method in ("kf", "kalman"):
        print("Running Kalman filter...")
        filt = KalmanFilter(ssm)
        filt.warmup(batch_size=tf.shape(y_obs)[0])
        res, wall_time_s, mem_rss, mem_gpu = _run_profiled(
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
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
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

    if method == "ekf":
        print("Running EKF...")
        filt = ExtendedKalmanFilter(ssm, joseph=True)
        filt.warmup(batch_size=tf.shape(y_obs)[0])
        res, wall_time_s, mem_rss, mem_gpu = _run_profiled(
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
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
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

    if method == "ukf":
        print("Running UKF...")
        filt = UnscentedKalmanFilter(
            ssm,
            alpha=float(cfg.get("alpha", 1e-3)),
            beta=float(cfg.get("beta", 2.0)),
            kappa=float(cfg.get("kappa", 0.0)),
            joseph=True,
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0])
        res, wall_time_s, mem_rss, mem_gpu = _run_profiled(
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
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
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

    if method in ("pf", "bootstrap"):
        print("Running bootstrap particle filter...")
        filt = BootstrapParticleFilter(
            ssm,
            resample=cfg.get("reweight", "auto"),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0], resample=cfg.get("reweight", "auto"))
        (x, w, diagnostics, parents), wall_time_s, mem_rss, mem_gpu = _run_profiled(
            lambda: filt.filter(
                y_obs,
                resample=cfg.get("reweight", "auto"),
                init_dist=cfg.get("init_dist"),
                init_seed=init_seed,
                init_particles=init_particles,
            )
        )
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
        }
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    flow_kind = _flow_kind(method)

    if flow_kind == "edh":
        print("Running EDH flow...")
        reweight = cfg.get("reweight")
        if reweight is None:
            reweight = _flow_reweight_default(method, "never")
        resample = cfg.get("resample")
        if resample is None:
            resample = _flow_resample_default(method, "never")
        filt = EDHFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=reweight,
            beta_schedule=_beta_schedule_from_cfg(cfg.get("beta_schedule")),
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=reweight, resample=resample)
        (x, w, diagnostics, parents), wall_time_s, mem_rss, mem_gpu = _run_profiled(
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                resample=resample,
                init_seed=init_seed,
                init_particles=init_particles,
            )
        )
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
        }
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    if flow_kind == "ledh":
        print("Running LEDH flow...")
        reweight = cfg.get("reweight")
        if reweight is None:
            reweight = _flow_reweight_default(method, "never")
        resample = cfg.get("resample")
        if resample is None:
            resample = _flow_resample_default(method, "never")
        filt = LEDHFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=reweight,
            beta_schedule=_beta_schedule_from_cfg(cfg.get("beta_schedule")),
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=reweight, resample=resample)
        (x, w, diagnostics, parents), wall_time_s, mem_rss, mem_gpu = _run_profiled(
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                resample=resample,
                init_seed=init_seed,
                init_particles=init_particles,
            )
        )
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
        }
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    if flow_kind == "stochastic_pf":
        print("Running stochastic particle flow...")
        reweight = cfg.get("reweight")
        if reweight is None:
            reweight = "never"
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
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=reweight, resample=resample)
        (x, w, diagnostics, parents), wall_time_s, mem_rss, mem_gpu = _run_profiled(
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                resample=resample,
                init_seed=init_seed,
                init_particles=init_particles,
            )
        )
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
        }
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    if method.startswith("kflow") or method.startswith("kernel"):
        print("Running kernel flow...")
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
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=reweight)
        (x, w, diagnostics, parents), wall_time_s, mem_rss, mem_gpu = _run_profiled(
            lambda: filt.filter(
                y_obs,
                init_dist=cfg.get("init_dist"),
                reweight=reweight,
                init_seed=init_seed,
                init_particles=init_particles,
            )
        )
        stats = _particles_to_stats(ssm, x, w)
        out = {
            "x_particles": x,
            "w": w,
            "mean": stats["mean"],
            "cov": stats["cov"],
            "diagnostics": diagnostics,
            "parents": parents,
            "kernel_type": kernel_type,
        }
        _add_pred_stats(out, diagnostics)
        if mem_rss:
            diagnostics["memory_rss"] = mem_rss
        if mem_gpu:
            diagnostics["memory_gpu"] = mem_gpu
        batch_size = int(y_obs.shape[0] or tf.shape(y_obs)[0].numpy())
        num_steps = int(y_obs.shape[1] or tf.shape(y_obs)[1].numpy())
        out.update(_runtime_and_memory(diagnostics, wall_time_s, num_steps, batch_size))
        return out

    raise ValueError(f"Unknown method '{method}'")
