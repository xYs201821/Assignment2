from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from src.benchmark import MemorySampler, summarize_gpu, summarize_rss, summarize_step_times
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.kalman import KalmanFilter
from src.filters.pf_bootstrap import BootstrapParticleFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.flows.edh import EDHFlow
from src.flows.ledh import LEDHFlow
from src.flows.kernel_embedded import KernelParticleFlow


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


def _particles_to_stats(ssm, x: tf.Tensor, w: Optional[tf.Tensor]) -> Dict[str, tf.Tensor]:
    if w is None:
        w = tf.ones(tf.shape(x)[:-1], dtype=tf.float32)
        w = w / tf.cast(tf.shape(x)[-2], tf.float32)
    mean = ssm.state_mean(x, w)
    cov = ssm.state_cov(x, w)
    return {"mean": mean, "cov": cov}


def _runtime_and_memory(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    step_times = diagnostics.get("step_time_s")
    if step_times is not None:
        out["runtime"] = summarize_step_times(np.asarray(step_times))
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
    memory_sampler = MemorySampler(sample_gpu=bool(cfg.get("sample_gpu", False)))
    mem_fn = memory_sampler.sample if cfg.get("track_memory", True) else None

    m0 = cfg.get("m0")
    P0 = cfg.get("P0")
    init_particles = cfg.get("init_particles")

    if method in ("kf", "kalman"):
        print("Running Kalman filter...")
        filt = KalmanFilter(ssm)
        filt.warmup(batch_size=tf.shape(y_obs)[0])
        res = filt.filter(y_obs, m0=m0, P0=P0, memory_sampler=mem_fn)
        mean = res["m_filt"]
        cov = res["P_filt"]
        batch = tf.shape(mean)[:-2]
        T = tf.shape(mean)[-2]
        x_particles = mean[..., tf.newaxis, :]
        w = _uniform_weights(batch, T, 1)
        diagnostics = {k: v for k, v in res.items() if k not in ("m_filt", "P_filt")}
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
        out.update(_runtime_and_memory(diagnostics))
        return out

    if method == "ekf":
        print("Running EKF...")
        filt = ExtendedKalmanFilter(ssm, joseph=True)
        filt.warmup(batch_size=tf.shape(y_obs)[0])
        res = filt.filter(y_obs, m0=m0, P0=P0, memory_sampler=mem_fn)
        mean = res["m_filt"]
        cov = res["P_filt"]
        batch = tf.shape(mean)[:-2]
        T = tf.shape(mean)[-2]
        x_particles = mean[..., tf.newaxis, :]
        w = _uniform_weights(batch, T, 1)
        diagnostics = {k: v for k, v in res.items() if k not in ("m_filt", "P_filt")}
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
        out.update(_runtime_and_memory(diagnostics))
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
        res = filt.filter(y_obs, m0=m0, P0=P0, memory_sampler=mem_fn)
        mean = res["m_filt"]
        cov = res["P_filt"]
        batch = tf.shape(mean)[:-2]
        T = tf.shape(mean)[-2]
        x_particles = mean[..., tf.newaxis, :]
        w = _uniform_weights(batch, T, 1)
        diagnostics = {k: v for k, v in res.items() if k not in ("m_filt", "P_filt")}
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
        out.update(_runtime_and_memory(diagnostics))
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
        x, w, diagnostics, parents = filt.filter(
            y_obs,
            resample=cfg.get("reweight", "auto"),
            init_dist=cfg.get("init_dist"),
            init_seed=cfg.get("init_seed"),
            init_particles=init_particles,
            memory_sampler=mem_fn,
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
        out.update(_runtime_and_memory(diagnostics))
        return out

    if method == "edh":
        print("Running EDH flow...")
        filt = EDHFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=cfg.get("reweight", "never"),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=cfg.get("reweight", "never"))
        x, w, diagnostics, parents = filt.filter(
            y_obs,
            init_dist=cfg.get("init_dist"),
            reweight=cfg.get("reweight", "never"),
            init_seed=cfg.get("init_seed"),
            init_particles=init_particles,
            memory_sampler=mem_fn,
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
        out.update(_runtime_and_memory(diagnostics))
        return out

    if method == "ledh":
        print("Running LEDH flow...")
        filt = LEDHFlow(
            ssm,
            num_lambda=int(cfg.get("num_lambda", 20)),
            num_particles=int(cfg.get("num_particles", 100)),
            ess_threshold=float(cfg.get("ess_threshold", 0.5)),
            reweight=cfg.get("reweight", "never"),
        )
        filt.warmup(batch_size=tf.shape(y_obs)[0], reweight=cfg.get("reweight", "never"))
        x, w, diagnostics, parents = filt.filter(
            y_obs,
            init_dist=cfg.get("init_dist"),
            reweight=cfg.get("reweight", "never"),
            init_seed=cfg.get("init_seed"),
            init_particles=init_particles,
            memory_sampler=mem_fn,
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
        out.update(_runtime_and_memory(diagnostics))
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
        x, w, diagnostics, parents = filt.filter(
            y_obs,
            init_dist=cfg.get("init_dist"),
            reweight=reweight,
            init_seed=cfg.get("init_seed"),
            init_particles=init_particles,
            memory_sampler=mem_fn,
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
        out.update(_runtime_and_memory(diagnostics))
        return out

    raise ValueError(f"Unknown method '{method}'")
