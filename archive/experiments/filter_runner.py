from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf

from .experiment_helper import print_particle_log_terms, timed_call
from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter, BootstrapParticleFilter
from src.filters.tfp_particle_filter import TFPParticleFilter
from src.flows.edh import EDHFlow
from src.flows.ledh import LEDHFlow
from src.flows.kernel_embedded import KernelParticleFlow


def run_filters(
    ssm,
    y_obs: tf.Tensor,
    cfg: Any,
    m0: Optional[tf.Tensor] = None,
    P0: Optional[tf.Tensor] = None,
    init_dist=None,
    tfp_init_dist=None,
    reset_fn=None,
) -> Dict[str, Any]:
    filter_set = _normalize_filters(getattr(cfg, "filters", None))
    kflow_kernel_types = _normalize_kernel_types(
        getattr(cfg, "kflow_kernel_types", None)
    ) if "kflow" in filter_set else []

    runtime: Dict[str, Dict[str, float]] = {}
    ekf_res = ukf_res = None
    pf_pack = tfp_pf_pack = edh_pack = ledh_pack = None
    kflow_pack: Dict[str, Any] = {}

    if "ekf" in filter_set:
        ekf = ExtendedKalmanFilter(ssm)
        ekf_res, runtime["ekf"] = timed_call(
            lambda: ekf.filter(y_obs, m0=m0, P0=P0),
            warmup=True,
            reset_fn=reset_fn,
        )
    if "ukf" in filter_set:
        ukf = UnscentedKalmanFilter(
            ssm,
            alpha=getattr(cfg, "ukf_alpha", 1.0),
            beta=getattr(cfg, "ukf_beta", 2.0),
            kappa=getattr(cfg, "ukf_kappa", 0.0),
        )
        ukf_res, runtime["ukf"] = timed_call(
            lambda: ukf.filter(y_obs, m0=m0, P0=P0),
            warmup=True,
            reset_fn=reset_fn,
        )

    if "pf" in filter_set:
        pf = BootstrapParticleFilter(
            ssm,
            num_particles=getattr(cfg, "pf_particles", 100),
            ess_threshold=getattr(cfg, "pf_ess_threshold", 0.5),
        )
        pf_out, runtime["pf"] = timed_call(
            lambda: pf.filter(
                y_obs,
                init_dist=init_dist,
                reweight=_reweight_mode(getattr(cfg, "pf_reweight", "auto")),
            ),
            warmup=True,
            reset_fn=reset_fn,
        )
        pf_x, pf_w, pf_diag, pf_parent = pf_out
        pf_pack = {
            "x": pf_x,
            "w": pf_w,
            "diagnostics": pf_diag,
            "parents": pf_parent,
            "ess_threshold": getattr(cfg, "pf_ess_threshold", 0.5),
            "num_particles": getattr(cfg, "pf_particles", 100),
        }
    if "tfp_pf" in filter_set:
        tfp_pf = TFPParticleFilter(
            ssm,
            num_particles=getattr(cfg, "pf_particles", 100),
            ess_threshold=getattr(cfg, "pf_ess_threshold", 0.5),
        )
        tfp_pf_out, runtime["tfp_pf"] = timed_call(
            lambda: tfp_pf.filter(
                y_obs,
                init_dist=tfp_init_dist if tfp_init_dist is not None else init_dist,
                reweight=_reweight_mode(getattr(cfg, "pf_reweight", "auto")),
            ),
            warmup=True,
            reset_fn=reset_fn,
        )
        tfp_pf_x, tfp_pf_w, tfp_pf_diag, tfp_pf_parent = tfp_pf_out
        tfp_pf_pack = {
            "x": tfp_pf_x,
            "w": tfp_pf_w,
            "diagnostics": tfp_pf_diag,
            "parents": tfp_pf_parent,
            "ess_threshold": getattr(cfg, "pf_ess_threshold", 0.5),
            "num_particles": getattr(cfg, "pf_particles", 100),
        }
    if "edh" in filter_set:
        edh = EDHFlow(
            ssm,
            num_lambda=getattr(cfg, "edh_num_lambda", 10),
            num_particles=getattr(cfg, "edh_particles", 100),
            ess_threshold=getattr(cfg, "edh_ess_threshold", 0.5),
        )
        edh_out, runtime["edh"] = timed_call(
            lambda: edh.filter(
                y_obs,
                init_dist=init_dist,
                reweight=_reweight_mode(getattr(cfg, "edh_reweight", "auto")),
            ),
            warmup=True,
            reset_fn=reset_fn,
        )
        edh_x, edh_w, edh_diag, edh_parent = edh_out
        edh_pack = {
            "x": edh_x,
            "w": edh_w,
            "diagnostics": edh_diag,
            "parents": edh_parent,
            "ess_threshold": getattr(cfg, "edh_ess_threshold", 0.5),
            "num_particles": getattr(cfg, "edh_particles", 100),
            "num_lambda": getattr(cfg, "edh_num_lambda", 10),
        }
        print_particle_log_terms(
            "EDH",
            edh_w,
            edh_diag,
            enabled=bool(getattr(edh, "print_log_terms", False)),
        )
    if "ledh" in filter_set:
        ledh = LEDHFlow(
            ssm,
            num_lambda=getattr(cfg, "edh_num_lambda", 10),
            num_particles=getattr(cfg, "edh_particles", 100),
            ess_threshold=getattr(cfg, "edh_ess_threshold", 0.5),
        )
        ledh_out, runtime["ledh"] = timed_call(
            lambda: ledh.filter(
                y_obs,
                init_dist=init_dist,
                reweight=_reweight_mode(getattr(cfg, "edh_reweight", "auto")),
            ),
            warmup=True,
            reset_fn=reset_fn,
        )
        ledh_x, ledh_w, ledh_diag, ledh_parent = ledh_out
        ledh_pack = {
            "x": ledh_x,
            "w": ledh_w,
            "diagnostics": ledh_diag,
            "parents": ledh_parent,
            "ess_threshold": getattr(cfg, "edh_ess_threshold", 0.5),
            "num_particles": getattr(cfg, "edh_particles", 100),
            "num_lambda": getattr(cfg, "edh_num_lambda", 10),
        }
        print_particle_log_terms(
            "LEDH",
            ledh_w,
            ledh_diag,
            enabled=bool(getattr(ledh, "print_log_terms", False)),
        )

    if "kflow" in filter_set:
        base_kwargs = {
            "num_particles": getattr(cfg, "kflow_particles", 100),
            "num_lambda": getattr(cfg, "kflow_num_lambda", 10),
            "alpha": getattr(cfg, "kflow_alpha", None),
            "localization_radius": getattr(cfg, "kflow_localization_radius", None),
        }
        extra_kwargs = {
            "ds_init": getattr(cfg, "kflow_ds_init", None),
            "adaptive_step": getattr(cfg, "kflow_adaptive_step", False),
            "adaptive_window": getattr(cfg, "kflow_adaptive_window", 20),
            "adaptive_factor": getattr(cfg, "kflow_adaptive_factor", 1.4),
            "adaptive_min": getattr(cfg, "kflow_adaptive_min", None),
            "adaptive_max": getattr(cfg, "kflow_adaptive_max", None),
            "debug": getattr(cfg, "kflow_debug", False),
            "debug_every": getattr(cfg, "kflow_debug_every", 1),
            "max_flow_norm": getattr(cfg, "kflow_max_flow_norm", None),
        }
        for kernel_type in kflow_kernel_types:
            kflow = KernelParticleFlow(
                ssm,
                kernel_type=kernel_type,
                **base_kwargs,
                **{k: v for k, v in extra_kwargs.items() if v is not None},
            )
            out, runtime[f"kflow_{kernel_type}"] = timed_call(
                lambda: kflow.filter(y_obs, init_dist=init_dist, reweight="never"),
                warmup=True,
                reset_fn=reset_fn,
            )
            kflow_x, _, kflow_diag, kflow_parent = out
            kflow_pack[kernel_type] = {
                "x": kflow_x,
                "diagnostics": kflow_diag,
                "parents": kflow_parent,
                "num_particles": getattr(cfg, "kflow_particles", 100),
                "num_lambda": getattr(cfg, "kflow_num_lambda", 10),
                "kernel_type": kernel_type,
            }

    return {
        "ekf": ekf_res,
        "ukf": ukf_res,
        "pf": pf_pack,
        "tfp_pf": tfp_pf_pack,
        "edh": edh_pack,
        "ledh": ledh_pack,
        "kflow": kflow_pack,
        "runtime": runtime,
    }


def _reweight_mode(mode):
    if isinstance(mode, bool):
        return 1 if mode else 0
    if isinstance(mode, str):
        mapping = {"never": 0, "auto": 1, "always": 2}
        if mode not in mapping:
            raise ValueError(f"Invalid reweight mode: {mode}")
        return mapping[mode]
    if isinstance(mode, int):
        if mode not in (0, 1, 2):
            raise ValueError("reweight int must be 0 (never), 1 (auto), or 2 (always)")
        return mode
    raise ValueError("reweight must be bool, int, or one of 'auto', 'never', 'always'")


def _normalize_filters(filters):
    if filters is None:
        return set()
    return {str(name).lower() for name in filters}


def _normalize_kernel_types(kernel_types):
    if kernel_types is None:
        return []
    out = []
    for name in kernel_types:
        key = str(name).lower()
        if key not in ("scalar", "diag"):
            raise ValueError("kflow_kernel_types must be in {'scalar', 'diag'}")
        out.append(key)
    return out
