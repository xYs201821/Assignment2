from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

from experiments.exp_utils import tag_from_cfg


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_kflow_method(method: str) -> bool:
    method = str(method).lower()
    return method.startswith("kflow") or method.startswith("kernel")


def _kflow_param_grid(
    kflow_cfg: Dict[str, Any],
    num_particles_flow: int,
    num_lambda_flow: int,
) -> Dict[str, List[Any]]:
    raw_particles = _as_list(kflow_cfg.get("num_particles"))
    if not raw_particles:
        kflow_particles: List[Any] = [int(num_particles_flow)]
    else:
        kflow_particles = [None if v is None else int(v) for v in raw_particles]

    raw_lambda = _as_list(kflow_cfg.get("num_lambda"))
    if not raw_lambda:
        kflow_lambda: List[Any] = [int(num_lambda_flow)]
    else:
        kflow_lambda = [None if v is None else int(v) for v in raw_lambda]

    grid = {
        "num_particles": kflow_particles,
        "num_lambda": kflow_lambda,
        "ll_grad_mode": _as_list(kflow_cfg.get("ll_grad_mode")),
        "alpha": _as_list(kflow_cfg.get("alpha")),
        "alpha_update_every": _as_list(kflow_cfg.get("alpha_update_every")),
        "ds_init": _as_list(kflow_cfg.get("ds_init")),
        "optimizer": _as_list(kflow_cfg.get("optimizer")),
        "optimizer_eps": _as_list(kflow_cfg.get("optimizer_eps")),
        "optimizer_beta_1": _as_list(kflow_cfg.get("optimizer_beta_1")),
        "optimizer_beta_2": _as_list(kflow_cfg.get("optimizer_beta_2")),
        "max_flow_norm": _as_list(kflow_cfg.get("max_flow_norm")),
        "localization_radius": _as_list(kflow_cfg.get("localization_radius")),
    }
    for key, values in grid.items():
        if key in ("num_particles", "num_lambda"):
            if not values:
                grid[key] = [None]
            continue
        if not values:
            grid[key] = [None]
    return grid


def _expand_kflow_methods(
    base_methods: List[str],
    kflow_grid: Dict[str, List[Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    if not any(_is_kflow_method(method) for method in base_methods):
        return base_methods, {}

    keys = list(kflow_grid.keys())
    varying = {key for key in keys if len(kflow_grid[key]) > 1}
    expanded: List[str] = []
    configs: Dict[str, Dict[str, Any]] = {}
    for method in base_methods:
        if not _is_kflow_method(method):
            expanded.append(method)
            continue
        for values in product(*(kflow_grid[key] for key in keys)):
            params = dict(zip(keys, values))
            suffix_cfg = {key: params[key] for key in keys if key in varying}
            if suffix_cfg:
                method_name = f"{method}__{tag_from_cfg(suffix_cfg)}"
            else:
                method_name = method
            expanded.append(method_name)
            configs[method_name] = params
    return expanded, configs


def _resolve_kflow_int(value: Any, fallback: int) -> int:
    if value is None:
        return int(fallback)
    return int(value)


def build_filter_cfg(
    num_particles_pf: int,
    num_particles_flow: int,
    num_lambda_flow: int,
    ukf_alpha: float | None = None,
    ukf_beta: float | None = None,
    ukf_kappa: float | None = None,
    ukf_jitter: float | None = None,
    ess_threshold_pf: float = 0.5,
    ess_threshold_flow: float = 0.5,
    reweight_pf: str = "auto",
    reweight_flow: str = "auto",
    methods: Iterable[str] | None = None,
    kflow_cfg: Dict[str, Any] | None = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ukf_cfg: Dict[str, Any] = {}
    if ukf_alpha is not None:
        ukf_cfg["alpha"] = float(ukf_alpha)
    if ukf_beta is not None:
        ukf_cfg["beta"] = float(ukf_beta)
    if ukf_kappa is not None:
        ukf_cfg["kappa"] = float(ukf_kappa)
    if ukf_jitter is not None:
        ukf_cfg["jitter"] = float(ukf_jitter)
    filter_cfg = {
        "kf": {},
        "ekf": {},
        "ukf": ukf_cfg,
        "pf": {
            "num_particles": int(num_particles_pf),
            "ess_threshold": float(ess_threshold_pf),
            "reweight": reweight_pf,
        },
        "edh": {
            "num_particles": int(num_particles_flow),
            "num_lambda": int(num_lambda_flow),
            "ess_threshold": float(ess_threshold_flow),
            "reweight": reweight_flow,
        },
        "ledh": {
            "num_particles": int(num_particles_flow),
            "num_lambda": int(num_lambda_flow),
            "ess_threshold": float(ess_threshold_flow),
            "reweight": reweight_flow,
        },
    }
    if methods is None:
        return [], filter_cfg

    methods_list = [str(m).lower() for m in methods]
    if kflow_cfg is None:
        return methods_list, filter_cfg

    kflow_grid = _kflow_param_grid(kflow_cfg, num_particles_flow, num_lambda_flow)
    methods_list, kflow_param_cfgs = _expand_kflow_methods(methods_list, kflow_grid)
    if kflow_param_cfgs:
        kflow_debug = bool(kflow_cfg.get("debug", False))
        kflow_ess_threshold = float(kflow_cfg.get("ess_threshold", ess_threshold_flow))
        kflow_reweight = str(kflow_cfg.get("reweight", reweight_flow))
        kflow_base = {
            "ess_threshold": kflow_ess_threshold,
            "reweight": kflow_reweight,
            "debug": kflow_debug,
        }
        for method_name, param_cfg in kflow_param_cfgs.items():
            method_cfg = dict(param_cfg)
            method_cfg["num_particles"] = _resolve_kflow_int(
                method_cfg.get("num_particles"),
                num_particles_flow,
            )
            method_cfg["num_lambda"] = _resolve_kflow_int(
                method_cfg.get("num_lambda"),
                num_lambda_flow,
            )
            method_cfg.update(kflow_base)
            filter_cfg[method_name] = method_cfg
    return methods_list, filter_cfg
