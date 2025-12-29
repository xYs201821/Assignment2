from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import yaml

from experiments.exp_helper import (
    aggregate_metrics_by_method,
    print_method_summary_table,
    print_metrics_compare,
    print_separator,
    record_metrics,
)
from experiments.filter_cfg import build_filter_cfg
from experiments.exp_utils import ensure_dir, save_npz, set_seed, tag_from_cfg
from experiments.runner import run_filter
from src.ssm import LinearGaussianSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp1_config.yaml")


def build_params(dx: int, dy: int, q_scale: float, r_scale: float) -> Dict[str, np.ndarray]:
    A = 0.9 * np.eye(dx, dtype=np.float32)
    for i in range(dx - 1):
        A[i, i + 1] = 0.1
    B = q_scale * np.eye(dx, dtype=np.float32)
    C = np.zeros((dy, dx), dtype=np.float32)
    obs_idx = np.linspace(0, dx - 1, dy, dtype=int)
    for i, idx in enumerate(obs_idx):
        C[i, idx] = 1.0
    D = r_scale * np.eye(dy, dtype=np.float32)
    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.eye(dx, dtype=np.float32)
    return {"A": A, "B": B, "C": C, "D": D, "m0": m0, "P0": P0, "obs_idx": obs_idx}


def build_ssm(params: Dict[str, np.ndarray], seed: int) -> LinearGaussianSSM:
    ssm = LinearGaussianSSM(
        A=params["A"],
        B=params["B"],
        C=params["C"],
        D=params["D"],
        m0=params["m0"],
        P0=params["P0"],
        seed=seed,
    )
    ssm.obs_indices = params["obs_idx"]
    return ssm


def kf_distance(kf_out: Dict[str, tf.Tensor], out: Dict[str, tf.Tensor]) -> Tuple[float, float]:
    mean_diff = tf.norm(kf_out["mean"] - out["mean"], axis=-1)
    cov_diff = tf.norm(kf_out["cov"] - out["cov"], axis=[-2, -1])
    mean_rmse = tf.sqrt(tf.reduce_mean(tf.square(mean_diff)))
    cov_rmse = tf.sqrt(tf.reduce_mean(tf.square(cov_diff)))
    return float(mean_rmse.numpy()), float(cov_rmse.numpy())


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _deep_set(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("override key cannot be empty")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _deep_set(cfg, key.strip(), value)
    return cfg


def _load_config(path: Path, overrides: List[str]) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return _apply_overrides(cfg, overrides)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 (linear-Gaussian) runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: key=value (dot-separated keys).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config, args.overrides)

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    filters_cfg = cfg.get("filters", {})
    pf_cfg = filters_cfg.get("pf", {})
    flow_cfg = filters_cfg.get("flow", {})
    kflow_cfg = filters_cfg.get("kflow", {})
    ukf_cfg = filters_cfg.get("ukf", {})

    out_root = Path(exp_cfg.get("output_root", "results/exp1_linear_gaussian"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 60))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in _as_list(exp_cfg.get("seeds", [42]))]
    dims = [int(d) for d in _as_list(exp_cfg.get("dims", [4]))]
    dy_values = _as_list(exp_cfg.get("dy"))
    dy_ratio = float(exp_cfg.get("dy_ratio", 0.25))

    pf_particles = [int(n) for n in _as_list(pf_cfg.get("num_particles", [100]))]
    flow_particles = [int(n) for n in _as_list(flow_cfg.get("num_particles", [100]))]
    base_methods = [
        str(m).lower()
        for m in _as_list(filters_cfg.get("methods", ["kf", "ekf", "ukf", "pf", "edh", "ledh"]))
    ]
    num_lambda_flow = int(flow_cfg.get("num_lambda", 20))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    q_scale = float(model_cfg.get("q_scale", 0.5))
    r_scale = float(model_cfg.get("r_scale", 0.3))
    for dx in dims:
        if dy_values:
            dy_candidates = [int(dy) for dy in dy_values]
        else:
            stride = int(round(1.0 / dy_ratio)) if dy_ratio > 0 else 4
            dy_candidates = [max(1, dx // max(1, stride))]
        for dy in dy_candidates:
            params = build_params(dx, dy, q_scale=q_scale, r_scale=r_scale)
            for N_pf in pf_particles:
                for N_flow in flow_particles:
                    cfg_tag = tag_from_cfg(
                        {
                            "dx": dx,
                            "dy": dy,
                            "Npf": N_pf,
                            "Nflow": N_flow,
                            "lambda": num_lambda_flow,
                            "B": batch_size,
                        }
                    )
                    methods, filter_cfg = build_filter_cfg(
                        num_particles_pf=N_pf,
                        num_particles_flow=N_flow,
                        num_lambda_flow=num_lambda_flow,
                        ukf_alpha=ukf_alpha,
                        ukf_beta=ukf_beta,
                        ukf_kappa=ukf_kappa,
                        ukf_jitter=ukf_jitter,
                        ess_threshold_pf=pf_ess_threshold,
                        ess_threshold_flow=flow_ess_threshold,
                        reweight_pf=pf_reweight,
                        reweight_flow=flow_reweight,
                        methods=base_methods,
                        kflow_cfg=kflow_cfg,
                    )
                    metrics_across_seeds: Dict[str, List[Dict[str, Any]]] = {
                        method: [] for method in methods
                    }
                    for seed in seeds:
                        set_seed(seed)
                        sim_ssm = build_ssm(params, seed=seed)
                        x_true, y_obs = sim_ssm.simulate(
                            T,
                            shape=(batch_size,),
                        )

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            method_ssm = build_ssm(params, seed=seed)
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "ekf", "ukf"):
                                method_cfg["m0"] = params["m0"]
                                method_cfg["P0"] = params["P0"]
                            method_cfg["init_seed"] = seed
                            out = run_filter(
                                method_ssm,
                                y_obs,
                                method,
                                **method_cfg,
                            )
                            outputs[method] = out

                        metrics_by_method: Dict[str, Dict[str, Any]] = {}
                        kf_out = outputs["kf"]
                        for method in methods:
                            out = outputs[method]
                            method_dir = per_seed_dir / method
                            extra_metrics = None
                            if method != "kf":
                                mean_rmse, cov_rmse = kf_distance(kf_out, out)
                                extra_metrics = {
                                    "kf_mean_rmse": mean_rmse,
                                    "kf_cov_rmse": cov_rmse,
                                }
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                extra_metrics=extra_metrics,
                                print_full=False,
                            )
                            metrics_by_method[method] = metrics
                            metrics_across_seeds[method].append(metrics)

                            diag = {
                                k: v
                                for k, v in out.get("diagnostics", {}).items()
                                if v is not None and not isinstance(v, dict)
                            }
                            diag["mean"] = out["mean"]
                            diag["cov"] = out["cov"]
                            diff = x_true - out["mean"]
                            diag["rmse_t"] = tf.norm(diff, axis=-1)
                            save_npz(method_dir / "diagnostics.npz", **diag)

                        print_separator(f"exp1_linear_gaussian {cfg_tag} seed{seed} summary")
                        print_method_summary_table(metrics_by_method, method_order=tuple(methods))
                        print_separator(f"exp1_linear_gaussian {cfg_tag} seed{seed} compare")
                        print_metrics_compare(metrics_by_method, method_order=tuple(methods))

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp1_linear_gaussian {cfg_tag} avg summary")
                        print_method_summary_table(mean_metrics, method_order=tuple(methods))
                        print_separator(f"exp1_linear_gaussian {cfg_tag} avg compare")
                        print_metrics_compare(mean_metrics, method_order=tuple(methods))


if __name__ == "__main__":
    main()
