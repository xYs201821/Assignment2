from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from experiments.exp_utils import build_init_dist, ensure_dir, save_npz, set_seed, tag_from_cfg
from experiments.filter_cfg import build_filter_cfg
from experiments.runner import run_filter
from src.metrics import split_obs_indices
from src.ssm import Lorenz96SSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp4_config.yaml")


def obs_stride_from_ratio(ratio: float) -> int:
    return int(round(1.0 / ratio))


def build_ssm(
    state_dim: int,
    obs_stride: int,
    dt: float,
    F: float,
    obs_op: str,
    q_scale: float,
    r_scale: float,
    seed: int,
) -> Lorenz96SSM:
    return Lorenz96SSM(
        state_dim=state_dim,
        obs_stride=obs_stride,
        dt=dt,
        F=F,
        obs_op=obs_op,
        q_scale=q_scale,
        r_scale=r_scale,
        seed=seed,
    )


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _particle_pairs(
    pf_particles: List[int],
    flow_particles: List[int],
    pair_particles: bool,
) -> List[Tuple[int, int]]:
    if not pf_particles and not flow_particles:
        raise ValueError("num_particles must be set for pf/flow.")
    if not pf_particles:
        pf_particles = list(flow_particles)
    if not flow_particles:
        flow_particles = list(pf_particles)
    if pair_particles:
        if len(pf_particles) == 1 and len(flow_particles) > 1:
            pf_particles = pf_particles * len(flow_particles)
        if len(flow_particles) == 1 and len(pf_particles) > 1:
            flow_particles = flow_particles * len(pf_particles)
        if len(pf_particles) != len(flow_particles):
            raise ValueError("pair_particles requires pf/flow lists of equal length.")
        return list(zip(pf_particles, flow_particles))
    return list(product(pf_particles, flow_particles))


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
    parser = argparse.ArgumentParser(description="Experiment 4 (Lorenz96) runner.")
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


def _lorenz96_extra_metrics(
    ssm: Lorenz96SSM,
    x_true: Any,
    mean: Any,
    include_energy: bool,
    include_final: bool,
) -> Dict[str, Any]:
    x_true_t = tf.convert_to_tensor(x_true, dtype=tf.float32)
    mean_t = tf.convert_to_tensor(mean, dtype=tf.float32)
    if x_true_t.shape.rank == 2:
        x_true_t = x_true_t[tf.newaxis, ...]
    if mean_t.shape.rank == 2:
        mean_t = mean_t[tf.newaxis, ...]

    diff = x_true_t - mean_t
    metrics: Dict[str, Any] = {
        "diverged": int(not np.isfinite(np.asarray(mean)).all()),
    }
    if include_energy:
        energy_true = tf.reduce_mean(tf.square(x_true_t), axis=-1)
        energy_mean = tf.reduce_mean(tf.square(mean_t), axis=-1)
        rmse_energy = tf.sqrt(tf.reduce_mean(tf.square(energy_true - energy_mean)))
        metrics["rmse_energy"] = float(rmse_energy.numpy())
    if include_final:
        rmse_final = tf.sqrt(tf.reduce_mean(tf.square(diff[:, -1, :])))
        metrics["rmse_final"] = float(rmse_final.numpy())
        obs_idx, unobs_idx = split_obs_indices(ssm, int(x_true_t.shape[-1]))
        if obs_idx is not None and len(obs_idx) > 0:
            diff_obs = tf.gather(diff, obs_idx, axis=-1)
            rmse_obs_final = tf.sqrt(tf.reduce_mean(tf.square(diff_obs[:, -1, :])))
            metrics["rmse_obs_final"] = float(rmse_obs_final.numpy())
        if unobs_idx is not None and len(unobs_idx) > 0:
            diff_unobs = tf.gather(diff, unobs_idx, axis=-1)
            rmse_unobs_final = tf.sqrt(tf.reduce_mean(tf.square(diff_unobs[:, -1, :])))
            metrics["rmse_unobs_final"] = float(rmse_unobs_final.numpy())
    return metrics


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config, args.overrides)

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    filters_cfg = cfg.get("filters", {})
    metrics_cfg = cfg.get("metrics", {})

    out_root = Path(exp_cfg.get("output_root", "results/exp4_lorenz96"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 40))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in _as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))

    dims = [int(d) for d in _as_list(model_cfg.get("dims", [40, 200, 1000]))]
    obs_ratios = [float(r) for r in _as_list(model_cfg.get("obs_ratios", [0.5, 0.25, 0.1]))]
    obs_ops = [str(o).lower() for o in _as_list(model_cfg.get("obs_ops", ["linear", "abs", "square"]))]
    dt = float(model_cfg.get("dt", 0.05))
    F = float(model_cfg.get("F", 8.0))
    q_scale = float(model_cfg.get("q_scale", 0.1))
    r_scale = float(model_cfg.get("r_scale", 0.5))
    x0_base = model_cfg.get("x0", 8.0)
    x0_noise = float(model_cfg.get("x0_noise", 0.01))

    base_methods = [
        str(m).lower()
        for m in _as_list(filters_cfg.get("methods", ["kflow_scalar", "kflow_diag", "pf"]))
    ]
    pf_cfg = filters_cfg.get("pf", {})
    flow_cfg = filters_cfg.get("flow", {})
    kflow_cfg = filters_cfg.get("kflow", {})
    ukf_cfg = filters_cfg.get("ukf", {})

    pf_particles = [int(n) for n in _as_list(pf_cfg.get("num_particles", [50]))]
    flow_particles = [int(n) for n in _as_list(flow_cfg.get("num_particles", [50]))]
    num_lambda_flow = int(flow_cfg.get("num_lambda", 60))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    metrics_base = metrics_cfg.get("base")
    if metrics_base is None:
        metrics_base = {}
    include_energy = bool(metrics_cfg.get("rmse_energy", True))
    include_final = bool(metrics_cfg.get("rmse_final", True))

    particle_pairs = _particle_pairs(pf_particles, flow_particles, pair_particles)

    for nx in dims:
        for ratio in obs_ratios:
            stride = obs_stride_from_ratio(ratio)
            if nx % stride != 0:
                continue
            for obs_op in obs_ops:
                for N_pf, N_flow in particle_pairs:
                    cfg_tag = tag_from_cfg(
                        {
                            "nx": nx,
                            "ratio": ratio,
                            "obs": obs_op,
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
                        sim_ssm = build_ssm(
                            state_dim=nx,
                            obs_stride=stride,
                            dt=dt,
                            F=F,
                            obs_op=obs_op,
                            q_scale=q_scale,
                            r_scale=r_scale,
                            seed=seed,
                        )
                        if isinstance(x0_base, (list, tuple, np.ndarray)):
                            x0 = np.array(x0_base, dtype=np.float32)
                        else:
                            x0 = np.ones(nx, dtype=np.float32) * float(x0_base)
                        x0 = x0 + np.random.normal(scale=x0_noise, size=nx).astype(np.float32)
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,), x0=x0)

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        init_dist = build_init_dist(sim_ssm.m0, sim_ssm.P0)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            method_ssm = build_ssm(
                                state_dim=nx,
                                obs_stride=stride,
                                dt=dt,
                                F=F,
                                obs_op=obs_op,
                                q_scale=q_scale,
                                r_scale=r_scale,
                                seed=seed,
                            )
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "kalman", "ekf", "ukf"):
                                method_cfg["m0"] = sim_ssm.m0
                                method_cfg["P0"] = sim_ssm.P0
                            else:
                                method_cfg["init_dist"] = init_dist
                            method_cfg["init_seed"] = seed
                            out = run_filter(method_ssm, y_obs, method, **method_cfg)
                            outputs[method] = out

                        metrics_by_method: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            out = outputs[method]
                            method_dir = per_seed_dir / method
                            extra_metrics = _lorenz96_extra_metrics(
                                sim_ssm,
                                x_true,
                                out["mean"],
                                include_energy=include_energy,
                                include_final=include_final,
                            )
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                metrics_cfg=metrics_base,
                                extra_metrics=extra_metrics,
                                prefix=f"exp4_lorenz96 {cfg_tag} seed{seed} {method}",
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
                            if seed == seeds[0] and method.startswith("kflow"):
                                diag["x_particles"] = out["x_particles"]
                                diag["w"] = out["w"]
                            save_npz(method_dir / "diagnostics.npz", **diag)

                        print_separator(f"exp4_lorenz96 {cfg_tag} seed{seed} summary")
                        print_method_summary_table(metrics_by_method, method_order=tuple(methods))
                        print_separator(f"exp4_lorenz96 {cfg_tag} seed{seed} compare")
                        print_metrics_compare(metrics_by_method, method_order=tuple(methods))

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp4_lorenz96 {cfg_tag} avg summary")
                        print_method_summary_table(mean_metrics, method_order=tuple(methods))
                        print_separator(f"exp4_lorenz96 {cfg_tag} avg compare")
                        print_metrics_compare(mean_metrics, method_order=tuple(methods))


if __name__ == "__main__":
    main()
