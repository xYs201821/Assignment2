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
from src.metrics import best_match_rmse, ospa_distance
from src.ssm import MultiTargetAcousticSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp3_config.yaml")


def make_sensors(K: int, area_size: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    radius = area_size * 0.4
    center = area_size * 0.5
    xs = center + radius * np.cos(angles)
    ys = center + radius * np.sin(angles)
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def extract_positions(x: np.ndarray, num_targets: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    x = x.reshape(x.shape[0], x.shape[1], num_targets, 4)
    return x[..., 0:2]


def build_ssm(
    num_targets: int,
    sensor_xy: Optional[np.ndarray],
    sigma_w: float,
    dt: float,
    Psi: float,
    d0: float,
    cov_eps_x: Optional[np.ndarray],
    m0: Optional[np.ndarray],
    P0: Optional[np.ndarray],
    seed: int,
    area_size: float,
    grid_size: Optional[int] = None,
) -> MultiTargetAcousticSSM:
    kwargs: Dict[str, Any] = {
        "num_targets": num_targets,
        "sensor_xy": sensor_xy,
        "area_size": area_size,
        "sigma_w": sigma_w,
        "dt": dt,
        "Psi": Psi,
        "d0": d0,
        "cov_eps_x": cov_eps_x,
        "m0": m0,
        "P0": P0,
        "seed": seed,
    }
    if grid_size is not None:
        kwargs["grid_size"] = int(grid_size)
    return MultiTargetAcousticSSM(
        **kwargs,
    )


def _parse_state_cfg(state_cfg: Any, num_targets: int, name: str) -> np.ndarray:
    arr = np.asarray(state_cfg, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 4 * num_targets:
            raise ValueError(
                f"{name} must have length {4 * num_targets} for {num_targets} targets"
            )
        arr = arr.reshape(num_targets, 4)
    elif arr.ndim == 2:
        if arr.shape != (num_targets, 4):
            raise ValueError(
                f"{name} must have shape ({num_targets}, 4), got {arr.shape}"
            )
    else:
        raise ValueError(f"{name} must be shape (4*M,) or (M,4), got ndim={arr.ndim}")
    return arr.reshape(4 * num_targets)


def _parse_cov_eps_x(cov_eps_cfg: Any) -> Optional[np.ndarray]:
    if cov_eps_cfg is None:
        return None
    arr = np.asarray(cov_eps_cfg, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError("cov_eps_x vector must have length 4")
        arr = np.diag(arr)
    if arr.ndim != 2 or arr.shape != (4, 4):
        raise ValueError(f"cov_eps_x must be 4x4 (per-target), got {arr.shape}")
    return arr


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
    parser = argparse.ArgumentParser(description="Experiment 3 (multi-target acoustic) runner.")
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


def _multitarget_metrics(
    x_true: Any,
    mean: Any,
    num_targets: int,
    ospa_cutoff: float,
    ospa_p: float,
    ospa_percentiles: Optional[List[float]] = None,
) -> Dict[str, Any]:
    x_true_np = np.asarray(x_true)
    mean_np = np.asarray(mean)
    if x_true_np.ndim == 2:
        x_true_np = x_true_np[np.newaxis, ...]
    if mean_np.ndim == 2:
        mean_np = mean_np[np.newaxis, ...]
    pos_true = extract_positions(x_true_np, num_targets)
    pos_est = extract_positions(mean_np, num_targets)

    batch_size, T, _, _ = pos_true.shape
    ospa_vals = np.zeros((batch_size, T), dtype=np.float32)
    match_vals = np.zeros((batch_size, T), dtype=np.float32)
    for b in range(batch_size):
        for t in range(T):
            ospa_vals[b, t] = ospa_distance(
                pos_true[b, t],
                pos_est[b, t],
                cutoff=ospa_cutoff,
                p=ospa_p,
            )
            match_vals[b, t] = best_match_rmse(pos_true[b, t], pos_est[b, t])

    metrics: Dict[str, Any] = {
        "ospa": float(np.mean(ospa_vals)),
        "ospa_final": float(np.mean(ospa_vals[:, -1])),
        "best_match_rmse": float(np.mean(match_vals)),
        "best_match_rmse_final": float(np.mean(match_vals[:, -1])),
        "diverged": int(not np.isfinite(mean_np).all()),
    }
    if ospa_percentiles:
        for pct in ospa_percentiles:
            key = f"ospa_p{int(round(pct))}"
            metrics[key] = float(np.percentile(ospa_vals, pct))
    return metrics


def _plot_multitarget_trajectories(
    path: Path,
    x_true: tf.Tensor,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    sensor_xy: np.ndarray,
    num_targets: int,
    plot_pf_particles: bool = False,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    import math
    import matplotlib.pyplot as plt

    x_true_np = np.asarray(x_true)
    pos_true = extract_positions(x_true_np, num_targets)[0]
    sensor_xy = np.asarray(sensor_xy)

    method_count = len(method_order)
    ncols = min(3, max(1, method_count))
    nrows = int(math.ceil(method_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.reshape(-1)
    else:
        axes_flat = [axes]

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, num_targets)))
    for idx, method in enumerate(method_order):
        ax = axes_flat[idx]
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], c="k", marker="^", s=30)
        for t_idx in range(num_targets):
            color = colors[t_idx % len(colors)]
            ax.plot(
                pos_true[:, t_idx, 0],
                pos_true[:, t_idx, 1],
                color=color,
                linestyle="-",
                linewidth=1.5,
            )
        mean = outputs.get(method, {}).get("mean")
        if mean is not None:
            pos_est = extract_positions(np.asarray(mean), num_targets)[0]
            for t_idx in range(num_targets):
                color = colors[t_idx % len(colors)]
                ax.plot(
                    pos_est[:, t_idx, 0],
                    pos_est[:, t_idx, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                )
        if plot_pf_particles and method.startswith("pf"):
            x_particles = outputs.get(method, {}).get("x_particles")
            if x_particles is not None:
                x_part = np.asarray(x_particles)
                if x_part.ndim >= 4:
                    x_part = x_part[0, -1]
                    x_part = x_part.reshape(x_part.shape[0], num_targets, 4)
                    for t_idx in range(num_targets):
                        color = colors[t_idx % len(colors)]
                        ax.scatter(
                            x_part[:, t_idx, 0],
                            x_part[:, t_idx, 1],
                            color=color,
                            s=8,
                            alpha=0.15,
                            marker=".",
                        )
        ax.set_title(method)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":")
    for idx in range(method_count, len(axes_flat)):
        axes_flat[idx].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config, args.overrides)

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    filters_cfg = cfg.get("filters", {})
    metrics_cfg = cfg.get("metrics", {})

    out_root = Path(exp_cfg.get("output_root", "results/exp3_multitarget_acoustic"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 60))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in _as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    plot_state = bool(exp_cfg.get("plot_state", False))
    plot_seed0_only = bool(exp_cfg.get("plot_seed0_only", True))
    show_plots = bool(exp_cfg.get("show_plots", False))

    targets = [int(m) for m in _as_list(model_cfg.get("targets", [2, 3, 5]))]
    sigma_ws = [float(s) for s in _as_list(model_cfg.get("sigma_ws", [0.05, 0.1, 0.3]))]
    area_size = float(model_cfg.get("area_size", 40.0))
    dt = float(model_cfg.get("dt", 1.0))
    Psi = float(model_cfg.get("Psi", 10.0))
    d0 = float(model_cfg.get("d0", 0.1))
    grid_sizes = [int(g) for g in _as_list(model_cfg.get("grid_size")) if g is not None]
    if grid_sizes:
        sensor_specs = [
            {"sensor_xy": None, "grid_size": grid_size, "K": grid_size * grid_size}
            for grid_size in grid_sizes
        ]
    else:
        sensors = [int(k) for k in _as_list(model_cfg.get("sensors", [3, 6, 12]))]
        sensor_specs = [
            {"sensor_xy": make_sensors(K, area_size=area_size), "grid_size": None, "K": K}
            for K in sensors
        ]

    cov_eps_x = _parse_cov_eps_x(model_cfg.get("cov_eps_x"))
    init_targets_cfg = model_cfg.get("init_targets")
    m0_cfg = model_cfg.get("m0")
    P0_cfg = model_cfg.get("P0")
    init_pos_std = model_cfg.get("init_pos_std")
    init_vel_std = model_cfg.get("init_vel_std")

    base_methods = [
        str(m).lower()
        for m in _as_list(filters_cfg.get("methods", ["ekf", "ukf", "pf", "edh", "ledh"]))
    ]
    pf_cfg = filters_cfg.get("pf", {})
    flow_cfg = filters_cfg.get("flow", {})
    kflow_cfg = filters_cfg.get("kflow", {})
    ukf_cfg = filters_cfg.get("ukf", {})

    pf_particles = [int(n) for n in _as_list(pf_cfg.get("num_particles", [1000]))]
    flow_particles = [int(n) for n in _as_list(flow_cfg.get("num_particles", [1000]))]
    num_lambda_flow = int(flow_cfg.get("num_lambda", 20))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    ospa_cutoff = float(metrics_cfg.get("ospa_cutoff", 20.0))
    ospa_p = float(metrics_cfg.get("ospa_p", 1.0))
    ospa_percentiles = _as_list(metrics_cfg.get("ospa_percentiles", [50, 90]))
    ospa_percentiles = [float(p) for p in ospa_percentiles]
    metrics_base = metrics_cfg.get("base")
    if metrics_base is None:
        metrics_base = {"rmse_obs": False}

    particle_pairs = _particle_pairs(pf_particles, flow_particles, pair_particles)

    for M in targets:
        if P0_cfg is None and (init_pos_std is not None or init_vel_std is not None):
            pos_std = float(init_pos_std) if init_pos_std is not None else 1.0
            vel_std = float(init_vel_std) if init_vel_std is not None else pos_std
            base = np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2]).astype(np.float32)
            P0 = np.kron(np.eye(M, dtype=np.float32), base).astype(np.float32)
        else:
            P0 = None if P0_cfg is None else np.array(P0_cfg, dtype=np.float32)
        for sensor_spec in sensor_specs:
            sensor_xy = sensor_spec["sensor_xy"]
            grid_size = sensor_spec["grid_size"]
            K = sensor_spec["K"]
            for sigma_w in sigma_ws:
                for N_pf, N_flow in particle_pairs:
                    cfg_tag_values = {
                        "M": M,
                        "K": K,
                        "sigma_w": sigma_w,
                        "Npf": N_pf,
                        "Nflow": N_flow,
                        "lambda": num_lambda_flow,
                        "B": batch_size,
                    }
                    if grid_size is not None:
                        cfg_tag_values["grid_size"] = grid_size
                    cfg_tag = tag_from_cfg(cfg_tag_values)
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
                        x0 = None
                        if init_targets_cfg is not None:
                            x0 = _parse_state_cfg(init_targets_cfg, M, "init_targets")
                        if m0_cfg is not None:
                            m0 = _parse_state_cfg(m0_cfg, M, "m0")
                        else:
                            m0 = x0
                        sim_ssm = build_ssm(
                            num_targets=M,
                            sensor_xy=sensor_xy,
                            sigma_w=sigma_w,
                            dt=dt,
                            Psi=Psi,
                            d0=d0,
                            cov_eps_x=cov_eps_x,
                            m0=m0,
                            P0=P0,
                            seed=seed,
                            area_size=area_size,
                            grid_size=grid_size,
                        )
                        sensor_xy_used = sensor_xy
                        if sensor_xy_used is None:
                            sensor_xy_used = sim_ssm.sensor_xy.numpy()
                        if x0 is None:
                            x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,))
                        else:
                            x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,), x0=x0)

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        init_dist = build_init_dist(sim_ssm.m0, sim_ssm.P0)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            method_ssm = build_ssm(
                                num_targets=M,
                                sensor_xy=sensor_xy_used,
                                sigma_w=sigma_w,
                                dt=dt,
                                Psi=Psi,
                                d0=d0,
                                cov_eps_x=cov_eps_x,
                                m0=m0,
                                P0=P0,
                                seed=seed,
                                area_size=area_size,
                                grid_size=grid_size,
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
                            extra_metrics = _multitarget_metrics(
                                x_true=x_true,
                                mean=out["mean"],
                                num_targets=M,
                                ospa_cutoff=ospa_cutoff,
                                ospa_p=ospa_p,
                                ospa_percentiles=ospa_percentiles,
                            )
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                metrics_cfg=metrics_base,
                                extra_metrics=extra_metrics,
                                prefix=f"exp3_multitarget_acoustic {cfg_tag} seed{seed} {method}",
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

                        print_separator(f"exp3_multitarget_acoustic {cfg_tag} seed{seed} summary")
                        print_method_summary_table(metrics_by_method, method_order=tuple(methods))
                        print_separator(f"exp3_multitarget_acoustic {cfg_tag} seed{seed} compare")
                        print_metrics_compare(metrics_by_method, method_order=tuple(methods))

                        if plot_state and (not plot_seed0_only or seed == seeds[0]):
                            plot_path = per_seed_dir / "state_trajectory.png"
                            plot_title = f"exp3_multitarget_acoustic {cfg_tag} seed{seed}"
                            _plot_multitarget_trajectories(
                                plot_path,
                                x_true,
                                outputs,
                                methods,
                                sensor_xy_used,
                                M,
                                plot_pf_particles=bool(exp_cfg.get("plot_pf_particles", False)),
                                title=plot_title,
                                show=show_plots,
                            )

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp3_multitarget_acoustic {cfg_tag} avg summary")
                        print_method_summary_table(mean_metrics, method_order=tuple(methods))
                        print_separator(f"exp3_multitarget_acoustic {cfg_tag} avg compare")
                        print_metrics_compare(mean_metrics, method_order=tuple(methods))


if __name__ == "__main__":
    main()
