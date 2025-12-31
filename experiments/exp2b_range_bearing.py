from __future__ import annotations

import argparse
import sys
from itertools import cycle, product
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
from src.metrics import rmse
from src.motion_model import ConstantVelocityMotionModel
from src.ssm import RangeBearingSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp2b_config.yaml")


def build_ssm(
    sigma_r: float,
    sigma_theta: float,
    dt: float,
    cov_eps_x: np.ndarray,
    jitter: float,
    seed: int,
) -> RangeBearingSSM:
    motion_model = ConstantVelocityMotionModel(
        dt=dt,
        cov_eps=cov_eps_x,
        seed=seed,
        jitter=jitter,
    )
    cov_eps_y = np.diag([sigma_r**2, sigma_theta ** 2]).astype(np.float32)
    return RangeBearingSSM(motion_model=motion_model, cov_eps_y=cov_eps_y, jitter=jitter, seed=seed)


def build_initial_state(dist: float, init_cfg: Dict[str, Any]) -> np.ndarray:
    x0_cfg = init_cfg.get("x0")
    if x0_cfg is not None:
        return np.array(x0_cfg, dtype=np.float32)
    speed = float(init_cfg.get("x0_speed", 0.5))
    return np.array([dist, 0.0, speed, 0.0], dtype=np.float32)


def build_initial_mean(x0: np.ndarray, init_cfg: Dict[str, Any]) -> np.ndarray:
    offset = init_cfg.get("m0_offset")
    if offset is None:
        offset = [0.1, 0.1, 0.0, 0.0]
    return x0 + np.array(offset, dtype=np.float32)


def build_initial_cov(init_cfg: Dict[str, Any]) -> np.ndarray:
    P0 = init_cfg.get("P0")
    if P0 is not None:
        return np.array(P0, dtype=np.float32)
    diag = init_cfg.get("P0_diag")
    if diag is None:
        diag = [1.0, 1.0, 0.5, 0.5]
    diag = np.array(diag, dtype=np.float32)
    return np.diag(diag)


def build_init_particles(
    m0: np.ndarray,
    P0: np.ndarray,
    num_particles: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dx = int(m0.shape[0])
    z = rng.standard_normal((batch_size, num_particles, dx)).astype(np.float32)
    z = z - z.mean(axis=1, keepdims=True)
    L = np.linalg.cholesky(P0).astype(np.float32)
    return m0.astype(np.float32) + np.matmul(z, L.T)


def _plot_state_trajectory(
    path: Path,
    x_true: tf.Tensor,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    title: Optional[str] = None,
    show: bool = False,
    interactive: bool = False,
    time_gap: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    x_true_np = np.asarray(x_true)[0]
    markevery = max(1, int(len(x_true_np) / 12))
    fig, ax = plt.subplots(figsize=(5, 5))
    line_true, = ax.plot(x_true_np[:, 0], x_true_np[:, 1], color="k", label="true", linestyle="-")
    style_cycle = cycle(["-", "--", "-.", ":", (0, (3, 1, 1, 1))])
    marker_cycle = cycle(["o", "s", "^", "v", "D", "x", "P", "*"])
    for method in method_order:
        mean = outputs.get(method, {}).get("mean")
        if mean is None:
            continue
        mean_np = np.asarray(mean)[0]
        line, = ax.plot(
            mean_np[:, 0],
            mean_np[:, 1],
            label=method,
            linestyle=next(style_cycle),
            marker=next(marker_cycle),
            markevery=markevery,
            markersize=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _ess_from_weights(w: np.ndarray) -> Optional[np.ndarray]:
    w_np = np.asarray(w, dtype=np.float64)
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if w_np.ndim != 3:
        return None
    w_sum = np.sum(w_np, axis=-1, keepdims=True)
    w_norm = np.divide(w_np, w_sum, out=np.zeros_like(w_np), where=w_sum > 0)
    ess_t = 1.0 / np.sum(np.square(w_norm), axis=-1)
    return ess_t


def _impoverishment_from_parents(parents: np.ndarray) -> Optional[np.ndarray]:
    parents_np = np.asarray(parents)
    if parents_np.ndim == 2:
        parents_np = parents_np[np.newaxis, ...]
    if parents_np.ndim != 3:
        return None
    batch, T, N = parents_np.shape
    if N == 0 or T == 0:
        return None
    unique_frac = np.zeros((batch, T), dtype=np.float32)
    for b in range(batch):
        for t in range(T):
            unique_frac[b, t] = np.unique(parents_np[b, t]).size / float(N)
    return 1.0 - unique_frac


def _plot_ess_over_time(
    path: Path,
    w: np.ndarray,
    ess_threshold: Optional[float] = None,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    ess_t = _ess_from_weights(w)
    if ess_t is None:
        return
    T = ess_t.shape[1]
    t = np.arange(T)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ess_mean = np.mean(ess_t, axis=0)
    ax.plot(t, ess_mean, color="C0", linewidth=1.6, label="ESS mean")
    if ess_t.shape[0] > 1 and band_percentiles is not None:
        p_lo, p_hi = band_percentiles
        ess_lo = np.percentile(ess_t, p_lo, axis=0)
        ess_hi = np.percentile(ess_t, p_hi, axis=0)
        ax.fill_between(
            t,
            ess_lo,
            ess_hi,
            color="C0",
            alpha=0.2,
            label=f"ESS p{int(p_lo)}-p{int(p_hi)}",
        )
    if ess_threshold is not None:
        N = np.asarray(w).shape[-1]
        ax.axhline(
            ess_threshold * float(N),
            color="C3",
            linestyle="--",
            linewidth=1.0,
            label="ESS threshold",
        )
    ax.set_xlabel("time")
    ax.set_ylabel("ESS")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="best")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_stability_series(
    path: Path,
    values: np.ndarray,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    arr = np.asarray(values)
    if arr.ndim == 1:
        mean = arr
        lo = hi = None
    else:
        flat = arr.reshape(-1, arr.shape[-1])
        mean = np.mean(flat, axis=0)
        if band_percentiles is None:
            lo = hi = None
        else:
            p_lo, p_hi = band_percentiles
            lo = np.percentile(flat, p_lo, axis=0)
            hi = np.percentile(flat, p_hi, axis=0)

    t = np.arange(mean.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(t, mean, color="C0", linewidth=1.6)
    if lo is not None and hi is not None:
        ax.fill_between(t, lo, hi, color="C0", alpha=0.25, linewidth=0)
    ax.set_xlabel("time")
    ax.grid(True, linestyle=":")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_stability_over_time(
    output_dir: Path,
    diagnostics: Dict[str, Any],
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    key_specs = [
        ("logdet_cov", "logdet_cov"),
        ("condH_log10_max", "condH_log10"),
        ("condJ_log10_max", "condJ_log10"),
        ("condK_log10_max", "condK_log10"),
        ("flow_norm_mean_max", "flow_norm_mean"),
    ]
    for key, label in key_specs:
        val = diagnostics.get(key)
        if val is None:
            continue
        title = f"stability_{label}"
        path = output_dir / f"{title}.png"
        _plot_stability_series(
            path,
            np.asarray(val),
            band_percentiles=band_percentiles,
            show=show,
            title=title,
        )


def _plot_particle_cloud(
    path: Path,
    x_particles: np.ndarray,
    w: np.ndarray,
    x_true: np.ndarray,
    time_indices: List[int],
    show: bool = False,
    title: Optional[str] = None,
    dims: Tuple[int, int] = (0, 1),
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    x_np = np.asarray(x_particles)
    w_np = np.asarray(w)
    x_true_np = np.asarray(x_true)
    if x_np.ndim == 3:
        x_np = x_np[np.newaxis, ...]
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if x_np.ndim != 4 or w_np.ndim != 3 or x_true_np.ndim < 3:
        return
    T = min(x_np.shape[1], w_np.shape[1], x_true_np.shape[1])
    times = [t for t in time_indices if 0 <= t < T]
    if not times:
        return
    b = 0
    if x_np.shape[0] == 0:
        return

    idx0, idx1 = dims
    if x_np.shape[-1] <= max(idx0, idx1):
        return

    weights: List[np.ndarray] = []
    w_max = 0.0
    for t in times:
        w_t = w_np[b, t]
        w_sum = np.sum(w_t)
        if w_sum > 0:
            w_t = w_t / w_sum
        weights.append(w_t)
        if w_t.size > 0:
            w_max = max(w_max, float(np.max(w_t)))
    if w_max <= 0:
        w_max = 1.0

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=w_max)
    markers = ["o", "^", "s"]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(
        x_true_np[b, :, idx0],
        x_true_np[b, :, idx1],
        color="k",
        linewidth=1.2,
        label="true",
    )
    for i, t in enumerate(times):
        post = x_np[b, t][:, [idx0, idx1]]
        ax.scatter(
            post[:, 0],
            post[:, 1],
            s=8.0,
            c=weights[i],
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            edgecolors="none",
            marker=markers[i % len(markers)],
            label=f"t={t}",
        )
    ax.set_xlabel(f"x{idx0}")
    ax.set_ylabel(f"x{idx1}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="weight")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _is_pf_method(method: str) -> bool:
    method = str(method).lower()
    return method in ("pf", "bootstrap") or method.startswith("pf")


def _is_pfpf_flow_method(method: str) -> bool:
    method = str(method).lower()
    return "pfpf" in method and (method.startswith("edh") or method.startswith("ledh"))


def _select_pre_resample_particles(out: Dict[str, Any]) -> Optional[np.ndarray]:
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    x_pre = diagnostics.get("x_pre")
    if x_pre is None:
        x_pre = diagnostics.get("x_pred")
    if x_pre is None:
        x_pre = out.get("x_particles") if isinstance(out, dict) else None
    return x_pre


def _select_pre_resample_weights(out: Dict[str, Any]) -> Optional[np.ndarray]:
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    w_pre = diagnostics.get("w_pre")
    if w_pre is None:
        w_pre = out.get("w") if isinstance(out, dict) else None
    return w_pre


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
    parser = argparse.ArgumentParser(description="Experiment 2b (range/bearing) runner.")
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
    init_cfg = cfg.get("init", {})

    out_root = Path(exp_cfg.get("output_root", "results/exp2b_range_bearing"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 80))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in _as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    save_particles_seed0 = bool(exp_cfg.get("save_particles_seed0", True))
    plot_state = bool(exp_cfg.get("plot_state", False))
    plot_seed0_only = bool(exp_cfg.get("plot_seed0_only", True))
    show_plots = bool(exp_cfg.get("show_plots", False))
    plot_interactive = bool(exp_cfg.get("plot_interactive", False))
    plot_time_gap = exp_cfg.get("plot_time_gap")
    if plot_time_gap is not None:
        plot_time_gap = int(plot_time_gap)
        if plot_time_gap <= 0:
            plot_time_gap = None
    plot_pf_ess = bool(exp_cfg.get("plot_pf_ess", False))
    plot_pf_ess_seed0_only = bool(exp_cfg.get("plot_pf_ess_seed0_only", True))
    plot_pf_ess_show = bool(exp_cfg.get("plot_pf_ess_show", False))
    plot_pf_degeneracy = bool(exp_cfg.get("plot_pf_degeneracy", False))
    plot_pf_degeneracy_seed0_only = bool(exp_cfg.get("plot_pf_degeneracy_seed0_only", True))
    plot_pf_degeneracy_show = bool(exp_cfg.get("plot_pf_degeneracy_show", False))
    plot_pf_degeneracy_times = [0, 9, 29]
    plot_pf_degeneracy_times = [t for t in plot_pf_degeneracy_times if 0 <= t < T]
    plot_stability = bool(exp_cfg.get("plot_stability", False))
    plot_stability_seed0_only = bool(exp_cfg.get("plot_stability_seed0_only", True))
    plot_stability_show = bool(exp_cfg.get("plot_stability_show", False))
    plot_stability_percentiles = exp_cfg.get("plot_stability_percentiles")
    if plot_stability_percentiles is None:
        plot_stability_percentiles = (25.0, 75.0)
    else:
        vals = list(plot_stability_percentiles) if isinstance(plot_stability_percentiles, (list, tuple)) else []
        if len(vals) >= 2:
            plot_stability_percentiles = (float(vals[0]), float(vals[1]))
        else:
            plot_stability_percentiles = (25.0, 75.0)

    distances = [float(d) for d in _as_list(model_cfg.get("distances", [0.5, 2.0, 10.0]))]
    sigma_thetas = [float(s) for s in _as_list(model_cfg.get("sigma_thetas", [1.0, 5.0, 15.0]))]
    sigma_rs = [float(s) for s in _as_list(model_cfg.get("sigma_rs", [0.05, 0.2, 1.0]))]
    dt = float(model_cfg.get("dt", 1.0))
    jitter = float(model_cfg.get("jitter", 1e-12))
    cov_eps_cfg = model_cfg.get("cov_eps")
    if cov_eps_cfg is not None:
        cov_eps_x = np.array(cov_eps_cfg, dtype=np.float32)
        if cov_eps_x.shape != (4, 4):
            raise ValueError("model.cov_eps must be a 4x4 matrix for range-bearing CV model")
    else:
        if "q_scale_v" not in model_cfg:
            raise ValueError("Set model.cov_eps (4x4) or model.q_scale_v for range-bearing CV model")
        q_scale_v = float(model_cfg.get("q_scale_v", 0.2))
        cov_eps_x = np.diag([0.0, 0.0, q_scale_v**2, q_scale_v**2]).astype(np.float32)

    base_methods = [
        str(m).lower()
        for m in _as_list(
            filters_cfg.get(
                "methods",
                [
                    "ekf",
                    "ukf",
                    "pf",
                    "edh",
                    "edh(pfpf)",
                    "ledh",
                    "ledh(pfpf)",
                    "kflow_scalar",
                    "kflow_diag",
                ],
            )
        )
    ]
    pf_particles = [int(n) for n in _as_list(pf_cfg.get("num_particles", [200]))]
    flow_particles = [int(n) for n in _as_list(flow_cfg.get("num_particles", [200]))]

    num_lambda_flow = int(flow_cfg.get("num_lambda", 20))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    particle_pairs = _particle_pairs(pf_particles, flow_particles, pair_particles)

    for dist in distances:
        x0 = build_initial_state(dist, init_cfg)
        m0 = build_initial_mean(x0, init_cfg)
        P0 = build_initial_cov(init_cfg)
        init_dist = build_init_dist(m0, P0)
        for sigma_theta in sigma_thetas:
            for sigma_r in sigma_rs:
                for N_pf, N_flow in particle_pairs:
                    cfg_tag = tag_from_cfg(
                        {
                            "dist": dist,
                            "sigma_theta": sigma_theta,
                            "sigma_r": sigma_r,
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
                        particle_counts = {
                            int(filter_cfg.get(method, {}).get("num_particles"))
                            for method in methods
                            if method not in ("kf", "kalman", "ekf", "ukf")
                            and filter_cfg.get(method, {}).get("num_particles") is not None
                        }
                        init_particles_by_n = {
                            n: build_init_particles(m0, P0, n, batch_size, seed)
                            for n in particle_counts
                        }
                        sim_ssm = build_ssm(
                            sigma_r=sigma_r,
                            sigma_theta=np.deg2rad(sigma_theta),
                            dt=dt,
                            cov_eps_x=cov_eps_x,
                            jitter=jitter,
                            seed=seed,
                        )
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,), x0=x0)

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            method_ssm = build_ssm(
                                sigma_r=sigma_r,
                                sigma_theta=np.deg2rad(sigma_theta),
                                dt=dt,
                                cov_eps_x=cov_eps_x,
                                jitter=jitter,
                                seed=seed,
                            )
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "kalman", "ekf", "ukf"):
                                method_cfg["m0"] = m0
                                method_cfg["P0"] = P0
                            else:
                                method_cfg["init_dist"] = init_dist
                                num_particles = method_cfg.get("num_particles")
                                if num_particles is not None:
                                    init_particles = init_particles_by_n.get(int(num_particles))
                                    if init_particles is not None:
                                        method_cfg["init_particles"] = init_particles
                            method_cfg["init_seed"] = seed
                            out = run_filter(
                                method_ssm,
                                y_obs,
                                method,
                                **method_cfg,
                            )
                            outputs[method] = out

                        metrics_by_method: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            out = outputs[method]
                            method_dir = per_seed_dir / method
                            extra_metrics = {
                                "rmse_pos": float(rmse(x_true, out["mean"], dims=[0, 1]).numpy()),
                                "rmse_vel": float(rmse(x_true, out["mean"], dims=[2, 3]).numpy()),
                            }
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                metrics_cfg={"rmse_obs": False},
                                extra_metrics=extra_metrics,
                                prefix=f"exp2b_range_bearing {cfg_tag} seed{seed} {method}",
                                print_full=False,
                            )
                            metrics_by_method[method] = metrics
                            metrics_across_seeds[method].append(metrics)

                            if _is_pf_method(method) or _is_pfpf_flow_method(method):
                                if plot_pf_ess and (
                                    not plot_pf_ess_seed0_only or seed == seeds[0]
                                ):
                                    w_pre = _select_pre_resample_weights(out)
                                    if w_pre is not None:
                                        ess_threshold = (
                                            pf_ess_threshold
                                            if _is_pf_method(method)
                                            else flow_ess_threshold
                                        )
                                        plot_path = method_dir / "pf_ess_over_time.png"
                                        _plot_ess_over_time(
                                            plot_path,
                                            w_pre,
                                            ess_threshold=ess_threshold,
                                            show=plot_pf_ess_show,
                                        )
                                if plot_pf_degeneracy and (
                                    not plot_pf_degeneracy_seed0_only or seed == seeds[0]
                                ):
                                    x_pre = _select_pre_resample_particles(out)
                                    w_pre = _select_pre_resample_weights(out)
                                    if x_pre is not None and w_pre is not None:
                                        plot_path = method_dir / "pf_particles_t0_9_29.png"
                                        _plot_particle_cloud(
                                            plot_path,
                                            x_particles=x_pre,
                                            w=w_pre,
                                            x_true=x_true,
                                            time_indices=plot_pf_degeneracy_times,
                                            show=plot_pf_degeneracy_show,
                                        )

                            diag = {
                                k: v
                                for k, v in out.get("diagnostics", {}).items()
                                if v is not None and not isinstance(v, dict)
                            }
                            diag["mean"] = out["mean"]
                            diag["cov"] = out["cov"]
                            diff = x_true - out["mean"]
                            diag["rmse_t"] = tf.norm(diff, axis=-1)
                            w_pre = _select_pre_resample_weights(out)
                            if w_pre is None:
                                w_pre = out.get("w")
                            if w_pre is not None and not out.get("is_gaussian", False):
                                ess_t = _ess_from_weights(np.asarray(w_pre))
                                if ess_t is not None:
                                    diag["ess_t"] = ess_t
                                parents = out.get("parents")
                                if parents is not None:
                                    impoverishment_t = _impoverishment_from_parents(parents)
                                    if impoverishment_t is not None:
                                        diag["impoverishment_t"] = impoverishment_t
                            if save_particles_seed0 and seed == seeds[0]:
                                diag["x_particles"] = out["x_particles"]
                                diag["w"] = out["w"]
                            save_npz(method_dir / "diagnostics.npz", **diag)
                            if plot_stability and (
                                not plot_stability_seed0_only or seed == seeds[0]
                            ):
                                diag_src = out.get("diagnostics", {})
                                if isinstance(diag_src, dict):
                                    _plot_stability_over_time(
                                        method_dir,
                                        diag_src,
                                        band_percentiles=plot_stability_percentiles,
                                        show=plot_stability_show,
                                    )

                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} summary")
                        print_method_summary_table(metrics_by_method, method_order=tuple(methods))
                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} compare")
                        print_metrics_compare(metrics_by_method, method_order=tuple(methods))

                        if plot_state and (not plot_seed0_only or seed == seeds[0]):
                            plot_path = per_seed_dir / "state_trajectory.png"
                            plot_title = f"exp2b_range_bearing {cfg_tag} seed{seed}"
                            _plot_state_trajectory(
                                plot_path,
                                x_true,
                                outputs,
                                methods,
                                title=plot_title,
                                show=show_plots,
                                interactive=plot_interactive,
                                time_gap=plot_time_gap,
                            )

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp2b_range_bearing {cfg_tag} avg summary")
                        print_method_summary_table(mean_metrics, method_order=tuple(methods))
                        print_separator(f"exp2b_range_bearing {cfg_tag} avg compare")
                        print_metrics_compare(mean_metrics, method_order=tuple(methods))


if __name__ == "__main__":
    main()
