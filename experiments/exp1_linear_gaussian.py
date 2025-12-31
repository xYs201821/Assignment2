from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _sensor_grid_positions(dx: int) -> np.ndarray:
    grid_size = int(np.sqrt(dx))
    if grid_size * grid_size != dx:
        raise ValueError(f"dx={dx} must be a perfect square for the spatial grid")
    coords = np.arange(1, grid_size + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def build_params(
    dx: int,
    dy: int,
    alpha: float,
    sigma_z: float,
    alpha0: float,
    alpha1: float,
    beta: float,
    p0_scale: float,
) -> Dict[str, np.ndarray]:
    if p0_scale <= 0:
        raise ValueError("p0_scale must be positive")
    alpha_f = np.float32(alpha)
    sigma_z_f = np.float32(sigma_z)
    alpha0_f = np.float32(alpha0)
    alpha1_f = np.float32(alpha1)
    beta_f = np.float32(beta)

    if dy <= 0 or dy > dx:
        raise ValueError("dy must satisfy 1 <= dy <= dx")

    A = alpha_f * np.eye(dx, dtype=np.float32)
    obs_idx = np.linspace(0, dx - 1, dy, dtype=int)
    C = np.zeros((dy, dx), dtype=np.float32)
    for i, idx in enumerate(obs_idx):
        C[i, idx] = 1.0

    sensor_xy = _sensor_grid_positions(dx)
    diff = sensor_xy[:, None, :] - sensor_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    Sigma = alpha0_f * np.exp(-dist2 / beta_f) + alpha1_f * np.eye(dx, dtype=np.float32)
    Sigma = Sigma.astype(np.float32) + 1e-6 * np.eye(dx, dtype=np.float32)

    B = np.linalg.cholesky(Sigma).astype(np.float32)
    D = sigma_z_f * np.eye(dy, dtype=np.float32)
    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.float32(p0_scale) * np.eye(dx, dtype=np.float32)
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


def _select_plot_dims(
    obs_idx: np.ndarray,
    dx: int,
    obs_dim: Optional[int],
    unobs_dim: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    obs_list = sorted(int(i) for i in obs_idx)
    obs_set = set(obs_list)
    unobs_list = [i for i in range(dx) if i not in obs_set]
    if obs_dim is None:
        if obs_list:
            obs_dim = obs_list[len(obs_list) // 2]
        else:
            obs_dim = None
    else:
        obs_dim = int(obs_dim)
        if obs_dim not in obs_set:
            raise ValueError(f"plot_kflow_obs_index={obs_dim} is not observed")
    if unobs_dim is None:
        if unobs_list:
            unobs_dim = unobs_list[len(unobs_list) // 2]
        else:
            unobs_dim = None
    else:
        unobs_dim = int(unobs_dim)
        if unobs_dim in obs_set:
            raise ValueError(f"plot_kflow_unobs_index={unobs_dim} is observed")
    return obs_dim, unobs_dim


def _extract_particles_at_time(
    out: Dict[str, Any],
    t_index: int,
    batch_index: int,
    use_pred: bool,
) -> Optional[np.ndarray]:
    if use_pred:
        x = out.get("diagnostics", {}).get("x_pred")
    else:
        x = out.get("x_particles")
    if x is None:
        return None
    x_np = np.asarray(x)
    if x_np.ndim < 4:
        return None
    t = min(t_index, x_np.shape[1] - 1)
    b = min(batch_index, x_np.shape[0] - 1)
    return x_np[b, t]


def _gaussian_contour(
    ax,
    mean2: np.ndarray,
    cov2: np.ndarray,
    n_levels: int = 8,
    n_std: float = 3.0,
) -> None:
    mean2 = np.asarray(mean2, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    cov2 = 0.5 * (cov2 + cov2.T)
    if cov2.shape != (2, 2):
        return
    std_x = np.sqrt(max(cov2[0, 0], 1e-12))
    std_y = np.sqrt(max(cov2[1, 1], 1e-12))
    xs = np.linspace(mean2[0] - n_std * std_x, mean2[0] + n_std * std_x, 120)
    ys = np.linspace(mean2[1] - n_std * std_y, mean2[1] + n_std * std_y, 120)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = np.stack([grid_x - mean2[0], grid_y - mean2[1]], axis=-1)
    try:
        inv_cov = np.linalg.inv(cov2)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov2 + 1e-8 * np.eye(2))
    dist2 = np.einsum("...i,ij,...j->...", pos, inv_cov, pos)
    z = np.exp(-0.5 * dist2)
    levels = np.linspace(z.min(), z.max(), n_levels + 2)[1:-1]
    ax.contour(grid_x, grid_y, z, levels=levels, cmap="viridis", linewidths=1.0)


def _find_method_key(
    outputs: Dict[str, Dict[str, Any]],
    exact: str,
    prefix: str,
) -> Optional[str]:
    if exact in outputs:
        return exact
    for key in sorted(outputs):
        if key.startswith(prefix):
            return key
    return None


def _plot_kflow_marginals(
    path: Path,
    contour_method: str,
    gaussian_out: Dict[str, Any],
    kflow_diag_out: Dict[str, Any],
    kflow_scalar_out: Dict[str, Any],
    obs_dim: int,
    unobs_dim: int,
    t_index: int,
    show: bool = False,
    plot_prior: bool = True,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mean = np.asarray(gaussian_out.get("mean"))
    cov = np.asarray(gaussian_out.get("cov"))
    if mean.ndim < 3 or cov.ndim < 4:
        return
    t = min(t_index, mean.shape[1] - 1)
    batch_index = 0
    idx = [unobs_dim, obs_dim]
    mean2 = mean[batch_index, t, idx]
    cov2 = cov[batch_index, t][np.ix_(idx, idx)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    panels = [
        ("matrix-valued", kflow_diag_out),
        ("scalar", kflow_scalar_out),
    ]
    for ax, (panel_title, kflow_out) in zip(axes, panels):
        _gaussian_contour(ax, mean2, cov2)
        if plot_prior:
            prior = _extract_particles_at_time(kflow_out, t, batch_index, use_pred=True)
            if prior is not None:
                prior_xy = prior[:, idx]
                ax.scatter(
                    prior_xy[:, 0],
                    prior_xy[:, 1],
                    s=22,
                    facecolors="none",
                    edgecolors="k",
                    linewidths=0.8,
                    alpha=0.8,
                )
        post = _extract_particles_at_time(kflow_out, t, batch_index, use_pred=False)
        if post is not None:
            post_xy = post[:, idx]
            ax.scatter(
                post_xy[:, 0],
                post_xy[:, 1],
                s=22,
                color="red",
                edgecolors="none",
                alpha=0.85,
            )
        ax.set_title(panel_title)
        ax.grid(True, linestyle=":")
        ax.set_xlabel(f"x{unobs_dim} (unobs)")
    axes[0].set_ylabel(f"x{obs_dim} (obs)")

    contour_label = contour_method.upper()
    legend_items = [Line2D([0], [0], color="C0", label=f"{contour_label} contour")]
    if plot_prior:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                markerfacecolor="none",
                linestyle="None",
                label="prior",
            )
        )
    legend_items.append(
        Line2D([0], [0], marker="o", color="red", linestyle="None", label="posterior")
    )
    axes[0].legend(handles=legend_items, fontsize=8, loc="best")
    # Keep the plot clean (no title); use filename for metadata instead.
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


def _plot_pf_degeneracy(
    path: Path,
    pf_out: Dict[str, Any],
    gaussian_out: Optional[Dict[str, Any]],
    x_true: np.ndarray,
    obs_dim: int,
    unobs_dim: int,
    t_index: int,
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    post = _extract_particles_at_time(pf_out, t_index, batch_index=0, use_pred=False)
    w = pf_out.get("w")
    if post is None or w is None:
        return
    w_np = np.asarray(w)
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if w_np.ndim != 3:
        return
    t = min(t_index, w_np.shape[1] - 1)
    w_t = w_np[0, t]
    w_sum = np.sum(w_t)
    if w_sum > 0:
        w_t = w_t / w_sum

    idx = [unobs_dim, obs_dim]
    post_xy = post[:, idx]

    w_max = np.max(w_t) if w_t.size > 0 else 0.0
    size_scale = w_t / (w_max + 1e-12)
    sizes = 12.0 + 80.0 * size_scale

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    if gaussian_out is not None:
        mean = np.asarray(gaussian_out.get("mean"))
        cov = np.asarray(gaussian_out.get("cov"))
        if mean.ndim >= 3 and cov.ndim >= 4:
            mean2 = mean[0, t, idx]
            cov2 = cov[0, t][np.ix_(idx, idx)]
            _gaussian_contour(axes[0], mean2, cov2)
    sc = axes[0].scatter(
        post_xy[:, 0],
        post_xy[:, 1],
        s=sizes,
        c=w_t,
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
    )
    mean = pf_out.get("mean")
    if mean is not None:
        mean_np = np.asarray(mean)
        if mean_np.ndim >= 3:
            mean_xy = mean_np[0, t, idx]
            axes[0].scatter(
                mean_xy[0],
                mean_xy[1],
                s=70,
                c="red",
                marker="x",
                linewidths=1.5,
            )
    x_true_np = np.asarray(x_true)
    if x_true_np.ndim >= 3:
        true_xy = x_true_np[0, t, idx]
        axes[0].scatter(
            true_xy[0],
            true_xy[1],
            s=70,
            c="black",
            marker="+",
            linewidths=1.5,
        )
    axes[0].set_xlabel(f"x{unobs_dim} (unobs)")
    axes[0].set_ylabel(f"x{obs_dim} (obs)")
    axes[0].grid(True, linestyle=":")
    fig.colorbar(sc, ax=axes[0], label="weight")

    w_sorted = np.sort(w_t)[::-1]
    axes[1].plot(w_sorted, color="C1", linewidth=1.4)
    axes[1].set_xlabel("particle (sorted)")
    axes[1].set_ylabel("weight")
    axes[1].grid(True, linestyle=":")
    if w_sum > 0:
        ess_val = 1.0 / np.sum(np.square(w_t))
        axes[1].set_title(f"ESS={ess_val:.1f}")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_ess_over_time(
    path: Path,
    w: np.ndarray,
    ess_threshold: Optional[float] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    ess_t = _ess_from_weights(w)
    if ess_t is None:
        return
    T = ess_t.shape[1]
    t = np.arange(T)
    ess_mean = np.mean(ess_t, axis=0)
    ess_min = np.min(ess_t, axis=0)
    ess_max = np.max(ess_t, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(t, ess_mean, color="C0", linewidth=1.6, label="ESS mean")
    if ess_t.shape[0] > 1:
        ax.fill_between(t, ess_min, ess_max, color="C0", alpha=0.2, label="ESS range")
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
    dy_ratio = float(exp_cfg.get("dy_ratio", 1.0))

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
    plot_kflow_marginals = bool(exp_cfg.get("plot_kflow_marginals", False))
    plot_kflow_seed0_only = bool(exp_cfg.get("plot_kflow_seed0_only", True))
    plot_kflow_show = bool(exp_cfg.get("plot_kflow_show", False))
    plot_kflow_time = exp_cfg.get("plot_kflow_time")
    plot_kflow_contour_method = str(exp_cfg.get("plot_kflow_contour_method", "kf")).lower()
    plot_kflow_prior = bool(exp_cfg.get("plot_kflow_prior", True))
    plot_kflow_obs_index = exp_cfg.get("plot_kflow_obs_index")
    plot_kflow_unobs_index = exp_cfg.get("plot_kflow_unobs_index")
    plot_pf_degeneracy = bool(exp_cfg.get("plot_pf_degeneracy", False))
    plot_pf_degeneracy_seed0_only = bool(exp_cfg.get("plot_pf_degeneracy_seed0_only", True))
    plot_pf_degeneracy_show = bool(exp_cfg.get("plot_pf_degeneracy_show", False))
    plot_pf_degeneracy_time = exp_cfg.get("plot_pf_degeneracy_time")
    plot_pf_ess = bool(exp_cfg.get("plot_pf_ess", False))
    plot_pf_ess_seed0_only = bool(exp_cfg.get("plot_pf_ess_seed0_only", True))
    plot_pf_ess_show = bool(exp_cfg.get("plot_pf_ess_show", False))
    plot_pf_obs_index = exp_cfg.get("plot_pf_obs_index")
    plot_pf_unobs_index = exp_cfg.get("plot_pf_unobs_index")
    plot_stability = bool(exp_cfg.get("plot_stability", False))
    plot_stability_seed0_only = bool(exp_cfg.get("plot_stability_seed0_only", True))
    plot_stability_show = bool(exp_cfg.get("plot_stability_show", False))
    plot_stability_percentiles = exp_cfg.get("plot_stability_percentiles")
    if plot_stability_percentiles is None:
        plot_stability_percentiles = (25.0, 75.0)
    else:
        vals = (
            list(plot_stability_percentiles)
            if isinstance(plot_stability_percentiles, (list, tuple))
            else []
        )
        if len(vals) >= 2:
            plot_stability_percentiles = (float(vals[0]), float(vals[1]))
        else:
            plot_stability_percentiles = (25.0, 75.0)
    if plot_kflow_time is None:
        plot_kflow_time = min(T - 1, 19)
    plot_kflow_time = int(plot_kflow_time)
    if T > 0:
        plot_kflow_time = max(0, min(plot_kflow_time, T - 1))
    if plot_pf_degeneracy_time is None:
        plot_pf_degeneracy_time = min(T - 1, 19)
    plot_pf_degeneracy_time = int(plot_pf_degeneracy_time)
    if T > 0:
        plot_pf_degeneracy_time = max(0, min(plot_pf_degeneracy_time, T - 1))

    alpha = float(model_cfg.get("alpha", 0.9))
    alpha0 = float(model_cfg.get("alpha0", 3.0))
    alpha1 = float(model_cfg.get("alpha1", 0.01))
    beta = float(model_cfg.get("beta", 20.0))
    sigma_z_values = [float(v) for v in _as_list(model_cfg.get("sigma_z", [0.5]))]
    p0_scale = float(model_cfg.get("p0_scale", 1e-3))
    for dx in dims:
        if dy_values:
            dy_candidates = [int(dy) for dy in dy_values]
        else:
            stride = int(round(1.0 / dy_ratio)) if dy_ratio > 0 else 4
            dy_candidates = [max(1, dx // max(1, stride))]
        for dy in dy_candidates:
            for sigma_z in sigma_z_values:
                params = build_params(
                    dx,
                    dy,
                    alpha=alpha,
                    sigma_z=sigma_z,
                    alpha0=alpha0,
                    alpha1=alpha1,
                    beta=beta,
                    p0_scale=p0_scale,
                )
                for N_pf in pf_particles:
                    for N_flow in flow_particles:
                        cfg_tag = tag_from_cfg(
                            {
                                "dx": dx,
                                "dy": dy,
                                "sigma_z": sigma_z,
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

                            if plot_kflow_marginals and (not plot_kflow_seed0_only or seed == seeds[0]):
                                contour_key = plot_kflow_contour_method
                                if contour_key not in outputs:
                                    print(
                                        f"plot_kflow_marginals: contour method '{contour_key}' not found; skipping."
                                    )
                                else:
                                    diag_key = _find_method_key(
                                        outputs,
                                        "kflow_diag",
                                        "kflow_diag",
                                    )
                                    scalar_key = _find_method_key(
                                        outputs,
                                        "kflow_scalar",
                                        "kflow_scalar",
                                    )
                                    if diag_key is None or scalar_key is None:
                                        print(
                                            "plot_kflow_marginals: missing kflow_diag/kflow_scalar outputs; skipping."
                                        )
                                    else:
                                        obs_dim, unobs_dim = _select_plot_dims(
                                            params["obs_idx"],
                                            dx,
                                            plot_kflow_obs_index,
                                            plot_kflow_unobs_index,
                                        )
                                        if obs_dim is None or unobs_dim is None:
                                            print(
                                                "plot_kflow_marginals: no unobserved dimension available; skipping."
                                            )
                                        else:
                                            plot_path = per_seed_dir / (
                                                f"kflow_marginals_{contour_key}_t{plot_kflow_time}.png"
                                            )
                                            plot_title = (
                                                f"exp1_linear_gaussian {cfg_tag} seed{seed} t={plot_kflow_time}"
                                            )
                                            _plot_kflow_marginals(
                                                plot_path,
                                                contour_method=contour_key,
                                                gaussian_out=outputs[contour_key],
                                                kflow_diag_out=outputs[diag_key],
                                                kflow_scalar_out=outputs[scalar_key],
                                                obs_dim=obs_dim,
                                                unobs_dim=unobs_dim,
                                                t_index=plot_kflow_time,
                                                show=plot_kflow_show,
                                                plot_prior=plot_kflow_prior,
                                                title=plot_title,
                                            )

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

                                if method.startswith("pf"):
                                    if plot_pf_ess and (
                                        not plot_pf_ess_seed0_only or seed == seeds[0]
                                    ):
                                        w = out.get("w")
                                        if w is not None:
                                            plot_path = method_dir / "pf_ess_over_time.png"
                                            _plot_ess_over_time(
                                                plot_path,
                                                w,
                                                ess_threshold=pf_ess_threshold,
                                                show=plot_pf_ess_show,
                                            )
                                    if plot_pf_degeneracy and (
                                        not plot_pf_degeneracy_seed0_only or seed == seeds[0]
                                    ):
                                        obs_dim, unobs_dim = _select_plot_dims(
                                            params["obs_idx"],
                                            dx,
                                            plot_pf_obs_index,
                                            plot_pf_unobs_index,
                                        )
                                        if obs_dim is None or unobs_dim is None:
                                            print(
                                                "plot_pf_degeneracy: no unobserved dimension available; skipping."
                                            )
                                        else:
                                            plot_path = method_dir / (
                                                f"pf_degeneracy_t{plot_pf_degeneracy_time}.png"
                                            )
                                            _plot_pf_degeneracy(
                                                plot_path,
                                                pf_out=out,
                                                gaussian_out=kf_out,
                                                x_true=x_true,
                                                obs_dim=obs_dim,
                                                unobs_dim=unobs_dim,
                                                t_index=plot_pf_degeneracy_time,
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
                                w = out.get("w")
                                if w is not None and not out.get("is_gaussian", False):
                                    ess_t = _ess_from_weights(np.asarray(w))
                                    if ess_t is not None:
                                        diag["ess_t"] = ess_t
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
