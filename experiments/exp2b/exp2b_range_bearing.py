from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.common.exp_helper import (
    SUMMARY_KEYS,
    aggregate_metrics_by_method,
    print_method_summary_table,
    print_metrics_compare,
    print_separator,
    record_metrics,
)
from experiments.common.exp_utils import (
    as_list,
    build_init_dist,
    cfg_section,
    cfg_subsection,
    ess_threshold_for_method,
    ensure_dir,
    ess_from_weights,
    expand_sweep_values,
    first_non_null,
    get_summary_keys_and_prefixes,
    is_particle_like_method,
    load_config,
    parse_percentile_band,
    parse_positive_int_or_none,
    particle_pairs,
    resolve_filter_model_cfg,
    resolve_plot_controls,
    resolve_optional_float_list,
    save_npz,
    select_pre_resample_weights,
    set_seed,
    tag_from_cfg,
)
from experiments.common.filter_cfg import build_filter_cfg
from experiments.common.plot_utils import plot_stability_over_time
from experiments.common.runner import run_filter
from src.metrics import rmse
from src.motion_model import ConstantVelocityMotionModel
from src.ssm import RangeBearingSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp2b_config.yaml")

_STABILITY_KEY_LABELS = {
    "condInfo_log10": "cond_Info_log10",
    "condA_log10_max": "cond_A_log10",
    "condJ_log10_max": "cond_J_log10",
    "logdetJ": "logdet_J",
    "logdet_cov": "logdet_cov",
    "condH_log10_max": "condH_log10",
    "condK_log10_max": "condK_log10",
    "flow_norm_mean_max": "flow_norm_mean",
}

_DEFAULT_STABILITY_KEYS = (
    "condInfo_log10",
    "condA_log10_max",
    "condJ_log10_max",
    "logdetJ",
    "logdet_cov",
    "condH_log10_max",
    "condK_log10_max",
    "flow_norm_mean_max",
)


def build_ssm(
    sigma_r: float,
    sigma_theta: float,
    dt: float,
    cov_eps_x: np.ndarray,
    jitter: float,
    seed: int,
    jacobian_r_min: Optional[float] = None,
) -> RangeBearingSSM:
    motion_model = ConstantVelocityMotionModel(
        dt=dt,
        cov_eps=cov_eps_x,
        seed=seed,
        jitter=jitter,
    )
    cov_eps_y = np.diag([sigma_r**2, sigma_theta ** 2]).astype(np.float32)
    return RangeBearingSSM(
        motion_model=motion_model,
        cov_eps_y=cov_eps_y,
        jitter=jitter,
        seed=seed,
        jacobian_r_min=jacobian_r_min,
    )


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


def build_initial_cov_filter(init_cfg: Dict[str, Any]) -> np.ndarray:
    """Optional filter-specific initial covariance."""
    P0 = init_cfg.get("P0_filter")
    if P0 is not None:
        return np.array(P0, dtype=np.float32)
    diag = init_cfg.get("P0_diag_filter")
    if diag is not None:
        diag = np.array(diag, dtype=np.float32)
        return np.diag(diag)
    return build_initial_cov(init_cfg)


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


def _plot_ess_over_time(
    path: Path,
    w: np.ndarray,
    ess_threshold: Optional[float] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    ess_t = ess_from_weights(w)
    if ess_t is None:
        return
    ess_np = np.asarray(ess_t)
    if ess_np.ndim == 1:
        ess_np = ess_np[np.newaxis, :]
    T = ess_np.shape[1]
    t = np.arange(T)
    ess_mean = np.mean(ess_np, axis=0)
    ess_min = np.min(ess_np, axis=0)
    ess_max = np.max(ess_np, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(t, ess_mean, color="C0", linewidth=1.6, label="ESS mean")
    if ess_np.shape[0] > 1:
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


def _flow_compare_group(method: str) -> Optional[Tuple[str, str]]:
    name = str(method).lower()
    if name.startswith("edh"):
        family = "edh"
    elif name.startswith("ledh"):
        family = "ledh"
    else:
        return None

    variant = ""
    if "(" in name and name.endswith(")"):
        variant = name[name.find("(") + 1 : -1].strip()
    if variant == "opt":
        subgroup = ""
    elif variant.endswith("_opt"):
        subgroup = variant[:-4]
    else:
        subgroup = variant
    return family, subgroup


def _flow_variant(method: str) -> str:
    name = str(method).lower()
    if "(" in name and name.endswith(")"):
        return name[name.find("(") + 1 : -1].strip()
    return ""


def _is_opt_flow_variant(method: str) -> bool:
    variant = _flow_variant(method)
    return variant == "opt" or variant.endswith("_opt")


def _stability_series_with_band(
    values: np.ndarray,
    band_percentiles: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr, None, None
    flat = arr.reshape(-1, arr.shape[-1])
    center = np.median(flat, axis=0)
    if band_percentiles is None:
        return center, None, None
    p_lo, p_hi = band_percentiles
    lo = np.percentile(flat, p_lo, axis=0)
    hi = np.percentile(flat, p_hi, axis=0)
    return center, lo, hi


def _schedule_series_with_band(
    values: np.ndarray,
    band_percentiles: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        center = np.asarray([float(arr)], dtype=np.float64)
        return center, None, None
    if arr.ndim == 1:
        flat = arr[np.newaxis, :]
    elif arr.ndim == 2:
        # Prefer [B, L]; fallback to transpose when first axis appears to be lambda.
        flat = arr if arr.shape[0] <= arr.shape[1] else arr.T
    else:
        # Flow diagnostics stack schedule vectors as [B, L, T].
        if arr.ndim == 3 and arr.shape[1] >= arr.shape[2]:
            arr_l = np.transpose(arr, (0, 2, 1))  # [B, T, L]
        else:
            # Fallback: choose the largest non-batch axis as lambda axis.
            lam_axis = 1 + int(np.argmax(arr.shape[1:])) if arr.ndim > 1 else 0
            arr_l = np.moveaxis(arr, lam_axis, -1)
        flat = arr_l.reshape(-1, arr_l.shape[-1])
    center = np.median(flat, axis=0)
    if band_percentiles is None:
        return center, None, None
    p_lo, p_hi = band_percentiles
    lo = np.percentile(flat, p_lo, axis=0)
    hi = np.percentile(flat, p_hi, axis=0)
    return center, lo, hi


def _schedule_time_len(values: np.ndarray) -> int:
    arr = np.asarray(values)
    if arr.ndim >= 3:
        return int(arr.shape[-1])
    return 1


def _schedule_series_at_time_with_band(
    values: np.ndarray,
    time_index: int,
    band_percentiles: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract lambda-series at a specific time index and aggregate over batch."""
    arr = np.asarray(values)
    if arr.ndim >= 3:
        t = max(0, min(int(time_index), int(arr.shape[-1]) - 1))
        arr_t = arr[..., t]
    else:
        arr_t = arr
    arr_t = np.asarray(arr_t)
    if arr_t.ndim == 1:
        flat = arr_t[np.newaxis, :]
    elif arr_t.ndim == 2:
        flat = arr_t
    else:
        flat = arr_t.reshape(-1, arr_t.shape[-1])
    center = np.median(flat, axis=0)
    if band_percentiles is None:
        return center, None, None
    p_lo, p_hi = band_percentiles
    lo = np.percentile(flat, p_lo, axis=0)
    hi = np.percentile(flat, p_hi, axis=0)
    return center, lo, hi


def _plot_beta_schedule_compare_groups(
    output_dir: Path,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    """Render Dai22-style beta-schedule comparison panels for flow method pairs."""
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[str, str], List[str]] = {}
    for method in method_order:
        group = _flow_compare_group(method)
        if group is None:
            continue
        grouped.setdefault(group, []).append(method)

    for (family, subgroup), methods in grouped.items():
        if len(methods) < 2:
            continue
        base_methods = [m for m in methods if not _is_opt_flow_variant(m)]
        opt_methods = [m for m in methods if _is_opt_flow_variant(m)]
        if not base_methods or not opt_methods:
            continue
        method_base = base_methods[0]
        method_opt = opt_methods[0]

        diag_base = outputs.get(method_base, {}).get("diagnostics", {})
        diag_opt = outputs.get(method_opt, {}).get("diagnostics", {})
        if not isinstance(diag_base, dict) or not isinstance(diag_opt, dict):
            continue

        beta_base_v = diag_base.get("beta_sched")
        beta_opt_v = diag_opt.get("beta_sched")
        beta_dot_base_v = diag_base.get("beta_dot_sched")
        beta_dot_opt_v = diag_opt.get("beta_dot_sched")
        if any(v is None for v in (beta_base_v, beta_opt_v, beta_dot_base_v, beta_dot_opt_v)):
            continue

        T_sched = min(
            _schedule_time_len(np.asarray(beta_base_v)),
            _schedule_time_len(np.asarray(beta_opt_v)),
            _schedule_time_len(np.asarray(beta_dot_base_v)),
            _schedule_time_len(np.asarray(beta_dot_opt_v)),
        )
        if T_sched <= 0:
            continue
        time_indices: List[int] = []
        for t in (1, 40, 70):
            t = int(t)
            if 0 <= t < T_sched and t not in time_indices:
                time_indices.append(t)

        group_tag = family if subgroup == "" else f"{family}_{subgroup}"
        for t_sel in time_indices:
            beta_base, beta_base_lo, beta_base_hi = _schedule_series_at_time_with_band(
                np.asarray(beta_base_v),
                t_sel,
                band_percentiles,
            )
            beta_opt, beta_opt_lo, beta_opt_hi = _schedule_series_at_time_with_band(
                np.asarray(beta_opt_v),
                t_sel,
                band_percentiles,
            )
            beta_dot_base, beta_dot_base_lo, beta_dot_base_hi = _schedule_series_at_time_with_band(
                np.asarray(beta_dot_base_v),
                t_sel,
                band_percentiles,
            )
            beta_dot_opt, beta_dot_opt_lo, beta_dot_opt_hi = _schedule_series_at_time_with_band(
                np.asarray(beta_dot_opt_v),
                t_sel,
                band_percentiles,
            )
            lengths = [
                beta_base.shape[0],
                beta_opt.shape[0],
                beta_dot_base.shape[0],
                beta_dot_opt.shape[0],
            ]
            L = int(min(lengths))
            if L <= 0:
                continue
            beta_base = beta_base[:L]
            beta_opt = beta_opt[:L]
            beta_dot_base = beta_dot_base[:L]
            beta_dot_opt = beta_dot_opt[:L]
            if beta_base_lo is not None and beta_base_hi is not None:
                beta_base_lo = beta_base_lo[:L]
                beta_base_hi = beta_base_hi[:L]
            if beta_opt_lo is not None and beta_opt_hi is not None:
                beta_opt_lo = beta_opt_lo[:L]
                beta_opt_hi = beta_opt_hi[:L]
            if beta_dot_base_lo is not None and beta_dot_base_hi is not None:
                beta_dot_base_lo = beta_dot_base_lo[:L]
                beta_dot_base_hi = beta_dot_base_hi[:L]
            if beta_dot_opt_lo is not None and beta_dot_opt_hi is not None:
                beta_dot_opt_lo = beta_dot_opt_lo[:L]
                beta_dot_opt_hi = beta_dot_opt_hi[:L]

            lam = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float64)

            fig, axes = plt.subplots(2, 2, figsize=(9, 7))

            ax = axes[0, 0]
            ax.plot(lam, beta_base, label=f"{method_base}", color="C0", linestyle="--")
            ax.plot(lam, beta_opt, label=f"{method_opt}", color="C1")
            if beta_base_lo is not None and beta_base_hi is not None:
                ax.fill_between(lam, beta_base_lo, beta_base_hi, color="C0", alpha=0.15, linewidth=0)
            if beta_opt_lo is not None and beta_opt_hi is not None:
                ax.fill_between(lam, beta_opt_lo, beta_opt_hi, color="C1", alpha=0.15, linewidth=0)
            ax.set_xlabel("lambda")
            ax.set_ylabel("beta(lambda)")
            ax.grid(True, linestyle=":")
            ax.legend(fontsize=8, loc="best")

            ax = axes[0, 1]
            ax.plot(lam, beta_opt - beta_base, color="C1")
            ax.set_xlabel("lambda")
            ax.set_ylabel("e(lambda)=beta_opt-beta_base")
            ax.grid(True, linestyle=":")

            ax = axes[1, 0]
            ax.plot(lam, beta_dot_base, label=f"{method_base}", color="C0", linestyle="--")
            ax.plot(lam, beta_dot_opt, label=f"{method_opt}", color="C1")
            if beta_dot_base_lo is not None and beta_dot_base_hi is not None:
                ax.fill_between(
                    lam,
                    beta_dot_base_lo,
                    beta_dot_base_hi,
                    color="C0",
                    alpha=0.15,
                    linewidth=0,
                )
            if beta_dot_opt_lo is not None and beta_dot_opt_hi is not None:
                ax.fill_between(
                    lam,
                    beta_dot_opt_lo,
                    beta_dot_opt_hi,
                    color="C1",
                    alpha=0.15,
                    linewidth=0,
                )
            ax.set_xlabel("lambda")
            ax.set_ylabel("beta_dot(lambda)")
            ax.grid(True, linestyle=":")
            ax.legend(fontsize=8, loc="best")

            ax = axes[1, 1]
            stiff_base_v = diag_base.get("condF_sched")
            stiff_opt_v = diag_opt.get("condF_sched")
            if stiff_base_v is not None and stiff_opt_v is not None:
                stiff_base, _, _ = _schedule_series_at_time_with_band(
                    np.asarray(stiff_base_v),
                    t_sel,
                    band_percentiles,
                )
                stiff_opt, _, _ = _schedule_series_at_time_with_band(
                    np.asarray(stiff_opt_v),
                    t_sel,
                    band_percentiles,
                )
                Ls = int(min(L, stiff_base.shape[0], stiff_opt.shape[0]))
                lam_s = lam[:Ls]
                ax.plot(lam_s, stiff_base[:Ls], label=f"{method_base}", color="C0", linestyle="--")
                ax.plot(lam_s, stiff_opt[:Ls], label=f"{method_opt}", color="C1")
                ax.set_yscale("log")
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "condF_sched unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_xlabel("lambda")
            ax.set_ylabel("R_stiff (condF)")
            ax.grid(True, linestyle=":")

            fig.suptitle(f"beta_schedule_compare_{group_tag} (t={t_sel})")
            fig.tight_layout()
            fig.savefig(output_dir / f"beta_schedule_compare_{group_tag}_t{t_sel}.png", dpi=150)
            np.savez_compressed(
                output_dir / f"beta_schedule_compare_{group_tag}_t{t_sel}.npz",
                t_index=np.array([t_sel], dtype=np.int32),
                lambda_grid=lam,
                beta_base=beta_base,
                beta_opt=beta_opt,
                beta_dot_base=beta_dot_base,
                beta_dot_opt=beta_dot_opt,
            )
            if show:
                plt.show()
            plt.close(fig)


def _plot_stability_compare_groups(
    output_dir: Path,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    keys: Optional[List[str]] = None,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    key_order = [str(k) for k in (keys if keys else _DEFAULT_STABILITY_KEYS)]
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for method in method_order:
        group = _flow_compare_group(method)
        if group is None:
            continue
        grouped.setdefault(group, []).append(method)

    for (family, subgroup), methods in grouped.items():
        if len(methods) < 2:
            continue
        group_tag = family if subgroup == "" else f"{family}_{subgroup}"
        for key in key_order:
            series_items: List[Tuple[str, np.ndarray]] = []
            for method in methods:
                diag = outputs.get(method, {}).get("diagnostics", {})
                if not isinstance(diag, dict):
                    continue
                val = diag.get(key)
                if val is None:
                    continue
                series_items.append((method, np.asarray(val)))
            if len(series_items) < 2:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for method, series in series_items:
                mean, lo, hi = _stability_series_with_band(series, band_percentiles)
                t = np.arange(mean.shape[0])
                ax.plot(t, mean, linewidth=1.6, label=method)
                if lo is not None and hi is not None:
                    ax.fill_between(t, lo, hi, alpha=0.15, linewidth=0)

            label = _STABILITY_KEY_LABELS.get(key, key)
            ax.set_xlabel("time")
            ax.grid(True, linestyle=":")
            ax.set_title(f"stability_compare_{group_tag}_{label}")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / f"stability_compare_{group_tag}_{label}.png", dpi=150)
            if show:
                plt.show()
            plt.close(fig)


def _plot_stability_compare_groups_panel(
    output_dir: Path,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    keys: Optional[List[str]] = None,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    """Render Dai22-style 2x2 stability panels for method pairs."""
    import matplotlib.pyplot as plt

    default_panel_keys = [
        "condInfo_log10",
        "condA_log10_max",
        "condJ_log10_max",
        "logdetJ",
    ]
    key_order_all = [str(k) for k in (keys if keys else _DEFAULT_STABILITY_KEYS)]
    panel_keys = [k for k in default_panel_keys if k in key_order_all]
    if len(panel_keys) < 2:
        return

    grouped: Dict[Tuple[str, str], List[str]] = {}
    for method in method_order:
        group = _flow_compare_group(method)
        if group is None:
            continue
        grouped.setdefault(group, []).append(method)

    for (family, subgroup), methods in grouped.items():
        if len(methods) < 2:
            continue
        group_tag = family if subgroup == "" else f"{family}_{subgroup}"

        # Keep only keys that are available for at least two methods in this group.
        keys_used: List[str] = []
        for key in panel_keys:
            count = 0
            for method in methods:
                diag = outputs.get(method, {}).get("diagnostics", {})
                if isinstance(diag, dict) and diag.get(key) is not None:
                    count += 1
            if count >= 2:
                keys_used.append(key)
        if len(keys_used) < 2:
            continue

        n = min(4, len(keys_used))
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(9, 7))
        axes_flat = axes.reshape(-1)

        for i in range(4):
            ax = axes_flat[i]
            if i >= n:
                ax.axis("off")
                continue
            key = keys_used[i]
            drew_any = False
            for method in methods:
                diag = outputs.get(method, {}).get("diagnostics", {})
                if not isinstance(diag, dict):
                    continue
                val = diag.get(key)
                if val is None:
                    continue
                mean, lo, hi = _stability_series_with_band(np.asarray(val), band_percentiles)
                t = np.arange(mean.shape[0])
                ax.plot(t, mean, linewidth=1.6, label=method)
                if lo is not None and hi is not None:
                    ax.fill_between(t, lo, hi, alpha=0.15, linewidth=0)
                drew_any = True
            if not drew_any:
                ax.axis("off")
                continue
            ax.set_xlabel("time")
            ax.set_ylabel(_STABILITY_KEY_LABELS.get(key, key))
            ax.grid(True, linestyle=":")
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"stability_compare_{group_tag}_panel")
        fig.tight_layout()
        fig.savefig(output_dir / f"stability_compare_{group_tag}_panel.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)


# ess_from_weights moved to exp_utils.ess_from_weights


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




def _select_pre_resample_particles(out: Dict[str, Any]) -> Optional[np.ndarray]:
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    x_pre = diagnostics.get("x_pre")
    if x_pre is None:
        x_pre = diagnostics.get("x_pred")
    if x_pre is None:
        x_pre = out.get("x_particles") if isinstance(out, dict) else None
    return x_pre




# Common config helpers are imported from exp_utils.


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


def _resolve_model_cov(model_cfg: Dict[str, Any]) -> np.ndarray:
    cov_eps_cfg = model_cfg.get("cov_eps")
    if cov_eps_cfg is not None:
        cov_eps_x = np.array(cov_eps_cfg, dtype=np.float32)
        if cov_eps_x.shape != (4, 4):
            raise ValueError("model.cov_eps must be a 4x4 matrix for range-bearing CV model")
        return cov_eps_x
    if "q_scale_v" not in model_cfg:
        raise ValueError("Set model.cov_eps (4x4) or model.q_scale_v for range-bearing CV model")
    q_scale_v = float(model_cfg.get("q_scale_v", 0.2))
    return np.diag([0.0, 0.0, q_scale_v**2, q_scale_v**2]).astype(np.float32)


def _resolve_filter_cov(
    model_cfg: Dict[str, Any],
    model_filter_cfg: Dict[str, Any],
    cov_eps_x: np.ndarray,
) -> np.ndarray:
    cov_eps_filter_cfg = first_non_null(
        model_filter_cfg.get("cov_eps"),
        model_filter_cfg.get("cov_eps_x"),
        model_cfg.get("cov_eps_filter"),
        model_cfg.get("cov_eps_x_filter"),
    )
    if cov_eps_filter_cfg is None:
        return cov_eps_x
    cov_eps_x_filter = np.array(cov_eps_filter_cfg, dtype=np.float32)
    if cov_eps_x_filter.shape != (4, 4):
        raise ValueError("filters.model.cov_eps must be a 4x4 matrix for range-bearing CV model")
    return cov_eps_x_filter


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, args.overrides)

    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    filters_cfg = cfg_section(cfg, "filters")
    model_filter_cfg = resolve_filter_model_cfg(cfg, filters_cfg)
    pf_cfg = cfg_subsection(filters_cfg, "pf", "filters")
    flow_cfg = cfg_subsection(filters_cfg, "flow", "filters")
    spf_cfg = cfg_subsection(filters_cfg, "stochastic_pf", "filters")
    kflow_cfg = cfg_subsection(filters_cfg, "kflow", "filters")
    ukf_cfg = cfg_subsection(filters_cfg, "ukf", "filters")
    init_cfg = cfg_section(cfg, "init")

    out_root = Path(exp_cfg.get("output_root", "results/exp2b_range_bearing"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 80))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    save_particles_seed0 = bool(exp_cfg.get("save_particles_seed0", True))
    plot_controls = resolve_plot_controls(
        exp_cfg,
        [
            ("plot_state", "plot_seed0_only", None),
            ("plot_pf_ess", "plot_pf_ess_seed0_only", "plot_pf_ess_show"),
            ("plot_pf_degeneracy", "plot_pf_degeneracy_seed0_only", "plot_pf_degeneracy_show"),
            ("plot_stability", "plot_stability_seed0_only", "plot_stability_show"),
        ],
    )
    plot_state = plot_controls["plot_state"]
    plot_seed0_only = plot_controls["plot_seed0_only"]
    show_plots = plot_controls["show_plots"]
    plot_interactive = plot_controls["plot_interactive"]
    plot_pf_ess = plot_controls["plot_pf_ess"]
    plot_pf_ess_seed0_only = plot_controls["plot_pf_ess_seed0_only"]
    plot_pf_ess_show = plot_controls["plot_pf_ess_show"]
    plot_pf_degeneracy = plot_controls["plot_pf_degeneracy"]
    plot_pf_degeneracy_seed0_only = plot_controls["plot_pf_degeneracy_seed0_only"]
    plot_pf_degeneracy_show = plot_controls["plot_pf_degeneracy_show"]
    plot_stability = plot_controls["plot_stability"]
    plot_stability_seed0_only = plot_controls["plot_stability_seed0_only"]
    plot_stability_show = plot_controls["plot_stability_show"]
    plot_time_gap = parse_positive_int_or_none(exp_cfg.get("plot_time_gap"))
    plot_pf_degeneracy_times = [0, 9, 29]
    plot_pf_degeneracy_times = [t for t in plot_pf_degeneracy_times if 0 <= t < T]
    plot_stability_percentiles = parse_percentile_band(
        exp_cfg.get("plot_stability_percentiles"),
        (25.0, 75.0),
    )
    plot_stability_keys_cfg = [str(k) for k in as_list(exp_cfg.get("plot_stability_keys"))]
    plot_stability_keys = plot_stability_keys_cfg or None

    summary_keys, exclude_prefixes = get_summary_keys_and_prefixes(exp_cfg, SUMMARY_KEYS)

    distances = [float(d) for d in as_list(model_cfg.get("distances", [0.5, 2.0, 10.0]))]
    sigma_thetas = [float(s) for s in as_list(model_cfg.get("sigma_thetas", [1.0, 5.0, 15.0]))]
    sigma_rs = [float(s) for s in as_list(model_cfg.get("sigma_rs", [0.05, 0.2, 1.0]))]
    sigma_thetas_filter = resolve_optional_float_list(
        first_non_null(
            model_filter_cfg.get("sigma_thetas"),
            model_filter_cfg.get("sigma_theta"),
            model_cfg.get("sigma_thetas_filter"),
            model_cfg.get("sigma_theta_filter"),
        )
    )
    sigma_rs_filter = resolve_optional_float_list(
        first_non_null(
            model_filter_cfg.get("sigma_rs"),
            model_filter_cfg.get("sigma_r"),
            model_cfg.get("sigma_rs_filter"),
            model_cfg.get("sigma_r_filter"),
        )
    )
    sigma_thetas_filter = expand_sweep_values(
        sigma_thetas_filter,
        sigma_thetas,
        "filters.model.sigma_thetas",
    )
    sigma_rs_filter = expand_sweep_values(
        sigma_rs_filter,
        sigma_rs,
        "filters.model.sigma_rs",
    )
    dt = float(model_cfg.get("dt", 1.0))
    jitter = float(model_cfg.get("jitter", 1e-12))
    jacobian_r_min = model_cfg.get("jacobian_r_min")
    if jacobian_r_min is not None:
        jacobian_r_min = float(jacobian_r_min)
    cov_eps_x = _resolve_model_cov(model_cfg)
    cov_eps_x_filter = _resolve_filter_cov(model_cfg, model_filter_cfg, cov_eps_x)
    jacobian_r_min_filter = model_filter_cfg.get("jacobian_r_min", jacobian_r_min)
    if jacobian_r_min_filter is not None:
        jacobian_r_min_filter = float(jacobian_r_min_filter)

    methods_cfg = filters_cfg.get("methods")
    base_methods = [
        str(m).lower()
        for m in as_list(
            methods_cfg
            if methods_cfg is not None
            else [
                "ekf",
                "ukf",
                "pf",
                "edh",
                "edh(pfpf)",
                "ledh",
                "ledh(pfpf)",
                "kflow_scalar",
                "kflow_diag",
            ]
        )
    ]
    pf_particles = [int(n) for n in as_list(pf_cfg.get("num_particles", [200]))]
    flow_particles = [int(n) for n in as_list(flow_cfg.get("num_particles", [200]))]

    num_lambda_flow = int(flow_cfg.get("num_lambda", 20))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    pairs_list = particle_pairs(pf_particles, flow_particles, pair_particles)

    for dist in distances:
        x0 = build_initial_state(dist, init_cfg)
        m0 = build_initial_mean(x0, init_cfg)
        P0 = build_initial_cov(init_cfg)
        P0_filter = build_initial_cov_filter(init_cfg)
        init_dist = build_init_dist(m0, P0_filter)
        for sigma_theta_idx, sigma_theta in enumerate(sigma_thetas):
            for sigma_r_idx, sigma_r in enumerate(sigma_rs):
                sigma_theta_filter = sigma_thetas_filter[sigma_theta_idx]
                sigma_r_filter = sigma_rs_filter[sigma_r_idx]
                for N_pf, N_flow in pairs_list:
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
                        flow_cfg=flow_cfg,
                        kflow_cfg=kflow_cfg,
                        stochastic_pf_cfg=spf_cfg,
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
                            n: build_init_particles(m0, P0_filter, n, batch_size, seed)
                            for n in particle_counts
                        }
                        sim_ssm = build_ssm(
                            sigma_r=sigma_r,
                            sigma_theta=np.deg2rad(sigma_theta),
                            dt=dt,
                            cov_eps_x=cov_eps_x,
                            jitter=jitter,
                            seed=seed,
                            jacobian_r_min=jacobian_r_min,
                        )
                        filter_ssm = build_ssm(
                            sigma_r=sigma_r_filter,
                            sigma_theta=np.deg2rad(sigma_theta_filter),
                            dt=dt,
                            cov_eps_x=cov_eps_x_filter,
                            jitter=jitter,
                            seed=seed,
                            jacobian_r_min=jacobian_r_min_filter,
                        )
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,), x0=x0)
                        rng_state = filter_ssm.rng.state.read_value()

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            filter_ssm.rng.state.assign(rng_state)
                            method_ssm = filter_ssm
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "kalman", "ekf", "ukf"):
                                method_cfg["m0"] = m0
                                method_cfg["P0"] = P0_filter
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

                            if is_particle_like_method(method):
                                if plot_pf_ess and (
                                    not plot_pf_ess_seed0_only or seed == seeds[0]
                                ):
                                    w_pre = select_pre_resample_weights(out)
                                    if w_pre is not None:
                                        ess_threshold = ess_threshold_for_method(
                                            method,
                                            pf_ess_threshold,
                                            flow_ess_threshold,
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
                                    w_pre = select_pre_resample_weights(out)
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
                            w_pre = select_pre_resample_weights(out)
                            if w_pre is None:
                                w_pre = out.get("w")
                            if w_pre is not None and not out.get("is_gaussian", False):
                                ess_t = ess_from_weights(np.asarray(w_pre))
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
                                    plot_stability_over_time(
                                        method_dir,
                                        diag_src,
                                        band_percentiles=plot_stability_percentiles,
                                        show=plot_stability_show,
                                        keys=plot_stability_keys,
                                    )

                        if plot_stability and (
                            not plot_stability_seed0_only or seed == seeds[0]
                        ):
                            _plot_stability_compare_groups(
                                per_seed_dir,
                                outputs,
                                methods,
                                keys=plot_stability_keys,
                                band_percentiles=plot_stability_percentiles,
                                show=plot_stability_show,
                            )
                            _plot_stability_compare_groups_panel(
                                per_seed_dir,
                                outputs,
                                methods,
                                keys=plot_stability_keys,
                                band_percentiles=plot_stability_percentiles,
                                show=plot_stability_show,
                            )
                            _plot_beta_schedule_compare_groups(
                                per_seed_dir,
                                outputs,
                                methods,
                                band_percentiles=plot_stability_percentiles,
                                show=plot_stability_show,
                            )

                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} summary")
                        print_method_summary_table(
                            metrics_by_method,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} compare")
                        print_metrics_compare(
                            metrics_by_method,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )

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
                        print_method_summary_table(
                            mean_metrics,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2b_range_bearing {cfg_tag} avg compare")
                        print_metrics_compare(
                            mean_metrics,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )


if __name__ == "__main__":
    main()
