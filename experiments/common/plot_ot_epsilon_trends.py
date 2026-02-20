from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OT epsilon trends from tuning results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/ot_epsilon_tuning_seed0_only_v3/exp3_ot_epsilon_search_results.json"),
        help="Path to exp3_ot_epsilon_search_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ot_epsilon_tuning_seed0_only_v3/ot_epsilon_trends.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=("rmse", "loss"),
        default="rmse",
        help="Tie-break objective when multiple records share same epsilon and steps.",
    )
    parser.add_argument(
        "--zoom-eps-max",
        type=float,
        default=1.0,
        help=(
            "Upper epsilon bound for the zoomed panels. "
            "Set <=0 to disable zoom and plot only full-range panels."
        ),
    )
    parser.add_argument(
        "--zoom-quantiles",
        type=float,
        nargs=2,
        default=[5.0, 95.0],
        metavar=("Q_LOW", "Q_HIGH"),
        help="Quantile range used to set robust y-limits on zoomed panels.",
    )
    return parser.parse_args()


def _best_per_epsilon(records: List[Dict[str, Any]], objective: str) -> List[Dict[str, Any]]:
    if objective not in ("rmse", "loss"):
        raise ValueError("objective must be either 'rmse' or 'loss'.")
    primary_key = "rmse_mean" if objective == "rmse" else "loss_mean"
    secondary_key = "loss_mean" if objective == "rmse" else "rmse_mean"
    by_eps: Dict[float, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        eps = float(rec["epsilon"])
        curr = by_eps.get(eps)
        # Prefer higher-step evaluations; if tie, prefer lower rmse.
        if curr is None:
            by_eps[eps] = rec
            continue
        steps_curr = int(curr.get("steps", 0))
        steps_new = int(rec.get("steps", 0))
        if steps_new > steps_curr:
            by_eps[eps] = rec
            continue
        if steps_new == steps_curr and (
            float(rec.get(primary_key, np.inf)),
            float(rec.get(secondary_key, np.inf)),
        ) < (
            float(curr.get(primary_key, np.inf)),
            float(curr.get(secondary_key, np.inf)),
        ):
            by_eps[eps] = rec
    return sorted(by_eps.values(), key=lambda r: float(r["epsilon"]))


def _robust_ylim(values: np.ndarray, q_low: float, q_high: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    if arr.size == 1:
        center = float(arr[0])
        pad = max(1e-6, abs(center) * 0.05)
        return (center - pad, center + pad)
    lo = float(np.percentile(arr, q_low))
    hi = float(np.percentile(arr, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if hi <= lo:
        hi = lo + max(1e-6, abs(lo) * 0.05)
    pad = 0.10 * (hi - lo)
    return (lo - pad, hi + pad)


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    selected = _best_per_epsilon(records, objective=str(args.objective))
    if not selected:
        raise ValueError(f"No successful records in {args.input}")

    eps = np.asarray([float(r["epsilon"]) for r in selected], dtype=np.float64)
    runtime = np.asarray([float(r["elapsed_sec"]) for r in selected], dtype=np.float64)
    loss = np.asarray([float(r["loss_mean"]) for r in selected], dtype=np.float64)
    rmse = np.asarray([float(r["rmse_mean"]) for r in selected], dtype=np.float64)

    zoom_enabled = float(args.zoom_eps_max) > 0.0
    q_low, q_high = float(args.zoom_quantiles[0]), float(args.zoom_quantiles[1])

    if zoom_enabled:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
        ax_runtime_full, ax_runtime_zoom = axes[0, 0], axes[0, 1]
        ax_loss_full, ax_loss_zoom = axes[1, 0], axes[1, 1]
        ax_rmse_full, ax_rmse_zoom = axes[2, 0], axes[2, 1]
        full_axes = [ax_runtime_full, ax_loss_full, ax_rmse_full]
        zoom_axes = [ax_runtime_zoom, ax_loss_zoom, ax_rmse_zoom]
    else:
        fig, axes_1d = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        full_axes = list(axes_1d)
        zoom_axes = []

    full_axes[0].plot(eps, runtime, marker="o", linewidth=1.4, markersize=3.5, color="#1f77b4")
    full_axes[0].set_ylabel("Runtime (sec)")
    full_axes[0].grid(True, linestyle=":")
    full_axes[0].set_title("OT epsilon tuning trends (full range)")

    full_axes[1].plot(eps, loss, marker="o", linewidth=1.4, markersize=3.5, color="#d62728")
    full_axes[1].set_ylabel("Loss")
    full_axes[1].grid(True, linestyle=":")

    full_axes[2].plot(eps, rmse, marker="o", linewidth=1.4, markersize=3.5, color="#2ca02c")
    full_axes[2].set_ylabel("RMSE")
    full_axes[2].set_xlabel("epsilon (log scale)")
    full_axes[2].grid(True, linestyle=":")

    for ax in full_axes:
        ax.set_xscale("log")

    if zoom_enabled:
        zoom_mask = eps <= float(args.zoom_eps_max)
        if np.any(zoom_mask):
            eps_z = eps[zoom_mask]
            runtime_z = runtime[zoom_mask]
            loss_z = loss[zoom_mask]
            rmse_z = rmse[zoom_mask]

            zoom_axes[0].plot(eps_z, runtime_z, marker="o", linewidth=1.4, markersize=3.5, color="#1f77b4")
            zoom_axes[0].set_ylabel("Runtime (sec)")
            zoom_axes[0].set_title(f"Zoomed range (epsilon <= {args.zoom_eps_max:g})")
            zoom_axes[0].set_ylim(*_robust_ylim(runtime_z, q_low=q_low, q_high=q_high))
            zoom_axes[0].grid(True, linestyle=":")

            zoom_axes[1].plot(eps_z, loss_z, marker="o", linewidth=1.4, markersize=3.5, color="#d62728")
            zoom_axes[1].set_ylabel("Loss")
            zoom_axes[1].set_ylim(*_robust_ylim(loss_z, q_low=q_low, q_high=q_high))
            zoom_axes[1].grid(True, linestyle=":")

            zoom_axes[2].plot(eps_z, rmse_z, marker="o", linewidth=1.4, markersize=3.5, color="#2ca02c")
            zoom_axes[2].set_ylabel("RMSE")
            zoom_axes[2].set_xlabel("epsilon (log scale)")
            zoom_axes[2].set_ylim(*_robust_ylim(rmse_z, q_low=q_low, q_high=q_high))
            zoom_axes[2].grid(True, linestyle=":")

            for ax in zoom_axes:
                ax.set_xscale("log")
        else:
            for ax in zoom_axes:
                ax.text(0.5, 0.5, "No points in zoom range", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
