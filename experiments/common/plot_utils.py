"""Shared plotting utilities for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def plot_stability_series(
    path: Path,
    values: np.ndarray,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    """Plot a stability metric time series with optional percentile bands.
    
    Args:
        path: Output file path
        values: Array of shape [T] or [..., T]
        band_percentiles: (low, high) percentiles for band, or None
        show: Whether to display the plot
        title: Optional plot title
    """
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


def plot_stability_over_time(
    output_dir: Path,
    diagnostics: Dict[str, Any],
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    """Plot stability diagnostics over time.
    
    Args:
        output_dir: Directory to save plots
        diagnostics: Dictionary containing diagnostic arrays
        band_percentiles: (low, high) percentiles for bands
        show: Whether to display plots
    """
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
        plot_stability_series(
            path,
            np.asarray(val),
            band_percentiles=band_percentiles,
            show=show,
            title=title,
        )


def plot_ess_over_time(
    path: Path,
    ess_by_method: Dict[str, np.ndarray],
    num_particles_by_method: Optional[Dict[str, int]] = None,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
    title: Optional[str] = None,
    ess_threshold: Optional[float] = None,
) -> None:
    """Plot ESS over time for multiple methods.
    
    Args:
        path: Output file path
        ess_by_method: Dict mapping method name to ESS array [B, T] or [T]
        num_particles_by_method: Optional dict mapping method to particle count
        band_percentiles: (low, high) percentiles for bands, or None for min/max
        show: Whether to display the plot
        title: Optional plot title
    """
    import matplotlib.pyplot as plt
    from itertools import cycle

    style_cycle = cycle(["-", "--", "-.", ":"])
    marker_cycle = cycle(["o", "s", "^", "v", "D", "P", "X"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    for method, ess in ess_by_method.items():
        ess_np = np.asarray(ess)
        if ess_np.ndim == 1:
            ess_np = ess_np[np.newaxis, :]
        
        ess_mean = np.mean(ess_np, axis=0)
        T = ess_mean.shape[0]
        t = np.arange(T)
        markevery = max(1, T // 10)
        
        label = method
        if num_particles_by_method and method in num_particles_by_method:
            N = num_particles_by_method[method]
            label = f"{method} (N={N})"
        
        ax.plot(
            t,
            ess_mean,
            label=label,
            linestyle=next(style_cycle),
            marker=next(marker_cycle),
            markevery=markevery,
            markersize=4,
        )
        
        if ess_np.shape[0] > 1:
            if band_percentiles is not None:
                p_lo, p_hi = band_percentiles
                lo = np.percentile(ess_np, p_lo, axis=0)
                hi = np.percentile(ess_np, p_hi, axis=0)
            else:
                lo = np.min(ess_np, axis=0)
                hi = np.max(ess_np, axis=0)
            ax.fill_between(t, lo, hi, alpha=0.2)

    if ess_threshold is not None:
        ax.axhline(
            float(ess_threshold),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold ({ess_threshold:.2g})",
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
