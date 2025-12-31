from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


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


def _logdet_from_cov(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = np.asarray(cov)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, eps)
    return np.sum(np.log(eigvals), axis=-1)


def _find_latest_diagnostics(root: Path) -> Path:
    candidates = [p for p in root.rglob("diagnostics.npz") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No diagnostics.npz found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_diagnostics_path(target: Optional[str], root: Path) -> Path:
    if target is None:
        return _find_latest_diagnostics(root)
    path = Path(target)
    if path.is_dir():
        diag = path / "diagnostics.npz"
        if not diag.exists():
            raise FileNotFoundError(f"diagnostics.npz not found in {path}")
        return diag
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stability diagnostics from existing diagnostics.npz files."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Path to diagnostics.npz or method directory. If omitted, uses the latest under --root.",
    )
    parser.add_argument(
        "--root",
        default="results",
        help="Root directory to search when target is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (defaults to diagnostics parent).",
    )
    parser.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=(25.0, 75.0),
        help="Percentile band for plots (default: 25 75).",
    )
    parser.add_argument(
        "--no-band",
        action="store_true",
        help="Disable percentile band shading.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    root = Path(args.root)
    diag_path = _resolve_diagnostics_path(args.target, root)
    output_dir = Path(args.output_dir) if args.output_dir else diag_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    band = None if args.no_band else tuple(float(x) for x in args.percentiles)
    diagnostics = dict(np.load(diag_path, allow_pickle=False))
    if "logdet_cov" not in diagnostics:
        cov = diagnostics.get("cov")
        if cov is not None:
            diagnostics["logdet_cov"] = _logdet_from_cov(cov)
    _plot_stability_over_time(output_dir, diagnostics, band_percentiles=band, show=args.show)
    print(f"Loaded: {diag_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
