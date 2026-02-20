from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.exp_utils import ensure_dir, load_config


def _canonical_dpf_method_name(value: Any) -> str:
    method = str(value).strip().lower()
    if method in ("kalman", "kf"):
        return "kalman"
    if method in (
        "baseline",
        "pf_baseline",
        "baseline_pf",
        "baselinepf",
        "pf_fixed",
        "fixed_phi",
        "fixedphi",
    ):
        return "baseline"
    if method in ("pf", "bootstrap", "bootstrap_pf", "bootstrappf"):
        return "pf"
    if method in ("soft", "soft_resampling", "softresampling"):
        return "soft"
    if method in ("ot", "ot_resampling", "otresampling"):
        return "ot"
    if method in ("diffusion", "diff", "diffres", "diffusion_resampling"):
        return "diffusion"
    if method in (
        "transformer",
        "particle_transformer",
        "particletransformer",
        "pt",
    ):
        return "transformer"
    return method


def _method_storage_dirs(input_root: Path, method: str) -> List[Path]:
    canonical = _canonical_dpf_method_name(method)
    if canonical == "baseline":
        return [input_root / "baseline", input_root / "pf_baseline"]
    return [input_root / canonical]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exp3 DPF diagnostics from saved seed trace files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/exp3_lgssm_dpf"),
        help="Experiment output root that contains per-method trace files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <input-root>/plots).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method list; otherwise inferred from subdirectories.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling window for gradient variance/SNR curves.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=10,
        help="Start plotting from this step index to avoid early outliers.",
    )
    parser.add_argument(
        "--band-low",
        type=float,
        default=25.0,
        help="Lower percentile for uncertainty band.",
    )
    parser.add_argument(
        "--band-high",
        type=float,
        default=75.0,
        help="Upper percentile for uncertainty band.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("exp3_config.yaml"),
        help="Config path used to resolve per-method num_particles for ESS normalization.",
    )
    return parser.parse_args()


def _discover_methods(input_root: Path, methods: Optional[List[str]]) -> List[str]:
    if methods:
        out: List[str] = []
        for raw in methods:
            method = _canonical_dpf_method_name(raw)
            if method not in out:
                out.append(method)
        return out
    out: List[str] = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        method = _canonical_dpf_method_name(path.name)
        if _list_trace_paths(path, method) and method not in out:
            out.append(method)
    return out


def _trace_glob_patterns_for_method(method: str) -> List[str]:
    method_tag = _canonical_dpf_method_name(method)
    # Keep legacy compatibility for old traces named as seed{seed}_trace.npz.
    patterns = {"seed*_trace.npz", f"*_seed*_{method_tag}_trace.npz"}
    if method_tag == "baseline":
        patterns.add("*_seed*_pf_baseline_trace.npz")
    return sorted(patterns)


def _list_trace_paths(method_dir: Path, method: str) -> List[Path]:
    found: Dict[Path, None] = {}
    for pattern in _trace_glob_patterns_for_method(method):
        for trace_path in sorted(method_dir.glob(pattern)):
            found[trace_path] = None
    return sorted(found.keys())


def _load_method_traces(method_dir: Path, method: str) -> List[Dict[str, np.ndarray]]:
    traces: List[Dict[str, np.ndarray]] = []
    for trace_path in _list_trace_paths(method_dir, method):
        with np.load(trace_path) as data:
            traces.append({k: np.asarray(data[k]) for k in data.files})
    return traces


def _stack_curves(curves: List[np.ndarray]) -> Optional[np.ndarray]:
    valid = [np.asarray(curve, dtype=np.float64).reshape(-1) for curve in curves if curve is not None]
    valid = [curve for curve in valid if curve.size > 0]
    if not valid:
        return None
    min_len = min(curve.shape[0] for curve in valid)
    if min_len <= 0:
        return None
    return np.stack([curve[:min_len] for curve in valid], axis=0)


def _collect_ess_stack(traces: List[Dict[str, np.ndarray]], key: str) -> Optional[np.ndarray]:
    chunks: List[np.ndarray] = []
    for trace in traces:
        if key not in trace:
            continue
        arr = np.asarray(trace[key], dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        chunks.append(arr)
    if not chunks:
        return None
    min_t = min(arr.shape[-1] for arr in chunks)
    if min_t <= 0:
        return None
    return np.concatenate([arr[:, :min_t] for arr in chunks], axis=0)


def _rolling_mean_var(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    win = max(int(window), 1)
    csum = np.cumsum(np.concatenate([[0.0], x]))
    csum2 = np.cumsum(np.concatenate([[0.0], np.square(x)]))
    mean = np.zeros_like(x)
    var = np.zeros_like(x)
    for idx in range(x.size):
        start = max(0, idx - win + 1)
        count = idx - start + 1
        sx = csum[idx + 1] - csum[start]
        sx2 = csum2[idx + 1] - csum2[start]
        m = sx / count
        v = sx2 / count - m * m
        mean[idx] = m
        var[idx] = max(v, 0.0)
    return mean, var


def _method_cfg_with_common(exp_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    method = _canonical_dpf_method_name(method)
    dpf_cfg_raw = exp_cfg.get("dpf", {})
    dpf_cfg = dpf_cfg_raw if isinstance(dpf_cfg_raw, dict) else {}

    out: Dict[str, Any] = {}
    common_cfg = dpf_cfg.get("common")
    if isinstance(common_cfg, dict):
        out.update(common_cfg)

    if method == "baseline":
        for key in ("pf_baseline", "baseline_pf", "baseline"):
            method_cfg = dpf_cfg.get(key)
            if isinstance(method_cfg, dict):
                out.update(method_cfg)
    else:
        method_cfg = dpf_cfg.get(method)
        if isinstance(method_cfg, dict):
            out.update(method_cfg)

    if method == "transformer":
        for alias in ("particle_transformer", "pt"):
            alias_cfg = dpf_cfg.get(alias)
            if isinstance(alias_cfg, dict):
                out.update(alias_cfg)
    if method == "diffusion":
        alias_cfg = dpf_cfg.get("diffres")
        if isinstance(alias_cfg, dict):
            out.update(alias_cfg)
    if method == "baseline":
        has_explicit_baseline_cfg = (
            isinstance(dpf_cfg.get("baseline"), dict)
            or isinstance(dpf_cfg.get("pf_baseline"), dict)
            or isinstance(dpf_cfg.get("baseline_pf"), dict)
        )
        if (not has_explicit_baseline_cfg) and isinstance(dpf_cfg.get("pf"), dict):
            out.update(dpf_cfg["pf"])
    if method == "pf":
        alias_cfg = dpf_cfg.get("bootstrap_pf")
        if isinstance(alias_cfg, dict):
            out.update(alias_cfg)
    return out


def _resolve_num_particles_by_method(config_path: Path, methods: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not config_path.exists():
        print(f"[warn] config not found: {config_path}; cannot normalize ESS by num_particles.")
        return out
    try:
        cfg = load_config(config_path, overrides=[])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to load config {config_path}: {exc}")
        return out

    for method in methods:
        cfg_m = _method_cfg_with_common(cfg, method)
        n = cfg_m.get("num_particles")
        if n is None:
            continue
        try:
            n_i = int(n)
        except Exception:  # noqa: BLE001
            continue
        if n_i > 0:
            out[method] = n_i
    return out


def _plot_curves(
    path: Path,
    series_by_method: Dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
    p_low: float,
    p_high: float,
    log_y: bool,
    start_step: int,
    show: bool,
) -> bool:
    import matplotlib.pyplot as plt

    plotted = False
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for method, stack in series_by_method.items():
        arr = np.asarray(stack, dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[-1] <= start_step:
            continue
        arr = arr[:, start_step:]
        if log_y:
            arr = np.maximum(arr, 1e-12)
        mean = np.mean(arr, axis=0)
        t = np.arange(start_step, start_step + mean.shape[0], dtype=np.int32)
        line, = ax.plot(t, mean, label=method, linewidth=1.6)
        color = line.get_color()
        if arr.shape[0] > 1:
            lo = np.percentile(arr, p_low, axis=0)
            hi = np.percentile(arr, p_high, axis=0)
            ax.fill_between(t, lo, hi, color=color, alpha=0.2, linewidth=0.0)
            ax.plot(t, lo, linestyle="--", linewidth=1.0, color=color, alpha=0.9)
            ax.plot(t, hi, linestyle="--", linewidth=1.0, color=color, alpha=0.9)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _plot_ess_median_curve(
    path: Path,
    ess_by_method: Dict[str, np.ndarray],
    *,
    show: bool,
) -> bool:
    import matplotlib.pyplot as plt

    plotted = False
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for method, stack in ess_by_method.items():
        arr = np.asarray(stack, dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] <= 0:
            continue
        median = np.nanmedian(arr, axis=0)
        t = np.arange(median.shape[0], dtype=np.int32)
        ax.plot(t, median, label=method, linewidth=1.8)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("step")
    ax.set_ylabel("ESS / N")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle("Median ESS / Num Particles Over Time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return True


def main() -> None:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root / "plots"
    )

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    ensure_dir(output_dir)

    methods = _discover_methods(input_root, args.methods)
    if not methods:
        raise RuntimeError(
            f"No methods with seed trace files were found under: {input_root}"
        )
    num_particles_by_method = _resolve_num_particles_by_method(
        args.config.expanduser().resolve(),
        methods,
    )

    traces_by_method: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for method in methods:
        traces: List[Dict[str, np.ndarray]] = []
        for method_dir in _method_storage_dirs(input_root, method):
            if not method_dir.exists():
                continue
            traces.extend(_load_method_traces(method_dir, method))
        if traces:
            traces_by_method[method] = traces
        else:
            print(f"[warn] no trace files found for method '{method}' under {input_root}")

    if not traces_by_method:
        raise RuntimeError("No trace data available for plotting.")

    p_low = float(args.band_low)
    p_high = float(args.band_high)
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("band percentiles must satisfy 0 <= low < high <= 100.")

    loss_by_method: Dict[str, np.ndarray] = {}
    rmse_by_method: Dict[str, np.ndarray] = {}
    grad_snr_by_method: Dict[str, np.ndarray] = {}
    grad_var_by_method: Dict[str, np.ndarray] = {}
    ess_by_method: Dict[str, np.ndarray] = {}

    for method, traces in traces_by_method.items():
        loss_stack = _stack_curves([trace.get("loss_history") for trace in traces])
        if loss_stack is not None:
            loss_by_method[method] = loss_stack

        rmse_stack = _stack_curves([trace.get("rmse_history") for trace in traces])
        if rmse_stack is not None:
            rmse_by_method[method] = rmse_stack

        grad_histories: List[np.ndarray] = []
        for trace in traces:
            grad = trace.get("grad_raw_norm_history")
            if grad is None:
                continue
            grad = np.asarray(grad, dtype=np.float64).reshape(-1)
            if grad.size > 0:
                grad_histories.append(grad)

        snr_curves: List[np.ndarray] = []
        var_curves: List[np.ndarray] = []
        for grad in grad_histories:
            roll_mean, roll_var = _rolling_mean_var(grad, window=args.rolling_window)
            snr = np.abs(roll_mean) / (np.sqrt(roll_var) + 1e-8)
            snr_curves.append(snr)
            var_curves.append(roll_var)

        snr_stack = _stack_curves(snr_curves)
        if snr_stack is not None:
            grad_snr_by_method[method] = snr_stack
        var_stack = _stack_curves(var_curves)
        if var_stack is not None:
            grad_var_by_method[method] = var_stack

        ess_stack = _collect_ess_stack(traces, key="ess_over_time")
        if ess_stack is not None:
            num_particles = num_particles_by_method.get(method)
            if num_particles is None:
                print(
                    f"[warn] num_particles missing for method '{method}'; "
                    "skip ESS/N curve for this method."
                )
                continue
            ess_by_method[method] = ess_stack / float(num_particles)

    ess_path = output_dir / "ess_over_time.png"
    ess_ok = _plot_ess_median_curve(
        ess_path,
        ess_by_method,
        show=bool(args.show),
    )
    if ess_ok:
        print(f"[saved] {ess_path}")
    else:
        print(f"[skip] {ess_path.name}: no available data")

    outputs = [
        (
            output_dir / "gradient_snr_curve.png",
            grad_snr_by_method,
            "gradient SNR",
            f"Gradient SNR (rolling window={int(args.rolling_window)})",
            True,
        ),
        (
            output_dir / "gradient_variance_curve.png",
            grad_var_by_method,
            "gradient variance",
            f"Gradient Variance (rolling window={int(args.rolling_window)})",
            True,
        ),
        (
            output_dir / "train_loss_curve.png",
            loss_by_method,
            "loss",
            "Train Loss Curve",
            True,
        ),
        (
            output_dir / "rmse_curve.png",
            rmse_by_method,
            "RMSE",
            "Train RMSE Curve",
            False,
        ),
    ]

    for path, payload, ylabel, title, log_y in outputs:
        ok = _plot_curves(
            path,
            payload,
            ylabel=ylabel,
            title=title,
            p_low=p_low,
            p_high=p_high,
            log_y=log_y,
            start_step=max(int(args.start_step), 0),
            show=bool(args.show),
        )
        if ok:
            print(f"[saved] {path}")
        else:
            print(f"[skip] {path.name}: no available data")


if __name__ == "__main__":
    main()
