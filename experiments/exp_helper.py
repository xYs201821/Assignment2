from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.exp_utils import ensure_dir, save_json
from src.metrics import evaluate as evaluate_metrics


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, np.bool_))


SUMMARY_KEYS = (
    "rmse_state",
    "rmse_obs",
    "rmse_y",
    "nll",
    "nees",
    "nis",
    "ess_mean",
    "runtime.total_s",
    "memory.peak_rss_mb",
)
SKIP_PRINT_KEYS = {"rank_hist"}


def _shape_hint(value: Any) -> Optional[Tuple[int, ...]]:
    if hasattr(value, "shape"):
        try:
            return tuple(int(dim) for dim in value.shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        shape = []
        current = value
        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(shape)
    return None


def _format_scalar(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not np.isfinite(val):
            return "nan" if np.isnan(val) else ("inf" if val > 0 else "-inf")
        return f"{val:.4f}"
    return str(value)


def _format_metric_value(value: Any) -> str:
    if _is_scalar(value):
        return _format_scalar(value)
    if isinstance(value, dict):
        parts = []
        for key, val in sorted(value.items(), key=lambda item: str(item[0])):
            parts.append(f"{key}={_format_metric_value(val)}")
        return "{" + ", ".join(parts) + "}"
    shape = _shape_hint(value)
    if shape is not None:
        return f"array(shape={shape})"
    return str(value)


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _aggregate_values(values: List[Any]) -> Any:
    values = [v for v in values if v is not None]
    if not values:
        return None
    first = values[0]
    if _is_scalar(first):
        nums = [float(v) for v in values if _is_scalar(v)]
        return float(np.mean(nums)) if nums else None
    if isinstance(first, dict):
        keys = {k for v in values if isinstance(v, dict) for k in v.keys()}
        out: Dict[Any, Any] = {}
        for key in sorted(keys, key=str):
            sub_vals = [v.get(key) for v in values if isinstance(v, dict) and key in v]
            agg = _aggregate_values(sub_vals)
            if agg is not None:
                out[key] = agg
        return out
    if isinstance(first, (list, tuple, np.ndarray)):
        arrays = []
        for v in values:
            arr = np.asarray(v)
            if arr.dtype.kind not in ("i", "f"):
                return None
            arrays.append(arr.astype(np.float64))
        try:
            stacked = np.stack(arrays, axis=0)
        except ValueError:
            return None
        return np.mean(stacked, axis=0)
    return None


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {}
    keys = {key for metrics in metrics_list for key in metrics.keys()}
    out: Dict[str, Any] = {}
    for key in sorted(keys, key=str):
        vals = [m.get(key) for m in metrics_list if key in m]
        agg = _aggregate_values(vals)
        if agg is not None:
            out[key] = agg
    return out


def aggregate_metrics_by_method(
    metrics_by_method: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    return {method: aggregate_metrics(vals) for method, vals in metrics_by_method.items()}


def print_separator(title: str, char: str = "=") -> None:
    line = char * 12
    print(f"{line} {title} {line}")


def print_metrics(prefix: str, metrics: Dict[str, Any]) -> None:
    print(f"[metrics] {prefix}")
    for key in sorted(metrics.keys()):
        if key in SKIP_PRINT_KEYS:
            continue
        print(f"  {key}: {_format_metric_value(metrics[key])}")


def _is_number_str(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _print_table(headers: List[str], rows: List[List[str]], sep: str) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))

    align_right = []
    for idx in range(len(headers)):
        if idx == 0:
            align_right.append(False)
            continue
        col_vals = [row[idx] for row in rows]
        numeric = all(val == "NA" or _is_number_str(val) for val in col_vals)
        align_right.append(numeric)

    def _format_row(row: List[str]) -> str:
        parts = []
        for idx, val in enumerate(row):
            if align_right[idx]:
                parts.append(val.rjust(widths[idx]))
            else:
                parts.append(val.ljust(widths[idx]))
        return sep.join(parts)

    print(_format_row(headers))
    for row in rows:
        print(_format_row(row))


def print_method_summary_table(
    metrics_by_method: Dict[str, Dict[str, Any]],
    method_order: Optional[Tuple[str, ...]] = None,
    keys: Optional[Tuple[str, ...]] = None,
    sep: str = " | ",
) -> None:
    if method_order is None:
        method_order = tuple(metrics_by_method.keys())
    keys = keys or SUMMARY_KEYS
    headers = ["method", *keys]
    rows: List[List[str]] = []
    for method in method_order:
        flat = _flatten_metrics(metrics_by_method.get(method, {}))
        row = [method]
        for key in keys:
            row.append(_format_metric_value(flat.get(key, "NA")))
        rows.append(row)
    _print_table(headers, rows, sep)


def print_method_summary(
    method: str,
    metrics: Dict[str, Any],
    keys: Optional[Tuple[str, ...]] = None,
    sep: str = " | ",
) -> None:
    flat = _flatten_metrics(metrics)
    keys = keys or SUMMARY_KEYS
    parts = [method]
    for key in keys:
        if key not in flat:
            continue
        parts.append(f"{key}={_format_metric_value(flat[key])}")
    print(sep.join(parts))


def print_metrics_compare(
    metrics_by_method: Dict[str, Dict[str, Any]],
    method_order: Optional[Tuple[str, ...]] = None,
    sep: str = " | ",
) -> None:
    if method_order is None:
        method_order = tuple(metrics_by_method.keys())
    flat_by_method = {
        method: _flatten_metrics(metrics)
        for method, metrics in metrics_by_method.items()
    }
    keys = sorted(
        {
            key
            for metrics in flat_by_method.values()
            for key in metrics.keys()
            if key not in SKIP_PRINT_KEYS
        },
        key=str,
    )
    header = ["metric", *method_order]
    rows: List[List[str]] = []
    for key in keys:
        row = [key]
        for method in method_order:
            value = flat_by_method.get(method, {}).get(key, "NA")
            row.append(_format_metric_value(value))
        rows.append(row)
    _print_table(header, rows, sep)


def record_metrics(
    ssm: Any,
    x_true: Any,
    y_obs: Any,
    outputs: Dict[str, Any],
    method_dir: Path,
    metrics_cfg: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
    print_full: bool = True,
) -> Dict[str, Any]:
    metrics = evaluate_metrics(ssm, x_true, y_obs, outputs, metrics_cfg=metrics_cfg)
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics["runtime"] = outputs.get("runtime", {})
    metrics["memory"] = outputs.get("memory", {})
    ensure_dir(method_dir)
    save_json(method_dir / "metrics.json", metrics)
    if print_full:
        print_metrics(prefix or str(method_dir), metrics)
    return metrics
