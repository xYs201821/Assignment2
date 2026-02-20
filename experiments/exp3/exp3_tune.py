from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp3.exp3_lgssm_dpf import _canonical_dpf_method, _run_single_seed
from experiments.common.exp_utils import as_list, cfg_section, ensure_dir, load_config

DEFAULT_CONFIG = Path(__file__).with_name("exp3_config.yaml")
DEFAULT_OUTPUT_DIR = Path("results/exp3_tune")
DEFAULT_TUNING_CONFIG = Path(__file__).with_name("exp3_tuning.yaml")


def _log(message: str) -> None:
    tqdm.write(message)


def _split_candidate_params_for_log(
    params: Dict[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source: Dict[str, Any]
    if keys:
        source = {k: params[k] for k in keys if k in params}
    else:
        source = dict(params)

    tune_params: Dict[str, Any] = {}
    train_params: Dict[str, Any] = {}
    for key in sorted(source.keys()):
        value = source[key]
        if key.startswith("train."):
            train_key = key.split(".", 1)[1]
            train_params[train_key] = value
        else:
            tune_params[key] = value
    return tune_params, train_params

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid tuning runner for exp3 LGSSM DPF methods.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to exp3 YAML config.")
    parser.add_argument(
        "--tuning-config",
        type=Path,
        default=None,
        help=(
            "Optional external tuning YAML with top-level `tuning` mapping. "
            "If omitted, auto-loads exp3_tuning.yaml when present; "
            "otherwise uses config.tuning."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save tuning results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Methods to tune (default: infer from tuning.methods, then from list-valued dpf params, "
            "otherwise use dpf.methods except kalman/baseline)."
        ),
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=("rmse", "loss", "rmse_phi", "combined"),
        default=None,
        help="Selection objective (default: tuning.objective or combined).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed override (default: experiment.seeds).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4096,
        help="Hard cap per-method candidate count to avoid accidental explosion.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="How many top candidates to keep in the per-method summary.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _method_cfg_base(dpf_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Resolve method configuration from `common` + `dpf.<method>` only."""
    out: Dict[str, Any] = {}
    common_cfg = dpf_cfg.get("common")
    if isinstance(common_cfg, dict):
        out.update(common_cfg)
    method_cfg = dpf_cfg.get(method)
    if isinstance(method_cfg, dict):
        out.update(method_cfg)
    return out


def _load_external_tuning_cfg(path: Path) -> Dict[str, Any]:
    ext = load_config(path, [])
    if not isinstance(ext, dict):
        raise ValueError(f"External tuning config must be a mapping: {path}")
    scoped = ext.get("tuning")
    if not isinstance(scoped, dict):
        raise ValueError(f"External tuning config 'tuning' must be a mapping: {path}")
    return dict(scoped)


def _merge_tuning_cfg(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if key == "methods" and isinstance(value, dict) and isinstance(out.get("methods"), dict):
            merged_methods = dict(out["methods"])
            for method_key, method_value in value.items():
                if (
                    isinstance(method_value, dict)
                    and isinstance(merged_methods.get(method_key), dict)
                ):
                    merged_leaf = dict(merged_methods[method_key])
                    merged_leaf.update(method_value)
                    merged_methods[method_key] = merged_leaf
                else:
                    merged_methods[method_key] = method_value
            out["methods"] = merged_methods
        else:
            out[key] = value
    return out


def _flatten_mapping(node: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    for key, value in node.items():
        key_str = str(key).strip()
        if not key_str:
            raise ValueError("Empty key is not allowed in tuning mappings.")
        path = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, dict):
            yield from _flatten_mapping(value, path)
        else:
            yield path, value


def _canonical_param_path(raw_path: str) -> str:
    parts = [p for p in str(raw_path).strip().split(".") if p]
    if not parts:
        raise ValueError("Tuning parameter path cannot be empty.")
    return ".".join(parts)


def _extract_overrides(
    mapping: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    fixed: Dict[str, Any] = {}
    sweep: Dict[str, List[Any]] = {}
    for raw_path, value in _flatten_mapping(mapping):
        path = _canonical_param_path(raw_path)
        if isinstance(value, list):
            if not value:
                continue
            vals = list(value)
            if len(vals) == 1:
                fixed[path] = vals[0]
            else:
                sweep[path] = vals
        else:
            fixed[path] = value
    return fixed, sweep


def _merge_overrides(
    fixed_base: Dict[str, Any],
    sweep_base: Dict[str, List[Any]],
    fixed_new: Dict[str, Any],
    sweep_new: Dict[str, List[Any]],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    fixed = dict(fixed_base)
    sweep = {k: list(v) for k, v in sweep_base.items()}

    for key, value in fixed_new.items():
        fixed[key] = value
        sweep.pop(key, None)
    for key, values in sweep_new.items():
        sweep[key] = list(values)
        fixed.pop(key, None)
    return fixed, sweep


def _resolve_method_overrides(
    method: str,
    dpf_cfg: Dict[str, Any],
    tuning_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    fixed: Dict[str, Any] = {}
    sweep: Dict[str, List[Any]] = {}

    method_cfg_base = _method_cfg_base(dpf_cfg, method)
    fixed_dpf, sweep_dpf = _extract_overrides(method_cfg_base)
    fixed, sweep = _merge_overrides(fixed, sweep, fixed_dpf, sweep_dpf)

    tuning_methods = tuning_cfg.get("methods")
    if isinstance(tuning_methods, dict):
        method_tune = tuning_methods.get(method)
        if method_tune is not None:
            if not isinstance(method_tune, dict):
                raise ValueError(f"tuning.methods.{method} must be a mapping.")
            fixed_tune, sweep_tune = _extract_overrides(method_tune)
            fixed, sweep = _merge_overrides(fixed, sweep, fixed_tune, sweep_tune)

    return fixed, sweep


def _method_tuning_param_keys(method: str, tuning_cfg: Dict[str, Any]) -> List[str]:
    tuning_methods = tuning_cfg.get("methods")
    if not isinstance(tuning_methods, dict):
        return []
    method_tune = tuning_methods.get(method)
    if method_tune is None:
        return []
    if not isinstance(method_tune, dict):
        raise ValueError(f"tuning.methods.{method} must be a mapping.")
    fixed_tune, sweep_tune = _extract_overrides(method_tune)
    return sorted(set(fixed_tune.keys()) | set(sweep_tune.keys()))


def _candidate_count(sweep: Dict[str, List[Any]]) -> int:
    count = 1
    for values in sweep.values():
        count *= len(values)
    return count


def _iter_candidates(
    fixed: Dict[str, Any],
    sweep: Dict[str, List[Any]],
) -> Iterator[Dict[str, Any]]:
    if not sweep:
        yield dict(fixed)
        return

    keys = sorted(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    for values in product(*value_lists):
        item = dict(fixed)
        item.update({k: v for k, v in zip(keys, values)})
        yield item


def _deep_set(mapping: Dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("Cannot set an empty path.")
    node = mapping
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def _objective_value(record: Dict[str, Any], objective: str) -> float:
    if record.get("status", "ok") != "ok":
        return float("inf")
    if objective == "combined":
        return float(record.get("combined_mean", float("inf")))
    return float(record.get(f"{objective}_mean", float("inf")))


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _finite_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr, ddof=0))


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        f = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}g}"


def _print_method_metric_table(method: str, records: Sequence[Dict[str, Any]]) -> None:
    headers = ["candidate", "train_sec", "rmse", "loss", "phi_rmse", "ess", "grad_raw", "status", "params"]
    rows: List[List[str]] = []
    for rec in sorted(records, key=lambda r: int(r.get("candidate_id", 0))):
        params_text = json.dumps(rec.get("params", {}), ensure_ascii=False, sort_keys=True)
        rows.append(
            [
                str(int(rec.get("candidate_id", 0))),
                _fmt_float(rec.get("train_sec_mean", float("nan"))),
                _fmt_float(rec.get("rmse_mean", float("nan"))),
                _fmt_float(rec.get("loss_mean", float("nan"))),
                _fmt_float(rec.get("rmse_phi_mean", float("nan"))),
                _fmt_float(rec.get("ess_mean", float("nan"))),
                _fmt_float(rec.get("grad_raw_mean", float("nan"))),
                str(rec.get("status", "")),
                params_text,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "-+-".join("-" * w for w in widths)
    _log(f"[exp3_tune][{method}] results table")
    _log(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    _log(sep)
    for row in rows:
        _log(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _write_method_table_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "train_sec",
        "rmse",
        "loss",
        "phi_rmse",
        "ess",
        "grad_raw",
        "status",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: int(r.get("candidate_id", 0))):
            writer.writerow(
                {
                    "candidate_id": int(rec.get("candidate_id", 0)),
                    "train_sec": rec.get("train_sec_mean", float("nan")),
                    "rmse": rec.get("rmse_mean", float("nan")),
                    "loss": rec.get("loss_mean", float("nan")),
                    "phi_rmse": rec.get("rmse_phi_mean", float("nan")),
                    "ess": rec.get("ess_mean", float("nan")),
                    "grad_raw": rec.get("grad_raw_mean", float("nan")),
                    "status": rec.get("status", ""),
                    "params": json.dumps(rec.get("params", {}), ensure_ascii=False, sort_keys=True),
                }
            )


def _topk(
    records: Sequence[Dict[str, Any]],
    k: int,
    objective: str,
) -> List[Dict[str, Any]]:
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    ok.sort(key=lambda r: (_objective_value(r, objective), float(r.get("rmse_mean", float("inf")))))
    return ok[: max(0, int(k))]


def _evaluate_candidate(
    candidate_id: int,
    method: str,
    params: Dict[str, Any],
    tune_param_keys: Sequence[str],
    seeds: Sequence[int],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dpf_cfg_base: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    objective: str,
    output_dir: Path,
) -> Dict[str, Any]:
    dpf_cfg = copy.deepcopy(dpf_cfg_base)
    method_cfg = dpf_cfg.get(method)
    if not isinstance(method_cfg, dict):
        method_cfg = {}
    for key, value in params.items():
        if key.startswith("common."):
            _deep_set(dpf_cfg, key, value)
        else:
            _deep_set(method_cfg, key, value)
    dpf_cfg[method] = method_cfg

    candidate_dir = output_dir / method / f"candidate_{candidate_id:04d}"
    ensure_dir(candidate_dir)

    t0 = time.time()
    per_seed: List[Dict[str, Any]] = []
    rmse_vals: List[float] = []
    rmse_phi_vals: List[float] = []
    loss_vals: List[float] = []
    ess_vals: List[float] = []
    train_sec_vals: List[float] = []
    grad_raw_vals: List[float] = []

    _log(f"[{method}] candidate={candidate_id:04d} seeds={list(map(int, seeds))}")
    tune_params, train_params = _split_candidate_params_for_log(
        params,
        keys=tune_param_keys,
    )
    _log(f"[{method}] tune_params={json.dumps(tune_params, default=_json_default, sort_keys=True)}")
    _log(f"[{method}] train_params={json.dumps(train_params, default=_json_default, sort_keys=True)}")
    try:
        for seed in seeds:
            t_seed0 = time.perf_counter()
            result = _run_single_seed(
                seed=int(seed),
                exp_cfg=exp_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                dpf_cfg=dpf_cfg,
                proposal_cfg=proposal_cfg,
                method=method,
                experiment_name=f"exp3_tune_{method}",
                experiment_dir=candidate_dir,
            )
            seed_elapsed = float(time.perf_counter() - t_seed0)
            rmse_final = float(result["rmse_final"])
            rmse_phi_final = float(result["rmse_phi_final"])
            loss_final = float(result["loss_final"])
            ess_final = float(result["ess_mean_final"])
            runtime_train_sec = float(result.get("runtime_train_sec", float("nan")))
            runtime_eval_init_sec = float(result.get("runtime_eval_init_sec", float("nan")))
            runtime_eval_final_sec = float(result.get("runtime_eval_final_sec", float("nan")))
            if method == "transformer":
                # For transformer, include auto-pretraining in train_sec by using wall clock.
                train_sec = seed_elapsed
                if np.isfinite(runtime_eval_init_sec) and np.isfinite(runtime_eval_final_sec):
                    train_sec = max(train_sec - runtime_eval_init_sec - runtime_eval_final_sec, 0.0)
            else:
                train_sec = runtime_train_sec
            grad_raw_hist = np.asarray(result.get("grad_raw_norm_history", []), dtype=np.float64).reshape(-1)
            grad_raw = float("nan")
            if grad_raw_hist.size > 0:
                grad_raw_finite = grad_raw_hist[np.isfinite(grad_raw_hist)]
                if grad_raw_finite.size > 0:
                    grad_raw = float(grad_raw_finite[-1])

            rmse_vals.append(rmse_final)
            rmse_phi_vals.append(rmse_phi_final)
            loss_vals.append(loss_final)
            ess_vals.append(ess_final)
            train_sec_vals.append(train_sec)
            grad_raw_vals.append(grad_raw)
            per_seed.append(
                {
                    "seed": int(seed),
                    "train_sec": train_sec,
                    "rmse_final": rmse_final,
                    "rmse_phi_final": rmse_phi_final,
                    "loss_final": loss_final,
                    "ess_mean_final": ess_final,
                    "grad_raw_final": grad_raw,
                }
            )
        status = "ok"
        error = None
    except Exception as exc:  
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        _log(f"[{method}] candidate={candidate_id:04d} failed error={error}")

    elapsed = float(time.time() - t0)
    if status == "ok" and rmse_vals:
        rmse_mean = float(np.mean(rmse_vals))
        rmse_phi_mean = float(np.mean(rmse_phi_vals))
        record: Dict[str, Any] = {
            "method": method,
            "candidate_id": int(candidate_id),
            "params": params,
            "seeds": [int(s) for s in seeds],
            "num_runs": int(len(rmse_vals)),
            "rmse_mean": rmse_mean,
            "rmse_std": float(np.std(rmse_vals, ddof=0)),
            "rmse_phi_mean": rmse_phi_mean,
            "rmse_phi_std": float(np.std(rmse_phi_vals, ddof=0)),
            "combined_mean": rmse_mean + rmse_phi_mean,
            "loss_mean": float(np.mean(loss_vals)),
            "loss_std": float(np.std(loss_vals, ddof=0)),
            "ess_mean": float(np.mean(ess_vals)),
            "ess_std": float(np.std(ess_vals, ddof=0)),
            "train_sec_mean": _finite_mean(train_sec_vals),
            "train_sec_std": _finite_std(train_sec_vals),
            "grad_raw_mean": _finite_mean(grad_raw_vals),
            "grad_raw_std": _finite_std(grad_raw_vals),
            "elapsed_sec": elapsed,
            "status": "ok",
            "per_seed": per_seed,
            "objective_value": float("nan"),
            "error": None,
        }
    else:
        record = {
            "method": method,
            "candidate_id": int(candidate_id),
            "params": params,
            "seeds": [int(s) for s in seeds],
            "num_runs": int(len(per_seed)),
            "rmse_mean": float("inf"),
            "rmse_std": float("nan"),
            "rmse_phi_mean": float("inf"),
            "rmse_phi_std": float("nan"),
            "combined_mean": float("inf"),
            "loss_mean": float("inf"),
            "loss_std": float("nan"),
            "ess_mean": float("nan"),
            "ess_std": float("nan"),
            "train_sec_mean": float("nan"),
            "train_sec_std": float("nan"),
            "grad_raw_mean": float("nan"),
            "grad_raw_std": float("nan"),
            "elapsed_sec": elapsed,
            "status": "failed",
            "per_seed": per_seed,
            "objective_value": float("inf"),
            "error": error,
        }
    record["objective_value"] = _objective_value(record, objective)
    _log(
        f"[{method}] done candidate={candidate_id:04d} status={record['status']} "
        f"objective={record['objective_value']:.6f} rmse={record['rmse_mean']:.6f} "
        f"rmse_phi={record['rmse_phi_mean']:.6f} loss={record['loss_mean']:.6f} "
        f"elapsed={record['elapsed_sec']:.1f}s"
    )
    return record


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "candidate_id",
        "status",
        "objective_value",
        "rmse_mean",
        "rmse_std",
        "rmse_phi_mean",
        "rmse_phi_std",
        "combined_mean",
        "loss_mean",
        "loss_std",
        "ess_mean",
        "ess_std",
        "train_sec_mean",
        "train_sec_std",
        "grad_raw_mean",
        "grad_raw_std",
        "elapsed_sec",
        "num_runs",
        "seeds",
        "params",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["seeds"] = ",".join(str(s) for s in row.get("seeds", []))
            row["params"] = json.dumps(row.get("params", {}), ensure_ascii=False, default=_json_default)
            writer.writerow({k: row.get(k, "") for k in fields})


def _resolve_methods(
    args_methods: List[str] | None,
    dpf_cfg: Dict[str, Any],
    tuning_cfg: Dict[str, Any],
) -> List[str]:
    if args_methods:
        out: List[str] = []
        for raw in args_methods:
            method = str(raw).strip()
            canonical = _canonical_dpf_method(method)
            if canonical != method:
                raise ValueError(f"--methods expects canonical method names; use '{canonical}' instead of '{raw}'.")
            if method not in out:
                out.append(method)
        return out

    methods: List[str] = []
    tuning_methods = tuning_cfg.get("methods")
    if isinstance(tuning_methods, dict):
        for raw in tuning_methods.keys():
            method = str(raw).strip()
            canonical = _canonical_dpf_method(method)
            if canonical != method:
                raise ValueError(
                    f"tuning.methods key '{raw}' is non-canonical; use '{canonical}'."
                )
            if method not in methods:
                methods.append(method)
    elif tuning_methods is not None:
        raise ValueError("tuning.methods must be a mapping.")

    if methods:
        return methods

    dpf_methods: List[str] = []
    for raw in as_list(dpf_cfg.get("methods", [])):
        method = str(raw).strip()
        canonical = _canonical_dpf_method(method)
        if canonical != method:
            raise ValueError(f"dpf.methods contains non-canonical method '{raw}'; use '{canonical}'.")
        dpf_methods.append(method)
    for method in dpf_methods:
        _, sweep = _resolve_method_overrides(method, dpf_cfg, tuning_cfg)
        if sweep and method not in methods:
            methods.append(method)
    if methods:
        return methods

    # Fallback: tune all trainable PF-style methods listed in config.
    for method in dpf_methods:
        if method in ("kalman", "baseline"):
            continue
        if method not in methods:
            methods.append(method)
    return methods


def main() -> None:
    args = _parse_args()
    ensure_dir(args.output_dir)

    cfg = load_config(args.config, [])
    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    proposal_cfg = cfg_section(cfg, "proposal")
    dpf_cfg = cfg_section(cfg, "dpf")
    train_cfg = cfg_section(cfg, "training")
    tuning_cfg = cfg_section(cfg, "tuning")
    tuning_config_used = None
    if args.tuning_config is not None:
        tuning_cfg_ext = _load_external_tuning_cfg(args.tuning_config)
        tuning_cfg = _merge_tuning_cfg(tuning_cfg, tuning_cfg_ext)
        tuning_config_used = str(args.tuning_config)
    else:
        default_tuning = DEFAULT_TUNING_CONFIG
        if default_tuning.exists():
            tuning_cfg_ext = _load_external_tuning_cfg(default_tuning)
            tuning_cfg = _merge_tuning_cfg(tuning_cfg, tuning_cfg_ext)
            tuning_config_used = str(default_tuning)

    objective = str(args.objective or tuning_cfg.get("objective", "combined")).strip().lower()
    if objective not in ("rmse", "loss", "rmse_phi", "combined"):
        raise ValueError("objective must be one of: rmse, loss, rmse_phi, combined.")
    seeds = [int(s) for s in (args.seeds if args.seeds is not None else as_list(exp_cfg.get("seeds", [0])))]
    methods = _resolve_methods(args.methods, dpf_cfg=dpf_cfg, tuning_cfg=tuning_cfg)
    if not methods:
        raise RuntimeError("No methods selected for tuning.")

    _log(f"[exp3_tune] config={args.config}")
    if tuning_config_used is not None:
        _log(f"[exp3_tune] tuning_config={tuning_config_used}")
    _log(f"[exp3_tune] output_dir={args.output_dir}")
    _log(f"[exp3_tune] methods={methods}")
    _log(f"[exp3_tune] seeds={seeds}")
    _log(f"[exp3_tune] objective={objective}")

    t0 = time.time()
    all_records: List[Dict[str, Any]] = []
    summary_by_method: Dict[str, Any] = {}

    for method in methods:
        fixed, sweep = _resolve_method_overrides(method, dpf_cfg=dpf_cfg, tuning_cfg=tuning_cfg)
        tune_param_keys = _method_tuning_param_keys(method, tuning_cfg)
        if not tune_param_keys:
            tune_param_keys = sorted(params_key for params_key in fixed.keys() | sweep.keys())
        num_candidates = _candidate_count(sweep)
        if num_candidates > int(args.max_candidates):
            raise ValueError(
                f"Method '{method}' has {num_candidates} candidates (> --max-candidates={args.max_candidates}). "
                f"Reduce list sizes before running."
            )
        _log(
            f"[exp3_tune] method={method} candidates={num_candidates} "
            f"sweep_keys={sorted(sweep.keys())}"
        )

        records_method: List[Dict[str, Any]] = []
        candidate_iter = tqdm(
            _iter_candidates(fixed=fixed, sweep=sweep),
            total=num_candidates,
            desc=f"exp3_tune:{method}",
            unit="cand",
            dynamic_ncols=True,
            leave=True,
        )
        for idx, params in enumerate(candidate_iter, start=1):
            rec = _evaluate_candidate(
                candidate_id=idx,
                method=method,
                params=params,
                tune_param_keys=tune_param_keys,
                seeds=seeds,
                exp_cfg=exp_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                dpf_cfg_base=dpf_cfg,
                proposal_cfg=proposal_cfg,
                objective=objective,
                output_dir=args.output_dir,
            )
            records_method.append(rec)
            all_records.append(rec)
            candidate_iter.set_postfix_str(
                f"done={idx}/{num_candidates} status={rec.get('status', 'ok')}",
                refresh=False,
            )

        _print_method_metric_table(method, records_method)
        method_table_csv = args.output_dir / f"{method}_results_table.csv"
        _write_method_table_csv(method_table_csv, records_method)
        _log(f"[exp3_tune] method={method} table_csv={method_table_csv}")

        topk = _topk(records_method, k=int(args.topk), objective=objective)
        best = topk[0] if topk else None
        summary_by_method[method] = {
            "objective": objective,
            "num_candidates": int(num_candidates),
            "sweep_keys": sorted(sweep.keys()),
            "fixed_params": fixed,
            "best": best,
            "topk": topk,
        }
        if best is None:
            _log(f"[exp3_tune] method={method} has no successful candidate.")
        else:
            _log(
                f"[exp3_tune] method={method} best candidate={best['candidate_id']:04d} "
                f"objective={best['objective_value']:.6f} "
                f"params={json.dumps(best['params'], default=_json_default)}"
            )

    elapsed = float(time.time() - t0)
    payload = {
        "config": str(args.config),
        "tuning_config": tuning_config_used,
        "output_dir": str(args.output_dir),
        "methods": methods,
        "seeds": seeds,
        "objective": objective,
        "elapsed_sec_total": elapsed,
        "summary_by_method": summary_by_method,
        "records": all_records,
    }

    json_path = args.output_dir / "exp3_tune_results.json"
    csv_path = args.output_dir / "exp3_tune_results.csv"
    best_path = args.output_dir / "exp3_tune_best_by_method.json"

    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_csv(csv_path, all_records)
    best_only = {m: v.get("best") for m, v in summary_by_method.items()}
    best_path.write_text(json.dumps(best_only, indent=2, default=_json_default), encoding="utf-8")

    _log(f"[exp3_tune] done elapsed={elapsed:.1f}s")
    _log(f"[exp3_tune] results_json={json_path}")
    _log(f"[exp3_tune] results_csv={csv_path}")
    _log(f"[exp3_tune] best_json={best_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
