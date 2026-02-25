from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import CommonConfig
from .experiment_helper import ensure_dir, make_seeds


RunOnceFn = Callable[[CommonConfig, Any, Optional[int]], Dict[str, Any]]


@dataclass
class ExperimentCase:
    name: str
    run_once: RunOnceFn
    common: CommonConfig
    cfg: Any


@dataclass
class SweepConfig:
    name: str
    param_grid: Dict[str, Sequence[Any]] = field(default_factory=dict)
    seeds: Optional[Sequence[Optional[int]]] = None
    num_seeds: Optional[int] = None
    base_seed: Optional[int] = None
    out_dir: str = "runs/experiments"
    save_metrics: bool = True


def build_case(case: str, common: Optional[CommonConfig] = None, cfg: Optional[Any] = None) -> ExperimentCase:
    key = case.lower()
    if key == "sv":
        from . import run_sv

        return ExperimentCase(
            name="sv",
            run_once=run_sv.run_sv_once,
            common=common or CommonConfig(),
            cfg=cfg or run_sv.SVConfig(),
        )
    if key == "rb":
        from . import run_rb

        return ExperimentCase(
            name="rb",
            run_once=run_rb.run_rb_once,
            common=common or CommonConfig(),
            cfg=cfg or run_rb.RBConfig(),
        )
    raise ValueError(f"Unknown case '{case}'. Use 'sv' or 'rb'.")


def _iter_grid(param_grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    if not param_grid:
        yield {}
        return
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _apply_overrides(
    common: CommonConfig,
    cfg: Any,
    overrides: Dict[str, Any],
) -> Tuple[CommonConfig, Any]:
    common = copy.deepcopy(common)
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key.startswith("common."):
            _set_nested_attr(common, key.split(".", 1)[1], value)
        elif key.startswith("cfg."):
            _set_nested_attr(cfg, key.split(".", 1)[1], value)
        else:
            _set_nested_attr(cfg, key, value)
    return common, cfg


def _resolve_seeds(cfg: SweepConfig) -> List[Optional[int]]:
    if cfg.seeds is not None:
        return list(cfg.seeds)
    if cfg.num_seeds is not None:
        return make_seeds(cfg.num_seeds, base_seed=cfg.base_seed)
    return [None]


def _slug(value: Any) -> str:
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _tag_from_overrides(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "base"
    parts = [f"{key}={_slug(value)}" for key, value in sorted(overrides.items())]
    return "__".join(parts)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _summarize_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        return {}
    metric_sets = [set(r.get("metrics", {}).keys()) for r in runs]
    keys = sorted(set.intersection(*metric_sets)) if metric_sets else []
    summary = {}
    for key in keys:
        vals = [float(r["metrics"][key]) for r in runs]
        mean, std = _mean_std(vals)
        summary[key] = {"mean": mean, "std": std}
    return summary


def _summarize_runtime(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not runs:
        return summary
    by_algo: Dict[str, Dict[str, List[float]]] = {}
    for run in runs:
        runtime = run.get("runtime", {})
        for algo, stats in runtime.items():
            entry = by_algo.setdefault(algo, {})
            for field in ("wall_s", "cpu_s", "warmup_wall_s", "warmup_cpu_s", "total_wall_s", "total_cpu_s"):
                if field in stats:
                    entry.setdefault(field, []).append(float(stats[field]))
                else:
                    if field.startswith("total_"):
                        warm = stats.get("warmup_wall_s", 0.0) if field.endswith("wall_s") else stats.get("warmup_cpu_s", 0.0)
                        base = stats.get("wall_s", 0.0) if field.endswith("wall_s") else stats.get("cpu_s", 0.0)
                        entry.setdefault(field, []).append(float(base + warm))
    for algo, fields in by_algo.items():
        summary[algo] = {}
        for field, vals in fields.items():
            mean, std = _mean_std(vals)
            summary[algo][field] = {"mean": mean, "std": std}
    return summary


def _save_run_metrics(run_dir: Path, seed: Optional[int], common: CommonConfig, cfg: Any, metrics: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    seed_label = "None" if seed is None else str(seed)
    out_dir = run_dir / f"seed{seed_label}"
    ensure_dir(out_dir)
    payload = {
        "common": asdict(common),
        "cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "metrics": metrics,
        "runtime": runtime,
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_sweep(case: ExperimentCase, sweep: SweepConfig) -> List[Dict[str, Any]]:
    base_dir = Path(sweep.out_dir) / case.name / sweep.name
    ensure_dir(base_dir)
    summaries: List[Dict[str, Any]] = []
    seeds = _resolve_seeds(sweep)
    for overrides in _iter_grid(sweep.param_grid):
        common, cfg = _apply_overrides(case.common, case.cfg, overrides)
        tag = _tag_from_overrides(overrides)
        run_dir = base_dir / tag
        ensure_dir(run_dir)
        common.out_dir = str(run_dir)
        runs = []
        for seed in seeds:
            out = case.run_once(common, cfg, seed)
            run_metrics = out.get("metrics", {})
            run_runtime = out.get("runtime", {})
            runs.append({"metrics": run_metrics, "runtime": run_runtime, "seed": seed})
            if sweep.save_metrics:
                _save_run_metrics(run_dir, seed, common, cfg, run_metrics, run_runtime)
        summary = {
            "case": case.name,
            "sweep": sweep.name,
            "overrides": overrides,
            "metrics": _summarize_metrics(runs),
            "runtime": _summarize_runtime(runs),
            "seeds": seeds,
        }
        _save_summary(run_dir, summary)
        summaries.append(summary)
    return summaries
