from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


# =============================================================================
# Configuration utilities
# =============================================================================

def as_list(value: Any) -> List[Any]:
    """Convert a value to a list, handling None and iterables."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def deep_set(cfg: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested key in a config dict using dot notation."""
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("override key cannot be empty")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply command-line overrides to a config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        deep_set(cfg, key.strip(), value)
    return cfg


def load_config(path: Path, overrides: List[str]) -> Dict[str, Any]:
    """Load a YAML config file and apply overrides."""
    raw = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return apply_overrides(cfg, overrides)


def as_mapping(value: Any, path: str) -> Dict[str, Any]:
    """Return config section as dict; raise for non-mapping values."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Read top-level config section with type validation."""
    return as_mapping(cfg.get(key), key)


def cfg_subsection(parent: Dict[str, Any], key: str, parent_path: str) -> Dict[str, Any]:
    """Read nested config section with type validation."""
    return as_mapping(parent.get(key), f"{parent_path}.{key}")


def first_non_null(*values: Any) -> Any:
    """Return the first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def resolve_filter_model_cfg(cfg: Dict[str, Any], filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve filter-side model config.

    Priority: `filters.model` overrides legacy top-level `model_filter`.
    """
    legacy_cfg = as_mapping(cfg.get("model_filter"), "model_filter")
    scoped_cfg = cfg_subsection(filters_cfg, "model", "filters")
    if legacy_cfg and scoped_cfg:
        merged = dict(legacy_cfg)
        merged.update(scoped_cfg)
        return merged
    return scoped_cfg or legacy_cfg


def resolve_optional_float_list(value: Any) -> List[float]:
    """Parse optional scalar/list into list[float]."""
    if value is None:
        return []
    return [float(v) for v in as_list(value)]


def expand_sweep_values(values: List[float], base_values: List[float], key: str) -> List[float]:
    """Expand optional sweep values to match base sweep length.

    Rules:
    - empty => copy `base_values`
    - length 1 => broadcast
    - same length as base => use directly
    """
    if not values:
        return list(base_values)
    if len(values) == 1:
        return values * len(base_values)
    if len(values) == len(base_values):
        return values
    raise ValueError(f"{key} must have length 1 or match sweep length {len(base_values)}")


def parse_percentile_band(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    """Parse percentile tuple with fallback.

    Accepts list/tuple with at least two entries; otherwise returns `default`.
    """
    if value is None:
        return default
    vals = list(value) if isinstance(value, (list, tuple)) else []
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1])
    return default


def parse_positive_int_or_none(value: Any) -> Optional[int]:
    """Parse optional integer; non-positive values map to None."""
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        return None
    return out


def clamp_time_index(value: Any, default: int, T: int) -> int:
    """Parse time index and clamp to [0, T-1] when T>0."""
    idx = default if value is None else int(value)
    if T > 0:
        idx = max(0, min(idx, T - 1))
    return idx


def resolve_plot_controls(
    exp_cfg: Dict[str, Any],
    specs: List[Tuple[str, str, Optional[str]]],
) -> Dict[str, bool]:
    """Resolve `plot_show` behavior and per-plot control flags.

    Each spec is (enabled_key, seed0_only_key, show_key_or_none).
    """
    plot_show = exp_cfg.get("plot_show")
    controls: Dict[str, bool] = {}
    if plot_show is None:
        controls["show_plots"] = bool(exp_cfg.get("show_plots", False))
        controls["plot_interactive"] = bool(exp_cfg.get("plot_interactive", False))
        for enabled_key, seed_key, show_key in specs:
            controls[enabled_key] = bool(exp_cfg.get(enabled_key, False))
            controls[seed_key] = bool(exp_cfg.get(seed_key, True))
            if show_key is not None:
                controls[show_key] = bool(exp_cfg.get(show_key, False))
        return controls

    enabled = bool(plot_show)
    show_plots = enabled and bool(exp_cfg.get("show_plots", False))
    controls["show_plots"] = show_plots
    controls["plot_interactive"] = enabled and bool(exp_cfg.get("plot_interactive", False))
    for enabled_key, seed_key, show_key in specs:
        controls[enabled_key] = enabled
        controls[seed_key] = bool(exp_cfg.get(seed_key, True))
        if show_key is not None:
            controls[show_key] = show_plots
    return controls


# =============================================================================
# Particle filter utilities
# =============================================================================

def particle_pairs(
    pf_particles: List[int],
    flow_particles: List[int],
    pair_particles: bool,
) -> List[Tuple[int, int]]:
    """Generate pairs of (pf_particles, flow_particles) for experiments."""
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


# =============================================================================
# Metrics utilities
# =============================================================================

def ess_from_weights(w: np.ndarray) -> Optional[np.ndarray]:
    """Compute effective sample size from particle weights.
    
    Args:
        w: Weights array of shape [B, T, N] or [T, N]
        
    Returns:
        ESS array of shape [B, T] or [T], or None if input is invalid.
    """
    w_np = np.asarray(w, dtype=np.float64)
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if w_np.ndim != 3:
        return None
    w_sum = np.sum(w_np, axis=-1, keepdims=True)
    w_norm = np.divide(w_np, w_sum, out=np.zeros_like(w_np), where=w_sum > 0)
    ess_t = 1.0 / np.sum(np.square(w_norm), axis=-1)
    return ess_t


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _nan_to_null(obj: Any) -> Any:
    """Recursively replace float NaN/Inf with None so JSON output is valid."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float("inf") or obj == float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_nan_to_null(payload), indent=2), encoding="utf-8")


def save_npz(path: Path, **arrays: Any) -> None:
    np.savez_compressed(str(path), **{k: _to_numpy(v) for k, v in arrays.items()})


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def slug(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def tag_from_cfg(cfg: Dict[str, Any]) -> str:
    parts = [f"{key}={slug(cfg[key])}" for key in sorted(cfg.keys())]
    return "__".join(parts) if parts else "base"


class GaussianInitDist:
    """Gaussian initial state sampler with .sample(shape, seed) interface."""

    def __init__(self, m0: Any, P0: Any):
        """Create a Gaussian initial distribution.
        
        Args:
            m0: Initial mean, shape [dx]
            P0: Initial covariance, shape [dx, dx]
        """
        import tensorflow_probability as tfp

        self._tfd = tfp.distributions
        self._m0 = tf.convert_to_tensor(m0, dtype=tf.float32)
        self._L0 = tf.linalg.cholesky(tf.convert_to_tensor(P0, dtype=tf.float32))

    def sample(self, shape, seed=None):
        """Sample initial states from N(m0, P0).
        
        Args:
            shape: Batch shape, e.g. [B, N]
            seed: Optional random seed
            
        Returns:
            Samples of shape [*shape, dx]
        """
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(
            self._m0, tf.concat([shape, [tf.shape(self._m0)[0]]], axis=0)
        )
        dist = self._tfd.MultivariateNormalTriL(loc=loc, scale_tril=self._L0)
        return dist.sample(seed=seed)


def build_init_dist(m0: Any, P0: Any) -> GaussianInitDist:
    """Build a Gaussian initial state sampler.
    
    Args:
        m0: Initial mean, shape [dx]
        P0: Initial covariance, shape [dx, dx]
        
    Returns:
        GaussianInitDist with .sample(shape, seed) method
    """
    return GaussianInitDist(m0, P0)


# =============================================================================
# Method type checking utilities
# =============================================================================

def is_pf_method(method: str) -> bool:
    """Check if method is a bootstrap particle filter."""
    method = str(method).lower()
    return method in ("pf", "bootstrap") or method.startswith("pf")


def is_stochastic_pf_method(method: str) -> bool:
    """Check if method is a stochastic particle flow."""
    method = str(method).lower()
    return method.startswith("stochastic_pf") or method.startswith("stochastic-pf") or method == "spf"


def is_pfpf_flow_method(method: str) -> bool:
    """Check if method is a PFPF flow (EDH/LEDH with PFPF)."""
    method = str(method).lower()
    return "pfpf" in method and (method.startswith("edh") or method.startswith("ledh"))


def is_particle_like_method(method: str) -> bool:
    """Check if method emits particle trajectories/weights."""
    return is_pf_method(method) or is_pfpf_flow_method(method) or is_stochastic_pf_method(method)


def ess_threshold_for_method(
    method: str,
    pf_threshold: float,
    flow_threshold: float,
) -> Optional[float]:
    """Get ESS threshold by method family."""
    if is_pf_method(method):
        return pf_threshold
    if is_pfpf_flow_method(method) or is_stochastic_pf_method(method):
        return flow_threshold
    return None


def select_pre_resample_weights(out: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract pre-resample weights from filter output."""
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    w_pre = diagnostics.get("w_pre")
    if w_pre is None:
        log_w_pre = diagnostics.get("log_w_pre")
        if log_w_pre is not None:
            w_pre = np.exp(np.asarray(log_w_pre))
    if w_pre is None:
        w_pre = out.get("w") if isinstance(out, dict) else None
    return w_pre


def get_summary_keys_and_prefixes(
    exp_cfg: Dict[str, Any],
    summary_keys: tuple,
) -> Tuple[tuple, Optional[tuple]]:
    """Get filtered summary keys and exclude prefixes based on config.
    
    Args:
        exp_cfg: Experiment config dict
        summary_keys: Default summary keys tuple
        
    Returns:
        (filtered_summary_keys, exclude_prefixes or None)
    """
    print_runtime_memory = exp_cfg.get("print_runtime_memory")
    if print_runtime_memory is None:
        print_runtime = bool(exp_cfg.get("print_runtime", True))
        print_memory = bool(exp_cfg.get("print_memory", True))
    else:
        print_runtime = bool(print_runtime_memory)
        print_memory = bool(print_runtime_memory)
    
    filtered_keys = summary_keys
    if not print_runtime:
        filtered_keys = tuple(k for k in filtered_keys if not k.startswith("runtime."))
    if not print_memory:
        filtered_keys = tuple(k for k in filtered_keys if not k.startswith("memory."))
    
    exclude_prefixes: List[str] = []
    if not print_runtime:
        exclude_prefixes.append("runtime.")
    if not print_memory:
        exclude_prefixes.append("memory.")
    
    return filtered_keys, tuple(exclude_prefixes) if exclude_prefixes else None
