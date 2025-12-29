from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def build_init_dist(m0: Any, P0: Any):
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    m0_tf = tf.convert_to_tensor(m0, dtype=tf.float32)
    P0_tf = tf.convert_to_tensor(P0, dtype=tf.float32)
    L0 = tf.linalg.cholesky(P0_tf)

    def _dist(shape):
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(m0_tf, tf.concat([shape, [tf.shape(m0_tf)[0]]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=L0)

    return _dist
