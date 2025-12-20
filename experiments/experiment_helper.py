import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

tfd = tfp.distributions


@dataclass
class CommonConfig:
    T: int = 300
    batch_size: int = 1
    seed: tuple[int, ...] = (123,)
    out_dir: str = "runs"
    save: bool = True
    show: bool = True


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def rmse_all(x_true: tf.Tensor, x_est: tf.Tensor) -> tf.Tensor:
    err2 = tf.square(x_true - x_est)
    return tf.sqrt(tf.reduce_mean(err2))


def innovation_rmse(ssm, y_obs: tf.Tensor, m_pred: tf.Tensor) -> tf.Tensor:
    y_pred = ssm.h(m_pred)
    v = ssm.innovation(y_obs, y_pred)
    return rmse_all(v, tf.zeros_like(v))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(str(path), **{k: to_numpy(v) for k, v in arrays.items()})


def print_metrics(prefix: str, metrics: Dict[str, Any]) -> None:
    keys = sorted(metrics.keys())
    print(prefix)
    for k in keys:
        v = metrics[k]
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")


def make_init_dist(m0: tf.Tensor, P0: tf.Tensor):
    L0 = tf.linalg.cholesky(P0)

    def _init(shape):
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(m0, tf.concat([shape, [tf.shape(m0)[-1]]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=L0)

    return _init


def aggregate_metrics(runs: list[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = np.array([r["metrics"][key] for r in runs], dtype=np.float32)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
