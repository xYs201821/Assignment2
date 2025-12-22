import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import time

tfd = tfp.distributions


@dataclass
class CommonConfig:
    T: int = 40
    batch_size: int = 1
    seed: tuple[int, ...] = (42,)
    out_dir: str = "runs"
    save: bool = True
    show: bool = True


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def rmse_all(x_true: tf.Tensor, x_est: tf.Tensor) -> tf.Tensor:
    err = x_true - x_est
    per_step_norm = tf.norm(err, axis=-1)
    return tf.sqrt(tf.reduce_mean(tf.square(per_step_norm)))


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
    print(f"==== {prefix} Metrics ====")
    for k in keys:
        v = metrics[k]
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")
    print("====")


def print_runtime(prefix: str, runtime: Dict[str, Dict[str, float]]) -> None:
    print(f"==== {prefix} Runtime ====")
    for algo in sorted(runtime.keys()):
        stats = runtime[algo]
        wall = stats.get("wall_s", 0.0)
        cpu = stats.get("cpu_s", 0.0)
        wwall = stats.get("warmup_wall_s", 0.0)
        wcpu = stats.get("warmup_cpu_s", 0.0)
        total_wall = stats.get("total_wall_s", wall + wwall)
        total_cpu = stats.get("total_cpu_s", cpu + wcpu)
        print(
            f"  {algo}: total_wall={total_wall:.6g}s total_cpu={total_cpu:.6g}s"
        )
    print("====")


def timed_call(fn, warmup=False, reset_fn=None):
    warmup_wall = 0.0
    warmup_cpu = 0.0
    if warmup:
        t0 = time.perf_counter()
        c0 = time.process_time()
        fn()
        warmup_wall = time.perf_counter() - t0
        warmup_cpu = time.process_time() - c0
        if reset_fn is not None:
            reset_fn()
    t1 = time.perf_counter()
    c1 = time.process_time()
    out = fn()
    wall = time.perf_counter() - t1
    cpu = time.process_time() - c1
    total_wall = wall + warmup_wall
    total_cpu = cpu + warmup_cpu
    return out, {
        "wall_s": wall,
        "cpu_s": cpu,
        "warmup_wall_s": warmup_wall,
        "warmup_cpu_s": warmup_cpu,
        "total_wall_s": total_wall,
        "total_cpu_s": total_cpu,
    }


def save_runtime(path: Path, runtime: Dict[str, Dict[str, float]]) -> None:
    rows = ["algo,wall_s,cpu_s,warmup_wall_s,warmup_cpu_s,total_wall_s,total_cpu_s"]
    for algo, stats in runtime.items():
        wall = stats.get("wall_s", 0.0)
        cpu = stats.get("cpu_s", 0.0)
        wwall = stats.get("warmup_wall_s", 0.0)
        wcpu = stats.get("warmup_cpu_s", 0.0)
        total_wall = wall + wwall
        total_cpu = cpu + wcpu
        rows.append(f"{algo},{wall:.6f},{cpu:.6f},{wwall:.6f},{wcpu:.6f},{total_wall:.6f},{total_cpu:.6f}")
    path.write_text("\n".join(rows), encoding="utf-8")

def print_particle_log_terms(
    name: str,
    w: tf.Tensor,
    diagnostics: Dict[str, Any],
    max_t: int = 5,
    top_k: int = 5,
    enabled: bool = False,
) -> None:
    if not enabled:
        return
    log_keys = ["loglik", "logtrans", "log_q", "log_q0", "log_det", "log_w"]
    if not all(k in diagnostics for k in log_keys):
        return
    w_np = to_numpy(w)
    diag_np = {k: to_numpy(diagnostics[k]) for k in log_keys}
    T = w_np.shape[1]
    steps = min(max_t, T)
    print(f"[{name} log terms]")
    for t in range(steps):
        wt = w_np[0, t]
        if wt.size <= top_k:
            top_idx = np.arange(wt.size)
        else:
            top_idx = np.argsort(wt)[::-1][:top_k]
        top_vals = wt[top_idx]
        print(f"  t={t} w_top_idx={top_idx.tolist()} w_top={np.round(top_vals, 6).tolist()}")
        for key in log_keys:
            arr = diag_np[key][0, t]
            arr_flat = arr.reshape(-1)
            if arr_flat.size <= top_k:
                arr_str = np.round(arr_flat, 6).tolist()
                print(f"    {key}={arr_str}")
            else:
                top_arr_idx = np.argsort(arr_flat)[::-1][:top_k]
                top_arr_vals = arr_flat[top_arr_idx]
                print(
                    f"    {key}: min={arr_flat.min():.6g} max={arr_flat.max():.6g} mean={arr_flat.mean():.6g}"
                )
                print(
                    f"      top_idx={top_arr_idx.tolist()} top={np.round(top_arr_vals, 6).tolist()}"
                )


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


def make_seeds(num_seeds: int, base_seed: int | None = None) -> list[int]:
    if num_seeds <= 0:
        return []
    if base_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(base_seed))
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=num_seeds)]
