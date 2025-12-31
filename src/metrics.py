from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.utility import cholesky_solve, quadratic_matmul, tf_cond


def _to_tensor(x: Any, dtype=tf.float32) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=dtype)


def rmse(x_true: Any, x_est: Any, dims: Optional[Sequence[int]] = None) -> tf.Tensor:
    x_true = _to_tensor(x_true)
    x_est = _to_tensor(x_est)
    err = x_true - x_est
    if dims is not None:
        err = tf.gather(err, dims, axis=-1)
    per_step = tf.norm(err, axis=-1)
    return tf.sqrt(tf.reduce_mean(tf.square(per_step)))


def rmse_per_t(x_true: Any, x_est: Any, dims: Optional[Sequence[int]] = None) -> tf.Tensor:
    x_true = _to_tensor(x_true)
    x_est = _to_tensor(x_est)
    err = x_true - x_est
    if dims is not None:
        err = tf.gather(err, dims, axis=-1)
    return tf.norm(err, axis=-1)


def rmse_from_error(err: Any) -> tf.Tensor:
    err = _to_tensor(err)
    per_step = tf.norm(err, axis=-1)
    return tf.sqrt(tf.reduce_mean(tf.square(per_step)))


def ess(w: Any, axis: int = -1) -> tf.Tensor:
    w = _to_tensor(w)
    return 1.0 / tf.reduce_sum(tf.square(w), axis=axis)


def _unique_parent_fraction(parents: Any) -> Optional[np.ndarray]:
    parents = np.asarray(parents)
    if parents.ndim == 2:
        parents = parents[np.newaxis, ...]
    if parents.ndim != 3:
        return None
    batch, T, N = parents.shape
    if N == 0 or T == 0:
        return None
    out = np.zeros((batch, T), dtype=np.float32)
    for b in range(batch):
        for t in range(T):
            out[b, t] = np.unique(parents[b, t]).size / float(N)
    return out


def nees(x_true: Any, mean: Any, cov: Any) -> tf.Tensor:
    x_true = _to_tensor(x_true)
    mean = _to_tensor(mean)
    cov = _stabilize_cov(_to_tensor(cov))
    err = (mean - x_true)[..., tf.newaxis]
    sol = cholesky_solve(cov, err, jitter=1e-6)
    val = tf.reduce_sum(err * sol, axis=(-2, -1))
    return val


def nis(innovation: Any, S: Any) -> tf.Tensor:
    innovation = _to_tensor(innovation)
    S = _to_tensor(S)
    S = _stabilize_cov(S)
    v = innovation[..., tf.newaxis]
    sol = cholesky_solve(S, v, jitter=1e-6)
    return tf.reduce_sum(v * sol, axis=(-2, -1))


def _stabilize_cov(cov: tf.Tensor, jitter: float = 1e-5) -> tf.Tensor:
    cov = _to_tensor(cov)
    cov = 0.5 * (cov + tf.linalg.matrix_transpose(cov))
    eye = tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype)
    return cov + eye * tf.cast(jitter, cov.dtype)


def gaussian_log_prob_from_error(error: Any, cov: Any, jitter: float = 1e-6) -> tf.Tensor:
    error = _to_tensor(error)
    cov = _stabilize_cov(_to_tensor(cov), jitter=jitter)
    chol = tf.linalg.cholesky(cov + jitter * tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype))
    sol = tf.linalg.triangular_solve(chol, error[..., tf.newaxis], lower=True)
    maha = tf.reduce_sum(tf.square(sol), axis=(-2, -1))
    dim = tf.cast(tf.shape(error)[-1], tf.float32)
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    return -0.5 * (dim * tf.math.log(2.0 * np.pi) + log_det + maha)


def gaussian_log_prob(y: Any, mean: Any, cov: Any, jitter: float = 1e-6) -> tf.Tensor:
    y = _to_tensor(y)
    mean = _to_tensor(mean)
    return gaussian_log_prob_from_error(y - mean, cov, jitter=jitter)

@tf.function(reduce_retracing=True)
def linearized_obs_stats(
    ssm,
    mean: tf.Tensor,
    cov: tf.Tensor,
    jitter: float = 1e-6,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mean = _to_tensor(mean)
    cov = _to_tensor(cov)
    batch_shape = tf.shape(mean)[:-1]
    batch_size = tf.reduce_prod(batch_shape)
    state_dim = tf.shape(mean)[-1]
    r_dim = tf.cast(ssm.r_dim, tf.int32)
    obs_dim = tf.cast(ssm.obs_dim, tf.int32)

    mean_flat = tf.reshape(mean, tf.stack([batch_size, state_dim]))
    r0 = tf.zeros(tf.stack([batch_size, r_dim]), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(mean_flat)
        y_pred_flat = ssm.h_with_noise(mean_flat, r0)
    H_x_flat = tape.batch_jacobian(y_pred_flat, mean_flat)
    with tf.GradientTape() as tape_r:
        tape_r.watch(r0)
        y_r = ssm.h_with_noise(mean_flat, r0)
    H_r_flat = tape_r.batch_jacobian(y_r, r0)

    H_x = tf.reshape(H_x_flat, tf.concat([batch_shape, [obs_dim, state_dim]], axis=0))
    H_r = tf.reshape(H_r_flat, tf.concat([batch_shape, [obs_dim, r_dim]], axis=0))
    y_pred = tf.reshape(y_pred_flat, tf.concat([batch_shape, [obs_dim]], axis=0))

    R = _to_tensor(ssm.cov_eps_y)
    R_eff = quadratic_matmul(H_r, R, H_r)
    S = quadratic_matmul(H_x, cov, H_x) + R_eff
    S = _stabilize_cov(S, jitter=jitter)
    return y_pred, S, H_x


def nll_from_particles(ssm, y: tf.Tensor, x_particles: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    loglik = ssm.observation_dist(x_particles).log_prob(y[..., tf.newaxis, :])
    w = _to_tensor(w)
    w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
    log_w = tf.math.log(w + tf.keras.backend.epsilon())
    logZ = tf.reduce_logsumexp(log_w + loglik, axis=-1)
    return -logZ


def nll_from_gaussian(ssm, y: tf.Tensor, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
    y_pred, S, _ = linearized_obs_stats(ssm, mean, cov)
    innovation = ssm.innovation(y, y_pred)
    return -gaussian_log_prob_from_error(innovation, S)


def rank_histogram(x_true: np.ndarray, particles: np.ndarray) -> np.ndarray:
    x_true = np.asarray(x_true)
    particles = np.asarray(particles)
    if x_true.ndim == 2:
        x_true = x_true[np.newaxis, ...]
    if particles.ndim == 3:
        particles = particles[np.newaxis, ...]
    _, T, N, dx = particles.shape
    counts = np.zeros((dx, N + 1), dtype=np.int64)
    for d in range(dx):
        p = particles[..., d]
        t = x_true[..., d]
        rank = np.sum(p < t[..., None], axis=-1)
        for r in rank.reshape(-1):
            counts[d, int(r)] += 1
    return counts


def coverage_from_gaussian(x_true: Any, mean: Any, cov: Any, sigmas: Iterable[float]) -> Dict[float, float]:
    x_true = _to_tensor(x_true)
    mean = _to_tensor(mean)
    cov = _to_tensor(cov)
    var = tf.linalg.diag_part(cov)
    std = tf.sqrt(tf.maximum(var, tf.keras.backend.epsilon()))
    out = {}
    for k in sigmas:
        k = float(k)
        inside = tf.abs(x_true - mean) <= k * std
        out[k] = float(tf.reduce_mean(tf.cast(inside, tf.float32)).numpy())
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values).reshape(-1)
    weights = np.asarray(weights).reshape(-1)
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]
    cum = np.cumsum(weights)
    if cum[-1] <= 0:
        return float(values[-1])
    target = q * cum[-1]
    return float(values[np.searchsorted(cum, target, side="left")])


def coverage_from_particles(
    x_true: np.ndarray,
    particles: np.ndarray,
    sigmas: Iterable[float],
    weights: Optional[np.ndarray] = None,
) -> Dict[float, float]:
    x_true = np.asarray(x_true)
    particles = np.asarray(particles)
    if x_true.ndim == 2:
        x_true = x_true[np.newaxis, ...]
    if particles.ndim == 3:
        particles = particles[np.newaxis, ...]
    _, T, N, dx = particles.shape
    out = {float(k): 0.0 for k in sigmas}
    total = T * dx
    for k in sigmas:
        k = float(k)
        alpha = 0.5 * (1.0 - math.erf(k / math.sqrt(2.0)))
        lo_q = alpha
        hi_q = 1.0 - alpha
        count = 0
        for t in range(T):
            for d in range(dx):
                vals = particles[0, t, :, d]
                if weights is not None:
                    w = np.asarray(weights)[0, t, :]
                    lo = weighted_quantile(vals, w, lo_q)
                    hi = weighted_quantile(vals, w, hi_q)
                else:
                    lo = np.quantile(vals, lo_q)
                    hi = np.quantile(vals, hi_q)
                if lo <= x_true[0, t, d] <= hi:
                    count += 1
        out[k] = count / float(total)
    return out


def _assignment_min_cost(cost: np.ndarray) -> Tuple[float, Tuple[int, ...]]:
    n, m = cost.shape
    if n == 0 or m == 0:
        return 0.0, tuple()
    if n == m and n <= 8:
        best = None
        best_perm = None
        for perm in itertools.permutations(range(m)):
            val = cost[np.arange(n), perm].sum()
            if best is None or val < best:
                best = val
                best_perm = perm
        return float(best), tuple(best_perm)
    # Fallback: greedy matching (simple but not optimal for large n)
    used = set()
    total = 0.0
    perm = [-1] * n
    for i in range(n):
        j = int(np.argmin(cost[i]))
        if j in used:
            j = int(np.argmin([c if k not in used else np.inf for k, c in enumerate(cost[i])]))
        used.add(j)
        perm[i] = j
        total += cost[i, j]
    return float(total), tuple(perm)


def ospa_distance(
    x_true: np.ndarray,
    x_est: np.ndarray,
    cutoff: float = 10.0,
    p: float = 1.0,
) -> float:
    x_true = np.asarray(x_true, dtype=np.float32)
    x_est = np.asarray(x_est, dtype=np.float32)
    n = x_true.shape[0]
    m = x_est.shape[0]
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(cutoff)
    cost = np.linalg.norm(x_true[:, None, :] - x_est[None, :, :], axis=-1)
    cost = np.minimum(cost, cutoff) ** p
    if n <= m:
        min_cost, _ = _assignment_min_cost(cost)
        total = (min_cost + (m - n) * (cutoff ** p)) / float(m)
    else:
        min_cost, _ = _assignment_min_cost(cost.T)
        total = (min_cost + (n - m) * (cutoff ** p)) / float(n)
    return float(total ** (1.0 / p))


def best_match_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=np.float32)
    x_est = np.asarray(x_est, dtype=np.float32)
    n = x_true.shape[0]
    m = x_est.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    cost = np.linalg.norm(x_true[:, None, :] - x_est[None, :, :], axis=-1)
    min_cost, perm = _assignment_min_cost(cost)
    if n <= m:
        matched = cost[np.arange(n), perm]
    else:
        matched = cost.T[np.arange(m), perm]
    return float(np.sqrt(np.mean(np.square(matched))))


def split_obs_indices(ssm, state_dim: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if hasattr(ssm, "obs_indices"):
        obs = np.asarray(getattr(ssm, "obs_indices"), dtype=np.int64)
    elif hasattr(ssm, "obs_mask"):
        mask = np.asarray(getattr(ssm, "obs_mask"), dtype=bool)
        obs = np.where(mask)[0]
    else:
        return None, None
    all_idx = np.arange(state_dim, dtype=np.int64)
    obs = np.unique(obs)
    unobs = np.setdiff1d(all_idx, obs)
    return obs, unobs


def default_metrics_config() -> Dict[str, Any]:
    return {
        "rmse": True,
        "rmse_obs": True,
        "rmse_unobs": True,
        "rmse_y": True,
        "nll": True,
        "nees": True,
        "nis": True,
        "ess": True,
        "impoverishment": True,
        "coverage_sigmas": (1.0, 2.0),
        "rank_hist": True,
        "flow_diagnostics": True,
    }


def _diag_array(diagnostics: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if not diagnostics:
        return None
    val = diagnostics.get(key)
    if val is None:
        return None
    return np.asarray(val)


def _flow_diagnostics_metrics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    dx_p95 = _diag_array(diagnostics, "dx_p95_max")
    if dx_p95 is not None:
        metrics["dx_p95_max_max"] = float(np.nanmax(dx_p95))
    condS = _diag_array(diagnostics, "condS_log10_max")
    if condS is not None:
        metrics["condS_log10_max_max"] = float(np.nanmax(condS))
    condK = _diag_array(diagnostics, "condK_log10_max")
    if condK is not None:
        metrics["condK_log10_max_max"] = float(np.nanmax(condK))
    logdet_cov = _diag_array(diagnostics, "logdet_cov")
    if logdet_cov is not None:
        metrics["logdet_cov_min"] = float(np.nanmin(logdet_cov))
    return metrics


def evaluate(
    ssm,
    x_true: Any,
    y_obs: Any,
    outputs: Dict[str, Any],
    metrics_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = default_metrics_config()
    if metrics_cfg:
        cfg.update(metrics_cfg)

    metrics: Dict[str, Any] = {}
    x_true_t = _to_tensor(x_true)
    y_obs_t = _to_tensor(y_obs)
    mean = outputs.get("mean")
    cov = outputs.get("cov")
    x_particles = outputs.get("x_particles")
    w = outputs.get("w")
    w_pre = outputs.get("w_pre")
    if w_pre is None:
        diagnostics = outputs.get("diagnostics")
        if isinstance(diagnostics, dict):
            w_pre = diagnostics.get("w_pre")
    parents = outputs.get("parents")
    is_gaussian = bool(outputs.get("is_gaussian", False))

    if cfg.get("rmse"):
        metrics["rmse_state"] = float(rmse(x_true_t, mean).numpy())

    obs_idx, unobs_idx = split_obs_indices(ssm, int(x_true_t.shape[-1]))
    if cfg.get("rmse_obs") and obs_idx is not None:
        metrics["rmse_obs"] = float(rmse(x_true_t, mean, dims=obs_idx).numpy())
    if cfg.get("rmse_unobs") and unobs_idx is not None:
        metrics["rmse_unobs"] = float(rmse(x_true_t, mean, dims=unobs_idx).numpy())

    if cfg.get("rmse_y"):
        if x_particles is not None:
            y_pred = ssm.h(x_particles)
            if w is None:
                y_mean = ssm.measurement_mean(y_pred)
            else:
                y_mean = ssm.measurement_mean(y_pred, w)
        else:
            y_mean = ssm.h(mean)
        innovation = ssm.innovation(y_obs_t, y_mean)
        metrics["rmse_y"] = float(rmse_from_error(innovation).numpy())

    if cfg.get("nll"):
        if is_gaussian:
            nll_t = nll_from_gaussian(ssm, y_obs_t, mean, cov)
        elif x_particles is not None and w is not None:
            nll_t = nll_from_particles(ssm, y_obs_t, x_particles, w)
        else:
            nll_t = nll_from_gaussian(ssm, y_obs_t, mean, cov)
        metrics["nll"] = float(tf.reduce_mean(nll_t).numpy())

    if cfg.get("nees") and cov is not None:
        nees_t = nees(x_true_t, mean, cov)
        metrics["nees"] = float(tf.reduce_mean(nees_t).numpy())

    if cfg.get("nis") and cov is not None:
        mean_nis = outputs.get("m_pred", mean)
        cov_nis = outputs.get("P_pred", cov)
        y_pred, S, _ = linearized_obs_stats(ssm, mean_nis, cov_nis)
        v = ssm.innovation(y_obs_t, y_pred)
        nis_t = nis(v, S)
        metrics["nis"] = float(tf.reduce_mean(nis_t).numpy())

    w_for_ess = w_pre if w_pre is not None else w
    if cfg.get("ess") and w_for_ess is not None and not is_gaussian:
        ess_t = ess(w_for_ess)
        metrics["ess_mean"] = float(tf.reduce_mean(ess_t).numpy())
        ess_len = tf.shape(ess_t)[-1]
        ess_min_source = tf.cond(ess_len > 1, lambda: ess_t[..., 1:], lambda: ess_t)
        metrics["ess_min"] = float(tf.reduce_min(ess_min_source).numpy())
        metrics["ess_final"] = float(tf.reduce_mean(ess_t[..., -1]).numpy())

    if cfg.get("impoverishment") and parents is not None and not is_gaussian:
        unique_frac = _unique_parent_fraction(parents)
        if unique_frac is not None:
            impoverishment = 1.0 - unique_frac
            metrics["impoverishment_mean"] = float(np.mean(impoverishment))
            metrics["impoverishment_final"] = float(np.mean(impoverishment[:, -1]))

    sigmas = cfg.get("coverage_sigmas")
    if sigmas:
        if is_gaussian or x_particles is None:
            metrics["coverage"] = coverage_from_gaussian(x_true_t, mean, cov, sigmas)
        else:
            metrics["coverage"] = coverage_from_particles(
                np.asarray(x_true),
                np.asarray(x_particles),
                sigmas,
                weights=np.asarray(w) if w is not None else None,
            )

    if cfg.get("rank_hist") and x_particles is not None and not is_gaussian:
        metrics["rank_hist"] = rank_histogram(
            np.asarray(x_true), np.asarray(x_particles)
        ).tolist()

    diagnostics = outputs.get("diagnostics")
    if cfg.get("flow_diagnostics") and isinstance(diagnostics, dict):
        metrics.update(_flow_diagnostics_metrics(diagnostics))

    return metrics
