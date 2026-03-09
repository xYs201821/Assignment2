from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


def sanitize_log_tensor(values: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(values)
    neg_large = tf.cast(-1e30, values.dtype)
    return tf.where(tf.math.is_finite(values), values, neg_large * tf.ones_like(values))


def normalize_log_weights(log_w: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_w = sanitize_log_tensor(log_w)
    logz = tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
    log_w_n = log_w - logz
    return log_w_n, tf.exp(log_w_n), tf.squeeze(logz, axis=-1)


def to_stateless_seed(seed: tf.Tensor | int) -> tf.Tensor:
    seed_t = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed_t.shape.rank == 0:
        return tf.stack([seed_t, seed_t + tf.constant(1, tf.int32)], axis=0)
    seed_t = tf.reshape(seed_t, [-1])
    if tf.shape(seed_t)[0] >= 2:
        return tf.stack([seed_t[0], seed_t[1]], axis=0)
    return tf.stack([seed_t[0], seed_t[0] + tf.constant(1, tf.int32)], axis=0)


def split_seed(seed: tf.Tensor, n: int) -> tf.Tensor:
    return tf.random.experimental.stateless_split(to_stateless_seed(seed), n)


def systematic_resample_indices(weights: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    shape = tf.shape(weights)
    n_particles = shape[-1]
    batch = tf.reduce_prod(shape[:-1])
    w2 = tf.reshape(weights, [batch, n_particles])
    cdf = tf.cumsum(w2, axis=-1)
    seed = to_stateless_seed(seed)
    u0 = tf.random.stateless_uniform(
        [batch, 1],
        seed=seed,
        minval=0.0,
        maxval=1.0 / tf.cast(n_particles, tf.float32),
        dtype=tf.float32,
    )
    js = tf.cast(tf.range(n_particles)[tf.newaxis, :], tf.float32)
    u = u0 + js / tf.cast(n_particles, tf.float32)
    idx = tf.searchsorted(cdf, u, side="left")
    idx = tf.clip_by_value(idx, 0, n_particles - 1)
    return tf.reshape(idx, shape)


def gather_particles(x: tf.Tensor, idx: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)
    batch = tf.reduce_prod(shape[:-2])
    x_flat = tf.reshape(x, [batch, shape[-2], shape[-1]])
    idx_flat = tf.reshape(idx, [batch, shape[-2]])
    out_flat = tf.gather(x_flat, idx_flat, batch_dims=1)
    return tf.reshape(out_flat, shape)


def pairwise_distance(x: tf.Tensor, y: tf.Tensor | None = None) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    y = x if y is None else tf.convert_to_tensor(y, dtype=x.dtype)
    x_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    y_sq = tf.reduce_sum(tf.square(y), axis=-1)[:, tf.newaxis, :]
    dist = x_sq - 2.0 * tf.matmul(x, y, transpose_b=True) + y_sq
    return tf.maximum(dist, tf.zeros_like(dist))


def sinkhorn_log_plan(
    log_a: tf.Tensor,
    log_b: tf.Tensor,
    cost: tf.Tensor,
    epsilon: float,
    num_iters: int,
) -> tf.Tensor:
    if int(num_iters) <= 0:
        raise ValueError("num_iters must be a positive integer.")

    eps = tf.cast(epsilon, cost.dtype)
    log_k = -cost / eps
    f = tf.zeros_like(log_a)
    g = tf.zeros_like(log_b)
    for _ in range(int(num_iters)):
        f = log_a - tf.reduce_logsumexp(log_k + g[:, tf.newaxis, :], axis=-1)
        g = log_b - tf.reduce_logsumexp(log_k + f[:, :, tf.newaxis], axis=-2)
    return f[:, :, tf.newaxis] + log_k + g[:, tf.newaxis, :]


def sinkhorn_matrix_scaling(
    a: tf.Tensor,
    b: tf.Tensor,
    cost: tf.Tensor,
    epsilon: float,
    num_iters: int,
) -> tf.Tensor:
    eps = tf.cast(epsilon, cost.dtype)
    k = tf.exp(-cost / eps)
    tiny = tf.cast(1e-16, cost.dtype)
    k = tf.maximum(k, tiny)

    u = tf.ones_like(a)
    v = tf.ones_like(b)
    for _ in range(int(num_iters)):
        kv = tf.einsum("bij,bj->bi", k, v)
        u = tf.math.divide_no_nan(a, tf.maximum(kv, tiny))
        ktu = tf.einsum("bij,bi->bj", k, u)
        v = tf.math.divide_no_nan(b, tf.maximum(ktu, tiny))

    plan = u[:, :, tf.newaxis] * k * v[:, tf.newaxis, :]
    plan = tf.maximum(plan, tiny)
    return plan


@dataclass
class StandardResampler:
    def resample(self, x: tf.Tensor, log_w: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        log_w, w, _ = normalize_log_weights(log_w)
        idx = systematic_resample_indices(w, seed)
        x_new = gather_particles(x, idx)
        n_particles = tf.shape(log_w)[-1]
        log_uniform = -tf.math.log(tf.cast(n_particles, log_w.dtype))
        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        return x_new, log_w_new, idx


@dataclass
class SoftResampler:
    """Soft resampling: sample from mixture q_i = λ·w_i + (1-λ)/N, then
    correct importance weights by w_new_i ∝ w_{idx_i} / q_{idx_i}."""

    lam: float = 0.95  # mixing coefficient; 1.0 → pure categorical, 0 → pure uniform

    def resample(self, x: tf.Tensor, log_w: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        log_w, _, _ = normalize_log_weights(log_w)
        lam = tf.cast(self.lam, log_w.dtype)
        n = tf.shape(log_w)[-1]
        log_uniform = -tf.math.log(tf.cast(n, log_w.dtype))
        log_lam = tf.math.log(tf.clip_by_value(lam, tf.constant(1e-6, log_w.dtype), tf.constant(1.0, log_w.dtype)))
        log_one_minus_lam = tf.math.log(tf.maximum(1.0 - lam, tf.constant(1e-6, log_w.dtype)))

        log_q = tf.reduce_logsumexp(
            tf.stack(
                [log_lam + log_w, log_one_minus_lam + log_uniform * tf.ones_like(log_w)],
                axis=0,
            ),
            axis=0,
        )
        # TF 2.19's stateless categorical can emit the out-of-range sentinel
        # `num_classes` when a row of logits is entirely non-finite.
        log_q = sanitize_log_tensor(log_q)

        idx = tf.random.stateless_categorical(log_q, num_samples=n, seed=to_stateless_seed(seed), dtype=tf.int32)
        x_new = gather_particles(x, idx)
        log_w_new = tf.gather(log_w, idx, batch_dims=1) - tf.gather(log_q, idx, batch_dims=1)
        log_w_new, _, _ = normalize_log_weights(log_w_new)
        return x_new, log_w_new, idx


@dataclass
class OTResampler:
    epsilon: float = 0.1
    num_iters: int = 50
    jitter: float = 1e-8

    def resample(self, x: tf.Tensor, log_w: tf.Tensor, seed: tf.Tensor | None = None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        log_w, w, _ = normalize_log_weights(log_w)
        seed = tf.constant([0, 1], dtype=tf.int32) if seed is None else to_stateless_seed(seed)
        n_particles = tf.shape(x)[-2]
        uniform_mass = tf.math.reciprocal(tf.cast(n_particles, x.dtype))
        log_uniform = tf.math.log(uniform_mass)
        log_b = tf.fill(tf.shape(log_w), log_uniform)
        b = tf.fill(tf.shape(log_w), uniform_mass)
        target_idx = systematic_resample_indices(w, seed)
        x_target = gather_particles(x, target_idx)
        x_target = tf.stop_gradient(x_target)
        cost = pairwise_distance(x, x_target)
        # log_plan = sinkhorn_log_plan(log_w, log_b, cost, epsilon=self.epsilon, num_iters=self.num_iters)
        # plan = tf.exp(log_plan)
        plan = sinkhorn_matrix_scaling(w, b, cost, epsilon=self.epsilon, num_iters=self.num_iters)
       
        col_mass = tf.reduce_sum(plan, axis=-2)
        weighted_sum = tf.einsum("bij,bid->bjd", plan, x)
        x_new = weighted_sum / tf.maximum(col_mass[..., tf.newaxis], tf.cast(self.jitter, x.dtype))
        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        parent = tf.argmax(plan, axis=-2, output_type=tf.int32)
        return x_new, log_w_new, parent
