"""Optimal-transport resampling utilities and DPF variant."""

from __future__ import annotations

import tensorflow as tf

from src.filters.dpf import DPFBase


def pairwise_distance(
    x: tf.Tensor,
    y: tf.Tensor | None = None,
    metric_fn=None,
) -> tf.Tensor:
    """Pairwise distances with an optional custom metric function.

    Shapes:
      x: [B, N, dx] or [N, dx]
      y: [B, M, dx] or [M, dx] (defaults to x)
      metric_fn: callable receiving broadcasted tensors
        x_exp=[B, N, 1, dx], y_exp=[B, 1, M, dx] and returning [B, N, M].
        If None, uses squared Euclidean distance.
    Returns:
      dist: [B, N, M] or [N, M]
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x_rank2 = x.shape.rank == 2
    if x_rank2:
        x = x[tf.newaxis, ...]
    if x.shape.rank != 3:
        raise ValueError("x must have shape [B, N, dx] or [N, dx].")

    if y is None:
        y = x
        y_rank2 = x_rank2
    else:
        y = tf.convert_to_tensor(y, dtype=x.dtype)
        y_rank2 = y.shape.rank == 2
        if y_rank2:
            y = y[tf.newaxis, ...]
        if y.shape.rank != 3:
            raise ValueError("y must have shape [B, M, dx] or [M, dx].")

    if metric_fn is None:
        x_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y_sq = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        dist = x_sq - 2.0 * tf.matmul(x, y, transpose_b=True) + tf.transpose(y_sq, perm=[0, 2, 1])
        dist = tf.maximum(dist, tf.zeros_like(dist))
    else:
        x_exp = x[:, :, tf.newaxis, :]
        y_exp = y[:, tf.newaxis, :, :]
        dist = tf.convert_to_tensor(metric_fn(x_exp, y_exp), dtype=x.dtype)
        if dist.shape.rank != 3:
            raise ValueError("metric_fn must return a tensor with shape [B, N, M].")
        tf.debugging.assert_equal(
            tf.shape(dist),
            tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(y)[1]]),
            message="metric_fn output shape must be [B, N, M].",
        )

    if x_rank2 and y_rank2:
        return dist[0]
    return dist


def sinkhorn_log_plan(
    log_a: tf.Tensor,
    log_b: tf.Tensor,
    cost: tf.Tensor,
    epsilon: float | tf.Tensor = 0.1,
    num_iters: int = 50,
) -> tf.Tensor:
    """Compute an entropic OT transport plan in log-domain.

    Shapes:
      log_a: [B, N]
      log_b: [B, M]
      cost: [B, N, M]
    Returns:
      log_pi: [B, N, M]
    """
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


def ot_resample_barycentric(
    x: tf.Tensor,
    log_w: tf.Tensor,
    epsilon: float | tf.Tensor = 0.1,
    num_iters: int = 50,
    jitter: float = 1e-8,
):
    """OT resampling with barycentric projection to equal-weight particles.

    Shapes:
      x: [B, N, dx] or [N, dx]
      log_w: [B, N] or [N]
    Returns:
      x_new: [B, N, dx] or [N, dx]
      log_w_new: [B, N] or [N] (uniform)
      parent_indices: [B, N] or [N] (argmax lineage proxy)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    squeeze_batch = False
    if x.shape.rank == 2:
        x = x[tf.newaxis, ...]
        squeeze_batch = True
    if x.shape.rank != 3:
        raise ValueError("x must have shape [B, N, dx] or [N, dx].")

    log_w = tf.convert_to_tensor(log_w, dtype=x.dtype)
    if log_w.shape.rank == 1:
        log_w = log_w[tf.newaxis, ...]
    if log_w.shape.rank != 2:
        raise ValueError("log_w must have shape [B, N] or [N].")

    log_w = log_w - tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
    num_particles = tf.shape(x)[-2]
    log_uniform = -tf.math.log(tf.cast(num_particles, x.dtype))
    log_b = tf.fill(tf.shape(log_w), log_uniform)

    cost = pairwise_distance(x)
    log_plan = sinkhorn_log_plan(log_w, log_b, cost, epsilon=epsilon, num_iters=num_iters)
    plan = tf.exp(log_plan)

    col_mass = tf.reduce_sum(plan, axis=-2)  # [B, N]
    weighted_sum = tf.einsum("bij,bid->bjd", plan, x)
    jitter_t = tf.convert_to_tensor(jitter, dtype=x.dtype)
    x_new = weighted_sum / tf.maximum(col_mass[..., tf.newaxis], jitter_t)

    log_w_new = tf.fill(tf.shape(log_w), log_uniform)
    parent_indices = tf.argmax(plan, axis=-2, output_type=tf.int32)

    if squeeze_batch:
        return x_new[0], log_w_new[0], parent_indices[0]
    return x_new, log_w_new, parent_indices


class OTResamplingDPF(DPFBase):
    """Differentiable PF using deterministic entropic OT resampling."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        ot_epsilon: float = 0.1,
        ot_num_iters: int = 50,
        ot_jitter: float = 1e-6,
        resample: str | int | bool = "auto",
        debug: bool = False,
        print: bool = False,
        proposal=None,
    ) -> None:
        super().__init__(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            debug=debug,
            print=print,
            proposal=proposal,
        )
        self.ot_epsilon = float(ot_epsilon)
        self.ot_num_iters = int(ot_num_iters)
        self.ot_jitter = float(ot_jitter)

    @staticmethod
    def pairwise_distance(
        x: tf.Tensor,
        y: tf.Tensor | None = None,
        metric_fn=None,
    ) -> tf.Tensor:
        """Class-level access to pairwise distance utility."""
        return pairwise_distance(x, y=y, metric_fn=metric_fn)

    @staticmethod
    def sinkhorn_log_plan(
        log_a: tf.Tensor,
        log_b: tf.Tensor,
        cost: tf.Tensor,
        epsilon: float | tf.Tensor = 0.1,
        num_iters: int = 50,
    ) -> tf.Tensor:
        """Class-level access to log-domain Sinkhorn utility."""
        return sinkhorn_log_plan(
            log_a=log_a,
            log_b=log_b,
            cost=cost,
            epsilon=epsilon,
            num_iters=num_iters,
        )

    def update_params(
        self,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        ot_epsilon=None,
        ot_num_iters=None,
        ot_jitter=None,
    ):
        super().update_params(
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )
        if ot_epsilon is not None:
            self.ot_epsilon = float(ot_epsilon)
        if ot_num_iters is not None:
            self.ot_num_iters = int(ot_num_iters)
        if ot_jitter is not None:
            self.ot_jitter = float(ot_jitter)

    def ot_resample_barycentric(
        self,
        x: tf.Tensor,
        log_w: tf.Tensor,
        epsilon: float | tf.Tensor | None = None,
        num_iters: int | None = None,
        jitter: float | tf.Tensor | None = None,
        metric_fn=None,
    ):
        """Class method interface for OT barycentric resampling."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        squeeze_batch = False
        if x.shape.rank == 2:
            x = x[tf.newaxis, ...]
            squeeze_batch = True
        if x.shape.rank != 3:
            raise ValueError("x must have shape [B, N, dx] or [N, dx].")

        log_w = tf.convert_to_tensor(log_w, dtype=x.dtype)
        if log_w.shape.rank == 1:
            log_w = log_w[tf.newaxis, ...]
        if log_w.shape.rank != 2:
            raise ValueError("log_w must have shape [B, N] or [N].")

        eps = self.ot_epsilon if epsilon is None else tf.cast(epsilon, x.dtype)
        iters = self.ot_num_iters if num_iters is None else int(num_iters)
        jit = self.ot_jitter if jitter is None else tf.cast(jitter, x.dtype)

        log_w = log_w - tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
        num_particles = tf.shape(x)[-2]
        log_uniform = -tf.math.log(tf.cast(num_particles, x.dtype))
        log_b = tf.fill(tf.shape(log_w), log_uniform)

        cost = self.pairwise_distance(x, metric_fn=metric_fn)
        log_plan = self.sinkhorn_log_plan(log_w, log_b, cost, epsilon=eps, num_iters=iters)
        plan = tf.exp(log_plan)

        col_mass = tf.reduce_sum(plan, axis=-2)  # [B, N]
        weighted_sum = tf.einsum("bij,bid->bjd", plan, x)
        x_new = weighted_sum / tf.maximum(col_mass[..., tf.newaxis], tf.cast(jit, x.dtype))

        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        parent_indices = tf.argmax(plan, axis=-2, output_type=tf.int32)

        if squeeze_batch:
            return x_new[0], log_w_new[0], parent_indices[0]
        return x_new, log_w_new, parent_indices

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor):
        return self.ot_resample_barycentric(
            x,
            log_w,
        )

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        ot_epsilon=None,
        ot_num_iters=None,
        ot_jitter=None,
        resample=None,
        proposal=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        if any(
            v is not None
            for v in (
                num_particles,
                ess_threshold,
                ot_epsilon,
                ot_num_iters,
                ot_jitter,
                resample,
                proposal,
            )
        ):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                ot_jitter=ot_jitter,
                resample=resample,
                proposal=proposal,
            )
        return super().filter(
            y,
            num_particles=None,
            ess_threshold=None,
            resample=None,
            proposal=None,
            init_dist=init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )


__all__ = [
    "pairwise_distance",
    "sinkhorn_log_plan",
    "ot_resample_barycentric",
    "OTResamplingDPF",
]
