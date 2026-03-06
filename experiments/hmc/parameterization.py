from __future__ import annotations

import tensorflow as tf


def sigma2_to_unconstrained(sigma2: tf.Tensor) -> tf.Tensor:
    """Map positive sigma^2 parameters to log-space."""
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    tf.debugging.assert_positive(sigma2, message="sigma^2 parameters must be positive.")
    return tf.math.log(sigma2)


def unconstrained_to_sigma2(unconstrained: tf.Tensor) -> tf.Tensor:
    """Map unconstrained log-parameters back to sigma^2 space."""
    unconstrained = tf.convert_to_tensor(unconstrained, dtype=tf.float32)
    return tf.exp(unconstrained)


def log_abs_det_jacobian(unconstrained: tf.Tensor) -> tf.Tensor:
    """Log abs det Jacobian for sigma^2 = exp(unconstrained)."""
    unconstrained = tf.convert_to_tensor(unconstrained, dtype=tf.float32)
    return tf.reduce_sum(unconstrained, axis=-1)


def sigma2_sd_to_log_rw_std(sigma2: tf.Tensor, sigma2_sd: tf.Tensor) -> tf.Tensor:
    """Convert sigma^2-scale proposal std to log-space RW std.

    If eta' = eta + eps with eps ~ N(0, s_eta^2) and sigma2 = exp(eta),
    this returns s_eta such that sigma2' = exp(eta') has the requested
    conditional standard deviation in sigma^2 space.
    """
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    sigma2_sd = tf.convert_to_tensor(sigma2_sd, dtype=tf.float32)
    ratio_sq = tf.square(sigma2_sd / sigma2)
    exp_s_sq = 0.5 * (1.0 + tf.sqrt(1.0 + 4.0 * ratio_sq))
    return tf.sqrt(tf.math.log(exp_s_sq))
