"""Standard (systematic) resampling DPF baseline."""

from __future__ import annotations

import tensorflow as tf

from src.filters.dpf.base import DPFBase


class StandardResamplingDPF(DPFBase):
    """DPF variant with classic systematic resampling and uniform reset."""

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor,
                      training: bool | None = None):
        w = tf.exp(log_w)
        parent_indices = self.systematic_resample(w, self.ssm.rng)
        x_new = self.resample_particles(x, parent_indices)
        n_particles = tf.shape(log_w)[-1]
        log_uniform = -tf.math.log(tf.cast(n_particles, log_w.dtype))
        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        return x_new, log_w_new, parent_indices


__all__ = [
    "StandardResamplingDPF",
]
