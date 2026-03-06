"""Soft-resampling DPF variant."""

from __future__ import annotations

import tensorflow as tf

from src.filters.dpf.base import DPFBase
import src.dtype_config as _dc


def categorical_sample(prob: tf.Tensor, num_samples: int, replacement: bool = True) -> tf.Tensor:
    """Sample categorical indices from a single probability vector."""
    if not replacement:
        raise NotImplementedError("Only replacement=True is implemented.")
    prob = tf.math.divide_no_nan(prob, tf.reduce_sum(prob))
    logits = tf.math.log(tf.clip_by_value(prob, 1e-12, 1.0))
    draws = tf.random.categorical(logits[tf.newaxis, :], num_samples=tf.cast(num_samples, tf.int32))
    return tf.cast(draws[0], tf.int32)


class SoftResamplingDPF(DPFBase):
    """DPF variant using mixture-with-uniform soft resampling."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        lam: float = 0.95,
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
        self.lam = self._validate_lam(lam)

    @staticmethod
    def _validate_lam(lam) -> tf.Tensor:
        lam = float(lam)
        if not (0.0 < lam <= 1.0):
            raise ValueError("lam must be in (0, 1].")
        return tf.convert_to_tensor(lam, dtype=_dc.DTYPE)

    def update_params(
        self,
        num_particles=None,
        ess_threshold=None,
        lam=None,
        resample=None,
        proposal=None,
    ):
        super().update_params(
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )
        if lam is not None:
            self.lam = self._validate_lam(lam)

    def soft_resampling_mixture(
        self,
        x: tf.Tensor,
        log_w: tf.Tensor,
        lam: tf.Tensor | None = None,
    ):
        """
        Soft-resampling by mixing normalized weights with a uniform proposal.
        """
        if lam is None:
            lam = self.lam
        lam = tf.cast(lam, x.dtype)
        log_lam = tf.math.log(tf.clip_by_value(lam, tf.constant(1e-6, x.dtype), tf.constant(1.0, x.dtype)))
        log_uniform = -tf.math.log(tf.cast(self.num_particles, x.dtype))
        log_one_minus_lam = tf.math.log(tf.maximum(1.0 - lam, tf.constant(1e-6, x.dtype)))

        log_q = tf.reduce_logsumexp(
            tf.stack(
                [
                    log_lam + log_w,
                    log_one_minus_lam + log_uniform + tf.zeros_like(log_w),
                ],
                axis=0,
            ),
            axis=0,
        )
        seed_pair = tf.cast(self.ssm.rng.make_seeds(2)[0], dtype=tf.int32)
        parent_indices = tf.random.stateless_categorical(
            tf.stop_gradient(log_q),
            num_samples=tf.cast(self.num_particles, tf.int32),
            seed=seed_pair,
            dtype=tf.int32, 
        )

        x_new = self.resample_particles(x, parent_indices)
        log_w_new = tf.gather(log_w, parent_indices, batch_dims=1) - tf.gather(log_q, parent_indices, batch_dims=1)
        log_w_new, _, _ = self._log_normalize(log_w_new)
        return x_new, log_w_new, parent_indices

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor,
                      training: bool | None = None):
        return self.soft_resampling_mixture(x, log_w, lam=self.lam)

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        lam=None,
        resample=None,
        proposal=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
        training: bool | None = None,
    ):
        if any(v is not None for v in (num_particles, ess_threshold, lam, resample, proposal)):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                lam=lam,
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
            training=training,
        )


__all__ = [
    "categorical_sample",
    "SoftResamplingDPF",
]
