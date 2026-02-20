"""Bootstrap particle filter implementation."""

from __future__ import annotations

import tensorflow as tf

from src.filters.particle import ParticleFilter


class BootstrapParticleFilter(ParticleFilter):
    """Bootstrap particle filter with optional resampling."""

    def __init__(self, ssm, num_particles=100, ess_threshold=0.5, resample="auto", debug=False, print=False):
        """Initialize with resampling configuration."""
        super().__init__(ssm, num_particles=num_particles, ess_threshold=ess_threshold, debug=debug, print=print)
        self.resample = self._normalize_reweight(resample)

    def update_params(self, num_particles=None, ess_threshold=None, resample=None):
        """Update particle count, ESS threshold, and resampling mode."""
        if num_particles is not None:
            self.num_particles = int(num_particles)
        if ess_threshold is not None:
            self.ess_threshold = tf.convert_to_tensor(ess_threshold, tf.float32)
        if resample is not None:
            self.resample = self._normalize_reweight(resample)

    def warmup(self, batch_size=1, T=2, resample=1, y=None):
        """Trace the filter function to reduce first-call overhead."""
        if y is None:
            y = tf.zeros([batch_size, T, self.ssm.obs_dim], dtype=tf.float32)
        _ = self.filter(y, resample=resample)

    def step(self, x_prev, log_w_prev, y_t, resample="auto"):
        """Bootstrap PF step: propagate, weight, and resample.

        Shapes:
          x_prev: [B, N, dx]
          log_w_prev: [B, N]
          y_t: [B, dy]
        Returns:
          x_pre: [B, N, dx]
          x_t: [B, N, dx]
          log_w_final: [B, N]
          w_final: [B, N]
          parent_indices: [B, N]
          log_w_pre: [B, N]
          logz_t: [B]
        """
        x_pred = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        loglik = self.ssm.observation_dist(x_pred).log_prob(y_t[..., tf.newaxis, :])
        log_w = log_w_prev + tf.cast(loglik, tf.float32)

        # Normalize weights before ESS/resampling decisions.
        log_w_norm, w_pre, logz_t = self._log_normalize(log_w)
        ess = self.ess(w_pre)

        if resample in (1, 2):
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w_pre, self.ssm.rng)
            batch_shape = tf.shape(x_pred)[:-2]
            no_rs_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            mask_do_rs = mask_do_rs[..., tf.newaxis]
            parent_indices = tf.where(mask_do_rs, rs_indices, no_rs_indices)

            x_t = self.resample_particles(x_pred, parent_indices)
            log_w_reset = -tf.math.log(N_float) * tf.ones_like(log_w_norm)
            log_w_final = tf.where(mask_do_rs, log_w_reset, log_w_norm)
        else:
            batch_shape = tf.shape(x_pred)[:-2]
            parent_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            log_w_final = log_w_norm
            x_t = x_pred
        w_final = tf.exp(log_w_final)
        log_w_pre = log_w_norm
        return x_pred, x_t, log_w_final, w_final, parent_indices, log_w_pre, logz_t

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        """Run the bootstrap particle filter over a sequence.

        Shapes:
          y: [T, dy] or [B, T, dy]
        Returns:
          x_seq: [B, T, N, dx]
          w_seq: [B, T, N]
          diagnostics: dict of per-step tensors
          parent_seq: [B, T, N]
        """
        if any(v is not None for v in (num_particles, ess_threshold, resample)):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                resample=resample,
            )
        y = self._normalize_y(y)

        x_init, log_w_init, _ = self._init_particles(
            y,
            init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )

        resample_mode = self.resample if resample is None else self._normalize_reweight(resample)
        return self._filter_loop(y, x_init, log_w_init, resample_mode)

    @tf.function(reduce_retracing=True)
    def _filter_loop(self, y, x_prev, log_w, resample):
        """Core filter loop (tf.function compiled)."""
        T = tf.shape(y)[1]

        x_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        x_pre_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        log_w_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        log_w_pre_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        parent_ta = tf.TensorArray(tf.int32, size=T, dynamic_size=False)
        logz_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)

        def _cond(t, _state):
            return t < T

        def _body(t, state):
            x_prev, log_w, tas = state
            (
                x_ta,
                x_pre_ta,
                w_ta,
                log_w_ta,
                log_w_pre_ta,
                parent_ta,
                logz_ta,
            ) = tas

            y_t = y[:, t, :]
            x_pre, x, log_w, w, parent_indices, log_w_pre, logz_t = self.step(
                x_prev,
                log_w,
                y_t,
                resample=resample,
            )
            x_prev = x

            x_pre_ta = x_pre_ta.write(t, x_pre)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            log_w_ta = log_w_ta.write(t, log_w)
            log_w_pre_ta = log_w_pre_ta.write(t, log_w_pre)
            parent_ta = parent_ta.write(t, parent_indices)
            logz_ta = logz_ta.write(t, logz_t)

            tas = (
                x_ta,
                x_pre_ta,
                w_ta,
                log_w_ta,
                log_w_pre_ta,
                parent_ta,
                logz_ta,
            )
            return t + 1, (x_prev, log_w, tas)

        tas = (
            x_ta,
            x_pre_ta,
            w_ta,
            log_w_ta,
            log_w_pre_ta,
            parent_ta,
            logz_ta,
        )
        _, (_, _, tas) = tf.while_loop(
            _cond,
            _body,
            (tf.constant(0), (x_prev, log_w, tas)),
        )
        (
            x_ta,
            x_pre_ta,
            w_ta,
            log_w_ta,
            log_w_pre_ta,
            parent_ta,
            logz_ta,
        ) = tas

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        log_w_seq = self._stack_and_permute(log_w_ta, tail_dims=1)
        x_pre_seq = self._stack_and_permute(x_pre_ta, tail_dims=2)
        log_w_pre_seq = self._stack_and_permute(log_w_pre_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)
        logz_seq = self._stack_and_permute(logz_ta, tail_dims=0)

        diagnostics = {
            "x": x_seq,
            "log_w": log_w_seq,
            "log_z": logz_seq,
            "x_pre": x_pre_seq,
            "log_w_pre": log_w_pre_seq,
            "parent_index": parent_seq,
        }
        return x_seq, w_seq, diagnostics, parent_seq
