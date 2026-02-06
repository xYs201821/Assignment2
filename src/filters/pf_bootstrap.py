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
          x_pred: [B, N, dx]
          x_t: [B, N, dx]
          log_w_final: [B, N]
          w_final: [B, N]
          parent_indices: [B, N]
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          w_pre: [B, N]
        """
        x_pred = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(x_pred, w_prev)
        P_pred = self.ssm.state_cov(x_pred, w_prev)
        loglik = self.ssm.observation_dist(x_pred).log_prob(y_t[..., tf.newaxis, :])
        log_w = log_w_prev + tf.cast(loglik, tf.float32)

        # Normalize weights before ESS/resampling decisions.
        log_w_norm, w, _ = self._log_normalize(log_w)
        w_pre = w
        ess = self.ess(w)

        if resample in (1, 2):
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w, self.ssm.rng)
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
        return x_pred, x_t, log_w_final, w_final, parent_indices, m_pred, P_pred, w_pre

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample=1,
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
        y = self._normalize_y(y)

        # Initialize particles outside tf.function to avoid retracing
        x_init, log_w_init, _ = self._init_particles(
            y,
            init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )

        return self._filter_loop(y, x_init, log_w_init, resample)

    @tf.function(reduce_retracing=True)
    def _filter_loop(self, y, x_prev, log_w, resample):
        """Core filter loop (tf.function compiled)."""
        T = tf.shape(y)[1]

        x_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        x_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_pre_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_prev_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        m_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        P_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        parent_ta = tf.TensorArray(tf.int32, size=T, dynamic_size=False)

        def _cond(t, _state):
            return t < T

        def _body(t, state):
            x_prev, log_w, tas = state
            x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta = tas

            y_t = y[:, t, :]
            w_prev = tf.exp(log_w)
            w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
            x_pred, x, log_w, w, parent_indices, m_pred, P_pred, w_pre = self.step(
                x_prev,
                log_w,
                y_t,
                resample=resample,
            )
            x_prev = x

            x_pred_ta = x_pred_ta.write(t, x_pred)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            w_pre_ta = w_pre_ta.write(t, w_pre)
            w_prev_ta = w_prev_ta.write(t, w_prev)
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)
            parent_ta = parent_ta.write(t, parent_indices)

            tas = (x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta)
            return t + 1, (x_prev, log_w, tas)

        tas = (x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta)
        _, (_, _, tas) = tf.while_loop(
            _cond,
            _body,
            (tf.constant(0), (x_prev, log_w, tas)),
        )
        x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta = tas

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)

        diagnostics = {
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "x_pred": self._stack_and_permute(x_pred_ta, tail_dims=2),
            "w_pre": self._stack_and_permute(w_pre_ta, tail_dims=1),
            "w_prev": self._stack_and_permute(w_prev_ta, tail_dims=1),
        }
        return x_seq, w_seq, diagnostics, parent_seq
