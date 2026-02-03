"""Shared base class for particle flow filters."""

from typing import Dict

import tensorflow as tf

from src.filters.particle import ParticleFilter
from src.filters.mixins import LinearizationMixin


class FlowBase(ParticleFilter, LinearizationMixin):
    """Shared scaffolding for particle flow filters."""

    def __init__(
        self,
        ssm,
        num_lambda=20,
        num_particles=100,
        ess_threshold=0.5,
        reweight="never",
        resample="never",
        init_from_particles="sample",
        debug=False,
        jitter=1e-12,
    ):
        """Initialize flow hyperparameters and resampling defaults."""
        super().__init__(ssm, num_particles=num_particles, ess_threshold=ess_threshold)
        self.num_lambda = int(num_lambda)
        self.default_reweight = reweight
        self.default_resample = resample
        self.init_from_particles = init_from_particles
        self.debug = bool(debug)
        self.jitter = jitter

    def _prior_from_sample(self, mu_tilde, w=None):
        """Compute weighted prior mean/covariance from particles.

        Shapes:
          mu_tilde: [B, N, dx]
          w: [B, N]
        Returns:
          m: [B, dx]
          P: [B, dx, dx]
        """
        if w is None:
            w = tf.ones_like(mu_tilde[..., 0]) / tf.cast(tf.shape(mu_tilde)[-2], tf.float32)
        m = tf.einsum("...n,...ni->...i", w, mu_tilde)
        mu_resid = mu_tilde - m[..., tf.newaxis, :]
        P = tf.einsum("...n,...ni,...nj->...ij", w, mu_resid, mu_resid)
        return m, P

    def _normalize_step_modes(self, reweight, resample):
        """Normalize reweight/resample flags into integer modes."""
        if reweight is None:
            reweight = self.default_reweight
        if resample is None:
            resample = self.default_resample
        reweight = self._normalize_reweight(reweight)
        resample = self._normalize_reweight(resample)
        return reweight, resample

    def _flow_supports_reweight(self):
        """Return True if the flow supports Jacobian-based reweighting."""
        return True

    def _flow_diag_keys(self):
        """Return fixed diagnostic keys for flow transport (empty if none)."""
        return ()

    @staticmethod
    def _broadcast_log_det(log_det, log_q0):
        """Broadcast log_det to match log_q0 rank when needed."""
        log_det = tf.convert_to_tensor(log_det, dtype=log_q0.dtype)
        rank_diff = tf.rank(log_q0) - tf.rank(log_det)

        def _pad_log_det():
            new_shape = tf.concat(
                [tf.shape(log_det), tf.ones(rank_diff, dtype=tf.int32)],
                axis=0,
            )
            return tf.reshape(log_det, new_shape)

        return tf.cond(rank_diff > 0, _pad_log_det, lambda: log_det)

    def _flow_reweight(
        self,
        log_w_prev,
        x_prev,
        mu_tilde,
        x_t,
        y_t,
        log_q0,
        log_det,
        trans_dist,
        reweight,
    ):
        """Compute updated log-weights after flow transport."""
        if reweight == 0:
            return log_w_prev
        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])
        log_prior = trans_dist.log_prob(x_t)
        log_det = self._broadcast_log_det(log_det, log_q0)
        log_q = log_q0 - log_det
        return log_w_prev + tf.cast(loglik + log_prior - log_q, tf.float32)

    # --- Abstract flow transport ---
    def _flow_transport(self, mu_tilde, y, m0, P, **kwargs):
        """Transport particles through a flow; return (x_next, log_det, diag).

        Shapes:
          mu_tilde: [B, N, dx]
          y: [B, dy]
          m0: [B, dx]
          P: [B, dx, dx]
          kwargs: optional flow-specific inputs (e.g., weights)
        Returns:
          x_next: [B, N, dx]
          log_det: [B] or [B, N]
          diag: dict or None
        """
        raise NotImplementedError

    def _propagate_particles(self, x_prev, seed=None):
        """Propagate particles through transition and return log-prob.

        Shapes:
          x_prev: [B, N, dx]
        Returns:
          mu_tilde: [B, N, dx]
          log_q0: [B, N]
        """
        mu_tilde = self.ssm.sample_transition(x_prev, seed=seed)
        return mu_tilde, self.ssm.transition_dist(x_prev).log_prob(mu_tilde)

    def sample(self, x_prev, y_t, w=None, seed=None):
        """Sample next particles and compute proposal log-prob corrections.

        Shapes:
          x_prev: [B, N, dx]
          y_t: [B, dy]
          w: [B, N] (optional)
        Returns:
          x_next: [B, N, dx]
          log_q: [B, N]
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
        """
        mu_tilde, log_q0 = self._propagate_particles(x_prev, seed=seed)
        if w is None:
            w = tf.ones_like(mu_tilde[..., 0])
        else:
            w = tf.convert_to_tensor(w, dtype=mu_tilde.dtype)
        w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w)
        P_pred = self.ssm.state_cov(mu_tilde, w)
        m, P = self._prior_from_sample(mu_tilde, w)

        x_next, log_det, _ = self._flow_transport(mu_tilde, y_t, m, P, w=w)
        return x_next, log_q0 - log_det, m_pred, P_pred

    def warmup(self, batch_size=1, reweight="auto", resample="never"):
        """Trace the step function to reduce first-call overhead."""
        x_spec = tf.TensorSpec(
            shape=[None, self.num_particles, self.ssm.state_dim],
            dtype=tf.float32,
        )
        y_spec = tf.TensorSpec(shape=[None, self.ssm.obs_dim], dtype=tf.float32)
        log_w_spec = tf.TensorSpec(shape=[None, self.num_particles], dtype=tf.float32)
        reweight, resample = self._normalize_step_modes(reweight, resample)
        _ = self.step.get_concrete_function(
            x_spec,
            log_w_spec,
            y_spec,
            reweight=reweight,
            resample=resample,
        )

        x = tf.zeros([batch_size, self.num_particles, self.ssm.state_dim], dtype=tf.float32)
        log_w = tf.zeros([batch_size, self.num_particles], dtype=tf.float32)
        y = tf.zeros([batch_size, self.ssm.obs_dim], dtype=tf.float32)
        _ = self.step(x, log_w, y, reweight=reweight, resample=resample)

    @tf.function(reduce_retracing=True)
    def step(
        self,
        x_prev,
        log_w_prev,
        y_t,
        reweight="auto",
        resample="never",
        **kwargs,
    ):
        """Propagate, flow-transport, and (optionally) resample particles.

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
          x_pre: [B, N, dx]
          w_pre: [B, N]
          flow_diag: dict or None
        """
        reweight, resample = self._normalize_step_modes(reweight, resample)
        if reweight != 0 and not self._flow_supports_reweight():
            if self.debug:
                tf.print(
                    "Reweight requested for",
                    self.__class__.__name__,
                    "but flow does not support it; disabling.",
                    summarize=4,
                )
            reweight = 0

        mu_tilde = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        trans_dist = self.ssm.transition_dist(x_prev)
        log_q0 = trans_dist.log_prob(mu_tilde)

        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w_prev)
        P_pred = self.ssm.state_cov(mu_tilde, w_prev)

        m, P = self._prior_from_sample(mu_tilde, w_prev)
        x_t, log_det, flow_diag = self._flow_transport(mu_tilde, y_t, m, P, w=w_prev)

        # Importance reweighting with flow Jacobian correction (if supported).
        log_w = self._flow_reweight(
            log_w_prev,
            x_prev,
            mu_tilde,
            x_t,
            y_t,
            log_q0,
            log_det,
            trans_dist,
            reweight,
        )

        log_w_norm, w, _ = self._log_normalize(log_w)
        x_pre = x_t
        w_pre = w

        if resample in (1, 2):
            ess = self.ess(w)
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w, self.ssm.rng)
            batch_shape_out = tf.shape(x_t)[:-2]
            no_rs_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape_out, [self.num_particles]], axis=0),
            )
            mask_do_rs = mask_do_rs[..., tf.newaxis]
            parent_indices = tf.where(mask_do_rs, rs_indices, no_rs_indices)

            x_t = self.resample_particles(x_t, parent_indices)
            log_w_reset = -tf.math.log(N_float) * tf.ones_like(log_w_norm)
            log_w_final = tf.where(mask_do_rs, log_w_reset, log_w_norm)
            w_final = tf.exp(log_w_final)
        else:
            batch_shape_out = tf.shape(x_t)[:-2]
            parent_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape_out, [self.num_particles]], axis=0),
            )
            log_w_final = log_w_norm
            w_final = w
        return (
            mu_tilde,
            x_t,
            log_w_final,
            w_final,
            parent_indices,
            m_pred,
            P_pred,
            x_pre,
            w_pre,
            flow_diag,
        )

    @tf.function(reduce_retracing=True)
    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        reweight="auto",
        resample="never",
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        """Run the flow filter over a full observation sequence.

        Shapes:
          y: [T, dy] or [B, T, dy]
        Returns:
          x_seq: [B, T, N, dx]
          w_seq: [B, T, N]
          diagnostics: dict of per-step tensors
          parent_seq: [B, T, N]
        """
        self.update_params(num_particles, ess_threshold)
        y = self._normalize_y(y)
        if reweight is None:
            reweight = self.default_reweight
        if resample is None:
            resample = self.default_resample
        reweight = self._normalize_reweight(reweight)
        resample = self._normalize_reweight(resample)

        T = tf.shape(y)[1]
        x_prev, log_w, parent_indices = self._init_particles(
            y,
            init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )

        x_ta = tf.TensorArray(tf.float32, size=T)
        x_pred_ta = tf.TensorArray(tf.float32, size=T)
        w_ta = tf.TensorArray(tf.float32, size=T)
        w_pre_ta = tf.TensorArray(tf.float32, size=T)
        w_prev_ta = tf.TensorArray(tf.float32, size=T)
        m_pred_ta = tf.TensorArray(tf.float32, size=T)
        P_pred_ta = tf.TensorArray(tf.float32, size=T)
        parent_ta = tf.TensorArray(tf.int32, size=T)
        step_time_ta = tf.TensorArray(tf.float32, size=T)
        x_pre_ta = tf.TensorArray(tf.float32, size=T)
        flow_diag_keys = list(self._flow_diag_keys())
        flow_diag_ta = tuple(tf.TensorArray(tf.float32, size=T) for _ in flow_diag_keys)
        x_invar = x_prev.shape
        log_w_invar = log_w.shape

        def _write_flow_diag(flow_diag_ta, flow_diag, t):
            if not flow_diag_keys:
                return flow_diag_ta
            return tuple(
                ta.write(t, tf.cast(flow_diag[key], tf.float32))
                for ta, key in zip(flow_diag_ta, flow_diag_keys)
            )

        def _cond(t, _state): return t < T

        def _body(t, state):
            x_prev, log_w, tas = state
            x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta, step_time_ta, x_pre_ta, flow_diag_ta = tas

            t_start = tf.timestamp()
            y_t = y[..., t, :]
            w_prev = tf.exp(log_w)
            w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
            (
                x_pred,
                x,
                log_w,
                w,
                parent_indices,
                m_pred,
                P_pred,
                x_pre,
                w_pre,
                flow_diag,
            ) = self.step(
                x_prev,
                log_w,
                y_t,
                reweight=reweight,
                resample=resample,
            )
            x = tf.ensure_shape(x, x_invar)
            log_w = tf.ensure_shape(log_w, log_w_invar)
            x_prev = x

            x_pred_ta = x_pred_ta.write(t, x_pred)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            w_pre_ta = w_pre_ta.write(t, w_pre)
            w_prev_ta = w_prev_ta.write(t, w_prev)
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)
            parent_ta = parent_ta.write(t, parent_indices)
            x_pre_ta = x_pre_ta.write(t, x_pre)
            flow_diag_ta = _write_flow_diag(flow_diag_ta, flow_diag, t)

            step_time = tf.cast(tf.timestamp() - t_start, tf.float32)
            step_time_ta = step_time_ta.write(t, step_time)

            tas = (x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta, step_time_ta, x_pre_ta, flow_diag_ta)
            return t + 1, (x_prev, log_w, tas)

        tas = (x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta, step_time_ta, x_pre_ta, flow_diag_ta)
        _, (x_prev, log_w, tas) = tf.while_loop(
            _cond,
            _body,
            (tf.constant(0), (x_prev, log_w, tas)),
        )
        x_ta, x_pred_ta, w_ta, w_pre_ta, w_prev_ta, m_pred_ta, P_pred_ta, parent_ta, step_time_ta, x_pre_ta, flow_diag_ta = tas

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)

        diagnostics = {
            "step_time_s": self._stack_and_permute(step_time_ta, tail_dims=0),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "x_pred": self._stack_and_permute(x_pred_ta, tail_dims=2),
            "x_pre": self._stack_and_permute(x_pre_ta, tail_dims=2),
            "w_pre": self._stack_and_permute(w_pre_ta, tail_dims=1),
            "w_prev": self._stack_and_permute(w_prev_ta, tail_dims=1),
        }
        if flow_diag_keys:
            for key, ta in zip(flow_diag_keys, flow_diag_ta):
                diagnostics[key] = self._stack_and_permute(ta, tail_dims=0)
        return x_seq, w_seq, diagnostics, parent_seq
