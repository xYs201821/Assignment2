"""Differentiable particle filters with pluggable resampling."""

from __future__ import annotations

import tensorflow as tf

from src.filters.particle import ParticleFilter


class _BootstrapProposal:
    """Default prior proposal q(x_t|x_{t-1}, y_t) = f(x_t|x_{t-1})."""

    def sample(self, ssm, x_prev, y_t, seed=None):
        del y_t
        return ssm.sample_transition(x_prev, seed=seed)

    def log_prob(self, ssm, x, x_prev, y_t):
        del y_t
        return ssm.transition_dist(x_prev).log_prob(x)


class DPFBase(ParticleFilter):
    """Base differentiable PF with common loop and pluggable resampling."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        resample: str | int | bool = "auto",
        debug: bool = False,
        print: bool = False,
        proposal=None,
    ) -> None:
        super().__init__(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            debug=debug,
            print=print,
        )
        self.resample = self._normalize_reweight(resample)
        self.proposal = proposal if proposal is not None else _BootstrapProposal()

    def set_proposal(self, proposal) -> None:
        """Set proposal.

        Proposal protocol:
          - callable(ssm, x_prev, y_t, seed=None) -> x_pred OR (x_pred, log_q)
          - object with .sample(ssm, x_prev, y_t, seed=None), optional .log_prob(...)
        """
        self.proposal = proposal if proposal is not None else _BootstrapProposal()

    def update_params(self, num_particles=None, ess_threshold=None, resample=None, proposal=None):
        """Update runtime hyperparameters."""
        super().update_params(num_particles=num_particles, ess_threshold=ess_threshold)
        if resample is not None:
            self.resample = self._normalize_reweight(resample)
        if proposal is not None:
            self.set_proposal(proposal)

    @staticmethod
    def _normalize_proposal_output(out):
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError("proposal sample output tuple/list must be (x_pred, log_q).")
            return out[0], out[1]
        return out, None

    def _sample_proposal(self, x_prev, y_t):
        seed = self.ssm._tfp_seed()
        proposal = self.proposal
        if callable(proposal):
            out = proposal(self.ssm, x_prev, y_t, seed=seed)
        elif hasattr(proposal, "sample"):
            out = proposal.sample(self.ssm, x_prev, y_t, seed=seed)
        else:
            raise TypeError("proposal must be callable or expose .sample(...).")

        x_pred, log_q = self._normalize_proposal_output(out)
        x_pred = tf.convert_to_tensor(x_pred, dtype=tf.float32)

        if log_q is None and hasattr(proposal, "log_prob"):
            log_q = proposal.log_prob(self.ssm, x_pred, x_prev, y_t)
        if log_q is not None:
            log_q = tf.convert_to_tensor(log_q, dtype=tf.float32)
        return x_pred, log_q

    def warmup(self, batch_size=1, T=2, resample=None, y=None):
        """Trace filter graph to reduce first-call overhead."""
        if y is None:
            y = tf.zeros([batch_size, T, self.ssm.obs_dim], dtype=tf.float32)
        if resample is None:
            resample = self.resample
        _ = self.filter(y, resample=resample)

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor):
        """Resampling implementation hook."""
        raise NotImplementedError

    def _identity_parent_indices(self, x: tf.Tensor) -> tf.Tensor:
        batch_shape = tf.shape(x)[:-2]
        return tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape, [self.num_particles]], axis=0),
        )

    def step(self, x_prev, log_w_prev, y_t, resample="auto"):
        """One DPF step: propagate, weight, normalize, and optional resample.

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
          logz_t: [B]
          ess: [B]
          resampled: [B] bool
        """
        x_pred, log_q = self._sample_proposal(x_prev, y_t)
        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(x_pred, w_prev)
        P_pred = self.ssm.state_cov(x_pred, w_prev)

        loglik = self.ssm.observation_dist(x_pred).log_prob(y_t[..., tf.newaxis, :])
        log_w = log_w_prev + tf.cast(loglik, log_w_prev.dtype)
        if log_q is not None:
            tf.debugging.assert_equal(
                tf.shape(log_q),
                tf.shape(log_w_prev),
                message="proposal log_q must have shape [B, N].",
            )
            log_f = self.ssm.transition_dist(x_prev).log_prob(x_pred)
            log_w = log_w + tf.cast(log_f - log_q, log_w_prev.dtype)
        log_w_norm, w_pre, logz_t = self._log_normalize(log_w)
        ess = self.ess(w_pre)

        resample = self._normalize_reweight(resample)
        if resample in (1, 2):
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            x_rs, log_w_rs, rs_indices = self.resample_step(x_pred, log_w_norm)
            no_rs_indices = self._identity_parent_indices(x_pred)

            mask_w = mask_do_rs[..., tf.newaxis]
            mask_x = mask_do_rs[..., tf.newaxis, tf.newaxis]
            parent_indices = tf.where(mask_w, rs_indices, no_rs_indices)
            x_t = tf.where(mask_x, x_rs, x_pred)
            log_w_final = tf.where(mask_w, log_w_rs, log_w_norm)
        else:
            parent_indices = self._identity_parent_indices(x_pred)
            x_t = x_pred
            log_w_final = log_w_norm
            mask_do_rs = tf.zeros_like(ess, dtype=tf.bool)

        w_final = tf.exp(log_w_final)
        return (
            x_pred,
            x_t,
            log_w_final,
            w_final,
            parent_indices,
            m_pred,
            P_pred,
            w_pre,
            logz_t,
            ess,
            mask_do_rs,
        )

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        """Run differentiable particle filter over a sequence.

        Shapes:
          y: [T, dy] or [B, T, dy]
        Returns:
          x_seq: [B, T, N, dx]
          w_seq: [B, T, N]
          diagnostics: dict of per-step tensors
          parent_seq: [B, T, N]
        """
        if any(v is not None for v in (num_particles, ess_threshold, resample, proposal)):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                resample=resample,
                proposal=proposal,
            )

        y = self._normalize_y(y)
        x_init, log_w_init, _ = self._init_particles(
            y,
            init_dist=init_dist,
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
        x_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_pre_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        w_prev_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        m_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        P_pred_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        parent_ta = tf.TensorArray(tf.int32, size=T, dynamic_size=False)
        ess_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        logz_ta = tf.TensorArray(tf.float32, size=T, dynamic_size=False)
        resampled_ta = tf.TensorArray(tf.bool, size=T, dynamic_size=False)

        logz_total = tf.zeros(tf.shape(log_w)[:-1], dtype=log_w.dtype)

        def _cond(t, _state):
            return t < T

        def _body(t, state):
            x_prev, log_w, logz_total, tas = state
            (
                x_ta,
                x_pred_ta,
                w_ta,
                w_pre_ta,
                w_prev_ta,
                m_pred_ta,
                P_pred_ta,
                parent_ta,
                ess_ta,
                logz_ta,
                resampled_ta,
            ) = tas

            y_t = y[:, t, :]
            w_prev = tf.exp(log_w)
            w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))

            (
                x_pred,
                x_t,
                log_w,
                w,
                parent_indices,
                m_pred,
                P_pred,
                w_pre,
                logz_t,
                ess,
                resampled,
            ) = self.step(
                x_prev,
                log_w,
                y_t,
                resample=resample,
            )
            x_prev = x_t
            logz_total = logz_total + logz_t

            x_pred_ta = x_pred_ta.write(t, x_pred)
            x_ta = x_ta.write(t, x_t)
            w_ta = w_ta.write(t, w)
            w_pre_ta = w_pre_ta.write(t, w_pre)
            w_prev_ta = w_prev_ta.write(t, w_prev)
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)
            parent_ta = parent_ta.write(t, parent_indices)
            ess_ta = ess_ta.write(t, ess)
            logz_ta = logz_ta.write(t, logz_t)
            resampled_ta = resampled_ta.write(t, resampled)

            tas = (
                x_ta,
                x_pred_ta,
                w_ta,
                w_pre_ta,
                w_prev_ta,
                m_pred_ta,
                P_pred_ta,
                parent_ta,
                ess_ta,
                logz_ta,
                resampled_ta,
            )
            return t + 1, (x_prev, log_w, logz_total, tas)

        tas = (
            x_ta,
            x_pred_ta,
            w_ta,
            w_pre_ta,
            w_prev_ta,
            m_pred_ta,
            P_pred_ta,
            parent_ta,
            ess_ta,
            logz_ta,
            resampled_ta,
        )
        _, (_, _, logz_total, tas) = tf.while_loop(
            _cond,
            _body,
            (tf.constant(0), (x_prev, log_w, logz_total, tas)),
        )
        (
            x_ta,
            x_pred_ta,
            w_ta,
            w_pre_ta,
            w_prev_ta,
            m_pred_ta,
            P_pred_ta,
            parent_ta,
            ess_ta,
            logz_ta,
            resampled_ta,
        ) = tas

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)

        diagnostics = {
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "x_pred": self._stack_and_permute(x_pred_ta, tail_dims=2),
            "w_pre": self._stack_and_permute(w_pre_ta, tail_dims=1),
            "w_prev": self._stack_and_permute(w_prev_ta, tail_dims=1),
            "ess": self._stack_and_permute(ess_ta, tail_dims=0),
            "logZ_t": self._stack_and_permute(logz_ta, tail_dims=0),
            "resampled": self._stack_and_permute(resampled_ta, tail_dims=0),
            "logZ_total": logz_total,
        }
        return x_seq, w_seq, diagnostics, parent_seq


__all__ = [
    "DPFBase",
]
