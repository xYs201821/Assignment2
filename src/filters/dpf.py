"""Differentiable particle filters with pluggable resampling."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from src.distributions import BootstrapProposal, Proposal
from src.filters.particle import ParticleFilter
import src.dtype_config as _dc

tfd = tfp.distributions


class DPFBase(ParticleFilter):
    """Base differentiable PF with common loop and pluggable resampling."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        resample: str | int | bool = "auto",
        stop_grad_through_time: bool = False,
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
        self.stop_grad_through_time = bool(stop_grad_through_time)
        self.proposal = proposal if proposal is not None else BootstrapProposal()

    def set_proposal(self, proposal) -> None:
        """Set proposal.

        Proposal protocol:
          - callable(ssm, x_prev, y_t, seed=None) -> x_pred OR (x_pred, log_q)
          - object with .sample(ssm, x_prev, y_t, seed=None), optional .log_prob(...)
        """
        self.proposal = proposal if proposal is not None else BootstrapProposal()

    def update_params(
        self,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        stop_grad_through_time=None,
    ):
        """Update runtime hyperparameters."""
        super().update_params(num_particles=num_particles, ess_threshold=ess_threshold)
        if resample is not None:
            self.resample = self._normalize_reweight(resample)
        if stop_grad_through_time is not None:
            self.stop_grad_through_time = bool(stop_grad_through_time)
        if proposal is not None:
            self.set_proposal(proposal)
        if stop_grad_through_time is not None:
            self.stop_grad_through_time = bool(stop_grad_through_time)

    @staticmethod
    def _normalize_proposal_output(out):
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError("proposal sample output tuple/list must be (x_pred, log_q).")
            return out[0], out[1]
        return out, None

    def _sample_proposal(self, x_prev, y_t, y_prev=None):
        seed = self.ssm._tfp_seed()
        proposal = self.proposal
        if callable(proposal):
            out = proposal(self.ssm, x_prev, y_t, seed=seed)
        elif hasattr(proposal, "sample"):
            if isinstance(proposal, Proposal):
                out = proposal.sample(
                    self.ssm,
                    x_prev,
                    y_t,
                    seed=seed,
                    y_prev=y_prev,
                )
            else:
                out = proposal.sample(self.ssm, x_prev, y_t, seed=seed)
        else:
            raise TypeError("proposal must be callable or expose .sample(...).")

        x_pred, log_q = self._normalize_proposal_output(out)
        x_pred = tf.convert_to_tensor(x_pred, dtype=_dc.DTYPE)

        if log_q is None and hasattr(proposal, "log_prob"):
            try:
                if isinstance(proposal, Proposal):
                    log_q = proposal.log_prob(
                        self.ssm,
                        x_pred,
                        x_prev,
                        y_t,
                        y_prev=y_prev,
                    )
                else:
                    log_q = proposal.log_prob(self.ssm, x_pred, x_prev, y_t)
            except (NotImplementedError, AttributeError):
                # Some SSMs may not expose transition_dist/log_prob; in that case
                # keep bootstrap semantics (no proposal correction).
                log_q = None
        if log_q is not None:
            log_q = tf.convert_to_tensor(log_q, dtype=_dc.DTYPE)
        return x_pred, log_q

    @staticmethod
    def _regularized_cholesky(cov: tf.Tensor, jitter: float = _dc.JITTER) -> tf.Tensor:
        cov = tf.convert_to_tensor(cov, dtype=_dc.DTYPE)
        cov = 0.5 * (cov + tf.linalg.matrix_transpose(cov))
        eye = tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype)
        return tf.linalg.cholesky(cov + tf.cast(jitter, cov.dtype) * eye)

    def _linearized_observation_cov(self, x_pred: tf.Tensor, cov_r: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if not callable(getattr(self.ssm, "h_with_noise", None)):
            raise NotImplementedError("h_with_noise is required for linearized observation fallback.")

        x_pred = tf.convert_to_tensor(x_pred, dtype=_dc.DTYPE)
        cov_r = tf.convert_to_tensor(cov_r, dtype=_dc.DTYPE)
        shape = tf.shape(x_pred)
        x_flat = tf.reshape(x_pred, [shape[0] * shape[1], shape[2]])
        r0 = tf.zeros([tf.shape(x_flat)[0], int(self.r_dim)], dtype=x_flat.dtype)

        ssm_jac = getattr(self.ssm, "jacobian_h_r", None)
        if callable(ssm_jac):
            H_r_flat, y_loc_flat = ssm_jac(x_flat, r0)
        else:
            with tf.GradientTape() as tape:
                tape.watch(r0)
                y_loc_flat = self.ssm.h_with_noise(x_flat, r0)
            H_r_flat = tape.batch_jacobian(y_loc_flat, r0)

        H_r_flat = tf.convert_to_tensor(H_r_flat, dtype=_dc.DTYPE)
        y_loc_flat = tf.convert_to_tensor(y_loc_flat, dtype=_dc.DTYPE)
        cov_eff_flat = tf.linalg.matmul(tf.linalg.matmul(H_r_flat, cov_r), H_r_flat, transpose_b=True)

        y_loc = tf.reshape(y_loc_flat, [shape[0], shape[1], tf.shape(y_loc_flat)[-1]])
        cov_eff = tf.reshape(
            cov_eff_flat,
            [shape[0], shape[1], tf.shape(cov_eff_flat)[-2], tf.shape(cov_eff_flat)[-1]],
        )
        return y_loc, cov_eff

    def _linearized_transition_cov(self, x_prev: tf.Tensor, cov_q: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if not callable(getattr(self.ssm, "f_with_noise", None)):
            raise NotImplementedError("f_with_noise is required for linearized transition fallback.")

        x_prev = tf.convert_to_tensor(x_prev, dtype=_dc.DTYPE)
        cov_q = tf.convert_to_tensor(cov_q, dtype=_dc.DTYPE)
        shape = tf.shape(x_prev)
        x_flat = tf.reshape(x_prev, [shape[0] * shape[1], shape[2]])
        q0 = tf.zeros([tf.shape(x_flat)[0], int(self.q_dim)], dtype=x_flat.dtype)

        ssm_jac = getattr(self.ssm, "jacobian_f_q", None)
        if callable(ssm_jac):
            F_q_flat, x_loc_flat = ssm_jac(x_flat, q0)
        else:
            with tf.GradientTape() as tape:
                tape.watch(q0)
                x_loc_flat = self.ssm.f_with_noise(x_flat, q0)
            F_q_flat = tape.batch_jacobian(x_loc_flat, q0)

        F_q_flat = tf.convert_to_tensor(F_q_flat, dtype=_dc.DTYPE)
        x_loc_flat = tf.convert_to_tensor(x_loc_flat, dtype=_dc.DTYPE)
        cov_eff_flat = tf.linalg.matmul(tf.linalg.matmul(F_q_flat, cov_q), F_q_flat, transpose_b=True)

        x_loc = tf.reshape(x_loc_flat, [shape[0], shape[1], tf.shape(x_loc_flat)[-1]])
        cov_eff = tf.reshape(
            cov_eff_flat,
            [shape[0], shape[1], tf.shape(cov_eff_flat)[-2], tf.shape(cov_eff_flat)[-1]],
        )
        return x_loc, cov_eff

    def _observation_log_prob(self, x_pred: tf.Tensor, y_t: tf.Tensor) -> tf.Tensor:
        # Preferred path: exact model likelihood from observation_dist.
        try:
            obs_dist = self.ssm.observation_dist(x_pred)
            if obs_dist is not None:
                return tf.cast(obs_dist.log_prob(y_t[..., tf.newaxis, :]), tf.float32)
        except (NotImplementedError, AttributeError):
            pass

        # Fallback: first-order Gaussianization using h/h_with_noise and cov_eps_y.
        if not callable(getattr(self.ssm, "h", None)):
            raise NotImplementedError(
                "observation_dist is unavailable and fallback requires ssm.h (and noise covariance)."
            )
        cov_r = getattr(self.ssm, "cov_eps_y", None)
        if cov_r is None:
            raise NotImplementedError(
                "observation_dist is unavailable and fallback requires ssm.cov_eps_y."
            )

        y_obs = tf.convert_to_tensor(y_t, dtype=_dc.DTYPE)[..., tf.newaxis, :]
        y_loc = tf.cast(self.ssm.h(x_pred), _dc.DTYPE)

        try:
            y_loc_lin, cov_eff = self._linearized_observation_cov(x_pred, cov_r)
            y_loc = y_loc_lin
        except Exception:  # noqa: BLE001
            cov_eff = tf.convert_to_tensor(cov_r, dtype=_dc.DTYPE)

        innovation_fn = getattr(self.ssm, "innovation", None)
        if callable(innovation_fn):
            innov = tf.cast(innovation_fn(y_obs, y_loc), _dc.DTYPE)
            zero = tf.zeros_like(y_loc)
            scale = self._regularized_cholesky(cov_eff)
            return tf.cast(tfd.MultivariateNormalTriL(loc=zero, scale_tril=scale).log_prob(innov), _dc.DTYPE)

        scale = self._regularized_cholesky(cov_eff)
        return tf.cast(tfd.MultivariateNormalTriL(loc=y_loc, scale_tril=scale).log_prob(y_obs), _dc.DTYPE)

    def _transition_log_prob(
        self,
        x_prev: tf.Tensor,
        x_pred: tf.Tensor,
        y_prev: tf.Tensor | None = None,
    ) -> tf.Tensor:
        # Preferred path: exact transition density.
        try:
            trans_dist = self.ssm.transition_dist(x_prev, y_prev=y_prev)
            if trans_dist is not None:
                return tf.cast(trans_dist.log_prob(x_pred), _dc.DTYPE)
        except (NotImplementedError, AttributeError):
            pass

        # Fallback: first-order Gaussianization using f/f_with_noise and cov_eps_x.
        if not callable(getattr(self.ssm, "f", None)):
            raise NotImplementedError(
                "transition_dist is unavailable and fallback requires ssm.f (and noise covariance)."
            )
        cov_q = getattr(self.ssm, "cov_eps_x", None)
        if cov_q is None:
            raise NotImplementedError(
                "transition_dist is unavailable and fallback requires ssm.cov_eps_x."
            )

        x_loc = tf.cast(self.ssm.f(x_prev), _dc.DTYPE)
        try:
            x_loc_lin, cov_eff = self._linearized_transition_cov(x_prev, cov_q)
            x_loc = x_loc_lin
        except Exception:  # noqa: BLE001
            cov_eff = tf.convert_to_tensor(cov_q, dtype=_dc.DTYPE)

        scale = self._regularized_cholesky(cov_eff)
        return tf.cast(tfd.MultivariateNormalTriL(loc=x_loc, scale_tril=scale).log_prob(x_pred), _dc.DTYPE)

    def warmup(self, batch_size=1, T=2, resample=None, y=None):
        """Trace filter graph to reduce first-call overhead."""
        if y is None:
            y = tf.zeros([batch_size, T, self.ssm.obs_dim], _dc.DTYPE)
        if resample is None:
            resample = self.resample
        _ = self.filter(y, resample=resample)

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor,
                      training: bool | None = None):
        """Resampling implementation hook."""
        raise NotImplementedError

    def _identity_parent_indices(self, x: tf.Tensor) -> tf.Tensor:
        batch_shape = tf.shape(x)[:-2]
        return tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape, [self.num_particles]], axis=0),
        )

    def step(self, x_prev, log_w_prev, y_t, resample="auto", y_prev=None,
             training: bool | None = None):
        """One DPF step: propagate, weight, normalize, and optional resample.

        Shapes:
          x_prev: [B, N, dx]
          log_w_prev: [B, N]
          y_t: [B, dy]
          y_prev: [B, dy] (used by observation-driven transitions, if applicable)
        Returns:
          x_pre: [B, N, dx]
          x_t: [B, N, dx]
          log_w_final: [B, N]
          w_final: [B, N]
          parent_indices: [B, N]
          log_w_pre: [B, N]
          logz_t: [B]
        """
        x_pred, log_q = self._sample_proposal(
            x_prev,
            y_t,
            y_prev=y_prev,
        )

        loglik = self._observation_log_prob(x_pred, y_t)
        log_w = log_w_prev + tf.cast(loglik, _dc.DTYPE)
        if log_q is not None:
            tf.debugging.assert_equal(
                tf.shape(log_q),
                tf.shape(log_w_prev),
                message="proposal log_q must have shape [B, N].",
            )
            log_f = self._transition_log_prob(
                x_prev,
                x_pred,
                y_prev=y_prev,
            )
            log_w = log_w + tf.cast(log_f - log_q, _dc.DTYPE)
        log_w_norm, w_pre, logz_t = self._log_normalize(log_w)
        ess = self.ess(w_pre)

        resample = self._normalize_reweight(resample)
        if resample in (1, 2):
            N_float = tf.cast(self.num_particles, _dc.DTYPE)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            no_rs_indices = self._identity_parent_indices(x_pred)
            # non-DMA copied error, have to avoid tf.cond inside tf.while_loop
            x_rs, log_w_rs, rs_indices = self.resample_step(
                x_pred, log_w_norm, training=training)

            mask_w = mask_do_rs[..., tf.newaxis]
            mask_x = mask_do_rs[..., tf.newaxis, tf.newaxis]
            parent_indices = tf.where(mask_w, rs_indices, no_rs_indices)
            x_t = tf.where(mask_x, x_rs, x_pred)
            log_w_final = tf.where(mask_w, log_w_rs, log_w_norm)
        else:
            parent_indices = self._identity_parent_indices(x_pred)
            x_t = x_pred
            log_w_final = log_w_norm

        w_final = tf.exp(log_w_final)
        log_w_pre = log_w_norm
        return (
            x_pred,
            x_t,
            log_w_final,
            w_final,
            parent_indices,
            log_w_pre,
            logz_t,
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
        training: bool | None = None,
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
        return self._filter_loop(y, x_init, log_w_init, resample_mode,
                                 training=training)

    @tf.function(reduce_retracing=True)
    def _filter_loop(self, y, x_prev, log_w, resample,
                     training: bool | None = None):
        """Core filter loop (tf.function compiled)."""
        T = tf.shape(y)[1]

        x_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)
        x_pre_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)
        w_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)
        log_w_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)
        log_w_pre_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)
        parent_ta = tf.TensorArray(tf.int32, size=T, dynamic_size=False)
        logz_ta = tf.TensorArray(_dc.DTYPE, size=T, dynamic_size=False)

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
            y_prev = tf.cond(
                tf.equal(t, 0),
                lambda: tf.zeros_like(y_t),
                lambda: y[:, t - 1, :],
            )

            (
                x_pre,
                x_t,
                log_w,
                w,
                parent_indices,
                log_w_pre,
                logz_t,
            ) = self.step(
                x_prev,
                log_w,
                y_t,
                resample=resample,
                y_prev=y_prev,
                training=training,
            )
            if bool(getattr(self, "stop_grad_through_time", False)):
                x_prev = tf.stop_gradient(x_t)
                log_w = tf.stop_gradient(log_w)
            else:
                x_prev = x_t

            x_pre_ta = x_pre_ta.write(t, x_pre)
            x_ta = x_ta.write(t, x_t)
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


__all__ = [
    "DPFBase",
]
