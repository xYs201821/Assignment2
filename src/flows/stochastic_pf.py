"""Stochastic particle flow (SDE-based) skeleton."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.flows.beta_schedule import BetaScheduleConfig, build_beta_schedule, _cond_number_f
from src.utility import cholesky_solve, quadratic_matmul


class StochasticParticleFlow(FlowBase):
    """Stochastic particle flow with Gaussian prior + Gauss-Newton likelihood."""

    def __init__(
        self,
        ssm,
        num_lambda: int = 20,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        reweight: str | bool | int = "never",
        debug: bool = False,
        jitter: float = 1e-6,
        diffusion: Optional[np.ndarray] = None,
        beta_schedule: BetaScheduleConfig | None = None,
        h_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        jacobian_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        scoreh_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        hvp_logh_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
    ) -> None:
        """Initialize flow parameters."""
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            reweight=reweight,
            init_from_particles="sample",
            debug=debug,
            jitter=jitter,
        )
        self.diffusion = diffusion
        if beta_schedule is None:
            beta_schedule = BetaScheduleConfig()
        if not isinstance(beta_schedule, BetaScheduleConfig):
            raise TypeError("beta_schedule must be a BetaScheduleConfig or None")
        mode = str(beta_schedule.mode).lower()
        if mode in ("linear", "lin"):
            mode = "linear"
        elif mode in ("optimal", "opt"):
            mode = "optimal"
        else:
            raise ValueError("beta_schedule must be 'linear' or 'optimal'")
        self.beta_schedule = BetaScheduleConfig(
            mode=mode,
            mu=float(beta_schedule.mu),
            guard=beta_schedule.guard,
        )
        self.h_fn = h_fn
        self.jacobian_fn = jacobian_fn
        self.scoreh_fn = scoreh_fn
        self.hvp_logh_fn = hvp_logh_fn

    def _flow_supports_reweight(self):
        """Stochastic flow does not provide a valid Jacobian correction."""
        return False

    def _inverse_from_cov(self, cov):
        """Compute a stable inverse via Cholesky solve."""
        cov = tf.convert_to_tensor(cov, dtype=tf.float32)
        eye = tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype)
        jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
        return cholesky_solve(cov, eye, jitter=jitter_val)

    @staticmethod
    def _score_log_prior_gaussian(x, m0, P_inv):
        """Gradient of log N(m0, P) with respect to x."""
        delta = x - m0[..., tf.newaxis, :]
        return -tf.einsum("...ij,...nj->...ni", P_inv, delta)

    @staticmethod
    def _hvp_log_prior_gaussian(P_inv, V):
        """Hessian-vector product of log N(m0, P) with respect to x."""
        return -tf.einsum("...ij,...nj->...ni", P_inv, V)

    def _likelihood_linearized(self, x, y, w=None):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y, tf.float32)

        num_particles = tf.shape(x)[-2]
        obs_dim = tf.shape(y)[-1]

        tf.debugging.assert_rank(
            y,
            2,
            message="y must have shape [batch, obs_dim]",
        )

        if w is None:
            x_mean = tf.reduce_mean(x, axis=-2)
        else:
            w = tf.convert_to_tensor(w, dtype=x.dtype)
            w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
            x_mean = self.ssm.state_mean(x, w)

        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        r0 = tf.zeros([tf.shape(x_mean)[0], r_dim], dtype=x.dtype)

        # shapes: [B, obs_dim, state_dim], [B, obs_dim]
        H_x, h_mean = self.jacobian_h_x(x_mean, r0)
        # shape: [B, obs_dim, r_dim]
        H_r, _ = self.jacobian_h_r(x_mean, r0)

        y_b = y[:, tf.newaxis, :]
        x_resid = x - x_mean[..., tf.newaxis, :]
        y_hat = h_mean[..., tf.newaxis, :] + tf.einsum("...ij,...nj->...ni", H_x, x_resid)

        # residual  r = y - h_lin(x)
        r = self.ssm.innovation(y_b, y_hat)  # [B, num_particles, obs_dim]

        # effective  covariance
        R = tf.convert_to_tensor(self.ssm.cov_eps_y, dtype=x.dtype)   # [r_dim, r_dim] or [...]
        R_eff = quadratic_matmul(H_r, R, H_r)

        return H_x, r, R_eff


    def _likelihood_terms(self, x, y, w=None):
        J, r, R_eff = self._likelihood_linearized(x, y, w=w)
        R_inv = self._inverse_from_cov(R_eff)  # [B, obs_dim, obs_dim]

        # gh = J^T R^{-1} r
        Rinv_r = tf.einsum("...ij,...nj->...ni", R_inv, r)           # [B, N, obs_dim]
        gh = tf.einsum("...ji,...nj->...ni", J, Rinv_r)              # [B, N, dx]

        # Info = J^T R^{-1} J
        RinvJ = tf.einsum("...ij,...jk->...ik", R_inv, J)            # [B, obs_dim, dx]
        Info = tf.einsum("...ji,...jk->...ik", J, RinvJ)             # [B, dx, dx]

        return gh, Info


    def _flow_drift(self, x, y, m0, P_inv, beta, beta_dot, w=None, gh=None, Info=None, Q=None):
        """
        Dai-style stochastic flow drift under:
        - prior: N(m0, P)
        - likelihood: Gauss-Newton linearization

        Shapes:
        x:   [..., N, state]
        y:   [batch, obs]
        m0:  [..., state]
        P_inv: [..., state, state]
        beta, beta_dot: [batch] (for current pseudo-time step)
        Returns:
        drift: [..., N, state] (λ-time drift)
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P_inv = tf.convert_to_tensor(P_inv, dtype=x.dtype)
        beta = tf.cast(beta, x.dtype)
        beta_dot = tf.cast(beta_dot, x.dtype)

        # prior score and info
        # P_inv = - Hess_p0
        g0 = self._score_log_prior_gaussian(x, m0, P_inv)         # [..., N, dx]

        # likelihood score + Info
        # Info = - Hess_h
        if gh is None or Info is None:
            gh, Info = self._likelihood_terms(x, y, w=w)               # gh: [B, N, dx], Info: [B, dx, dx]

        # intermediate score 
        beta_gh = beta[..., tf.newaxis, tf.newaxis]
        gp = g0 + beta_gh * gh                                    # [..., N, dx]

        # M = -Hess_p = P^{-1} + beta*Info, shared across particles
        beta_info = beta[..., tf.newaxis, tf.newaxis]                            # [B, 1, 1]
        M = P_inv + beta_info * Info                                             # [B, dx, dx]

        jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
        M_sym = 0.5 * (M + tf.linalg.matrix_transpose(M))
        eye = tf.eye(tf.shape(M_sym)[-1], batch_shape=tf.shape(M_sym)[:-2], dtype=M_sym.dtype)
        M_robust = M_sym + tf.cast(jitter_val, M_sym.dtype) * eye
        L = tf.linalg.cholesky(M_robust)                                         # [B, dx, dx]

        def _chol_solve(rhs):
            rhs_t = tf.linalg.matrix_transpose(rhs)                              # [B, dx, N]
            sol_t = tf.linalg.cholesky_solve(L, rhs_t)                           # [B, dx, N]
            return tf.linalg.matrix_transpose(sol_t)                             # [B, N, dx]

        # Solve M^{-1} gh and M^{-1} gp
        Minv_gh = _chol_solve(gh)                                                # [B, N, dx]
        Minv_gp = _chol_solve(gp)                                                # [B, N, dx]

        # M^{-1} Info M^{-1} gp
        Info_Minv_gp = tf.einsum("bij,bnj->bni", Info, Minv_gp)                  # [B, N, dx]
        Minv_Info_Minv_gp = _chol_solve(Info_Minv_gp)                            # [B, N, dx]

        # diffusion Q  [..., dx, dx]
        if Q is None:
            if self.diffusion is None:
                Q = tf.zeros([tf.shape(x)[-1], tf.shape(x)[-1]], dtype=x.dtype)
            else:
                Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)

        Qgp = tf.einsum("ij,...nj->...ni", Q, gp)                 # [..., N, dx]
        beta_dot_term = beta_dot[..., tf.newaxis, tf.newaxis]
        drift = 0.5 * Qgp - 0.5 * beta_dot_term * Minv_Info_Minv_gp + beta_dot_term * Minv_gh
        return drift

    def _flow_transport(self, x, y, m0, P, w=None):
        """Apply Euler-Maruyama integration over pseudo-time."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P = tf.convert_to_tensor(P, dtype=x.dtype)
        P0_inv = self._inverse_from_cov(P)

        num_steps = tf.cast(self.num_lambda, tf.int32)
        state_dim = tf.shape(x)[-1]
        batch = tf.shape(x)[0]
        x_invar = tf.TensorShape([None, None, None])
        if self.diffusion is None:
            Q = tf.zeros([state_dim, state_dim], dtype=x.dtype)
            cholQ = None
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)
            jitter_val = 0.0 if self.jitter is None else float(self.jitter)
            if jitter_val > 0.0:
                eye = tf.eye(state_dim, dtype=x.dtype)
                Q_chol = Q + tf.cast(jitter_val, x.dtype) * eye
            else:
                Q_chol = Q
            cholQ = tf.linalg.cholesky(Q_chol)

        if self.beta_schedule.mode == "optimal":
            _, Info0 = self._likelihood_terms(x, y, w=w)
            beta, beta_dot, dl = build_beta_schedule(
                "optimal",
                num_lambda=self.num_lambda,
                dtype=x.dtype,
                P0_inv=P0_inv,
                Info=Info0,
                jitter=self.jitter,
                mu=self.beta_schedule.mu,
            )
        else:
            beta, beta_dot, dl = build_beta_schedule(
                "linear",
                num_lambda=self.num_lambda,
                dtype=x.dtype,
            )
        beta = tf.convert_to_tensor(beta, dtype=x.dtype)
        beta_dot = tf.convert_to_tensor(beta_dot, dtype=x.dtype)
        tf.debugging.assert_rank(
            beta,
            2,
            message="beta must have shape [batch, num_lambda]",
        )
        tf.debugging.assert_rank(
            beta_dot,
            2,
            message="beta_dot must have shape [batch, num_lambda]",
        )
        tf.debugging.assert_equal(
            tf.shape(beta)[-1],
            self.num_lambda,
            message="beta schedule length must match num_lambda",
        )
        guard_on = tf.cast(bool(self.beta_schedule.guard), tf.bool)
        dl = tf.cast(dl, x.dtype)
        sqrt_dl = tf.sqrt(dl)
        beta_base = tf.linspace(
            tf.cast(0.0, x.dtype),
            tf.cast(1.0, x.dtype),
            self.num_lambda + 1,
        )[:-1]
        beta_base = tf.broadcast_to(beta_base[tf.newaxis, :], tf.shape(beta))
        beta_dot_base = tf.ones_like(beta_base)

        for j in tf.range(num_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(x, x_invar)],
            )
            beta_j = tf.gather(beta, j, axis=1)
            beta_dot_j = tf.gather(beta_dot, j, axis=1)
            beta_base_j = tf.gather(beta_base, j, axis=1)
            beta_dot_base_j = tf.gather(beta_dot_base, j, axis=1)
            beta_eff = tf.broadcast_to(beta_j, [batch])
            beta_dot_eff = tf.broadcast_to(beta_dot_j, [batch])
            gh, Info = self._likelihood_terms(x, y, w=w)
            beta_eff_col = beta_eff[:, tf.newaxis]
            beta_dot_eff_col = beta_dot_eff[:, tf.newaxis]
            beta_base_col = tf.broadcast_to(beta_base_j, [batch])[:, tf.newaxis]
            beta_dot_base_col = tf.broadcast_to(beta_dot_base_j, [batch])[:, tf.newaxis]
            cond_opt = _cond_number_f(
                P0_inv,
                Info,
                Q,
                beta_eff_col,
                beta_dot_eff_col,
                dtype=x.dtype,
                jitter=self.jitter,
            )
            cond_base = _cond_number_f(
                P0_inv,
                Info,
                Q,
                beta_base_col,
                beta_dot_base_col,
                dtype=x.dtype,
                jitter=self.jitter,
            )
            cond_opt = tf.reshape(cond_opt, [batch])
            cond_base = tf.reshape(cond_base, [batch])
            finite_opt = tf.math.is_finite(cond_opt)
            finite_base = tf.math.is_finite(cond_base)
            use_opt = tf.logical_and(
                finite_opt,
                tf.logical_or(~finite_base, cond_opt <= cond_base),
            )
            use_opt = tf.logical_or(~guard_on, use_opt)
            beta_eff = tf.where(use_opt, beta_eff, tf.broadcast_to(beta_base_j, [batch]))
            beta_dot_eff = tf.where(use_opt, beta_dot_eff, tf.broadcast_to(beta_dot_base_j, [batch]))
            #beta_eff = beta_eff + 0.5 * beta_dot_eff * dl # midpoint discretization
            #beta_eff = tf.clip_by_value(beta_eff, 0.0, 1.0)
            drift = self._flow_drift(
                x,
                y,
                m0,
                P0_inv,
                beta_eff,
                beta_dot_eff,
                w=w,
                gh=gh,
                Info=Info,
                Q=Q,
            )

            if cholQ is None:
                x = x + drift * dl
            else:
                xi = self.ssm.rng.normal(tf.shape(x), dtype=x.dtype)
                noise = tf.einsum("ij,bnj->bni", cholQ, xi) * sqrt_dl
                x = x + drift * dl + noise
            x = tf.ensure_shape(x, x_invar)
        log_det = tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
        return x, log_det, None
        
