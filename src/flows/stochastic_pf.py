"""Stochastic particle flow (SDE-based) skeleton."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.flows.flow_base import FlowBase
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
        beta_mode: str = "linear",
        beta_schedule: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
        self.beta_mode = str(beta_mode).lower()
        self.beta_schedule = beta_schedule
        self.beta, self.beta_dot = self._beta_schedule(beta_schedule)
        self.h_fn = h_fn
        self.jacobian_fn = jacobian_fn
        self.scoreh_fn = scoreh_fn
        self.hvp_logh_fn = hvp_logh_fn

    def _flow_supports_reweight(self):
        """Stochastic flow does not provide a valid Jacobian correction."""
        return False

    def _beta_schedule(self, beta_schedule):
        """Return beta and beta_dot schedules for pseudo-time."""
        if beta_schedule is not None:
            beta, beta_dot = beta_schedule
            beta = tf.convert_to_tensor(beta, dtype=tf.float32)
            beta_dot = tf.convert_to_tensor(beta_dot, dtype=tf.float32)
            return beta, beta_dot

        mode = self.beta_mode
        if mode in ("linear", "lin"):
            beta = tf.linspace(0.0, 1.0, self.num_lambda + 1)[1:]
            beta_dot = tf.ones_like(beta)
            return beta, beta_dot
        raise ValueError("beta_mode must be 'linear' or provide beta_schedule")

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

    def _likelihood_linearized(self, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y, tf.float32)

        batch_shape = tf.shape(x)[:-2]
        num_particles = tf.shape(x)[-2]
        state_dim = tf.shape(x)[-1]
        obs_dim = tf.shape(y)[-1]

        def _broadcast_y():
            return tf.broadcast_to(
                y[..., tf.newaxis, :],
                tf.concat([batch_shape, [num_particles, obs_dim]], axis=0),
            )

        x_rank = tf.rank(x)
        y_rank = tf.rank(y)
        y_b = tf.cond(tf.equal(y_rank, x_rank - 1), _broadcast_y, lambda: y)

        r_dim = tf.cast(self.ssm.r_dim, tf.int32)

        r0 = tf.zeros(tf.concat([batch_shape, [num_particles, r_dim]], axis=0), dtype=x.dtype)

        # shapes: [..., N, obs_dim, state_dim], [..., N, obs_dim]
        H_x, h = self.jacobian_h_x(x, r0)
        # shape: [..., N, obs_dim, r_dim]
        H_r, _ = self.jacobian_h_r(x, r0)

        # residual  r = y - h(x)  
        r = self.ssm.innovation(y_b, h)  # [..., num_particles, obs_dim]

        # effective  covariance
        R = tf.convert_to_tensor(self.ssm.cov_eps_y, dtype=x.dtype)   # [r_dim, r_dim] or [...]
        R_eff = quadratic_matmul(H_r, R, H_r)                         

        return H_x, r, R_eff


    def _likelihood_terms(self, x, y):
        J, r, R_eff= self._likelihood_linearized(x, y)
        R_inv = self._inverse_from_cov(R_eff)  # [..., N, obs_dim, obs_dim]

        # gh = J^T R^{-1} r
        Rinv_r = tf.einsum("...nij,...nj->...ni", R_inv, r)          # [..., N, M]
        gh = tf.einsum("...nji,...nj->...ni", J, Rinv_r)             # [..., N, D]

        # Info = J^T R^{-1} J
        RinvJ = tf.einsum("...nij,...njk->...nik", R_inv, J)         # [..., N, M, D]
        Info  = tf.einsum("...nji,...njk->...nik", J, RinvJ)         # [..., N, D, D]

        return gh, Info


    def _flow_drift(self, x, y, m0, P, beta, beta_dot):
        """
        Dai-style stochastic flow drift under:
        - prior: N(m0, P)
        - likelihood: Gauss-Newton linearization

        Shapes:
        x:   [..., N, state]
        y:   [..., obs] or [..., N, obs] (your _likelihood_linearized handles it)
        m0:  [..., state]
        P:   [..., state, state]
        beta, beta_dot: scalars (for current pseudo-time step)
        Returns:
        drift: [..., N, state]
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P = tf.convert_to_tensor(P, dtype=x.dtype)
        beta = tf.cast(beta, x.dtype)
        beta_dot = tf.cast(beta_dot, x.dtype)

        # prior score and info
        P_inv = self._inverse_from_cov(P)                         # [..., dx, dx]
        g0 = self._score_log_prior_gaussian(x, m0, P_inv)         # [..., N, dx]

        # likelihood score + Info
        gh, Info = self._likelihood_terms(x, y)                   # gh: [..., N, dx], Info: [..., N, dx, dx]

        # intermediate score 
        gp = g0 + beta * gh                                       # [..., N, dx]

        # A = -Hess_p = P^{-1} + beta*Info  (SPD) 
        P_inv_b = P_inv[..., tf.newaxis, :, :]                    # [..., 1, dx, dx]
        A = P_inv_b + beta * Info                                 # [..., N, dx, dx]

        # Solve A^{-1} gh and A^{-1} gp
        Ainv_gh = cholesky_solve(A, gh, jitter=float(self.jitter))  # [..., N, dx]
        Ainv_gp = cholesky_solve(A, gp, jitter=float(self.jitter))  # [..., N, dx]

        # A^{-1} Info A^{-1} gp
        Info_Ainv_gp = tf.einsum("...nij,...nj->...ni", Info, Ainv_gp)           # [..., N, dx]
        Ainv_Info_Ainv_gp = cholesky_solve(A, Info_Ainv_gp, jitter=float(self.jitter))

        # diffusion Q  [..., dx, dx]
        if self.diffusion is None:
            Q = tf.zeros([tf.shape(x)[-1], tf.shape(x)[-1]], dtype=x.dtype)
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)

        Qgp = tf.einsum("ij,...nj->...ni", Q, gp)                 # [..., N, dx]
        drift = 0.5 * Qgp - 0.5 * beta_dot * Ainv_Info_Ainv_gp + beta_dot * Ainv_gh
        return drift

    def _flow_transport(self, x, y, m0, P, **kwargs):
        """Apply Euler-Maruyama integration over pseudo-time."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P = tf.convert_to_tensor(P, dtype=x.dtype)

        num_steps = int(self.num_lambda)
        dl = tf.constant(1.0, dtype=x.dtype) / tf.cast(num_steps, x.dtype)
        sqrt_dl = tf.sqrt(dl)
        state_dim = tf.shape(x)[-1]

        if self.diffusion is None:
            cholQ = None
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)
            jitter_val = 0.0 if self.jitter is None else float(self.jitter)
            if jitter_val > 0.0:
                eye = tf.eye(state_dim, dtype=x.dtype)
                Q = Q + tf.cast(jitter_val, x.dtype) * eye
            cholQ = tf.linalg.cholesky(Q)

        for j in range(num_steps):
            beta = self.beta[j]
            beta_dot = self.beta_dot[j]

            drift = self._flow_drift(x, y, m0, P, beta, beta_dot)

            if cholQ is None:
                x = x + drift * dl
            else:
                xi = self.ssm.rng.normal(tf.shape(x), dtype=x.dtype)
                noise = tf.einsum("ij,...nj->...ni", cholQ, xi) * sqrt_dl
                x = x + drift * dl + noise

        log_det = tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
        return x, log_det, None
        
