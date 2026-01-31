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
        optimal_beta: bool = False,
        h_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        jacobian_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        scoreh_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        hvp_logh_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
        beta_guard: bool = False,
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
        self.optimal_beta = bool(optimal_beta) or self.beta_mode in ("optimal", "opt")
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
        def _ensure_rank2(t):
            t = tf.convert_to_tensor(t, dtype=tf.float32)
            t_rank = tf.rank(t)
            tf.debugging.assert_rank_in(
                t,
                [1, 2],
                message="beta schedules must be rank 1 or rank 2 tensors",
            )
            return tf.cond(tf.equal(t_rank, 1), lambda: t[tf.newaxis, :], lambda: t)

        if beta_schedule is not None:
            beta, beta_dot = beta_schedule
            beta = tf.convert_to_tensor(beta, dtype=tf.float32)
            beta_dot = tf.convert_to_tensor(beta_dot, dtype=tf.float32)
            return _ensure_rank2(beta), _ensure_rank2(beta_dot)

        mode = self.beta_mode
        if mode in ("linear", "lin"):
            beta = tf.linspace(0.0, 1.0, self.num_lambda + 1)[:-1][tf.newaxis, :]
            beta_dot = tf.ones_like(beta)
            return beta, beta_dot
        if mode in ("optimal", "opt"):
            beta = tf.linspace(0.0, 1.0, self.num_lambda + 1)[:-1][tf.newaxis, :]
            beta_dot = tf.ones_like(beta)
            return beta, beta_dot
        raise ValueError("beta_mode must be 'linear', 'optimal', or provide beta_schedule")

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


    def _flow_drift(self, x, y, m0, P_inv, beta, beta_dot, w=None):
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
        drift: [..., N, state]
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P_inv = tf.convert_to_tensor(P_inv, dtype=x.dtype)
        beta = tf.cast(beta, x.dtype)
        beta_dot = tf.cast(beta_dot, x.dtype)
        tf.debugging.assert_rank(beta, 1)
        tf.debugging.assert_rank(beta_dot, 1)
        tf.debugging.assert_equal(tf.shape(beta)[0], tf.shape(x)[0])
        tf.debugging.assert_equal(tf.shape(beta_dot)[0], tf.shape(x)[0])

        # prior score and info
        # P_inv = - Hess_p0
        g0 = self._score_log_prior_gaussian(x, m0, P_inv)         # [..., N, dx]

        # likelihood score + Info
        # Info = - Hess_h
        gh, Info = self._likelihood_terms(x, y, w=w)               # gh: [B, N, dx], Info: [B, dx, dx]

        # intermediate score 
        beta_gh = beta[..., tf.newaxis, tf.newaxis]
        gp = g0 + beta_gh * gh                                    # [..., N, dx]

        # M = -Hess_p = P^{-1} + beta*Info  (SPD), shared across particles
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
        if self.diffusion is None:
            Q = tf.zeros([tf.shape(x)[-1], tf.shape(x)[-1]], dtype=x.dtype)
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)

        Qgp = tf.einsum("ij,...nj->...ni", Q, gp)                 # [..., N, dx]
        beta_dot_term = beta_dot[..., tf.newaxis, tf.newaxis]
        drift = 0.5 * Qgp - 0.5 * beta_dot_term * Minv_Info_Minv_gp + beta_dot_term * Minv_gh
        return drift

    @tf.function
    def solve_optimal_beta_schedule(
        self,
        P0_inv,
        Info,
        mu=1.0,
        num_lambda=None,
        tol=1e-6,
        max_bisect=50,
        max_bracket=30,
        v0_low=0.0,
        v0_high=1.0,
        beta_floor=0.0,
        target_beta=1.0,
        debug_shapes: bool = False,
    ):
        """Solve optimal beta schedule via shooting+bisection with RK4 inner integrator.

        Args:
          P0_inv: [batch, dx, dx] prior precision (no particle dimension).
          Info: [batch, dx, dx] information matrix (no particle dimension).
          mu: scalar coefficient in beta ODE.
          num_lambda: number of integration steps (defaults to self.num_lambda).
          tol: bisection tolerance on v0 interval.
          max_bisect: maximum bisection iterations.
          max_bracket: maximum bracket expansions for v0_high.
          v0_low/v0_high: initial bracket for beta_dot(0).
          beta_floor: optional lower bound for beta in M(beta) to keep SPD (>=0).
          target_beta: desired beta(1) target (default 1.0).
        Returns:
          beta: [batch, num_lambda] schedule (start-of-step beta values, float32).
          beta_dot: [batch, num_lambda] schedule (start-of-step beta_dot values, float32).
          v0_star: [batch] optimal initial beta_dot(0), float32.
        """
        compute_dtype = tf.float32
        P0_inv = tf.cast(P0_inv, compute_dtype)
        Info = tf.cast(Info, compute_dtype)
        mu = tf.cast(mu, compute_dtype)
        tol = tf.cast(tol, compute_dtype)
        target_beta = tf.cast(target_beta, compute_dtype)
        beta_floor = tf.cast(beta_floor, compute_dtype)

        tf.debugging.assert_rank(Info, 3)
        tf.debugging.assert_rank(P0_inv, 3)
        tf.debugging.assert_equal(tf.shape(Info)[0], tf.shape(P0_inv)[0])

        Info = 0.5 * (Info + tf.linalg.matrix_transpose(Info))
        P0_inv = 0.5 * (P0_inv + tf.linalg.matrix_transpose(P0_inv))

        if debug_shapes:
            tf.print(
                "solve_optimal_beta_schedule shapes",
                "P0_inv:", tf.shape(P0_inv),
                "Info:", tf.shape(Info),
                "num_lambda:", self.num_lambda,
            )

        num_steps = self.num_lambda if num_lambda is None else num_lambda
        num_steps = tf.cast(num_steps, tf.int32)
        h = tf.cast(1.0, compute_dtype) / tf.cast(num_steps, compute_dtype)

        trace_info = tf.linalg.trace(Info)
        v_lo = tf.cast(v0_low, compute_dtype) + tf.zeros_like(trace_info)
        v_hi = tf.cast(v0_high, compute_dtype) + tf.zeros_like(trace_info)

        def inverse_spd(M):
            jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
            jitter_val = tf.cast(jitter_val, M.dtype)
            eye = tf.eye(tf.shape(M)[-1], batch_shape=tf.shape(M)[:-2], dtype=M.dtype)
            M_sym = 0.5 * (M + tf.linalg.matrix_transpose(M)) + jitter_val * eye
            L = tf.linalg.cholesky(M_sym)
            return tf.linalg.cholesky_solve(L, eye)

        def dkappa_dbeta(beta):
            beta = tf.cast(beta, compute_dtype)
            beta = tf.maximum(beta, beta_floor) # prevent negative beta
            beta_b = beta[..., tf.newaxis, tf.newaxis]
            M_beta = P0_inv + beta_b * Info
            M_inv = inverse_spd(M_beta)
            tr_M = tf.linalg.trace(M_beta)
            tr_M_inv = tf.linalg.trace(M_inv)
            Minv_Info = tf.linalg.matmul(M_inv, Info)
            tr_minv_info_minv = tf.linalg.trace(tf.linalg.matmul(Minv_Info, M_inv))
            return trace_info * tr_M_inv - tr_M * tr_minv_info_minv

        def ode_rhs(beta, v):
            return v, mu * dkappa_dbeta(beta)

        def rk4_step(beta, v):
        # ode integrator
            k1b, k1v = ode_rhs(beta, v)
            k2b, k2v = ode_rhs(beta + 0.5 * h * k1b, v + 0.5 * h * k1v)
            k3b, k3v = ode_rhs(beta + 0.5 * h * k2b, v + 0.5 * h * k2v)
            k4b, k4v = ode_rhs(beta + h * k3b, v + h * k3v)
            beta_next = beta + (h / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
            v_next = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            beta_next = tf.maximum(beta_next, beta_floor) # prevent negative beta
            return beta_next, v_next

        def integrate_endpoint(v0):
            beta = tf.zeros_like(v0)
            v = v0
            i0 = tf.constant(0, dtype=tf.int32)
            if debug_shapes:
                tf.print(
                    "integrate_endpoint entry",
                    "v0:", tf.shape(v0),
                    "beta:", tf.shape(beta),
                    "v:", tf.shape(v),
                )

            def cond(i, beta, v):
                return i < num_steps

            def body(i, beta, v):
                beta, v = rk4_step(beta, v)
                if debug_shapes:
                    def _print():
                        tf.print(
                            "integrate_endpoint step",
                            "i:", i,
                            "beta:", tf.shape(beta),
                            "v:", tf.shape(v),
                        )
                        return 0

                    _ = tf.cond(tf.equal(i, 0), _print, lambda: 0)
                return i + 1, beta, v
            _, beta, _ = tf.while_loop(cond, body, [i0, beta, v])
            return beta

        def residual(v0):
            return integrate_endpoint(v0) - target_beta

        # --- bracket v0_high until residual(v_hi) >= 0 ---
        r_hi = residual(v_hi)
        i0 = tf.constant(0, dtype=tf.int32)

        def bracket_cond(i, v_lo, v_hi, r_hi):
            need = tf.reduce_any(r_hi < 0.0)
            return tf.logical_and(i < tf.cast(max_bracket, tf.int32), need)

        def bracket_body(i, v_lo, v_hi, r_hi):
            v_hi_new = tf.where(r_hi < 0.0, v_hi * 2.0, v_hi)
            r_hi_new = residual(v_hi_new)
            return i + 1, v_lo, v_hi_new, r_hi_new

        _, v_lo, v_hi, r_hi = tf.while_loop(bracket_cond, bracket_body, [i0, v_lo, v_hi, r_hi])
        

        # --- bisection on v0 ---
        r_lo = residual(v_lo)
        tf.debugging.assert_less_equal(r_lo, 0.0)
        tf.debugging.assert_greater_equal(r_hi, 0.0)
        k0 = tf.constant(0, dtype=tf.int32)

        def bisect_cond(k, v_lo, v_hi, r_lo, r_hi):
            width = v_hi - v_lo
            need = tf.reduce_any(width > tol)
            return tf.logical_and(k < tf.cast(max_bisect, tf.int32), need)

        def bisect_body(k, v_lo, v_hi, r_lo, r_hi):
            v_mid = 0.5 * (v_lo + v_hi)
            r_mid = residual(v_mid)
            use_hi = r_mid >= 0.0
            v_hi = tf.where(use_hi, v_mid, v_hi)
            r_hi = tf.where(use_hi, r_mid, r_hi)
            v_lo = tf.where(use_hi, v_lo, v_mid)
            r_lo = tf.where(use_hi, r_lo, r_mid)
            return k + 1, v_lo, v_hi, r_lo, r_hi

        _, v_lo, v_hi, _, _ = tf.while_loop(
            bisect_cond,
            bisect_body,
            [k0, v_lo, v_hi, r_lo, r_hi],
        )

        v0_star = 0.5 * (v_lo + v_hi)

        # --- integrate full trajectory with v0_star ---
        beta0 = tf.zeros_like(v0_star)
        v0 = v0_star
        ta_beta = tf.TensorArray(compute_dtype, size=num_steps + 1)
        ta_v = tf.TensorArray(compute_dtype, size=num_steps + 1)
        ta_beta = ta_beta.write(0, beta0)
        ta_v = ta_v.write(0, v0)

        def traj_cond(i, beta, v, ta_beta, ta_v):
            return i < num_steps

        def traj_body(i, beta, v, ta_beta, ta_v):
            beta, v = rk4_step(beta, v)
            if debug_shapes:
                def _print():
                    tf.print(
                        "integrate_trajectory step",
                        "i:", i,
                        "beta:", tf.shape(beta),
                        "v:", tf.shape(v),
                    )
                    return 0

                _ = tf.cond(tf.equal(i, 0), _print, lambda: 0)
            ta_beta = ta_beta.write(i + 1, beta)
            ta_v = ta_v.write(i + 1, v)
            return i + 1, beta, v, ta_beta, ta_v

        i0 = tf.constant(0, dtype=tf.int32)
        _, _, _, ta_beta, ta_v = tf.while_loop(
            traj_cond,
            traj_body,
            [i0, beta0, v0, ta_beta, ta_v],
        )

        beta_grid = ta_beta.stack()
        v_grid = ta_v.stack()

        beta_grid = tf.linalg.matrix_transpose(beta_grid)
        v_grid = tf.linalg.matrix_transpose(v_grid)
        beta = tf.cast(beta_grid[:, :-1], tf.float32)
        beta_dot = tf.cast(v_grid[:, :-1], tf.float32)
        v0_star = tf.cast(v0_star, tf.float32)
        return beta, beta_dot, v0_star

    @tf.function
    def _flow_transport(self, x, y, m0, P, **kwargs):
        """Apply Euler-Maruyama integration over pseudo-time."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m0 = tf.convert_to_tensor(m0, dtype=x.dtype)
        P = tf.convert_to_tensor(P, dtype=x.dtype)
        w = kwargs.get("w")
        beta_mu = kwargs.get("beta_mu")
        beta_override = kwargs.get("beta_override")
        beta_dot_override = kwargs.get("beta_dot_override")
        P0_inv = self._inverse_from_cov(P)

        num_steps = tf.cast(self.num_lambda, tf.int32)
        dl = tf.constant(1.0, dtype=x.dtype) / tf.cast(num_steps, x.dtype)
        sqrt_dl = tf.sqrt(dl)
        state_dim = tf.shape(x)[-1]
        batch = tf.shape(x)[0]
        x_invar = tf.TensorShape([None, None, None])
        if self.diffusion is None:
            cholQ = None
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)
            jitter_val = 0.0 if self.jitter is None else float(self.jitter)
            if jitter_val > 0.0:
                eye = tf.eye(state_dim, dtype=x.dtype)
                Q = Q + tf.cast(jitter_val, x.dtype) * eye
            cholQ = tf.linalg.cholesky(Q)

        beta_opt = None
        beta_dot_opt = None
        need_info = self.optimal_beta
        Info = None
        if need_info:
            _, Info = self._likelihood_terms(x, y, w=w)

        if beta_override is not None:
            beta_opt = tf.convert_to_tensor(beta_override, dtype=x.dtype)
            beta_dot_opt = tf.convert_to_tensor(beta_dot_override, dtype=x.dtype)
        elif self.optimal_beta:
            if beta_mu is None:
                beta_mu = 1.0
            beta_opt, beta_dot_opt, _ = self.solve_optimal_beta_schedule(
                P0_inv,
                Info,
                mu=beta_mu,
            )

        beta_base = self.beta
        beta_dot_base = self.beta_dot

        if self.diffusion is None:
            Q = tf.zeros([state_dim, state_dim], dtype=x.dtype)
        else:
            Q = tf.convert_to_tensor(self.diffusion, dtype=x.dtype)
        eps = 1e-6 if self.jitter is None or self.jitter <= 0.0 else float(self.jitter)
        
        def _cond_number(F):
            s = tf.linalg.svd(F, compute_uv=False)
            s_min = tf.reduce_min(s, axis=-1)
            s_max = tf.reduce_max(s, axis=-1)
            return s_max / (s_min + tf.cast(eps, s.dtype))
        for j in tf.range(num_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(x, x_invar)],
            )
            if beta_opt is not None:
                beta_j = tf.gather(beta_opt, j, axis=1)
                beta_dot_j = tf.gather(beta_dot_opt, j, axis=1)
                if self.beta_guard and P0_inv is not None:
                    _, Info_step = self._likelihood_terms(x, y, w=w)
                    beta_base_j = tf.gather(beta_base, j, axis=1)
                    beta_dot_base_j = tf.gather(beta_dot_base, j, axis=1)
                    # per step j, for opt/base respectively
                    M_opt = P0_inv + beta_j[...,None,None] * Info_step          # [B, dx, dx]
                    M_base= P0_inv + beta_base_j[...,None,None] * Info_step         # [B, dx, dx]
                    Minv_opt = self._inverse_from_cov(M_opt)                               # [B, dx, dx]
                    Minv_base = self._inverse_from_cov(M_base)                               # [B, dx, dx]

                    term1_opt = tf.einsum("ij,...jk->...ik", Q, M_opt)            # Q M
                    term2_opt = tf.linalg.matmul(Minv_opt, Info_step)                          # M^{-1} Info
                    term1_base = tf.einsum("ij,...jk->...ik", Q, M_base)            # Q M
                    term2_base = tf.linalg.matmul(Minv_base, Info_step)                          # M^{-1} Info

                    F_opt = -0.5 * (term1_opt + beta_dot_j[..., tf.newaxis, tf.newaxis] * term2_opt)
                    F_base = -0.5 * (term1_base + beta_dot_base_j[..., tf.newaxis, tf.newaxis] * term2_base)
                    kappa_opt = _cond_number(F_opt)
                    kappa_base = _cond_number(F_base)
                    use_opt = kappa_opt <= kappa_base

                    beta_j = tf.where(use_opt, beta_j, beta_base_j)
                    beta_dot_j = tf.where(use_opt, beta_dot_j, beta_dot_base_j)
            else:
                beta_j = tf.gather(self.beta, j, axis=1)
                beta_dot_j = tf.gather(self.beta_dot, j, axis=1)

            beta_eff = tf.broadcast_to(beta_j, [batch])
            beta_dot_eff = tf.broadcast_to(beta_dot_j, [batch])
            #beta_eff = beta_eff + 0.5 * beta_dot_eff * dl # midpoint discretization
            #beta_eff = tf.clip_by_value(beta_eff, 0.0, 1.0)
            drift = self._flow_drift(x, y, m0, P0_inv, beta_eff, beta_dot_eff, w=w)

            if cholQ is None:
                x = x + drift * dl
            else:
                xi = self.ssm.rng.normal(tf.shape(x), dtype=x.dtype)
                noise = tf.einsum("ij,bnj->bni", cholQ, xi) * sqrt_dl
                x = x + drift * dl + noise
            x = tf.ensure_shape(x, x_invar)
        log_det = tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
        return x, log_det, None
        
