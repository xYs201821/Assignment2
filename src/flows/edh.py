"""Exact Daum-Huang (EDH) particle flow implementation."""

import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.flows.beta_schedule import BetaScheduleConfig, build_beta_schedule, _cond_number_f
from src.flows.diagnostics import _cond_from_matrix, _cond_from_rect
from src.utility import cholesky_solve, quadratic_matmul


class EDHFlow(FlowBase):
    """EDH flow using global linearization of the observation model."""

    def __init__(
        self,
        ssm,
        num_lambda=20,
        num_particles=100,
        ess_threshold=0.5,
        reweight="auto",
        debug=False,
        jitter=1e-5,
        beta_schedule: BetaScheduleConfig | None = None,
    ):
        """Initialize EDH flow parameters."""
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
            solver_steps=beta_schedule.solver_steps,
            max_bisect=beta_schedule.max_bisect,
            max_bracket=beta_schedule.max_bracket,
            tol=beta_schedule.tol,
        )

    def _flow_diag_keys(self):
        return (
            "condInfo_log10",
            "condA_log10_max",
            "condJ_log10_max",
            "logdetJ",
            "beta_sched",
            "beta_dot_sched",
            "condF_sched",
            "condCov_log10",
            "logdet_cov",
        )

    @staticmethod
    def _edh_flow_solution(lam, H, P, R, y_tilde, m0, jitter):
        """Compute EDH linear flow parameters A and b at pseudo-time lam.

        Shapes:
          H: [B, dy, dx]
          P: [B, dx, dx]
          R: [B, dy, dy]
          y_tilde: [B, dy]
          m0: [B, dx]
        Returns:
          A: [B, dx, dx]
          b: [B, dx]
          S: [B, dy, dy]
        """
        # S = lam * H P H^T + R and K = P H^T S^{-1}.
        HPH = quadratic_matmul(H, P, H)
        lam_b = lam[..., tf.newaxis, tf.newaxis]
        S = lam_b * HPH + R
        jitter_val = float(jitter) if jitter is not None else 0.0
        if jitter_val > 0.0:
            eye = tf.eye(tf.shape(S)[-1], batch_shape=tf.shape(S)[:-2], dtype=S.dtype)
            S = S + tf.cast(jitter_val, S.dtype) * eye
        RHS = tf.linalg.matmul(H, P, transpose_b=True)
        K_T = cholesky_solve(S, RHS, jitter=jitter_val)
        K = tf.linalg.matrix_transpose(K_T)

        # Linear flow: dx/dlambda = A x + b.
        A = -0.5 * tf.linalg.matmul(K, H)
        I = tf.eye(tf.shape(A)[-1], batch_shape=tf.shape(A)[:-2], dtype=A.dtype)
        b = tf.einsum("bij,bjk,bk->bi", I + lam_b * A, K, y_tilde)
        Am0 = tf.einsum("bij,bjk,bk->bi", I + 2.0 * lam_b * A, A, m0)
        b = b + Am0
        return A, b, S
    
    def _flow_transport(self, mu_tilde, y, m0, P, w=None):
        """Integrate the EDH flow over pseudo-time to transport particles.

        Shapes:
          mu_tilde: [B, N, dx]
          y: [B, dy]
          m0: [B, dx]
          P: [B, dx, dx]
        Returns:
          mu: [B, N, dx]
          logdet: [B]
          diagnostics: dict of per-step metrics
        """
        mu = mu_tilde
        m_bar = tf.identity(m0)
        R = self.ssm.cov_eps_y
        batch_shape = tf.shape(m_bar)[:-1]
        state_dim = tf.shape(m_bar)[-1]
        obs_dim = tf.shape(y)[-1]
        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        I = tf.eye(state_dim, batch_shape=batch_shape, dtype=mu.dtype)

        r0_sched = tf.zeros(tf.concat([batch_shape, [r_dim]], axis=0), dtype=mu.dtype)
        H0, _ = self.jacobian_h_x(m0, r0_sched)
        H_r0, _ = self.jacobian_h_r(m0, r0_sched)
        R_eff0 = quadratic_matmul(H_r0, R, H_r0)
        R_inv0 = self._inverse_from_cov(R_eff0)
        RinvJ = tf.einsum("bij,bjk->bik", R_inv0, H0)
        Info = tf.einsum("bji,bjk->bik", H0, RinvJ)
        P0_inv = self._inverse_from_cov(P)

        guard_on = bool(self.beta_schedule.guard) and self.beta_schedule.mode == "optimal"
        if self.beta_schedule.mode == "optimal":
            beta, beta_dot, dl = build_beta_schedule(
                "optimal",
                num_lambda=self.num_lambda,
                dtype=mu.dtype,
                P0_inv=P0_inv,
                Info=Info,
                jitter=self.jitter,
                mu=self.beta_schedule.mu,
                solver_steps=self.beta_schedule.solver_steps,
                max_bisect=self.beta_schedule.max_bisect,
                max_bracket=self.beta_schedule.max_bracket,
                tol=self.beta_schedule.tol,
            )
            if guard_on:
                beta_base, beta_dot_base, _ = build_beta_schedule(
                    "linear",
                    num_lambda=self.num_lambda,
                    dtype=mu.dtype,
                )
                beta_base = tf.broadcast_to(beta_base, tf.shape(beta))
                beta_dot_base = tf.broadcast_to(beta_dot_base, tf.shape(beta_dot))
        else:
            beta, beta_dot, dl = build_beta_schedule(
                "linear",
                num_lambda=self.num_lambda,
                dtype=mu.dtype,
            )
        tf.debugging.assert_equal(
            tf.shape(beta)[-1],
            self.num_lambda,
            message="beta schedule length must match num_lambda",
        )
        jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
        eps = jitter_val if jitter_val > 0.0 else 1e-6
        eps_t = tf.cast(eps, mu.dtype)
        log10_base = tf.math.log(10.0)
        cond_f_sched = _cond_number_f(
            P0_inv,
            Info,
            None,
            beta,
            beta_dot,
            dtype=mu.dtype,
            jitter=self.jitter,
            eps=eps,
        )
        cond_f_sched = tf.where(
            tf.math.is_finite(cond_f_sched),
            cond_f_sched,
            tf.ones_like(cond_f_sched),
        )
        cond_info = _cond_from_matrix(Info, eps_t)
        cond_info = tf.where(
            tf.math.is_finite(cond_info),
            cond_info,
            tf.ones_like(cond_info),
        )
        cond_info_log10 = tf.cast(
            tf.math.log(cond_info + eps_t) / log10_base,
            tf.float32,
        )
        logdet = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        condA_log10_max = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        condJ_log10_max = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        r0 = tf.zeros(tf.concat([batch_shape, [r_dim]], axis=0), dtype=mu.dtype)

        for j in range(self.num_lambda):
            beta_j = beta[:, j]
            beta_dot_j = beta_dot[:, j]
            H, h_m = self.jacobian_h_x(m_bar, r0)
            H_r, _ = self.jacobian_h_r(m_bar, r0)
            Hm = tf.einsum("bij,bj->bi", H, m_bar)
            v = self.ssm.innovation(y, h_m)
            y_tilde = v + Hm
            R_eff = quadratic_matmul(H_r, R, H_r)
            if guard_on:
                beta_base_j = beta_base[:, j]
                beta_dot_base_j = beta_dot_base[:, j]
                step_scale_opt = dl * beta_dot_j
                step_scale_base = dl * beta_dot_base_j
                A_opt, b_opt, _ = self._edh_flow_solution(
                    beta_j,
                    H,
                    P,
                    R_eff,
                    y_tilde,
                    m0,
                    self.jitter,
                )
                A_base, b_base, _ = self._edh_flow_solution(
                    beta_base_j,
                    H,
                    P,
                    R_eff,
                    y_tilde,
                    m0,
                    self.jitter,
                )
                J_opt = I + step_scale_opt[..., tf.newaxis, tf.newaxis] * A_opt
                J_base = I + step_scale_base[..., tf.newaxis, tf.newaxis] * A_base
                if self.jitter and self.jitter > 0.0:
                    J_opt = J_opt + tf.cast(self.jitter, J_opt.dtype) * I
                    J_base = J_base + tf.cast(self.jitter, J_base.dtype) * I
                finite_J_opt = tf.reduce_all(tf.math.is_finite(J_opt), axis=[-2, -1])
                finite_J_base = tf.reduce_all(tf.math.is_finite(J_base), axis=[-2, -1])
                J_opt_safe = tf.where(finite_J_opt[..., tf.newaxis, tf.newaxis], J_opt, I)
                J_base_safe = tf.where(finite_J_base[..., tf.newaxis, tf.newaxis], J_base, I)
                condJ_opt = _cond_from_rect(J_opt_safe, eps_t)
                condJ_base = _cond_from_rect(J_base_safe, eps_t)
                finite_opt = tf.logical_and(tf.math.is_finite(condJ_opt), finite_J_opt)
                finite_base = tf.logical_and(tf.math.is_finite(condJ_base), finite_J_base)
                use_opt = tf.logical_and(
                    finite_opt,
                    tf.logical_or(~finite_base, condJ_opt <= condJ_base),
                )
                step_scale = tf.where(use_opt, step_scale_opt, step_scale_base)
                A = tf.where(use_opt[..., tf.newaxis, tf.newaxis], A_opt, A_base)
                b = tf.where(use_opt[..., tf.newaxis], b_opt, b_base)
            else:
                step_scale = dl * beta_dot_j
                A, b, _ = self._edh_flow_solution(beta_j, H, P, R_eff, y_tilde, m0, self.jitter)
            Am = tf.einsum("bij,bj->bi", A, m_bar)
            # Euler update for mean flow ODE: m_bar += delta * (A m_bar + b).
            m_bar_next = m_bar + step_scale[..., tf.newaxis] * (Am + b)
            J = I + step_scale[..., tf.newaxis, tf.newaxis] * A
            if self.jitter and self.jitter > 0.0:
                J = J + tf.cast(self.jitter, J.dtype) * I
            sign, lad = tf.linalg.slogdet(J)
            finite_A = tf.reduce_all(tf.math.is_finite(A), axis=[-2, -1])
            finite_b = tf.reduce_all(tf.math.is_finite(b), axis=-1)
            finite_step = tf.math.is_finite(step_scale)
            finite_J = tf.reduce_all(tf.math.is_finite(J), axis=[-2, -1])
            bad = tf.logical_or(tf.equal(sign, 0.0), tf.logical_not(finite_J))
            bad = tf.logical_or(bad, tf.logical_not(finite_A))
            bad = tf.logical_or(bad, tf.logical_not(finite_b))
            bad = tf.logical_or(bad, tf.logical_not(finite_step))
            bad = tf.logical_or(bad, tf.logical_not(tf.math.is_finite(lad)))
            lad = tf.where(bad, tf.zeros_like(lad), lad)
            logdet = logdet + lad
            condA = _cond_from_rect(A, eps_t)
            condA = tf.where(
                tf.logical_or(tf.logical_not(tf.math.is_finite(condA)), bad),
                tf.ones_like(condA),
                condA,
            )
            condA_log10 = tf.math.log(condA + eps_t) / log10_base
            condA_log10_max = tf.maximum(condA_log10_max, tf.cast(condA_log10, tf.float32))
            condJ = _cond_from_rect(J, eps_t)
            condJ = tf.where(
                tf.logical_or(tf.logical_not(tf.math.is_finite(condJ)), bad),
                tf.ones_like(condJ),
                condJ,
            )
            condJ_log10 = tf.math.log(condJ + eps_t) / log10_base
            condJ_log10_max = tf.maximum(condJ_log10_max, tf.cast(condJ_log10, tf.float32))
            # Particle transport with Euler discretization.
            Ax = tf.einsum("bij,bnj->bni", A, mu)
            mu_next = mu + step_scale[..., tf.newaxis, tf.newaxis] * (
                Ax + b[..., tf.newaxis, :]
            )
            m_bar = tf.where(bad[..., tf.newaxis], m_bar, m_bar_next)
            mu = tf.where(bad[..., tf.newaxis, tf.newaxis], mu, mu_next)

        # Compute diagnostics outside the loop
        w_uniform = tf.ones(tf.shape(mu)[:-1], dtype=mu.dtype) / tf.cast(tf.shape(mu)[-2], mu.dtype)
        cov = self.ssm.state_cov(mu, w_uniform)
        # Ensure symmetric and add jitter for numerical stability
        cov = 0.5 * (cov + tf.linalg.matrix_transpose(cov))
        jitter_val = tf.maximum(eps_t, tf.cast(1e-5, cov.dtype))
        jitter_eye = jitter_val * tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype)
        cov_stable = cov + jitter_eye
        cond_cov = _cond_from_matrix(cov_stable, eps_t)
        cond_cov_log10 = tf.math.log(cond_cov + eps_t) / log10_base
        s = tf.linalg.svd(cov_stable, compute_uv=False)
        s = tf.maximum(s, eps_t)
        logdet_cov = tf.reduce_sum(tf.math.log(s), axis=-1)

        diagnostics = {
            "condInfo_log10": cond_info_log10,
            "condA_log10_max": condA_log10_max,
            "condJ_log10_max": condJ_log10_max,
            "logdetJ": logdet,
            "beta_sched": tf.cast(beta, tf.float32),
            "beta_dot_sched": tf.cast(beta_dot, tf.float32),
            "condF_sched": tf.cast(cond_f_sched, tf.float32),
            "condCov_log10": cond_cov_log10,
            "logdet_cov": logdet_cov,
        }
        return mu, logdet, diagnostics
