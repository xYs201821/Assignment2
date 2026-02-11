"""Localized Exact Daum-Huang (LEDH) particle flow implementation."""

import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.flows.beta_schedule import BetaScheduleConfig, build_beta_schedule
from src.flows.diagnostics import _cond_from_matrix, _cond_from_rect
from src.utility import cholesky_solve, quadratic_matmul


class LEDHFlow(FlowBase):
    """LEDH flow using per-particle linearization."""

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
        """Initialize LEDH flow parameters."""
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            reweight=reweight,
            init_from_particles="per_particle",
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
            "condCov_log10",
            "logdet_cov",
        )

    @staticmethod
    def _ledh_flow_solution(lam, H, P, R, y_tilde, m0, jitter):
        """Compute LEDH linear flow parameters A and b at pseudo-time lam.

        Shapes:
          H: [B, N, dy, dx]
          P: [B, dx, dx] or [B, N, dx, dx]
          R: [B, N, dy, dy]
          y_tilde: [B, N, dy]
          m0: [B, N, dx]
        Returns:
          A: [B, N, dx, dx]
          b: [B, N, dx]
        """
        # S = lam * H P H^T + R and K = P H^T S^{-1}.
        HPH = quadratic_matmul(H, P, H)
        lam_b = lam[..., tf.newaxis, tf.newaxis]
        S = lam_b * HPH + R
        jitter_val = float(jitter) if jitter is not None else 0.0
        if jitter_val > 0.0:
            eye = tf.eye(tf.shape(S)[-1], batch_shape=tf.shape(S)[:-2], dtype=S.dtype)
            S = S + tf.cast(jitter_val, S.dtype) * eye
        PHt = tf.linalg.matmul(P, H, transpose_b=True)
        K_T = cholesky_solve(S, tf.linalg.matrix_transpose(PHt), jitter=jitter_val)
        K = tf.linalg.matrix_transpose(K_T)

        # Linear flow: dx/dlambda = A x + b (per particle).
        A = -0.5 * tf.linalg.matmul(K, H)
        I = tf.eye(tf.shape(A)[-1], batch_shape=tf.shape(A)[:-2], dtype=A.dtype)
        b = tf.einsum("bnij,bnjk,bnk->bni", I + lam_b * A, K, y_tilde)
        Am0 = tf.einsum("bnij,bnjk,bnk->bni", I + 2.0 * lam_b * A, A, m0)
        b = b + Am0
        return A, b

    def _flow_transport(self, mu_tilde, y, m0, P, w=None):
        """Integrate the LEDH flow over lambda to transport particles.

        Shapes:
          mu_tilde: [B, N, dx]
          y: [B, dy]
          m0: [B, dx]
          P: [B, dx, dx]
        Returns:
          mu: [B, N, dx]
          logdet: [B, N]
          diagnostics: dict of per-step metrics
        """
        mu = mu_tilde
        batch_shape = tf.shape(mu)[:-2]
        N = tf.shape(mu)[-2]
        state_dim = tf.shape(mu)[-1]
        obs_dim = tf.shape(y)[-1]
        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        I = tf.eye(
            state_dim,
            batch_shape=tf.concat([batch_shape, [N]], axis=0),
            dtype=mu.dtype,
        )
        P_exp = tf.broadcast_to(
            P[..., tf.newaxis, :, :],
            tf.concat([batch_shape, [N, state_dim, state_dim]], axis=0),
        )
        R = self.ssm.cov_eps_y
        guard_on = bool(self.beta_schedule.guard) and self.beta_schedule.mode == "optimal"
        if self.beta_schedule.mode == "optimal":
            if w is None:
                mu_mean = tf.reduce_mean(mu, axis=-2)
            else:
                w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
                mu_mean = self.ssm.state_mean(mu, w)
            r0_sched = tf.zeros(tf.concat([batch_shape, [r_dim]], axis=0), dtype=mu.dtype)
            H0, _ = self.jacobian_h_x(mu_mean, r0_sched)
            H_r0, _ = self.jacobian_h_r(mu_mean, r0_sched)
            R_eff0 = quadratic_matmul(H_r0, R, H_r0)
            R_inv0 = self._inverse_from_cov(R_eff0)
            RinvJ = tf.einsum("bij,bjk->bik", R_inv0, H0)
            Info = tf.einsum("bji,bjk->bik", H0, RinvJ)
            P0_inv = self._inverse_from_cov(P)
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
            beta = beta[:, tf.newaxis, :]
            beta_dot = beta_dot[:, tf.newaxis, :]
            beta = tf.broadcast_to(beta, tf.concat([batch_shape, [N, self.num_lambda]], axis=0))
            beta_dot = tf.broadcast_to(beta_dot, tf.concat([batch_shape, [N, self.num_lambda]], axis=0))
            if guard_on:
                beta_base, beta_dot_base, _ = build_beta_schedule(
                    "linear",
                    num_lambda=self.num_lambda,
                    dtype=mu.dtype,
                )
                beta_base = tf.broadcast_to(beta_base[:, tf.newaxis, :], tf.shape(beta))
                beta_dot_base = tf.broadcast_to(beta_dot_base[:, tf.newaxis, :], tf.shape(beta_dot))
        else:
            beta, beta_dot, dl = build_beta_schedule(
                "linear",
                num_lambda=self.num_lambda,
                dtype=mu.dtype,
            )
            beta = beta[:, tf.newaxis, :]
            beta_dot = beta_dot[:, tf.newaxis, :]
            beta = tf.broadcast_to(beta, tf.concat([batch_shape, [N, self.num_lambda]], axis=0))
            beta_dot = tf.broadcast_to(beta_dot, tf.concat([batch_shape, [N, self.num_lambda]], axis=0))
        tf.debugging.assert_equal(
            tf.shape(beta)[-1],
            self.num_lambda,
            message="beta schedule length must match num_lambda",
        )
        R_exp = R
        if tf.rank(R_exp) == 2:
            R_exp = R_exp[tf.newaxis, tf.newaxis, ...]
        R_exp = tf.broadcast_to(
            R_exp,
            tf.concat([batch_shape, [N, obs_dim, obs_dim]], axis=0),
        )
        if y.shape.rank is not None and y.shape.rank == 2:
            y_broadcast = tf.broadcast_to(
                y[..., tf.newaxis, :],
                tf.concat([batch_shape, [N, obs_dim]], axis=0),
            )
        else:
            y_broadcast = y

        jitter_val = 0.0 if self.jitter is None else float(self.jitter)
        eps = jitter_val if jitter_val > 0.0 else 1e-12
        eps_t = tf.cast(eps, mu.dtype)
        logdet = tf.zeros(tf.shape(mu)[:-1], dtype=tf.float32)
        r0 = tf.zeros(tf.concat([batch_shape, [N, r_dim]], axis=0), dtype=mu.dtype)

        for j in range(self.num_lambda):
            beta_j = beta[:, :, j]
            beta_dot_j = beta_dot[:, :, j]

            H, h = self.jacobian_h_x(mu, r0)
            H_r, _ = self.jacobian_h_r(mu, r0)
            v = self.ssm.innovation(y_broadcast, h)
            Hx = tf.einsum("bnij,bnj->bni", H, mu)
            y_tilde = v + Hx

            R_eff = quadratic_matmul(H_r, R_exp, H_r)
            if guard_on:
                beta_base_j = beta_base[:, :, j]
                beta_dot_base_j = beta_dot_base[:, :, j]
                step_scale_opt = dl * beta_dot_j
                step_scale_base = dl * beta_dot_base_j
                A_opt, b_opt = self._ledh_flow_solution(
                    beta_j,
                    H,
                    P_exp,
                    R_eff,
                    y_tilde,
                    mu,
                    self.jitter,
                )
                A_base, b_base = self._ledh_flow_solution(
                    beta_base_j,
                    H,
                    P_exp,
                    R_eff,
                    y_tilde,
                    mu,
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
                A, b = self._ledh_flow_solution(beta_j, H, P_exp, R_eff, y_tilde, mu, self.jitter)

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

            # Particle transport with Euler discretization.
            Ax = tf.einsum("bnij,bnj->bni", A, mu)
            mu_next = mu + step_scale[..., tf.newaxis] * (Ax + b)
            mu = tf.where(bad[..., tf.newaxis], mu, mu_next)

        log10_base = tf.math.log(10.0)
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
            "condCov_log10": cond_cov_log10,
            "logdet_cov": logdet_cov,
        }
        return mu, logdet, diagnostics
