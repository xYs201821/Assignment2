"""Localized Exact Daum-Huang (LEDH) particle flow implementation."""

import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.flows.beta_schedule import BetaScheduleConfig, build_beta_schedule
from src.flows.diagnostics import LEDHDiagnostics
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
        )

    def _inverse_from_cov(self, cov: tf.Tensor) -> tf.Tensor:
        cov = tf.convert_to_tensor(cov, dtype=tf.float32)
        eye = tf.eye(tf.shape(cov)[-1], batch_shape=tf.shape(cov)[:-2], dtype=cov.dtype)
        jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
        return cholesky_solve(cov, eye, jitter=jitter_val)

    def _flow_diag_keys(self):
        return (
            "dx_p95_max",
            "condS_log10_max",
            "condH_log10_max",
            "condJ_log10_max",
            "flow_norm_mean_max",
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
        lam = tf.convert_to_tensor(lam, dtype=HPH.dtype)
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
        b = tf.einsum("...nij,...njk,...nk->...ni", I + lam_b * A, K, y_tilde)
        Am0 = tf.einsum("...nij,...njk,...nk->...ni", I + 2.0 * lam_b * A, A, m0)
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
        if self.beta_schedule.mode == "optimal":
            r0_sched = tf.zeros(tf.concat([batch_shape, [N, r_dim]], axis=0), dtype=mu.dtype)
            H0, _ = self.jacobian_h_x(mu, r0_sched)
            H_r0, _ = self.jacobian_h_r(mu, r0_sched)
            R_eff0 = quadratic_matmul(H_r0, R, H_r0)
            R_inv0 = self._inverse_from_cov(R_eff0)
            RinvJ = tf.einsum("...ij,...jk->...ik", R_inv0, H0)
            Info = tf.einsum("...ji,...jk->...ik", H0, RinvJ)
            P0_inv = self._inverse_from_cov(P)
            P0_inv_exp = tf.broadcast_to(P0_inv[:, tf.newaxis, :, :], tf.shape(Info))
            batch = tf.shape(Info)[0]
            num_particles = tf.shape(Info)[1]
            state_dim = tf.shape(Info)[-1]
            bn = batch * num_particles
            Info_flat = tf.reshape(Info, [bn, state_dim, state_dim])
            P0_inv_flat = tf.reshape(P0_inv_exp, [bn, state_dim, state_dim])
            beta_flat, beta_dot_flat, dl = build_beta_schedule(
                "optimal",
                num_lambda=self.num_lambda,
                dtype=mu.dtype,
                P0_inv=P0_inv_flat,
                Info=Info_flat,
                jitter=self.jitter,
                mu=self.beta_schedule.mu,
            )
            beta = tf.reshape(beta_flat, [batch, num_particles, self.num_lambda])
            beta_dot = tf.reshape(beta_dot_flat, [batch, num_particles, self.num_lambda])
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
        dl_t = tf.cast(dl, mu.dtype)
        R_exp = R
        if tf.rank(R_exp) == 2:
            R_exp = R_exp[tf.newaxis, tf.newaxis, ...]
        R_exp = tf.broadcast_to(
            R_exp,
            tf.concat([batch_shape, [N, obs_dim, obs_dim]], axis=0),
        )
        m0_exp = tf.broadcast_to(
            m0[..., tf.newaxis, :],
            tf.concat([batch_shape, [N, state_dim]], axis=0),
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
        r0 = tf.zeros(tf.concat([batch_shape, [N, r_dim]], axis=0), dtype=tf.float32)
        log10_base = tf.math.log(tf.cast(10.0, mu.dtype))
        diag = LEDHDiagnostics(self, batch_shape, eps_t, log10_base)
        for i in range(self.num_lambda):
            beta_j = tf.gather(beta, i, axis=2)
            beta_dot_j = tf.gather(beta_dot, i, axis=2)
            beta_j = tf.cast(beta_j, mu.dtype)
            beta_dot_j = tf.cast(beta_dot_j, mu.dtype)
            step_scale = dl_t * beta_dot_j

            H, h = self.jacobian_h_x(mu, r0)
            H_r, _ = self.jacobian_h_r(mu, r0)
            v = self.ssm.innovation(y_broadcast, h)
            Hx = tf.einsum("...nij,...nj->...ni", H, mu)
            y_tilde = v + Hx

            R_eff = quadratic_matmul(H_r, R_exp, H_r)
            A, b = self._ledh_flow_solution(beta_j, H, P_exp, R_eff, y_tilde, m0_exp, self.jitter)
            H_mean = tf.reduce_mean(H, axis=-3)
            H_r_mean = tf.reduce_mean(H_r, axis=-3)
            R_use = R_exp[..., 0, :, :]
            R_eff_mean = quadratic_matmul(H_r_mean, R_use, H_r_mean)
            beta_mean = tf.reduce_mean(beta_j, axis=-1)
            S_mean = beta_mean[..., tf.newaxis, tf.newaxis] * quadratic_matmul(H_mean, P, H_mean) + R_eff_mean
            if jitter_val > 0.0:
                eye = tf.eye(tf.shape(S_mean)[-1], batch_shape=tf.shape(S_mean)[:-2], dtype=S_mean.dtype)
                S_mean = S_mean + tf.cast(jitter_val, S_mean.dtype) * eye

            J = I + step_scale[..., tf.newaxis, tf.newaxis] * A
            if self.jitter and self.jitter > 0.0:
                J = J + tf.cast(self.jitter, J.dtype) * I
            sign, lad = tf.linalg.slogdet(J)
            if self.debug:
                tf.debugging.assert_greater(
                    tf.abs(sign),
                    0.0,
                    message="LEDH flow Jacobian is singular; reduce step size or check model.",
                )
            bad = tf.equal(sign, 0.0)
            lad = tf.where(bad, tf.zeros_like(lad), lad)
            logdet += lad
            flow = tf.einsum("...nij,...nj->...ni", A, mu) + b
            flow = tf.where(bad[..., tf.newaxis], tf.zeros_like(flow), flow)
            # Particle transport with Euler discretization.
            dx = step_scale[..., tf.newaxis] * flow

            Ax = tf.einsum("...nij,...nj->...ni", A, mu)
            mu_next = mu + step_scale[..., tf.newaxis] * (Ax + b)
            mu = tf.where(bad[..., tf.newaxis], mu, mu_next)
            diag.update(H, J, S_mean, flow, dx)

        diagnostics = diag.finalize(mu, eps)
        return mu, logdet, diagnostics
