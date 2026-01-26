"""Exact Daum-Huang (EDH) particle flow implementation."""

import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.flows.diagnostics import EDHDiagnostics
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
        S = lam * HPH + R
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
        b = tf.einsum("...ij,...jk,...k->...i", I + lam * A, K, y_tilde)
        Am0 = tf.einsum("...ij,...jk,...k->...i", I + 2.0 * lam * A, A, m0)
        b = b + Am0
        return A, b, S
    
    def _flow_transport(self, mu_tilde, y, m0, P, **kwargs):
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

        delta = 1.0 / float(self.num_lambda)
        delta_t = tf.cast(delta, mu.dtype)
        jitter_val = 0.0 if self.jitter is None else float(self.jitter)
        eps = jitter_val if jitter_val > 0.0 else 1e-12
        eps_t = tf.cast(eps, mu.dtype)
        lam = 0.0
        logdet = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        r0 = tf.zeros(tf.concat([batch_shape, [r_dim]], axis=0), dtype=tf.float32)
        log10_base = tf.math.log(tf.cast(10.0, mu.dtype))
        diag = EDHDiagnostics(self, batch_shape, eps_t, log10_base)
        for i in range(self.num_lambda):
            lam = lam + delta
            H, h_m = self.jacobian_h_x(m_bar, r0)
            H_r, _ = self.jacobian_h_r(m_bar, r0)
            Hm = tf.einsum("...ij,...j->...i", H, m_bar)
            v = self.ssm.innovation(y, h_m)
            y_tilde = v + Hm
            R_eff = quadratic_matmul(H_r, R, H_r)
            A, b, S = self._edh_flow_solution(lam, H, P, R_eff, y_tilde, m0, self.jitter)
            Am = tf.einsum("...ij,...j->...i", A, m_bar)
            # Euler update for mean flow ODE: m_bar += delta * (A m_bar + b).
            m_bar = m_bar + delta * (Am + b)
            J = I + delta * A
            if self.jitter and self.jitter > 0.0:
                J = J + tf.cast(self.jitter, J.dtype) * I
            sign, lad = tf.linalg.slogdet(J)
            if self.debug:
                tf.debugging.assert_greater(
                    tf.abs(sign),
                    0.0,
                    message="EDH flow Jacobian is singular; reduce step size or check model.",
                )
            bad = tf.equal(sign, 0.0)
            lad = tf.where(bad, tf.zeros_like(lad), lad)
            logdet += lad
            flow = tf.einsum("...ij,...nj->...ni", A, mu) + b[..., tf.newaxis, :]
            flow = tf.where(bad[..., tf.newaxis, tf.newaxis], tf.zeros_like(flow), flow)
            # Particle transport with Euler discretization.
            dx = delta_t * flow
            Ax = tf.einsum("...ij,...nj->...ni", A, mu)
            mu_next = mu + delta * (Ax + b[..., tf.newaxis, :])
            mu = tf.where(bad[..., tf.newaxis, tf.newaxis], mu, mu_next)
            diag.update(H, J, S, flow, dx)

        diagnostics = diag.finalize(mu, eps)
        return mu, logdet, diagnostics
