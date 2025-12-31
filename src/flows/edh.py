import tensorflow as tf
from src.flows.flow_base import FlowBase
from src.utility import cholesky_solve, quadratic_matmul


class EDHFlow(FlowBase):
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
        HPH = quadratic_matmul(H, P, H)
        S = lam * HPH + R
        jitter_val = float(jitter) if jitter is not None else 0.0
        if jitter_val > 0.0:
            eye = tf.eye(tf.shape(S)[-1], batch_shape=tf.shape(S)[:-2], dtype=S.dtype)
            S = S + tf.cast(jitter_val, S.dtype) * eye
        RHS = tf.linalg.matmul(H, P, transpose_b=True)
        K_T = cholesky_solve(S, RHS, jitter=jitter_val)
        K = tf.linalg.matrix_transpose(K_T)

        A = -0.5 * tf.linalg.matmul(K, H)
        I = tf.eye(tf.shape(A)[-1], batch_shape=tf.shape(A)[:-2], dtype=A.dtype)
        b = tf.einsum("...ij,...jk,...k->...i", I + lam * A, K, y_tilde)
        Am0 = tf.einsum("...ij,...jk,...k->...i", I + 2.0 * lam * A, A, m0)
        b = b + Am0
        return A, b, S

    def _flow_transport(self, mu_tilde, y, m0, P):
        mu = mu_tilde
        m_bar = tf.identity(m0)
        R = self.ssm.cov_eps_y
        batch_shape = tf.shape(m_bar)[:-1]
        batch_size = tf.reduce_prod(batch_shape)
        state_dim = tf.shape(m_bar)[-1]
        obs_dim = tf.shape(y)[-1]
        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        I = tf.eye(state_dim, batch_shape=batch_shape, dtype=mu.dtype)
        m_bar_flat_shape = tf.stack([batch_size, state_dim])
        r0_flat_shape = tf.stack([batch_size, r_dim])
        H_shape = tf.concat([batch_shape, tf.stack([obs_dim, state_dim])], axis=0)
        H_r_shape = tf.concat([batch_shape, tf.stack([obs_dim, r_dim])], axis=0)
        h_m_shape = tf.concat([batch_shape, tf.reshape(obs_dim, [1])], axis=0)

        delta = 1.0 / float(self.num_lambda)
        delta_t = tf.cast(delta, mu.dtype)
        jitter_val = 0.0 if self.jitter is None else float(self.jitter)
        eps = jitter_val if jitter_val > 0.0 else 1e-12
        eps_t = tf.cast(eps, mu.dtype)
        lam = 0.0
        logdet = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        r0_flat = tf.zeros(tf.stack([batch_size , r_dim]), dtype=tf.float32)
        dx_p95_max = tf.zeros(batch_shape, dtype=tf.float32)
        condS_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        condH_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        condJ_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        flow_norm_mean_max = tf.zeros(batch_shape, dtype=tf.float32)
        log10_base = tf.math.log(tf.cast(10.0, mu.dtype))
        for i in range(self.num_lambda):
            lam = lam + delta

            m_bar_flat = tf.reshape(m_bar, m_bar_flat_shape)
            H_flat, h_m_flat = self.jacobian_h_x(m_bar_flat, r0_flat)
            H_r_flat, _ = self.jacobian_h_r(m_bar_flat, r0_flat)
            H = tf.reshape(H_flat, H_shape)
            h_m = tf.reshape(h_m_flat, h_m_shape)
            H_r = tf.reshape(H_r_flat, H_r_shape)
            condH = self._cond_from_rect(H, eps_t)
            condH_log10 = tf.math.log(condH + eps_t) / log10_base
            condH_log10_max = tf.maximum(condH_log10_max, condH_log10)
            Hm = tf.einsum("...ij,...j->...i", H, m_bar)
            v = self.ssm.innovation(y, h_m)
            y_tilde = v + Hm
            R_eff = quadratic_matmul(H_r, R, H_r)
            A, b, S = self._edh_flow_solution(lam, H, P, R_eff, y_tilde, m0, self.jitter)
            Am = tf.einsum("...ij,...j->...i", A, m_bar)
            m_bar = m_bar + delta * (Am + b)
            J = I + delta * A
            if self.jitter and self.jitter > 0.0:
                J = J + tf.cast(self.jitter, J.dtype) * I
            condJ = self._cond_from_rect(J, eps_t)
            condJ_log10 = tf.math.log(condJ + eps_t) / log10_base
            condJ_log10_max = tf.maximum(condJ_log10_max, condJ_log10)
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
            flow_norm_mean = tf.reduce_mean(tf.norm(flow, axis=-1), axis=-1)
            flow_norm_mean_max = tf.maximum(flow_norm_mean_max, flow_norm_mean)
            dx = delta_t * flow
            dx_norm = tf.norm(dx, axis=-1)
            dx_p95 = self._percentile(dx_norm, 95.0)
            dx_p95_max = tf.maximum(dx_p95_max, dx_p95)
            condS = self._cond_from_matrix(S, eps_t)
            condS_log10 = tf.math.log(condS + eps_t) / log10_base
            condS_log10_max = tf.maximum(condS_log10_max, condS_log10)
            Ax = tf.einsum("...ij,...nj->...ni", A, mu)
            mu_next = mu + delta * (Ax + b[..., tf.newaxis, :])
            mu = tf.where(bad[..., tf.newaxis, tf.newaxis], mu, mu_next)

        cov = self._cov_from_particles(mu, eps)
        cond_cov = self._cond_from_matrix(cov, eps_t)
        cond_cov_log10 = tf.math.log(cond_cov + eps_t) / log10_base
        logdet_cov = self._logdet_cov_from_particles(mu, eps)
        diagnostics = {
            "dx_p95_max": dx_p95_max,
            "condS_log10_max": condS_log10_max,
            "condH_log10_max": condH_log10_max,
            "condJ_log10_max": condJ_log10_max,
            "flow_norm_mean_max": flow_norm_mean_max,
            "condCov_log10": cond_cov_log10,
            "logdet_cov": logdet_cov,
        }
        return mu, logdet, diagnostics
