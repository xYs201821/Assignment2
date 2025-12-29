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
    ):
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            reweight=reweight,
            init_from_particles="sample",
            debug=debug,
        )

    @staticmethod
    def _edh_flow_solution(lam, H, P, R, y_tilde, m0):
        HPH = quadratic_matmul(H, P, H)
        S = lam * HPH + R
        RHS = tf.linalg.matmul(H, P, transpose_b=True)
        K_T = cholesky_solve(S, RHS)
        K = tf.linalg.matrix_transpose(K_T)

        A = -0.5 * tf.linalg.matmul(K, H)
        I = tf.eye(tf.shape(A)[-1], batch_shape=tf.shape(A)[:-2], dtype=A.dtype)
        b = tf.einsum("...ij,...jk,...k->...i", I + lam * A, K, y_tilde)
        Am0 = tf.einsum("...ij,...jk,...k->...i", I + 2.0 * lam * A, A, m0)
        b = b + Am0
        return A, b

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
        lam = 0.0
        logdet = tf.zeros(tf.shape(mu)[:-2], dtype=tf.float32)
        r0_flat = tf.zeros(tf.stack([batch_size , r_dim]), dtype=tf.float32)
        flow_norm_max = tf.zeros(batch_shape, dtype=tf.float32)
        for i in range(self.num_lambda):
            lam = lam + delta

            m_bar_flat = tf.reshape(m_bar, m_bar_flat_shape)
            H_flat, h_m_flat = self.jacobian_h_x(m_bar_flat, r0_flat)
            H_r_flat, _ = self.jacobian_h_r(m_bar_flat, r0_flat)
            H = tf.reshape(H_flat, H_shape)
            h_m = tf.reshape(h_m_flat, h_m_shape)
            H_r = tf.reshape(H_r_flat, H_r_shape)
            Hm = tf.einsum("...ij,...j->...i", H, m_bar)
            v = self.ssm.innovation(y, h_m)
            y_tilde = v + Hm
            R_eff = quadratic_matmul(H_r, R, H_r)
            A, b = self._edh_flow_solution(lam, H, P, R_eff, y_tilde, m0)
            Am = tf.einsum("...ij,...j->...i", A, m_bar)
            m_bar = m_bar + delta * (Am + b)
            J = I + delta * A
            sign, lad = tf.linalg.slogdet(J)
            tf.debugging.assert_greater(
                sign,
                0.0,
                message="EDH flow Jacobian has non-positive determinant; reduce step size or check model.",
            )
            logdet += lad
            flow = tf.einsum("...ij,...nj->...ni", A, mu) + b[..., tf.newaxis, :]
            flow_norm = tf.reduce_mean(tf.norm(flow, axis=-1), axis=-1)
            flow_norm_max = tf.maximum(flow_norm_max, flow_norm)
            Ax = tf.einsum("...ij,...nj->...ni", A, mu)
            mu = mu + delta * (Ax + b[..., tf.newaxis, :])

        return mu, logdet, flow_norm_max
