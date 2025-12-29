import tensorflow as tf
from src.flows.flow_base import FlowBase
from src.utility import cholesky_solve, quadratic_matmul


class LEDHFlow(FlowBase):
    def __init__(
        self,
        ssm,
        num_lambda=20,
        num_particles=100,
        ess_threshold=0.5,
        reweight="auto",
        debug=False,
        jitter=1e-6,
    ):
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            reweight=reweight,
            init_from_particles="per_particle",
            debug=debug,
        )

    @staticmethod
    def _ledh_flow_solution(lam, H, P, R, y_tilde, m0):
        HPH = quadratic_matmul(H, P, H)
        S = lam * HPH + R
        PHt = tf.linalg.matmul(P, H, transpose_b=True)
        K_T = cholesky_solve(S, tf.linalg.matrix_transpose(PHt))
        K = tf.linalg.matrix_transpose(K_T)

        A = -0.5 * tf.linalg.matmul(K, H)
        I = tf.eye(tf.shape(A)[-1], batch_shape=tf.shape(A)[:-2], dtype=A.dtype)
        b = tf.einsum("...nij,...njk,...nk->...ni", I + lam * A, K, y_tilde)
        Am0 = tf.einsum("...nij,...njk,...nk->...ni", I + 2.0 * lam * A, A, m0)
        b = b + Am0
        return A, b

    def _flow_transport(self, mu_tilde, y, m0, P):
        mu = mu_tilde
        batch_shape = tf.shape(mu)[:-2]
        batch_size = tf.reduce_prod(batch_shape)
        N = tf.shape(mu)[-2]
        state_dim = tf.shape(mu)[-1]
        obs_dim = tf.shape(y)[-1]
        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        I = tf.eye(
            state_dim,
            batch_shape=tf.concat([batch_shape, [N]], axis=0),
            dtype=mu.dtype,
        )
        mu_flat_shape = tf.stack([batch_size * N, state_dim])
        H_shape = tf.concat([batch_shape, tf.stack([N, obs_dim, state_dim])], axis=0)
        h_shape = tf.concat([batch_shape, tf.stack([N, obs_dim])], axis=0)
        H_r_shape = tf.concat([batch_shape, tf.stack([N, obs_dim, r_dim])], axis=0)

        P_exp = tf.broadcast_to(
            P[..., tf.newaxis, :, :],
            tf.concat([batch_shape, [N, state_dim, state_dim]], axis=0),
        )
        R = self.ssm.cov_eps_y
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

        delta = 1.0 / float(self.num_lambda)
        lam = 0.0
        logdet = tf.zeros(tf.shape(mu)[:-1], dtype=tf.float32)
        r0_flat = tf.zeros(tf.stack([batch_size * N, r_dim]), dtype=tf.float32)
        flow_norm_max = tf.zeros(batch_shape, dtype=tf.float32)
        for _ in range(self.num_lambda):
            lam = lam + delta

            mu_flat = tf.reshape(mu, mu_flat_shape)

            H_flat, h_flat = self.jacobian_h_x(mu_flat, r0_flat)
            H_r_flat, _ = self.jacobian_h_r(mu_flat, r0_flat)

            H = tf.reshape(H_flat, H_shape)
            h = tf.reshape(h_flat, h_shape)
            H_r = tf.reshape(H_r_flat, H_r_shape)

            v = self.ssm.innovation(y_broadcast, h)
            Hx = tf.einsum("...nij,...nj->...ni", H, mu)
            y_tilde = v + Hx

            R_eff = quadratic_matmul(H_r, R_exp, H_r)
            A, b = self._ledh_flow_solution(lam, H, P_exp, R_eff, y_tilde, m0_exp)

            J = I + delta * A
            sign, lad = tf.linalg.slogdet(J)
            tf.debugging.assert_greater(
                sign,
                0.0,
                message="LEDH flow Jacobian has non-positive determinant; reduce step size or check model.",
            )
            logdet += lad
            flow = tf.einsum("...nij,...nj->...ni", A, mu) + b
            flow_norm = tf.reduce_mean(tf.norm(flow, axis=-1), axis=-1)
            flow_norm_max = tf.maximum(flow_norm_max, flow_norm)

            Ax = tf.einsum("...nij,...nj->...ni", A, mu)
            mu = mu + delta * (Ax + b)

        return mu, logdet, flow_norm_max
