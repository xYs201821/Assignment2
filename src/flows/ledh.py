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
        prior_stats="ekf",
        reweight="auto",
        debug=False,
        jitter=1e-12,
    ):
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            prior_stats=prior_stats,
            reweight=reweight,
            init_from_particles="per_particle",
            debug=debug,
        )

    def _init_ekf_tracker(self, batch_shape, state_dim, tracker):
        if tracker is None:
            P0 = tf.convert_to_tensor(self.ssm.P0, dtype=tf.float32)
            P_prev = tf.broadcast_to(
                P0,
                tf.concat([batch_shape, [self.num_particles, state_dim, state_dim]], axis=0),
            )
            m0 = tf.convert_to_tensor(self.ssm.m0, dtype=tf.float32)
            m_prev = tf.broadcast_to(
                m0,
                tf.concat([batch_shape, [self.num_particles, state_dim]], axis=0),
            )
        else:
            m_prev = tracker["m"]
            P_prev = tracker["P"]
        return m_prev, P_prev

    def _prior_from_sample(self, mu_tilde, w, batch_shape, state_dim):
        m = mu_tilde
        P = self.ssm.state_cov(mu_tilde, w)
        P = tf.broadcast_to(
            P[..., tf.newaxis, :, :],
            tf.concat([batch_shape, [self.num_particles, state_dim, state_dim]], axis=0),
        )
        return m, P

    def _prior_from_ekf(self, m_prev, P_prev, batch_shape, batch_size, state_dim, q_dim):
        del batch_size, q_dim  # unused in per-particle EKF path
        m_prev_flat = tf.reshape(m_prev, tf.stack([tf.reduce_prod(batch_shape) * self.num_particles, state_dim]))
        P_prev_flat = tf.reshape(
            P_prev,
            tf.stack([tf.reduce_prod(batch_shape) * self.num_particles, state_dim, state_dim]),
        )
        m_pred_flat, P_pred_flat = self._ekf.predict(m_prev_flat, P_prev_flat)
        m_pred = tf.reshape(
            m_pred_flat,
            tf.concat([batch_shape, [self.num_particles, state_dim]], axis=0),
        )
        P_pred = tf.reshape(
            P_pred_flat,
            tf.concat([batch_shape, [self.num_particles, state_dim, state_dim]], axis=0),
        )
        return m_pred, P_pred

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

        P_exp = P
        R = self.ssm.cov_eps_y
        R_exp = R
        if tf.rank(R_exp) == 2:
            R_exp = R_exp[tf.newaxis, tf.newaxis, ...]
        R_exp = tf.broadcast_to(
            R_exp,
            tf.concat([batch_shape, [N, obs_dim, obs_dim]], axis=0),
        )
        m0_exp = m0
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
        for _ in range(self.num_lambda):
            lam = lam + delta

            mu_flat = tf.reshape(mu, mu_flat_shape)

            H_flat, h_flat = self._jacobian(
                lambda x: self.ssm.h_with_noise(x, r0_flat),
                mu_flat,
            )
            H_r_flat, _ = self._jacobian(
                lambda r: self.ssm.h_with_noise(mu_flat, r),
                r0_flat,
            )

            H = tf.reshape(H_flat, H_shape)
            h = tf.reshape(h_flat, h_shape)
            H_r = tf.reshape(H_r_flat, H_r_shape)

            v = self.ssm.innovation(y_broadcast, h)
            Hx = tf.einsum("...nij,...nj->...ni", H, mu)
            y_tilde = v + Hx

            R_eff = quadratic_matmul(H_r, R_exp, H_r)
            HPH = quadratic_matmul(H, P_exp, H)
            S = lam * HPH + R_eff
            PHt = tf.linalg.matmul(P_exp, H, transpose_b=True)
            K_T = cholesky_solve(S, tf.linalg.matrix_transpose(PHt))
            K = tf.linalg.matrix_transpose(K_T)

            A = -0.5 * tf.linalg.matmul(K, H)
            b = tf.einsum("...nij,...njk,...nk->...ni", I + lam * A, K, y_tilde)
            Am0 = tf.einsum("...nij,...njk,...nk->...ni", I + 2.0 * lam * A, A, m0_exp)
            b = b + Am0

            sign, lad = tf.linalg.slogdet(I + delta * A)
            tf.debugging.assert_greater(
                sign,
                0.0,
                message="LEDH flow Jacobian has non-positive determinant; reduce step size or check model.",
            )
            logdet += lad

            Ax = tf.einsum("...nij,...nj->...ni", A, mu)
            mu = mu + delta * (Ax + b)

        return mu, logdet

    def _posterior_from_ekf(self, x_t, w_final, P_pred, y_t, batch_shape, batch_size, state_dim):
        del w_final  # not used in EKF path
        obs_dim = tf.shape(y_t)[-1]
        y_broadcast = tf.broadcast_to(
            y_t[..., tf.newaxis, :],
            tf.concat([batch_shape, [self.num_particles, obs_dim]], axis=0),
        )
        x_t_flat = tf.reshape(x_t, tf.stack([batch_size * self.num_particles, state_dim]))
        P_pred_flat = tf.reshape(
            P_pred,
            tf.stack([batch_size * self.num_particles, state_dim, state_dim]),
        )
        eye = tf.eye(state_dim, batch_shape=tf.shape(P_pred_flat)[:-2], dtype=P_pred_flat.dtype)
        P_pred_flat = tf.where(tf.math.is_finite(P_pred_flat), P_pred_flat, tf.zeros_like(P_pred_flat))
        P_pred_flat = P_pred_flat + eye * self.jitter
        y_flat = tf.reshape(y_broadcast, tf.stack([batch_size * self.num_particles, -1]))
        m_filt_flat, P_filt_flat, _, _ = self._ekf.update(
            x_t_flat,
            P_pred_flat,
            y_flat,
            joseph=True,
        )
        m_filt = tf.reshape(
            m_filt_flat,
            tf.concat([batch_shape, [self.num_particles, state_dim]], axis=0),
        )
        P_filt = tf.reshape(
            P_filt_flat,
            tf.concat([batch_shape, [self.num_particles, state_dim, state_dim]], axis=0),
        )
        return {"m": m_filt, "P": P_filt}
