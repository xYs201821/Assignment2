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
        jitter=1e-5,
    ):
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

    @staticmethod
    def _ledh_flow_solution(lam, H, P, R, y_tilde, m0, jitter):
        HPH = quadratic_matmul(H, P, H)
        S = lam * HPH + R
        jitter_val = float(jitter) if jitter is not None else 0.0
        if jitter_val > 0.0:
            eye = tf.eye(tf.shape(S)[-1], batch_shape=tf.shape(S)[:-2], dtype=S.dtype)
            S = S + tf.cast(jitter_val, S.dtype) * eye
        PHt = tf.linalg.matmul(P, H, transpose_b=True)
        K_T = cholesky_solve(S, tf.linalg.matrix_transpose(PHt), jitter=jitter_val)
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
        delta_t = tf.cast(delta, mu.dtype)
        jitter_val = 0.0 if self.jitter is None else float(self.jitter)
        eps = jitter_val if jitter_val > 0.0 else 1e-12
        eps_t = tf.cast(eps, mu.dtype)
        lam = 0.0
        logdet = tf.zeros(tf.shape(mu)[:-1], dtype=tf.float32)
        r0_flat = tf.zeros(tf.stack([batch_size * N, r_dim]), dtype=tf.float32)
        dx_p95_max = tf.zeros(batch_shape, dtype=tf.float32)
        condS_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        condH_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        condJ_log10_max = tf.zeros(batch_shape, dtype=tf.float32)
        flow_norm_mean_max = tf.zeros(batch_shape, dtype=tf.float32)
        log10_base = tf.math.log(tf.cast(10.0, mu.dtype))
        for _ in range(self.num_lambda):
            lam = lam + delta

            mu_flat = tf.reshape(mu, mu_flat_shape)

            H_flat, h_flat = self.jacobian_h_x(mu_flat, r0_flat)
            H_r_flat, _ = self.jacobian_h_r(mu_flat, r0_flat)

            H = tf.reshape(H_flat, H_shape)
            h = tf.reshape(h_flat, h_shape)
            H_r = tf.reshape(H_r_flat, H_r_shape)
            condH_particles = self._cond_from_rect(H, eps_t)
            condH_mean = tf.reduce_mean(condH_particles, axis=-1)
            condH_log10 = tf.math.log(condH_mean + eps_t) / log10_base
            condH_log10_max = tf.maximum(condH_log10_max, condH_log10)

            v = self.ssm.innovation(y_broadcast, h)
            Hx = tf.einsum("...nij,...nj->...ni", H, mu)
            y_tilde = v + Hx

            R_eff = quadratic_matmul(H_r, R_exp, H_r)
            A, b = self._ledh_flow_solution(lam, H, P_exp, R_eff, y_tilde, m0_exp, self.jitter)
            H_mean = tf.reduce_mean(H, axis=-3)
            H_r_mean = tf.reduce_mean(H_r, axis=-3)
            R_use = R_exp[..., 0, :, :]
            R_eff_mean = quadratic_matmul(H_r_mean, R_use, H_r_mean)
            S_mean = lam * quadratic_matmul(H_mean, P, H_mean) + R_eff_mean
            if jitter_val > 0.0:
                eye = tf.eye(tf.shape(S_mean)[-1], batch_shape=tf.shape(S_mean)[:-2], dtype=S_mean.dtype)
                S_mean = S_mean + tf.cast(jitter_val, S_mean.dtype) * eye
            condS = self._cond_from_matrix(S_mean, eps_t)
            condS_log10 = tf.math.log(condS + eps_t) / log10_base
            condS_log10_max = tf.maximum(condS_log10_max, condS_log10)

            J = I + delta * A
            if self.jitter and self.jitter > 0.0:
                J = J + tf.cast(self.jitter, J.dtype) * I
            condJ_particles = self._cond_from_rect(J, eps_t)
            condJ_mean = tf.reduce_mean(condJ_particles, axis=-1)
            condJ_log10 = tf.math.log(condJ_mean + eps_t) / log10_base
            condJ_log10_max = tf.maximum(condJ_log10_max, condJ_log10)
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
            flow_norm_mean = tf.reduce_mean(tf.norm(flow, axis=-1), axis=-1)
            flow_norm_mean_max = tf.maximum(flow_norm_mean_max, flow_norm_mean)
            dx = delta_t * flow
            dx_norm = tf.norm(dx, axis=-1)
            dx_p95 = self._percentile(dx_norm, 95.0)
            dx_p95_max = tf.maximum(dx_p95_max, dx_p95)

            Ax = tf.einsum("...nij,...nj->...ni", A, mu)
            mu_next = mu + delta * (Ax + b)
            mu = tf.where(bad[..., tf.newaxis], mu, mu_next)

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
