import tensorflow as tf

from src.filters.base import GaussianFilter
from src.utility import block_diag, cholesky_solve, quadratic_matmul, tf_cond, weighted_mean


class UnscentedKalmanFilter(GaussianFilter):
    def __init__(self, ssm, alpha=1e-3, beta=2.0, kappa=0.0, debug=False, print=False):
        super().__init__(ssm, debug=debug, print=print)
        self.state_dim = self.ssm.state_dim
        self.obs_dim = self.ssm.obs_dim

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.Wm, self.Wc, self.lamb = self._build_ukf_weights(self.state_dim)
        self._maybe_print()

    def _build_ukf_weights(self, n: tf.Tensor):
        alpha = self.alpha
        beta = self.beta
        kappa = self.kappa

        lamb = alpha ** 2 * (n + kappa) - n
        self.lamb = lamb

        num_sigma = 2 * n + 1
        w = 1.0 / (2.0 * (n + lamb))
        w0 = lamb / (n + lamb)
        Wm = tf.concat([tf.reshape(w0, [1]), tf.fill([num_sigma - 1], w)], axis=0)
        Wc = tf.concat([tf.reshape(w0 + (1.0 - alpha**2 + beta), [1]), tf.fill([num_sigma - 1], w)], axis=0)
        return Wm, Wc, lamb

    def generate_sigma_points(self, m, P):
        n = tf.shape(P)[-1]
        n_f = tf.cast(n, tf.float32)
        lamb = self.alpha ** 2 * (n_f + self.kappa) - n_f
        num_sigma = 2 * n + 1
        w = 1.0 / (2.0 * (n_f + lamb))
        w0 = lamb / (n_f + lamb)
        W_m = tf.concat([tf.reshape(w0, [1]), tf.fill([num_sigma - 1], w)], axis=0)
        W_c = tf.concat(
            [tf.reshape(w0 + (1.0 - self.alpha**2 + self.beta), [1]), tf.fill([num_sigma - 1], w)],
            axis=0,
        )
        scale = tf.cast(n_f + lamb, P.dtype)
        P_scaled = P * scale

        P_sqrt = tf.linalg.cholesky(P_scaled)
        P_sqrt = tf.linalg.matrix_transpose(P_sqrt)
        m_exp = m[..., tf.newaxis, :]
        X_plus = m_exp + P_sqrt
        X_minus = m_exp - P_sqrt

        X = tf.concat([m_exp, X_plus, X_minus], axis=1)
        return X, tf.convert_to_tensor(W_m, dtype=tf.float32), tf.convert_to_tensor(W_c, dtype=tf.float32)
    
    def propagate_sigma_points(self, func, X):
        return func(X)

    def unscented_transform(self, func, m, P, mean_fn=None, residual_fn=None):
        if mean_fn is None:
            mean_fn = lambda y, W: weighted_mean(y, W, axis=1)
        if residual_fn is None:
            residual_fn = lambda y, y_mean: y - y_mean[..., tf.newaxis, :]
        X, Wm, Wc = self.generate_sigma_points(m, P)
        Y = self.propagate_sigma_points(func, X)

        y_mean = mean_fn(Y, Wm)

        Y_c = residual_fn(Y, y_mean)
        cov = tf.einsum("...i,...ij,...ik->...jk", Wc, Y_c, Y_c)

        return y_mean, cov, X, Y, Wm, Wc

    @tf.function(reduce_retracing=True)
    def predict(self, m_prev, P_prev):
        q0 = tf.zeros(tf.concat([tf.shape(m_prev)[:-1], [self.q_dim]], axis=0), dtype=tf.float32)
        m_aug = tf.concat([m_prev, q0], axis=-1)
        P_aug = block_diag(P_prev, self.cov_eps_x)

        def f_aug(z):
            x = z[..., :self.state_dim]
            q = z[..., self.state_dim:]
            return self.ssm.f_with_noise(x, q)

        m_pred, P_pred, _, _, _, _ = self.unscented_transform(
            f_aug, m_aug, P_aug, mean_fn=self.ssm.state_mean, residual_fn=self.ssm.state_residual
        )
        return m_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update(self, m_pred, P_pred, y, joseph=True):
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)
        m_aug = tf.concat([m_pred, r0], axis=-1)
        P_aug = block_diag(P_pred, self.cov_eps_y)

        def h_aug(z):
            x = z[..., :self.state_dim]
            r = z[..., self.state_dim:]
            return self.ssm.h_with_noise(x, r)

        y_pred, S, X, Y, Wm, Wc = self.unscented_transform(
            h_aug, m_aug, P_aug, mean_fn=self.ssm.measurement_mean, residual_fn=self.ssm.measurement_residual
        )

        v = self.ssm.innovation(y, y_pred)

        X_x = X[:, :, :self.state_dim]
        X_c = self.ssm.state_residual(X_x, m_pred)
        Y_c = self.ssm.measurement_residual(Y, y_pred)

        C_xy = tf.einsum("...i,...ij,...ik->...jk", Wc, X_c, Y_c)

        RHS_T = tf.linalg.matrix_transpose(C_xy)
        K_T = cholesky_solve(S, RHS_T)
        K = tf.linalg.matrix_transpose(K_T)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)

        KSK_T = quadratic_matmul(K, S, K)
        P_filt = P_pred - KSK_T
        if joseph:
            P_filt = 0.5 * (P_filt + tf.linalg.matrix_transpose(P_filt))

        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)

        return m_filt, P_filt, cond_P, cond_S
