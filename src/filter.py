import tensorflow as tf
import numpy as np
from src.utility import tf_cond, cholesky_solve, weighted_mean, block_diag, quadratic_matmul

class BaseFilter(tf.Module):
    def __init__(self, ssm, **kwargs):
        super().__init__()
        self.ssm = ssm
        self.state_dim = ssm.state_dim
        self.obs_dim = ssm.obs_dim

    def filter(self, y, **kwargs):
        raise NotImplementedError
    
class GaussianFilter(BaseFilter):
    def __init__(self, ssm):
        super().__init__(ssm)
        #eps = tf.constant(1e-16, dtype=tf.float32)

        self.cov_eps_x = tf.convert_to_tensor(ssm.cov_eps_x, dtype=tf.float32)
        self.cov_eps_y = tf.convert_to_tensor(ssm.cov_eps_y, dtype=tf.float32)

        self.q_dim = int(self.cov_eps_x.shape[0]) # dim of transition noise
        self.r_dim = int(self.cov_eps_y.shape[0]) # dim of observation noise

        self.cov_eps_x = self.cov_eps_x #+ eps * tf.eye(self.q_dim, dtype=tf.float32)
        self.cov_eps_y = self.cov_eps_y #+ eps * tf.eye(self.r_dim, dtype=tf.float32)
    def filter(self, y, **kwargs):
        raise NotImplementedError
    
    def predict(self, m_prev, P_prev):
        raise NotImplementedError
    
    def update(self, m_pred, P_pred, y):
        raise NotImplementedError

    def filter(self, y, joseph=True, m0=None, P0=None):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if (len(y.shape) == 2):
            # [T, dy] → [1, T, dy]
            y = y[tf.newaxis, :]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]

        m_pred = tf.convert_to_tensor(m0 if m0 is not None else self.ssm.m0, dtype=tf.float32)
        if (len(m_pred.shape) == 1):
            # [dx] → [batch, dx]
            m_pred = m_pred[tf.newaxis, :]  
            m_pred = tf.tile(m_pred, [batch_size, 1])
        P_pred = tf.convert_to_tensor(P0 if P0 is not None else self.ssm.P0, dtype=tf.float32)
        if (len(P_pred.shape) == 2):
            # [dx, dx] → [batch, dx, dx]
            P_pred = P_pred[tf.newaxis, :, :]
            P_pred = tf.tile(P_pred, [batch_size, 1, 1])
        m_filt_list = []
        P_filt_list = []
        m_pred_list = []
        P_pred_list = []
        cond_P_list = []
        cond_S_list = []

        for t in range(T):
            y_t = y[:, t, :]
            m_pred_list.append(m_pred)
            P_pred_list.append(P_pred)
            m_filt, P_filt, cond_P, cond_S = self.update(m_pred, P_pred, y_t)
            m_filt_list.append(m_filt)
            P_filt_list.append(P_filt)
            m_pred, P_pred = self.predict(m_filt, P_filt)
            cond_P_list.append(cond_P)
            cond_S_list.append(cond_S)
        m_filt = tf.stack(m_filt_list, axis=1)
        P_filt = tf.stack(P_filt_list, axis=1)
        m_pred = tf.stack(m_pred_list, axis=1)
        P_pred = tf.stack(P_pred_list, axis=1)
        cond_P = tf.stack(cond_P_list, axis=1)
        cond_S = tf.stack(cond_S_list, axis=1)
        return {
            "m_filt": m_filt,
            "P_filt": P_filt,
            "m_pred": m_pred,
            "P_pred": P_pred,
            "cond_P": cond_P,
            "cond_S": cond_S,
        }

    @staticmethod
    def _kalman_gain(C, P, R):
        S = quadratic_matmul(C, P, C) + R # [batch, dy, dy]
        RHS = tf.linalg.matmul(C, P, transpose_b=True) 
        K_transpose = cholesky_solve(S, RHS) # S K' = CP'
        K = tf.linalg.matrix_transpose(K_transpose)

        return K, S

    @staticmethod
    def _joseph_update(C, P, K, R):
        I = tf.eye(tf.shape(P)[-1], batch_shape=tf.shape(P)[:-2], dtype=P.dtype)
        I_KC = I - tf.linalg.matmul(K, C)
        return quadratic_matmul(I_KC, P, I_KC) + quadratic_matmul(K, R, K)

class KalmanFilter(GaussianFilter):
    def __init__(self, ssm):
        super().__init__(ssm)
        self.A = self.ssm.A
        self.C = self.ssm.C
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0

    def predict(self, m_prev, P_prev):
        m_pred = tf.einsum('ij,...j->...i', self.A, m_prev) # [batch, dx]
        P_pred = quadratic_matmul(self.A, P_prev, self.A) + self.cov_eps_x # [batch, dx, dx]
        return m_pred, P_pred # [batch, dx], [batch, dx, dx]

    def update(self, m_pred, P_pred, y, joseph=True):
        y_pred = self.ssm.h(m_pred)
        v = self.ssm.innovation(y, y_pred)

        K, S = self._kalman_gain(self.C, P_pred, self.cov_eps_y)
       
        m_filt = m_pred + tf.einsum('...ij,...j->...i', K, v) # [batch, dx]
        if joseph:
            P_filt = self._joseph_update(self.C, P_pred, K, self.cov_eps_y)
        else:
            # standard form: P_filt = P_pred - K S K'
            P_filt = P_pred - quadratic_matmul(K, S, K)
        
        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)
        return m_filt, P_filt, cond_P, cond_S # [batch, dx], [batch, dx, dx], [batch], [batch]

class ExtendedKalmanFilter(GaussianFilter):
    def __init__(self, ssm):
        super().__init__(ssm)
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0

    def _jacobian(self, func, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)

        return tape.batch_jacobian(y, x)

    @tf.function(reduce_retracing=True)
    def predict(self, m_prev, P_prev):
        q0 = tf.zeros(tf.concat([tf.shape(m_prev)[:-1], [self.q_dim]], axis=0), dtype=tf.float32)
        m_pred = self.ssm.f_with_noise(m_prev, q0)
        F_x = self._jacobian(lambda x: self.ssm.f_with_noise(x, q0), m_prev)
        F_q = self._jacobian(lambda q: self.ssm.f_with_noise(m_prev, q), q0)

        P_pred = quadratic_matmul(F_x, P_prev, F_x) + quadratic_matmul(F_q, self.cov_eps_x, F_q)
        return m_pred, P_pred
    
    @tf.function(reduce_retracing=True)
    def update(self, m_pred, P_pred, y, joseph=True):
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)
    
        y_pred = self.ssm.h_with_noise(m_pred, r0)
        v = self.ssm.innovation(y, y_pred)

        H_x = self._jacobian(lambda x: self.ssm.h_with_noise(x, r0), m_pred)
        H_r = self._jacobian(lambda r: self.ssm.h_with_noise(m_pred, r), r0)

        R_eff = quadratic_matmul(H_r, self.cov_eps_y, H_r)
        K, S = self._kalman_gain(H_x, P_pred, R_eff)

        m_filt = m_pred + tf.einsum('...ij,...j->...i', K, v)

        if joseph:
            P_filt = self._joseph_update(H_x, P_pred, K, R_eff)
        else:
            P_filt = P_pred - quadratic_matmul(K, S, K)

        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)
        return m_filt, P_filt, cond_P, cond_S

class UnscentedKalmanFilter(GaussianFilter):
    
    def __init__(self, ssm, alpha=1e-3, beta=2.0, kappa=0.0):
        super().__init__(ssm)
        self.state_dim = self.ssm.state_dim
        self.obs_dim = self.ssm.obs_dim

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.Wm, self.Wc, self.lamb = self._build_ukf_weights(self.state_dim)

    def _build_ukf_weights(self, n: int):
        alpha = self.alpha
        beta = self.beta
        kappa = self.kappa

        lamb = alpha ** 2 * (n + kappa) - n
        self.lamb = lamb

        num_sigma = 2 * n + 1
        w = 1.0 / (2.0 * (n + lamb))
        w0 = lamb / (n + lamb)
        Wm = tf.concat([tf.reshape(w0, [1]),
                    tf.fill([num_sigma - 1], w)], axis=0)
        Wc = tf.concat([tf.reshape(w0 + (1.0 - alpha**2 + beta), [1]),
                    tf.fill([num_sigma - 1], w)], axis=0)
        return Wm, Wc, lamb

    def generate_sigma_points(self, m, P):

        n = int(P.shape[-1])
        W_m, W_c, lamb = self._build_ukf_weights(n)
        scale = n + lamb
        P_scaled = P * scale                                # [batch, n, n]

        P_sqrt = tf.linalg.cholesky(P_scaled)                    # [batch, n, n]
        P_sqrt = tf.linalg.matrix_transpose(P_sqrt) # the row of P_sqrt should be sigma directions
        m_exp = m[..., tf.newaxis, :]                         # [batch, 1, n]
        X_plus = m_exp + P_sqrt                              # [batch, n, n]
        X_minus = m_exp - P_sqrt                           

        X = tf.concat([m_exp, X_plus, X_minus], axis=1)     # [batch, 2n+1, n]
        return X, tf.convert_to_tensor(W_m, dtype=tf.float32), tf.convert_to_tensor(W_c, dtype=tf.float32)


    def propagate_sigma_points(self, func, X):
        """
        func: [batch, n, d] -> [batch, n, d]
        X   : [batch, 2n+1, n]
        returns Y: [batch, 2n+1, d]
        """
        return func(X) # [batch, 2n+1, d]

    def unscented_transform(self, func, m, P, mean_fn=None, residual_fn=None):

        if mean_fn is None:
            mean_fn = lambda y, W: weighted_mean(y, W, axis=1)
        if residual_fn is None:
            residual_fn = lambda y, y_mean: y - y_mean[..., tf.newaxis, :]
        X, Wm, Wc = self.generate_sigma_points(m, P)        
        Y = self.propagate_sigma_points(func, X)        

        # Wm: [2n+1], Y: [batch, 2n+1, d] -> [batch, d]
        y_mean = mean_fn(Y, Wm)

        Y_c = residual_fn(Y, y_mean)   # [batch, 2n+1, d]
        cov = tf.einsum('...i,...ij,...ik->...jk', Wc, Y_c, Y_c)  # [batch, d, d]

        return y_mean, cov, X, Y, Wm, Wc

    def predict(self, m_prev, P_prev):

        q0 = tf.zeros(tf.concat([tf.shape(m_prev)[:-1], [self.q_dim]], axis=0), dtype=tf.float32)
        m_aug = tf.concat([m_prev, q0], axis=-1)
        P_aug = block_diag(P_prev, self.cov_eps_x)

        def f_aug(z):
            x = z[..., :self.state_dim]
            q = z[..., self.state_dim:]
            return self.ssm.f_with_noise(x, q)

        m_pred, P_pred, _, _, _, _ = self.unscented_transform(
            f_aug, m_aug, P_aug,
             mean_fn=self.ssm.state_mean, residual_fn=self.ssm.state_residual
        )
        
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, joseph=True):
        # UT of h around (m_pred, P_pred)
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)
        m_aug = tf.concat([m_pred, r0], axis=-1)
        P_aug = block_diag(P_pred, self.cov_eps_y)

        def h_aug(z):
            x = z[..., :self.state_dim]
            r = z[..., self.state_dim:]
            return self.ssm.h_with_noise(x, r)

        y_pred, S, X, Y, Wm, Wc = self.unscented_transform(
            h_aug, m_aug, P_aug,
             mean_fn=self.ssm.measurement_mean, residual_fn=self.ssm.measurement_residual
        )  

        v = self.ssm.innovation(y, y_pred)                                # [batch, dy]

        X_x = X[:, :, :self.state_dim]
        X_c = self.ssm.state_residual(X_x, m_pred)           
        Y_c = self.ssm.measurement_residual(Y, y_pred)         

        # cross-cov: C_xy[b, j, k] = sum_i Wc[i] * X_c[b, i, j] * Y_c[b, i, k]
        C_xy = tf.einsum('...i,...ij,...ik->...jk', Wc, X_c, Y_c)  # [batch, dx, dy]

        RHS_T = tf.linalg.matrix_transpose(C_xy)
        K_T = cholesky_solve(S, RHS_T)                  # [batch, dy, dx]
        K = tf.linalg.matrix_transpose(K_T)           # [batch, dx, dy]

        # mean update: m_filt = m_pred + K v
        m_filt = m_pred + tf.einsum('...ij,...j->...i', K, v)  # [batch, dx]

        # cov update: P_filt = P_pred - K S K^T, no easy joseph form this time
        KSK_T = quadratic_matmul(K, S, K)
        P_filt = P_pred - KSK_T
        if joseph: # simply enforce symmetry
            P_filt = 0.5 * (P_filt + tf.linalg.matrix_transpose(P_filt))
        
        cond_P = tf_cond(P_filt)                      
        cond_S = tf_cond(S)                          

        return m_filt, P_filt, cond_P, cond_S

class ParticleFilter(BaseFilter):
    def __init__(self, ssm, num_particles=100, ess_threshold=0.5):
        super().__init__(ssm)
        self.num_particles = num_particles
        self.ssm = ssm
        self.ess_threshold = tf.convert_to_tensor(ess_threshold, dtype=tf.float32)

    @staticmethod
    def _normalize_weights(weights):
        log_norm = tf.reduce_logsumexp(weights, axis=-1)
        w = tf.exp(weights - log_norm)
        return w, log_norm

    @tf.function(reduce_retracing=True)
    def filter(self, y, resample=True):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if (len(y.shape) == 2):
            y = y[tf.newaxis, :] # [T, dy] → [batch, T, dy]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]

        x_particles = self.ssm.sample_initial_state(batch_size, self.num_particles)
        weights = tf.ones([batch_size, self.num_particles], dtype=tf.float32) / self.num_particles

        for t in range(T):
            y_t = y[:, t, :]
            x_particles, weights = self.update(x_particles, weights, y_t)
        return x_particles, weights