import tensorflow as tf
import numpy as np
from src.utility import tf_cond, cholesky_solve, weighted_mean

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
        self.cov_eps_x = tf.convert_to_tensor(ssm.cov_eps_x, dtype=tf.float32)
        self.cov_eps_y = tf.convert_to_tensor(ssm.cov_eps_y, dtype=tf.float32)

    def filter(self, y, **kwargs):
        raise NotImplementedError
    
    def predict(self, m_prev, P_prev):
        raise NotImplementedError
    
    def update(self, m_pred, P_pred, y):
        raise NotImplementedError

    def filter(self, y, joseph=True):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if (len(y.shape) == 2):
            # [T, dy] → [1, T, dy]
            y = y[tf.newaxis, :]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]

        m_pred = tf.convert_to_tensor(self.ssm.m0, dtype=tf.float32)
        if (len(m_pred.shape) == 1):
            # [dx] → [batch, dx]
            m_pred = m_pred[tf.newaxis, :]  
            m_pred = tf.tile(m_pred, [batch_size, 1])
        P_pred = tf.convert_to_tensor(self.ssm.P0, dtype=tf.float32)
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

class KalmanFilter(GaussianFilter):
    def __init__(self, ssm):
        super().__init__(ssm)
        self.A = self.ssm.A
        self.C = self.ssm.C
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0

    def predict(self, m_prev, P_prev):
        m_pred = tf.einsum('ij,bj->bi', self.A, m_prev) # [batch, dx]
        P_pred = tf.einsum('ij,bjk,lk->bil', self.A, P_prev, self.A) + self.cov_eps_x # [batch, dx, dx]
        return m_pred, P_pred # [batch, dx], [batch, dx, dx]

    def update(self, m_pred, P_pred, y, joseph=True):
        y_pred = self.ssm.h(m_pred)
        v = y - y_pred

        S = tf.einsum('ij,bjk,lk->bil', self.C, P_pred, self.C) + self.cov_eps_y # [batch, dy, dy]
        RHS = tf.einsum('bij,kj->bik', P_pred, self.C) 
        RHS_transpose = tf.transpose(RHS, perm=[0, 2, 1])
        
        K_transpose = cholesky_solve(S, RHS_transpose) # S K' = CP'
        K = tf.transpose(K_transpose, perm=[0, 2, 1])
        m_filt = m_pred + tf.einsum('bij,bj->bi', K, v) # [batch, dx]
        if joseph:
            # Joseph form: P_filt = (I - KC) P_pred (I - KC)' + K R K'
            I = tf.eye(tf.shape(P_pred)[1], batch_shape=[tf.shape(P_pred)[0]], dtype=tf.float32)
            I_KC = I - tf.einsum('bij,jk->bik', K, self.C)
            P_filt = tf.einsum('bij,bjk,blk->bil', I_KC, P_pred, I_KC) + tf.einsum('bij,jk,blk->bil', K, self.cov_eps_y, K)
        else:
            # standard form: P_filt = P_pred - K S K'
            P_filt = P_pred - tf.einsum('bij,bjk,blk->bil', K, S, K)
        
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

    def predict(self, m_prev, P_prev):
        m_pred = self.ssm.f(m_prev)
        F = self._jacobian(self.ssm.f, m_prev)
        P_pred = tf.einsum('bij,bjk,blk->bil', F, P_prev, F) + self.cov_eps_x
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, joseph=True):
        y_pred = self.ssm.h(m_pred)
        v = self.ssm.innovation(y, y_pred)

        H = self._jacobian(self.ssm.h, m_pred)
        S = tf.einsum('bij,bjk,blk->bil', H, P_pred, H) + self.cov_eps_y
        RHS = tf.einsum('bij,bkj->bik', P_pred, H)
        RHS_transpose = tf.transpose(RHS, perm=[0, 2, 1])

        K_transpose = cholesky_solve(S, RHS_transpose)
        K = tf.transpose(K_transpose, perm=[0, 2, 1])

        m_filt = m_pred + tf.einsum('bij,bj->bi', K, v)

        if joseph:
            I = tf.eye(tf.shape(P_pred)[1], batch_shape=[tf.shape(m_pred)[0]], dtype=tf.float32)
            I_KH = I - tf.einsum('bij,bjk->bik', K, H)
            P_filt = tf.einsum('bij,bjk,blk->bil', I_KH, P_pred, I_KH) + tf.einsum('bij,jk,blk->bil', K, self.cov_eps_y, K)
        else:
            P_filt = P_pred - tf.einsum('bij,bjk,blk->bil', K, S, K)

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

        self._build_ukf_weights()

    def _build_ukf_weights(self):
        n = self.state_dim
        alpha = self.alpha
        beta = self.beta
        kappa = self.kappa

        lamb = alpha ** 2 * (n + kappa) - n
        self.lamb = lamb

        num_sigma = 2 * n + 1

        Wm = np.full(num_sigma, 1.0 / (2.0 * (n + lamb)), dtype=np.float32)
        Wm[0] = lamb / (n + lamb)

        Wc = np.copy(Wm)
        Wc[0] = lamb / (n + lamb) + (1.0 - alpha ** 2 + beta)

        self.Wm = tf.constant(Wm, dtype=tf.float32)  # [2n+1]
        self.Wc = tf.constant(Wc, dtype=tf.float32)  # [2n+1]

    def generate_sigma_points(self, m, P):
        """
        Generate sigma points for each batch.

        m: [batch, n]
        P: [batch, n, n]

        returns X: [batch, 2L+1, n]
        """
        lamb = self.lamb

        scale = self.state_dim + lamb
        P_scaled = P * scale                                # [batch, n, n]

        P_sqrt = tf.linalg.cholesky(P_scaled)                    # [batch, n, n]
        P_sqrt = tf.transpose(P_sqrt, perm=[0, 2, 1])  # the row of P_sqrt should be sigma directions
        m_exp = m[:, tf.newaxis, :]                         # [batch, 1, n]
        X_plus = m_exp + P_sqrt                              # [batch, n, n]
        X_minus = m_exp - P_sqrt                           

        X = tf.concat([m_exp, X_plus, X_minus], axis=1)     # [batch, 2n+1, n]
        return X


    def propagate_sigma_points(self, func, X):
        """
        func: [batch, n] -> [batch, d]
        X   : [batch, 2n+1, n]
        returns Y: [batch, 2n+1, d]
        """
        batch_size = tf.shape(X)[0]
        num_sigma = tf.shape(X)[1]

        X_flat = tf.reshape(X, [batch_size * num_sigma, -1])   # [batch*(2n+1), n]
        Y_flat = func(X_flat)                          # [batch * (2n+1), d]
        d = tf.shape(Y_flat)[1]
        Y = tf.reshape(Y_flat, [batch_size, num_sigma, d])     # [batch, 2n+1, d]
        return Y

    def unscented_transform(self, func, m, P, noise_cov=None, mean_fn=None, residual_fn=None):
        """
        Unscented transform of Gaussian N(m, P) through func:

        func: [batch, n] -> [batch, d],  transformation
        m   : [batch, n]
        P   : [batch, n, n]
        noise_cov: [d, d] or None
        mean_fn: [batch, n, d] -> [batch, d], function to compute mean of Y
        residual_fn: [batch, n, d], [batch, d] -> [batch, n, d], function to compute residual of Y and y_mean

        returns:
            y_mean: [batch, d]
            cov   : [batch, d, d]
            X     : [batch, 2n+1, n]
            Y     : [batch, 2n+1, d]
        """
        if mean_fn is None:
            mean_fn = lambda y, W: weighted_mean(y, W, axis=1)
        if residual_fn is None:
            residual_fn = lambda y, y_mean: y - y_mean[:, tf.newaxis, :]
        X = self.generate_sigma_points(m, P)        
        Y = self.propagate_sigma_points(func, X)        

        # Wm: [2n+1], Y: [batch, 2n+1, d] -> [batch, d]
        y_mean = mean_fn(Y, self.Wm)

        Y_c = residual_fn(Y, y_mean)   # [batch, 2n+1, d]
        cov = tf.einsum('i,bij,bik->bjk', self.Wc, Y_c, Y_c)  # [Batch, d, d]

        if noise_cov is not None:
            cov = cov + noise_cov

        return y_mean, cov, X, Y

    def predict(self, m_prev, P_prev):
        """
        UKF prediction:

            (m_prev, P_prev) --UT through f--> (m_pred, P_pred)

        m_prev : [batch, dx]
        P_prev : [batch, dx, dx]
        """
        m_pred, P_pred, _, _ = self.unscented_transform(
            self.ssm.f, m_prev, P_prev, noise_cov=self.cov_eps_x,
             mean_fn=self.ssm.state_mean, residual_fn=self.ssm.state_residual
        )
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, joseph=True):
        # UT of h around (m_pred, P_pred)
        y_pred, S, X, Y = self.unscented_transform(
            self.ssm.h, m_pred, P_pred, noise_cov=self.cov_eps_y,
             mean_fn=self.ssm.measurement_mean, residual_fn=self.ssm.measurement_residual
        )  

        v = self.ssm.innovation(y, y_pred)                                # [batch, dy]

        X_c = X - m_pred[:, tf.newaxis, :]             
        Y_c = Y - y_pred[:, tf.newaxis, :]            

        # cross-cov: C_xy[b, j, k] = sum_i Wc[i] * X_c[b, i, j] * Y_c[b, i, k]
        C_xy = tf.einsum('i,bij,bik->bjk', self.Wc, X_c, Y_c)  # [batch, dx, dy]

        RHS_T = tf.transpose(C_xy, perm=[0, 2, 1])
        K_T = cholesky_solve(S, RHS_T)                  # [batch, dy, dx]
        K = tf.transpose(K_T, perm=[0, 2, 1])           # [batch, dx, dy]

        # mean update: m_filt = m_pred + K v
        m_filt = m_pred + tf.einsum('bij,bj->bi', K, v)  # [batch, dx]

        # cov update: P_filt = P_pred - K S K^T, no easy joseph form this time
        KSK_T = tf.einsum('bij,bjk, blk->bil', K, S, K)
        P_filt = P_pred - KSK_T
        if joseph: # simply enforce symmetry
            P_filt = 0.5 * (P_filt + tf.transpose(P_filt, perm=[0, 2, 1]))
        
        cond_P = tf_cond(P_filt)                      
        cond_S = tf_cond(S)                          

        return m_filt, P_filt, cond_P, cond_S
