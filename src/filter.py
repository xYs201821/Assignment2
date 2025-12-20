import tensorflow as tf
import numpy as np
from src.distributions import BootstrapProposal
from src.utility import tf_cond, cholesky_solve, weighted_mean, block_diag, quadratic_matmul

class BaseFilter(tf.Module):
    def __init__(self, ssm, **kwargs):
        super().__init__()
        self.ssm = ssm
        self.state_dim = ssm.state_dim
        self.obs_dim = ssm.obs_dim

    def filter(self, y, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def _stack_and_permute(ta, tail_dims=1):
        seq = ta.stack()
        rank = tf.rank(seq)
        batch_rank = rank - (1 + tail_dims)
        prefix = tf.range(1, 1 + batch_rank)
        perm = tf.concat([prefix, [0], tf.range(1 + batch_rank, rank)], axis=0)
        return tf.transpose(seq, perm)

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
        m_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        m_pred_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_pred_ta = tf.TensorArray(dtype=tf.float32, size=T)
        cond_P_ta = tf.TensorArray(dtype=tf.float32, size=T)
        cond_S_ta = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            y_t = y[:, t, :]
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)
            
            m_filt, P_filt, cond_P, cond_S = self.update(m_pred, P_pred, y_t, joseph=joseph)
            
            m_filt_ta = m_filt_ta.write(t, m_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            cond_P_ta = cond_P_ta.write(t, cond_P)
            cond_S_ta = cond_S_ta.write(t, cond_S)
            
            m_pred, P_pred = self.predict(m_filt, P_filt)

        return {
            "m_filt": self._stack_and_permute(m_filt_ta, tail_dims=1),
            "P_filt": self._stack_and_permute(P_filt_ta, tail_dims=2),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "cond_P": self._stack_and_permute(cond_P_ta, tail_dims=0),
            "cond_S": self._stack_and_permute(cond_S_ta, tail_dims=0),
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

    def _build_ukf_weights(self, n: tf.Tensor):
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
    def __init__(self, ssm, proposal=None, num_particles=100, ess_threshold=0.5):
        super().__init__(ssm)
        self.num_particles = int(num_particles)
        self.ess_threshold = tf.convert_to_tensor(ess_threshold, tf.float32)
        self.proposal = proposal if proposal is not None else BootstrapProposal()

    @staticmethod
    def _log_normalize(log_w):
        logZ = tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
        log_w_norm = log_w - logZ
        w = tf.exp(log_w_norm)
        return log_w_norm, w, tf.squeeze(logZ, axis=-1)

    @staticmethod
    def ess(w):
        return 1.0 / tf.reduce_sum(tf.square(w), axis=-1)

    @staticmethod
    def systematic_resample(w, rng):
        shape = tf.shape(w)
        N = shape[-1]
        B = tf.reduce_prod(shape[:-1])
        w2 = tf.reshape(w, [B, N])
        cdf = tf.cumsum(w2, axis=-1)

        u0 = rng.uniform([B, 1], 0.0, 1.0 / tf.cast(N, tf.float32), dtype=tf.float32)
        js = tf.cast(tf.range(N)[tf.newaxis, :], tf.float32)
        u = u0 + js / tf.cast(N, tf.float32)

        idx = tf.searchsorted(cdf, u, side='left')
        idx = tf.clip_by_value(idx, 0, N - 1)
        return tf.reshape(idx, shape)

    @staticmethod
    def resample_particles(x, idx):
        shape = tf.shape(x)
        B = tf.reduce_prod(shape[:-2])
        x_flatten = tf.reshape(x, [B, shape[-2], shape[-1]])
        idx_flatten = tf.reshape(idx, [B, shape[-2]])
        out_flatten = tf.gather(x_flatten, idx_flatten, batch_dims=1)
        return tf.reshape(out_flatten, shape)

    def step(self, x_prev, log_w_prev, y_t, resample="auto"):
        x_t = self.proposal.sample(self.ssm, x_prev, y_t, seed=self.ssm._tfp_seed())

        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])
        logtrans = self.ssm.transition_dist(x_prev).log_prob(x_t)
        logq = self.proposal.log_prob(self.ssm, x_t, x_prev, y_t)

        log_w = log_w_prev + tf.cast(loglik + logtrans - logq, tf.float32)

        log_w_norm, w, logZ = self._log_normalize(log_w)
        ess = self.ess(w)

        if resample in ("auto", "always"):
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == "always":
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w, self.ssm.rng)

            batch_shape = tf.shape(x_t)[:-2]
            no_rs_indices = tf.broadcast_to(tf.range(self.num_particles, dtype=tf.int32),
             tf.concat([batch_shape, [self.num_particles]], axis=0))
            mask_do_rs = mask_do_rs[..., tf.newaxis]
            parent_indices = tf.where(mask_do_rs, rs_indices, no_rs_indices)

            x_t = self.resample_particles(x_t, parent_indices)
            log_w_reset = -tf.math.log(N_float) * tf.ones_like(log_w_norm)
            log_w_final = tf.where(mask_do_rs, log_w_reset, log_w_norm)
        else:
            batch_shape = tf.shape(x_t)[:-2]
            parent_indices = tf.broadcast_to(tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0))
            log_w_final = log_w_norm
        w_final = tf.exp(log_w_final)
        return x_t, log_w_final, w_final, ess, logZ, parent_indices

    @tf.function(reduce_retracing=True)
    def filter(self, y, resample="auto", init_dist=None):
        y = tf.convert_to_tensor(y, tf.float32)
        if tf.rank(y) == 2:
            y = y[tf.newaxis, ...]   # [1, T, dy]

        if isinstance(resample, bool):
            resample = "auto" if resample else "never"
        if resample not in ("auto", "never", "always"):
            raise ValueError("resample must be True/False or one of 'auto', 'never', 'always'")

        batch_shape = tf.shape(y)[:-2]
        T = tf.shape(y)[-2]
        N = self.num_particles

        if init_dist is None:
            x = self.ssm.sample_initial_state(tf.concat([batch_shape, [N]], axis=0), return_log_prob=False)
        else:
            if not callable(init_dist):
                raise TypeError("init_dist must be a callable: init_dist(shape) -> tfd.Distribution")
            dist = init_dist(tf.concat([batch_shape, [N]], axis=0))
            x = tf.cast(dist.sample(seed=self.ssm._tfp_seed()), dtype=tf.float32)
        log_w = -tf.math.log(tf.cast(N, tf.float32)) * tf.ones(tf.concat([batch_shape, [N]], axis=0), tf.float32)
        parent_indices = tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape, [self.num_particles]], axis=0),
        )

        ess_ta = tf.TensorArray(tf.float32, size=T)
        logZ_ta = tf.TensorArray(tf.float32, size=T)
        x_ta = tf.TensorArray(tf.float32, size=T)
        w_ta = tf.TensorArray(tf.float32, size=T)
        parent_ta = tf.TensorArray(tf.int32, size=T)

        for t in tf.range(T):
            y_t = y[..., t, :]  # [..., dy]
            x, log_w, w, ess, logZ, parent_indices = self.step(x, log_w, y_t, resample=resample)
            ess_ta = ess_ta.write(t, ess)
            logZ_ta = logZ_ta.write(t, logZ)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            parent_ta = parent_ta.write(t, parent_indices)

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)
        
        diagnostics = {
            "ess": self._stack_and_permute(ess_ta, tail_dims=0),
            "logZ": self._stack_and_permute(logZ_ta, tail_dims=0),
        }
        return x_seq, w_seq, diagnostics, parent_seq
