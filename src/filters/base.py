import tensorflow as tf

from src.utility import cholesky_solve, quadratic_matmul

FILTER_PRINT_INFO = {
    "BaseFilter": {
        "title": "BaseFilter",
        "fields": ("state_dim", "obs_dim", "q_dim", "r_dim"),
    },
    "GaussianFilter": {
        "title": "GaussianFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "KalmanFilter": {
        "title": "KalmanFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "ExtendedKalmanFilter": {
        "title": "ExtendedKalmanFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "UnscentedKalmanFilter": {
        "title": "UnscentedKalmanFilter",
        "fields": ("state_dim", "obs_dim", "alpha", "beta", "kappa"),
    },
    "ParticleFilter": {
        "title": "ParticleFilter",
        "fields": ("num_particles", "ess_threshold"),
    },
    "BootstrapParticleFilter": {
        "title": "BootstrapParticleFilter",
        "fields": ("num_particles", "ess_threshold"),
    },
}


class BaseFilter(tf.Module):
    def __init__(self, ssm, debug=False, print=False, **kwargs):
        super().__init__()
        self.ssm = ssm
        self.state_dim = ssm.state_dim
        self.obs_dim = ssm.obs_dim
        self.q_dim = ssm.q_dim
        self.r_dim = ssm.r_dim
        self.debug = bool(debug)
        self.print = bool(print)
        if type(self) is BaseFilter:
            self._maybe_print()
        
    def filter(self, y, **kwargs):
        raise NotImplementedError

    def _maybe_print(self):
        if self.print:
            self._print_debug_info()

    def _print_debug_info(self):
        info = FILTER_PRINT_INFO.get(type(self).__name__)
        if not info:
            return
        title = info.get("title", type(self).__name__)
        fields = info.get("fields", ())
        parts = [title]
        for field in fields:
            if hasattr(self, field):
                parts.append(f"{field}=")
                parts.append(getattr(self, field))
        tf.print("[filter]", *parts)

    @staticmethod
    def _stack_and_permute(ta, tail_dims=1):
        seq = ta.stack()
        rank = tf.rank(seq)
        batch_rank = rank - (1 + tail_dims)
        prefix = tf.range(1, 1 + batch_rank)
        perm = tf.concat([prefix, [0], tf.range(1 + batch_rank, rank)], axis=0)
        return tf.transpose(seq, perm)


class GaussianFilter(BaseFilter):
    def __init__(self, ssm, debug=False, print=False):
        super().__init__(ssm, debug=debug, print=print)
        self.cov_eps_x = tf.convert_to_tensor(ssm.cov_eps_x, dtype=tf.float32)
        self.cov_eps_y = tf.convert_to_tensor(ssm.cov_eps_y, dtype=tf.float32)
        if type(self) is GaussianFilter:
            self._maybe_print()


    def predict(self, m_prev, P_prev):
        raise NotImplementedError

    def update(self, m_pred, P_pred, y):
        raise NotImplementedError

    def filter(self, y, joseph=True, m0=None, P0=None):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if len(y.shape) == 2:
            y = y[tf.newaxis, :]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]

        m_pred = tf.convert_to_tensor(m0 if m0 is not None else self.ssm.m0, dtype=tf.float32)
        if len(m_pred.shape) == 1:
            m_pred = m_pred[tf.newaxis, :]
            m_pred = tf.tile(m_pred, [batch_size, 1])
        P_pred = tf.convert_to_tensor(P0 if P0 is not None else self.ssm.P0, dtype=tf.float32)
        if len(P_pred.shape) == 2:
            P_pred = P_pred[tf.newaxis, :, :]
            P_pred = tf.tile(P_pred, [batch_size, 1, 1])

        m_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        m_pred_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_pred_ta = tf.TensorArray(dtype=tf.float32, size=T)
        cond_P_ta = tf.TensorArray(dtype=tf.float32, size=T)
        cond_S_ta = tf.TensorArray(dtype=tf.float32, size=T)
        step_time_ta = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            t_start = tf.timestamp()
            y_t = y[:, t, :]
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)

            m_filt, P_filt, cond_P, cond_S = self.update(m_pred, P_pred, y_t, joseph=joseph)

            m_filt_ta = m_filt_ta.write(t, m_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            cond_P_ta = cond_P_ta.write(t, cond_P)
            cond_S_ta = cond_S_ta.write(t, cond_S)
            step_time = tf.cast(tf.timestamp() - t_start, tf.float32)
            step_time_ta = step_time_ta.write(t, step_time)

            m_pred, P_pred = self.predict(m_filt, P_filt)

        return {
            "m_filt": self._stack_and_permute(m_filt_ta, tail_dims=1),
            "P_filt": self._stack_and_permute(P_filt_ta, tail_dims=2),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "cond_P": self._stack_and_permute(cond_P_ta, tail_dims=0),
            "cond_S": self._stack_and_permute(cond_S_ta, tail_dims=0),
            "step_time_s": self._stack_and_permute(step_time_ta, tail_dims=0),
        }

    @staticmethod
    def _kalman_gain(C, P, R):
        S = quadratic_matmul(C, P, C) + R
        RHS = tf.linalg.matmul(C, P, transpose_b=True)
        K_transpose = cholesky_solve(S, RHS)
        K = tf.linalg.matrix_transpose(K_transpose)
        return K, S

    @staticmethod
    def _joseph_update(C, P, K, R):
        I = tf.eye(tf.shape(P)[-1], batch_shape=tf.shape(P)[:-2], dtype=P.dtype)
        I_KC = I - tf.linalg.matmul(K, C)
        return quadratic_matmul(I_KC, P, I_KC) + quadratic_matmul(K, R, K)
