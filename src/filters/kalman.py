"""Linear Gaussian Kalman filter implementation."""

import tensorflow as tf

from src.filters.base import GaussianFilter
from src.utility import cholesky_solve, quadratic_matmul, tf_cond


class KalmanFilter(GaussianFilter):
    """Kalman filter for linear Gaussian state-space models."""

    def __init__(self, ssm, debug=False, print=False):
        """Initialize with linear model matrices from the SSM."""
        super().__init__(ssm, debug=debug, print=print)
        self.A = self.ssm.A
        self.C = self.ssm.C
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0
        self._maybe_print()

    @tf.function
    def predict(self, m_prev, P_prev):
        """Kalman predict: m = A m, P = A P A^T + Q.

        Shapes:
          m_prev: [B, dx]
          P_prev: [B, dx, dx]
        Returns:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
        """
        m_pred = tf.einsum("ij,...j->...i", self.A, m_prev)
        P_pred = quadratic_matmul(self.A, P_prev, self.A) + self.cov_eps_x
        return m_pred, P_pred

    @tf.function
    def update_joseph(self, m_pred, P_pred, y):
        """Kalman update using Joseph stabilized covariance.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        y_pred = self.ssm.h(m_pred)
        v = self.ssm.innovation(y, y_pred)

        K, S = self._kalman_gain(self.C, P_pred, self.cov_eps_y)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)
        P_filt = self._joseph_update(self.C, P_pred, K, self.cov_eps_y)

        return m_filt, P_filt

    @tf.function
    def update_naive(self, m_pred, P_pred, y):
        """Kalman update using the naive covariance formula.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        y_pred = self.ssm.h(m_pred)
        v = self.ssm.innovation(y, y_pred)

        K, S = self._kalman_gain(self.C, P_pred, self.cov_eps_y)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)
        P_filt = P_pred - quadratic_matmul(K, S, K)

        return m_filt, P_filt

    def update(self, m_pred, P_pred, y):
        """Default update entry point (Joseph form)."""
        return self.update_joseph(m_pred, P_pred, y)

    def filter(self, y, joseph=True, m0=None, P0=None, memory_sampler=None):
        """Filter a sequence with optional Joseph form selection.

        Shapes:
          y: [T, dy] or [B, T, dy]
          m0: [dx] or [B, dx]
          P0: [dx, dx] or [B, dx, dx]
        Returns:
          m_filt: [B, T, dx]
          P_filt: [B, T, dx, dx]
          m_pred: [B, T, dx]
          P_pred: [B, T, dx, dx]
        """
        self.update = self.update_joseph if joseph else self.update_naive
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
        step_time_ta = tf.TensorArray(dtype=tf.float32, size=T)
        mem_rss_ta = tf.TensorArray(dtype=tf.float32, size=T) if memory_sampler is not None else None
        mem_gpu_ta = tf.TensorArray(dtype=tf.float32, size=T) if memory_sampler is not None else None

        for t in tf.range(T):
            t_start = tf.timestamp()
            y_t = y[:, t, :]

            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)

            m_filt, P_filt, m_pred, P_pred = self.step(m_pred, P_pred, y_t)

            m_filt_ta = m_filt_ta.write(t, m_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            cond_P_ta = cond_P_ta.write(t, tf_cond(P_filt))
            step_time = tf.cast(tf.timestamp() - t_start, tf.float32)
            step_time_ta = step_time_ta.write(t, step_time)
            if memory_sampler is not None:
                sample = memory_sampler()
                rss = None
                gpu = None
                if isinstance(sample, dict):
                    rss = sample.get("rss")
                    gpu = sample.get("gpu")
                elif isinstance(sample, (tuple, list)):
                    if len(sample) > 0:
                        rss = sample[0]
                    if len(sample) > 1:
                        gpu = sample[1]
                else:
                    rss = sample
                if rss is not None:
                    mem_rss_ta = mem_rss_ta.write(t, tf.cast(rss, tf.float32))
                if mem_gpu_ta is not None:
                    gpu_val = 0.0 if gpu is None else gpu
                    mem_gpu_ta = mem_gpu_ta.write(t, tf.cast(gpu_val, tf.float32))

        out = {
            "m_filt": self._stack_and_permute(m_filt_ta, tail_dims=1),
            "P_filt": self._stack_and_permute(P_filt_ta, tail_dims=2),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "cond_P": self._stack_and_permute(cond_P_ta, tail_dims=0),
            "step_time_s": self._stack_and_permute(step_time_ta, tail_dims=0),
        }
        if mem_rss_ta is not None:
            out["memory_rss"] = self._stack_and_permute(mem_rss_ta, tail_dims=0)
        if mem_gpu_ta is not None:
            out["memory_gpu"] = self._stack_and_permute(mem_gpu_ta, tail_dims=0)
        return out

    @staticmethod
    def _kalman_gain(C, P, R):
        """Compute Kalman gain and innovation covariance."""
        # S = C P C^T + R and K = P C^T S^{-1}.
        S = quadratic_matmul(C, P, C) + R
        RHS = tf.linalg.matmul(C, P, transpose_b=True)
        K_transpose = cholesky_solve(S, RHS)
        K = tf.linalg.matrix_transpose(K_transpose)
        return K, S

    @staticmethod
    def _joseph_update(C, P, K, R):
        """Joseph-form covariance update."""
        I = tf.eye(tf.shape(P)[-1], batch_shape=tf.shape(P)[:-2], dtype=P.dtype)
        I_KC = I - tf.linalg.matmul(K, C)
        return quadratic_matmul(I_KC, P, I_KC) + quadratic_matmul(K, R, K)
