"""Linear Gaussian Kalman filter implementation."""

import tensorflow as tf

from src.filters.base import GaussianFilter
from src.utility import cholesky_solve, quadratic_matmul


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

    def filter(self, y, joseph=True, m0=None, P0=None):
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
        # Set update method before entering the compiled graph
        self.update = self.update_joseph if joseph else self.update_naive
        return super().filter(y, m0=m0, P0=P0)

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
