"""Extended Kalman filter for nonlinear state-space models."""

import tensorflow as tf

from src.filters.base import GaussianFilter
from src.filters.mixins import LinearizationMixin
from src.utility import quadratic_matmul, tf_cond


class ExtendedKalmanFilter(GaussianFilter, LinearizationMixin):
    """Extended Kalman filter using first-order linearization."""

    def __init__(self, ssm, joseph=True, debug=False, print=False):
        """Initialize EKF with optional Joseph covariance update."""
        super().__init__(ssm, joseph=joseph, debug=debug, print=print)
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0
        self._maybe_print()

    @tf.function(reduce_retracing=True)
    def predict(self, m_prev, P_prev): 
        """EKF predict using Jacobians of the transition function.

        Shapes:
          m_prev: [B, dx]
          P_prev: [B, dx, dx]
        Returns:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
        """
        q0 = tf.zeros(tf.concat([tf.shape(m_prev)[:-1], [self.q_dim]], axis=0), dtype=tf.float32)
        F_x, m_pred = self.jacobian_f_x(m_prev, q0)
        F_q, _ = self.jacobian_f_q(m_prev, q0)

        # P = F_x P F_x^T + F_q Q F_q^T.
        P_pred = quadratic_matmul(F_x, P_prev, F_x) + quadratic_matmul(F_q, self.cov_eps_x, F_q)
        tf.debugging.assert_all_finite(m_pred, "EKF predict produced NaN/Inf in m_pred")
        tf.debugging.assert_all_finite(P_pred, "EKF predict produced NaN/Inf in P_pred")
        return m_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update_joseph(self, m_pred, P_pred, y):
        """EKF update using Joseph stabilized covariance.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)

        H_x, y_pred = self.jacobian_h_x(m_pred, r0)
        H_r, _ = self.jacobian_h_r(m_pred, r0)
        v = self.ssm.innovation(y, y_pred)

        # Effective observation noise via linearized measurement noise.
        R_eff = quadratic_matmul(H_r, self.cov_eps_y, H_r)
        K, S = self._kalman_gain(H_x, P_pred, R_eff)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)
        P_filt = self._joseph_update(H_x, P_pred, K, R_eff)

        return m_filt, P_filt

    @tf.function
    def update_naive(self, m_pred, P_pred, y):
        """EKF update using naive covariance update.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)

        H_x, y_pred = self.jacobian_h_x(m_pred, r0)
        H_r, _ = self.jacobian_h_r(m_pred, r0)
        v = self.ssm.innovation(y, y_pred)

        # Effective observation noise via linearized measurement noise.
        R_eff = quadratic_matmul(H_r, self.cov_eps_y, H_r)
        K, S = self._kalman_gain(H_x, P_pred, R_eff)
    
        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)
        P_filt = P_pred - quadratic_matmul(K, S, K)

        return m_filt, P_filt
