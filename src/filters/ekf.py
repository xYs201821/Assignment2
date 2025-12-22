import tensorflow as tf

from src.filters.base import GaussianFilter
from src.filters.mixins import LinearizationMixin
from src.utility import quadratic_matmul, tf_cond


class ExtendedKalmanFilter(GaussianFilter, LinearizationMixin):
    def __init__(self, ssm, debug=False, print=False):
        super().__init__(ssm, debug=debug, print=print)
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0
        self._maybe_print()

    @tf.function(reduce_retracing=True)
    def predict(self, m_prev, P_prev): 
        q0 = tf.zeros(tf.concat([tf.shape(m_prev)[:-1], [self.q_dim]], axis=0), dtype=tf.float32)
        F_x, m_pred = self._jacobian(lambda x: self.ssm.f_with_noise(x, q0), m_prev)
        F_q, _ = self._jacobian(lambda q: self.ssm.f_with_noise(m_prev, q), q0)

        P_pred = quadratic_matmul(F_x, P_prev, F_x) + quadratic_matmul(F_q, self.cov_eps_x, F_q)
        tf.debugging.assert_all_finite(m_pred, "EKF predict produced NaN/Inf in m_pred")
        tf.debugging.assert_all_finite(P_pred, "EKF predict produced NaN/Inf in P_pred")
        return m_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update(self, m_pred, P_pred, y, joseph=True):
        r0 = tf.zeros(tf.concat([tf.shape(m_pred)[:-1], [self.r_dim]], axis=0), dtype=tf.float32)


        H_x, y_pred = self._jacobian(lambda x: self.ssm.h_with_noise(x, r0), m_pred)
        H_r, _ = self._jacobian(lambda r: self.ssm.h_with_noise(m_pred, r), r0)
        v = self.ssm.innovation(y, y_pred)

        R_eff = quadratic_matmul(H_r, self.cov_eps_y, H_r)
        K, S = self._kalman_gain(H_x, P_pred, R_eff)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)

        if joseph:
            P_filt = self._joseph_update(H_x, P_pred, K, R_eff)
        else:
            P_filt = P_pred - quadratic_matmul(K, S, K)

        tf.debugging.assert_all_finite(H_x, "EKF update produced NaN/Inf in H_x")
        tf.debugging.assert_all_finite(H_r, "EKF update produced NaN/Inf in H_r")
        tf.debugging.assert_all_finite(S, "EKF update produced NaN/Inf in S")
        tf.debugging.assert_all_finite(K, "EKF update produced NaN/Inf in K")
        tf.debugging.assert_all_finite(m_filt, "EKF update produced NaN/Inf in m_filt")
        tf.debugging.assert_all_finite(P_filt, "EKF update produced NaN/Inf in P_filt")
        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)
        return m_filt, P_filt, cond_P, cond_S
