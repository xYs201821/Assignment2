import tensorflow as tf

from src.filters.base import GaussianFilter
from src.utility import quadratic_matmul, tf_cond


class KalmanFilter(GaussianFilter):
    def __init__(self, ssm, debug=False, print=False):
        super().__init__(ssm, debug=debug, print=print)
        self.A = self.ssm.A
        self.C = self.ssm.C
        self.m0 = self.ssm.m0
        self.P0 = self.ssm.P0
        self._maybe_print()

    def predict(self, m_prev, P_prev):
        m_pred = tf.einsum("ij,...j->...i", self.A, m_prev)
        P_pred = quadratic_matmul(self.A, P_prev, self.A) + self.cov_eps_x
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, joseph=True):
        y_pred = self.ssm.h(m_pred)
        v = self.ssm.innovation(y, y_pred)

        K, S = self._kalman_gain(self.C, P_pred, self.cov_eps_y)

        m_filt = m_pred + tf.einsum("...ij,...j->...i", K, v)
        if joseph:
            P_filt = self._joseph_update(self.C, P_pred, K, self.cov_eps_y)
        else:
            P_filt = P_pred - quadratic_matmul(K, S, K)

        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)
        return m_filt, P_filt, cond_P, cond_S
