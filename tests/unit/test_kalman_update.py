import pytest
import tensorflow as tf

from src.filters.kalman import KalmanFilter

pytestmark = pytest.mark.unit


def test_kalman_predict_closed_form(lgssm_1d):
    kf = KalmanFilter(lgssm_1d)
    m_prev = tf.constant([[0.0]], dtype=tf.float32)
    P_prev = tf.constant([[[1.0]]], dtype=tf.float32)
    m_pred, P_pred = kf.predict(m_prev, P_prev)

    expected_m = m_prev
    expected_P = P_prev + lgssm_1d.cov_eps_x
    tf.debugging.assert_near(m_pred, expected_m, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(P_pred, expected_P, atol=1e-6, rtol=1e-6)


def test_kalman_update_closed_form(lgssm_1d):
    kf = KalmanFilter(lgssm_1d)
    m_pred = tf.constant([[0.0]], dtype=tf.float32)
    P_pred = tf.constant([[[2.0]]], dtype=tf.float32) + lgssm_1d.jitter * tf.eye(1, batch_shape=[1])
    y = tf.constant([[1.0]], dtype=tf.float32)

    m_filt, P_filt = kf.update_joseph(m_pred, P_pred, y)

    R = lgssm_1d.cov_eps_y
    S = P_pred + R
    K = P_pred / S
    K_s = tf.squeeze(K, axis=-1)
    m_expected = m_pred + K_s * (y - m_pred)
    P_expected = (1.0 - K) * P_pred * (1.0 - K) + K * R * K

    tf.debugging.assert_near(m_filt, m_expected, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(P_filt, P_expected, atol=1e-6, rtol=1e-6)
