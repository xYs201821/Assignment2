import pytest
import tensorflow as tf

from src.filters import UnscentedKalmanFilter
from tests.testhelper import make_spd

pytestmark = pytest.mark.unit


def test_sigma_points_moment_matching(lgssm_2d):
    ukf = UnscentedKalmanFilter(lgssm_2d, alpha=1e-1, beta=2.0, kappa=0.0)
    n = ukf.state_dim
    batch = 3

    m = tf.random.normal([batch, n], dtype=tf.float32)
    P = make_spd(batch, n, eps=1e-3)

    X, Wm, Wc = ukf.generate_sigma_points(m, P)

    m_rec = tf.einsum("i,bin->bn", Wm, X)
    Xc = X - m_rec[:, tf.newaxis, :]
    P_rec = tf.einsum("i,bin,bim->bnm", Wc, Xc, Xc)

    tf.debugging.assert_near(tf.reduce_sum(Wm), 1.0, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(m_rec, m, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P_rec, P, atol=1e-3, rtol=1e-3)
