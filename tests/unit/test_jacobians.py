import pytest
import tensorflow as tf

import src.dtype_config as _dc
from src.filter import ExtendedKalmanFilter

pytestmark = pytest.mark.unit


def test_ekf_jacobian_linear_matches_A_C(lgssm_3d):
    ekf = ExtendedKalmanFilter(lgssm_3d)

    dx = lgssm_3d.state_dim
    batch = 4
    x = tf.random.normal([batch, dx], dtype=_dc.DTYPE)

    F, _ = ekf._jacobian(lgssm_3d.f, x)
    H, _ = ekf._jacobian(lgssm_3d.h, x)

    tf.debugging.assert_near(
        F,
        tf.broadcast_to(lgssm_3d.A[tf.newaxis, :, :], tf.shape(F)),
        atol=1e-3,
        rtol=1e-3,
    )
    tf.debugging.assert_near(
        H,
        tf.broadcast_to(lgssm_3d.C[tf.newaxis, :, :], tf.shape(H)),
        atol=1e-3,
        rtol=1e-3,
    )
