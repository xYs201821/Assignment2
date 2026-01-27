import numpy as np
import pytest
import tensorflow as tf

from src.motion_model import ConstantVelocityMotionModel

pytestmark = pytest.mark.unit


def test_constant_velocity_A_structure(constant_velocity_motion_model):
    mm = constant_velocity_motion_model
    dt = float(mm.dt.numpy())
    expected = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(mm.A.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_constant_velocity_invalid_cov_eps():
    cov_bad = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        _ = ConstantVelocityMotionModel(dt=0.1, cov_eps=cov_bad)


def test_constant_velocity_f_matches_A(constant_velocity_motion_model):
    mm = constant_velocity_motion_model
    x = tf.constant([[1.0, -2.0, 0.5, -0.5]], dtype=tf.float32)
    y = mm.f(x)
    expected = tf.einsum("ij,bj->bi", mm.A, x)
    tf.debugging.assert_near(y, expected, atol=1e-6, rtol=1e-6)


def test_constant_turn_rate_motion_model_zero_omega(constant_turn_rate_motion_model):
    mm = constant_turn_rate_motion_model
    x = tf.constant([[1.0, -2.0, 3.0, 0.5, 0.0]], dtype=tf.float32)
    y = mm.f(x)

    dt = tf.cast(mm.dt, tf.float32)
    expected_x = x[:, 0] + x[:, 2] * dt * tf.cos(x[:, 3])
    expected_y = x[:, 1] + x[:, 2] * dt * tf.sin(x[:, 3])

    tf.debugging.assert_near(y[:, 0], expected_x, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 1], expected_y, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 2], x[:, 2], atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 3], x[:, 3], atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 4], x[:, 4], atol=1e-5, rtol=1e-5)


def test_constant_turn_rate_motion_model_turning(constant_turn_rate_motion_model):
    mm = constant_turn_rate_motion_model
    x = tf.constant([[1.0, 2.0, 3.0, 0.25, 0.2]], dtype=tf.float32)
    y = mm.f(x)

    px, py, v, psi, omega = tf.unstack(x, axis=-1)
    dt = tf.cast(mm.dt, tf.float32)
    psi_next = psi + omega * dt
    radius = v / omega
    px_expected = px + radius * (tf.sin(psi_next) - tf.sin(psi))
    py_expected = py + radius * (-tf.cos(psi_next) + tf.cos(psi))

    tf.debugging.assert_near(y[:, 0], px_expected, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 1], py_expected, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(y[:, 3], psi_next, atol=1e-5, rtol=1e-5)
