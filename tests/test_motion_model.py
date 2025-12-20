import tensorflow as tf


def test_ctrv_motion_model_shapes(constant_turn_rate_motion_model):
    mm = constant_turn_rate_motion_model
    x = tf.zeros([3, mm.state_dim], dtype=tf.float32)
    y = mm.f(x)
    assert y.shape == x.shape


def test_ctrv_motion_model_zero_omega(constant_turn_rate_motion_model):
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
