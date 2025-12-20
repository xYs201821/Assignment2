# tests/test_rangebear.py
import numpy as np
import tensorflow as tf
from tests.testhelper import assert_all_finite

def test_range_bearing_ssm(range_bearing_ssm):
    T = 80
    batch_size = 16

    x_traj, y_traj = range_bearing_ssm.simulate(T=T, shape=(batch_size, ))

    dx = range_bearing_ssm.state_dim
    dy = range_bearing_ssm.obs_dim

    assert x_traj.shape == (batch_size, T, dx)
    assert y_traj.shape == (batch_size, T, dy)

    # NaN / Inf checks
    assert_all_finite(x_traj, y_traj)

    # Geometric checks
    rng = y_traj[..., 0]      # [batch, T]
    bearing = y_traj[..., 1]  # [batch, T]

    # positive range
    mean_rng = tf.reduce_mean(rng)
    assert mean_rng > 0.0
    sigma_r = tf.sqrt(range_bearing_ssm.cov_eps_y[0, 0])
    assert tf.reduce_min(rng) > -3.0 * sigma_r

    # bearing in [-pi, pi]
    pi = tf.constant(np.pi, dtype=tf.float32)
    assert tf.reduce_max(bearing) <= pi
    assert tf.reduce_min(bearing) >= -pi
