import numpy as np
import tensorflow as tf

from tests.testhelper import assert_all_finite

def test_sv_simulate_shapes_and_nan(sv_model):
    T = 50
    batch_size = 8

    x_traj, y_traj = sv_model.simulate(T=T, batch_size=batch_size, seed=0)

    dx = sv_model.state_dim  
    dy = sv_model.obs_dim    

    assert x_traj.shape == (batch_size, T, dx)
    assert y_traj.shape == (batch_size, T, dy)

    assert_all_finite(x_traj, y_traj)


def test_sv_stationary_distribution_timewise(sv_model):
    """Check the empirical distribution of log-volatility x_t
    should be close to the stationary distribution N(0, sigma^2 / (1 - alpha^2)),
    using batch statistics at each time step."""
    alpha = float(sv_model.alpha.numpy())
    sigma = float(sv_model.sigma.numpy())

    var_theoretical = sigma ** 2 / (1.0 - alpha ** 2)

    T = 250
    batch_size = 256
    x_traj, y_traj = sv_model.simulate(T=T, batch_size=batch_size, seed=123)

    x = x_traj[..., 0]

    mean_t = tf.reduce_mean(x, axis=0)          # [T]
    var_t = tf.math.reduce_variance(x, axis=0)  # [T]

    burn_in = 100
    mean_ss = mean_t[burn_in:]    # [T - burn_in]
    var_ss = var_t[burn_in:]      # [T - burn_in]

    # mean statistics [T - burn_in]
    mean_avg_bias = float(tf.reduce_mean(mean_ss).numpy())
    mean_max_abs  = float(tf.reduce_max(tf.abs(mean_ss)).numpy())

    # var statistics [T - burn_in]
    rel_err_var_time = tf.abs(var_ss - var_theoretical) / var_theoretical
    rel_err_var_avg  = float(tf.reduce_mean(rel_err_var_time).numpy())
    rel_err_var_max  = float(tf.reduce_max(rel_err_var_time).numpy())

    assert abs(mean_avg_bias) < 3.0 * float(tf.sqrt(var_theoretical / batch_size)) # 5 sigma as threshold

    assert mean_max_abs < 4.0 * float(tf.sqrt(var_theoretical / batch_size))

    assert rel_err_var_avg < 3.0 * float(tf.sqrt(2.0 / (batch_size - 1.0)))

    assert rel_err_var_max < 4.0 * float(tf.sqrt(2.0 / (batch_size - 1.0)))
