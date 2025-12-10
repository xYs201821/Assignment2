# tests/test_lgssm.py
import tensorflow as tf
from src.filter import Filter


def test_lgssm_simulate_shapes_and_nan(lgssm_2d):
    """Basic sanity check for LinearGaussianSSM.simulate."""
    T = 50
    batch_size = 4

    x_traj, y_traj = lgssm_2d.simulate(T=T, batch_size=batch_size, seed=0)

    dx = lgssm_2d.state_dim
    dy = lgssm_2d.obs_dim

    # shape check
    assert x_traj.shape == (batch_size, T, dx)
    assert y_traj.shape == (batch_size, T, dy)

    # NaN / Inf check
    assert not tf.reduce_any(tf.math.is_nan(x_traj))
    assert not tf.reduce_any(tf.math.is_nan(y_traj))
    assert not tf.reduce_any(tf.math.is_inf(x_traj))
    assert not tf.reduce_any(tf.math.is_inf(y_traj))


def test_kalman_filter_close_to_tfp(lgssm_3d, sim_data_3d, tfp_ref_3d):
    """
    Test the accuracy of our Kalman filter implementation by:
      1) Comparing filtered means to the true state trajectory.
      2) Ensuring implementation is numerically close to TFP's LGSSM filter.
    """
    x_traj = sim_data_3d["x_traj"]   # [1, T, dx]
    y_traj = sim_data_3d["y_traj"]   # [1, T, dy]
    x_true = x_traj[0]               # [T, dx] 

    # TFP reference (ground truth Kalman recursion)
    m_tfp, P_tfp = tfp_ref_3d        # m_tfp: [T, dx], P_tfp: [T, dx, dx]

    # Our Kalman filter (Joseph form)
    kalman = Filter(lgssm_3d)
    res = kalman.filter(y_traj, joseph=True)
    m_filt = res["m_filt"][0]        # [T, dx]
    P_filt = res["P_filt"][0]        # [T, dx, dx]

    mse_tfp  = tf.reduce_mean((m_tfp  - x_true) ** 2)   # scalar
    mse_filt = tf.reduce_mean((m_filt - x_true) ** 2)   # scalar

    rel_diff = tf.abs(mse_filt - mse_tfp) / (mse_tfp + 1e-12)
    assert rel_diff < 1e-4


    mse_between = tf.reduce_mean((m_filt - m_tfp) ** 2)
    max_m_diff  = tf.reduce_max(tf.abs(m_filt - m_tfp))
    max_P_diff  = tf.reduce_max(tf.abs(P_filt - P_tfp))

    assert mse_between < 1e-8
    assert max_m_diff  < 1e-4
    assert max_P_diff  < 1e-4


def test_kalman_filter_joseph_and_std(lgssm_3d, sim_data_3d):
    """
    Compare Joseph-stabilized covariance update vs standard update.
    They should give (numerically) almost identical results.
    """
    y_traj = sim_data_3d["y_traj"]

    kalman = Filter(lgssm_3d)
    res_j = kalman.filter(y_traj, joseph=True)
    res_s = kalman.filter(y_traj, joseph=False)

    m_j = res_j["m_filt"][0]   # [T, dx]
    m_s = res_s["m_filt"][0]
    P_j = res_j["P_filt"][0]   # [T, dx, dx]
    P_s = res_s["P_filt"][0]

    max_diff_m = tf.reduce_max(tf.abs(m_j - m_s))
    max_diff_P = tf.reduce_max(tf.abs(P_j - P_s))

    assert max_diff_m < 1e-5
    assert max_diff_P < 1e-5
