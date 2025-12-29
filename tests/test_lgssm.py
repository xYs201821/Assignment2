# tests/test_lgssm.py
import inspect
import tensorflow as tf
from src.filter import KalmanFilter
from tests.testhelper import assert_all_finite

def _kalman_filter(kalman, y_traj, joseph=True):
    if "joseph" in inspect.signature(kalman.filter).parameters:
        return kalman.filter(y_traj, joseph=joseph)
    if not joseph:
        kalman.update = kalman.update_naive
    return kalman.filter(y_traj)

def test_lgssm_simulate_shapes_and_nan(lgssm_2d):
    """Basic sanity check for LinearGaussianSSM.simulate."""
    T = 50
    batch_size = 4

    x_traj, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size, ))

    dx = lgssm_2d.state_dim
    dy = lgssm_2d.obs_dim

    # shape check
    assert x_traj.shape == (batch_size, T, dx)
    assert y_traj.shape == (batch_size, T, dy)

    # NaN / Inf check
    assert_all_finite(x_traj, y_traj)
    


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
    kalman = KalmanFilter(lgssm_3d)
    res = _kalman_filter(kalman, y_traj, joseph=True)
    m_filt = res["m_filt"][0]        # [T, dx]
    P_filt = res["P_filt"][0]        # [T, dx, dx]

    mse_tfp  = tf.reduce_mean((m_tfp  - x_true) ** 2)   # scalar
    mse_filt = tf.reduce_mean((m_filt - x_true) ** 2)   # scalar
    tf.debugging.assert_near(mse_filt, mse_tfp, atol=1e-8, rtol=1e-8)


    mse_between = tf.reduce_mean((m_filt - m_tfp) ** 2)
    max_m_diff  = tf.reduce_max(tf.abs(m_filt - m_tfp))
    max_P_diff  = tf.reduce_max(tf.abs(P_filt - P_tfp))
    
    tf.debugging.assert_near(mse_between, 0.0, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(max_m_diff, 0.0, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(max_P_diff, 0.0, atol=1e-4, rtol=1e-4)


def test_kalman_filter_joseph_and_std(lgssm_3d, sim_data_3d):
    """
    Compare Joseph-stabilized covariance update vs standard update.
    They should give (numerically) almost identical results.
    """
    y_traj = sim_data_3d["y_traj"]

    kalman_j = KalmanFilter(lgssm_3d)
    kalman_s = KalmanFilter(lgssm_3d)
    res_j = _kalman_filter(kalman_j, y_traj, joseph=True)
    res_s = _kalman_filter(kalman_s, y_traj, joseph=False)

    m_j = res_j["m_filt"][0]   # [T, dx]
    m_s = res_s["m_filt"][0]
    P_j = res_j["P_filt"][0]   # [T, dx, dx]
    P_s = res_s["P_filt"][0]


    tf.debugging.assert_near(m_j, m_s, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P_j, P_s, atol=1e-4, rtol=1e-4)
