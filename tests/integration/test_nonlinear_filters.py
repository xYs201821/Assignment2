import pytest
import tensorflow as tf

from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter
from tests.testhelper import assert_all_finite, assert_psd, assert_symmetric

pytestmark = pytest.mark.integration


def test_ekf_matches_tfp_on_linear(lgssm_3d, sim_data_3d, tfp_ref_3d):
    y = sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_ref_3d

    ekf = ExtendedKalmanFilter(lgssm_3d, joseph=True)
    out = ekf.filter(y)

    m = out["m_filt"][0]
    P = out["P_filt"][0]

    tf.debugging.assert_near(m, m_tfp, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P, P_tfp, atol=1e-4, rtol=1e-4)


def test_ukf_matches_tfp_on_linear(lgssm_3d, sim_data_3d, tfp_ref_3d):
    y = sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_ref_3d

    ukf = UnscentedKalmanFilter(lgssm_3d, alpha=1e-1, joseph=True)
    out = ukf.filter(y)

    m = out["m_filt"][0]
    P = out["P_filt"][0]

    tf.debugging.assert_near(m, m_tfp, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P, P_tfp, atol=1e-4, rtol=1e-4)


def test_ekf_ukf_runs_sv(sv_model):
    sv = sv_model
    sv.cov_eps_y = tf.eye(sv.obs_dim, dtype=tf.float32) * 1e-4

    _, y = sv.simulate(T=25, shape=(2,))

    ekf = ExtendedKalmanFilter(sv, joseph=True)
    ukf = UnscentedKalmanFilter(sv, joseph=True)

    out_e = ekf.filter(y)
    out_u = ukf.filter(y)

    assert_all_finite(out_e["m_filt"], out_e["P_filt"], out_e["cond_P"])
    assert_all_finite(out_u["m_filt"], out_u["P_filt"], out_u["cond_P"])

    assert_symmetric(out_e["P_filt"])
    assert_symmetric(out_u["P_filt"])

    assert_psd(out_e["P_filt"], eps=-1e-6)
    assert_psd(out_u["P_filt"], eps=-1e-6)


def test_ekf_ukf_runs_range_bearing(range_bearing_ssm):
    rb = range_bearing_ssm
    rb.cov_eps_x = tf.convert_to_tensor(rb.motion_model.cov_eps, dtype=tf.float32)
    rb.m0 = tf.constant([1.0, 1.0, 1.0, 0.7], dtype=tf.float32)
    rb.P0 = tf.eye(rb.state_dim, dtype=tf.float32) * 0.1

    _, y = rb.simulate(T=25, shape=(2,))

    ekf = ExtendedKalmanFilter(rb, joseph=True)
    ukf = UnscentedKalmanFilter(rb, alpha=1e-1, joseph=True)

    out_e = ekf.filter(y)
    out_u = ukf.filter(y)

    assert_all_finite(out_e["m_filt"], out_e["P_filt"], out_e["cond_P"])
    assert_all_finite(out_u["m_filt"], out_u["P_filt"], out_u["cond_P"])

    assert_symmetric(out_e["P_filt"])
    assert_symmetric(out_u["P_filt"])

    assert_psd(out_e["P_filt"], eps=-1e-6)
    assert_psd(out_u["P_filt"], eps=-1e-6)
