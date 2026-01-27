import pytest
import tensorflow as tf

from src.filters import KalmanFilter

pytestmark = pytest.mark.integration


def test_kf_matches_tfp_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    y_traj = sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_ref_3d

    kf = KalmanFilter(lgssm_3d)
    out = kf.filter(y_traj, joseph=True)

    m_filt = out["m_filt"][0]
    P_filt = out["P_filt"][0]

    tf.debugging.assert_near(m_filt, m_tfp, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P_filt, P_tfp, atol=1e-4, rtol=1e-4)


def test_kf_joseph_matches_naive(lgssm_3d, sim_data_3d):
    y_traj = sim_data_3d["y_traj"]

    kf_j = KalmanFilter(lgssm_3d)
    kf_n = KalmanFilter(lgssm_3d)

    out_j = kf_j.filter(y_traj, joseph=True)
    out_n = kf_n.filter(y_traj, joseph=False)

    tf.debugging.assert_near(out_j["m_filt"], out_n["m_filt"], atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(out_j["P_filt"], out_n["P_filt"], atol=1e-5, rtol=1e-5)
