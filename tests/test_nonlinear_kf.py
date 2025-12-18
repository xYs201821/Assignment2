import tensorflow as tf
from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter
from tests.testhelper import assert_all_finite, assert_symmetric, assert_psd, make_spd

def test_ekf_matches_tfp_on_linear(lgssm_3d, sim_data_3d, tfp_ref_3d):
    # test EKF on linear ssm, should be close to TFP's KF
    _, y = sim_data_3d["x_traj"], sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_ref_3d   

    ekf = ExtendedKalmanFilter(lgssm_3d)
    out = ekf.filter(y, joseph=True)

    m = out["m_filt"][0]   # [T, dx]
    P = out["P_filt"][0]   # [T, dx, dx]

    tf.debugging.assert_near(m, m_tfp, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P, P_tfp, atol=1e-4, rtol=1e-4)

def test_ukf_matches_tfp_on_linear(lgssm_3d, sim_data_3d, tfp_ref_3d):
    # test UKF on linear ssm
    _, y = sim_data_3d["x_traj"], sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_ref_3d  

    ukf = UnscentedKalmanFilter(lgssm_3d, alpha=1e-1) # use a larger alpha to make UKF more accurate for linear system
    out = ukf.filter(y, joseph=True)

    m = out["m_filt"][0]   # [T, dx]
    P = out["P_filt"][0]   # [T, dx, dx]

    tf.debugging.assert_near(m, m_tfp, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P, P_tfp, atol=1e-4, rtol=1e-4)

def test_ekf_jacobian_linear_matches_A_C(lgssm_3d):
    ekf = ExtendedKalmanFilter(lgssm_3d)

    dx = lgssm_3d.state_dim
    batch = 4
    x = tf.random.normal([batch, dx], dtype=tf.float32)

    F = ekf._jacobian(lgssm_3d.f, x)  # [batch, dx, dx]
    H = ekf._jacobian(lgssm_3d.h, x)  # [batch, dy, dx]

    tf.debugging.assert_near(F, tf.broadcast_to(lgssm_3d.A[tf.newaxis, :, :], tf.shape(F)),
                             atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(H, tf.broadcast_to(lgssm_3d.C[tf.newaxis, :, :], tf.shape(H)),
                             atol=1e-5, rtol=1e-5)



def test_sigma_points_moment_matching(lgssm_2d):
    ukf = UnscentedKalmanFilter(lgssm_2d, alpha=1e-1, beta=2.0, kappa=0.0)
    n = ukf.state_dim
    batch = 3

    m = tf.random.normal([batch, n], dtype=tf.float32)
    P = make_spd(batch, n, eps=1e-3)

    X, Wm, Wc = ukf.generate_sigma_points(m, P)  # [batch, 2n+1, n]

    m_rec = tf.einsum("i,bin->bn", Wm, X)
    Xc = X - m_rec[:, tf.newaxis, :]
    P_rec = tf.einsum("i,bin,bim->bnm", Wc, Xc, Xc)

    tf.debugging.assert_near(m_rec, m, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(P_rec, P, atol=1e-3, rtol=1e-3)

def test_sv_ekf_ukf_runs(sv_model):
    sv = sv_model
    sv.cov_eps_y = tf.eye(sv.obs_dim, dtype=tf.float32) * 1e-4

    _, y = sv.simulate(T=30, shape=(2, ))

    ekf = ExtendedKalmanFilter(sv)
    ukf = UnscentedKalmanFilter(sv)

    out_e = ekf.filter(y, joseph=True)
    out_u = ukf.filter(y, joseph=True)

    assert_all_finite(out_e["m_filt"], out_e["P_filt"], out_e["cond_P"], out_e["cond_S"])
    assert_all_finite(out_u["m_filt"], out_u["P_filt"], out_u["cond_P"], out_u["cond_S"])

    assert_symmetric(out_e["P_filt"])
    assert_symmetric(out_u["P_filt"])

    assert_psd(out_e["P_filt"], eps=-1e-6)
    assert_psd(out_u["P_filt"], eps=-1e-6)


def test_range_bearing_ekf_ukf_runs(range_bearing_ssm):
    rb = range_bearing_ssm
    rb.cov_eps_x = tf.convert_to_tensor(rb.motion_model.cov_eps, dtype=tf.float32)

    rb.m0 = tf.constant([1.0, 1.0, 1.0, 0.7], dtype=tf.float32)
    rb.P0 = tf.eye(rb.state_dim, dtype=tf.float32) * 0.1

    _, y = rb.simulate(T=30, shape=(2, ))

    ekf = ExtendedKalmanFilter(rb)
    ukf = UnscentedKalmanFilter(rb)

    out_e = ekf.filter(y, joseph=True)
    out_u = ukf.filter(y, joseph=True)

    assert_all_finite(out_e["m_filt"], out_e["P_filt"], out_e["cond_P"], out_e["cond_S"])
    assert_all_finite(out_u["m_filt"], out_u["P_filt"], out_u["cond_P"], out_u["cond_S"])

    assert_symmetric(out_e["P_filt"])
    assert_symmetric(out_u["P_filt"])

    assert_psd(out_e["P_filt"], eps=-1e-6)
    assert_psd(out_u["P_filt"], eps=-1e-6)