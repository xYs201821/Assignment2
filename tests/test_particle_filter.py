import tensorflow as tf

from src.filter import ExtendedKalmanFilter, BootstrapParticleFilter, UnscentedKalmanFilter
from src.utility import weighted_mean
from tests.testhelper import assert_all_finite


def test_particle_filter_runs(sv_model):
    T = 100
    batch_size = 2

    _, y_traj = sv_model.simulate(T=T, shape=(batch_size,))

    pf = BootstrapParticleFilter(sv_model, num_particles=200, ess_threshold=0.5, resample="always")
    x_particles, w, diagnostics, parent_indices = pf.filter(y_traj)

    dx = sv_model.state_dim
    N = pf.num_particles

    assert x_particles.shape == (batch_size, T, N, dx)
    assert w.shape == (batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    tf.debugging.assert_equal(tf.shape(diagnostics["step_time_s"])[0], T)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess, diagnostics["step_time_s"])

    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    tf.debugging.assert_greater_equal(tf.reduce_min(w), 0.0)

    N_float = tf.cast(N, ess.dtype)
    tf.debugging.assert_greater_equal(tf.reduce_min(ess), 1.0 - 1e-3)
    tf.debugging.assert_less_equal(tf.reduce_max(ess), N_float + 1e-3)


def test_particle_filter_sv_compares(sv_model):
    T = 100
    batch_size = 2

    x_traj, y_traj = sv_model.simulate(T=T, shape=(batch_size,))

    pf = BootstrapParticleFilter(sv_model, num_particles=500, ess_threshold=0.5, resample="always")
    x_particles, w, _, _ = pf.filter(y_traj)

    ekf = ExtendedKalmanFilter(sv_model, joseph=True)
    ekf_out = ekf.filter(y_traj)
    ukf = UnscentedKalmanFilter(sv_model, joseph=True)
    ukf_out = ukf.filter(y_traj)
    x_true = x_traj
    x_pf = weighted_mean(x_particles, w, axis=-2)
    x_ekf = ekf_out["m_filt"]
    x_ukf = ukf_out["m_filt"]
    assert_all_finite(x_pf, x_ekf, x_ukf)

    mse_pf = tf.reduce_mean((x_pf - x_true) ** 2)
    mse_ekf = tf.reduce_mean((x_ekf - x_true) ** 2)
    mse_ukf = tf.reduce_mean((x_ukf - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_pf, mse_ekf * 2.0 + 1e-4)
    tf.debugging.assert_less_equal(mse_pf, mse_ukf * 2.0 + 1e-4)

def test_pf_matches_kf_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    # Bootstrap PF should track the true state similarly to TFP on linear-Gaussian models.
    T = 100
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    pf = BootstrapParticleFilter(lgssm_3d, num_particles=800, ess_threshold=0.5, resample="always")
    x_particles, w, _, _ = pf.filter(y)

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_pf = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_pf, mse_tfp * 3.0 + 1e-4)

def test_particle_filter_range_bearing_compares(range_bearing_ssm):
    rb = range_bearing_ssm
    rb.cov_eps_x = tf.convert_to_tensor(rb.motion_model.cov_eps, dtype=tf.float32)
    rb.m0 = tf.constant([1.0, 1.0, 1.0, 0.7], dtype=tf.float32)
    rb.P0 = tf.eye(rb.state_dim, dtype=tf.float32) * 0.1

    T = 40
    batch_size = 2

    x_traj, y_traj = rb.simulate(T=T, shape=(batch_size,))

    pf = BootstrapParticleFilter(rb, num_particles=800, ess_threshold=0.5, resample="always")
    x_particles, w, _, _ = pf.filter(y_traj)

    ekf = ExtendedKalmanFilter(rb, joseph=True)
    ukf = UnscentedKalmanFilter(rb, joseph=True)
    ekf_out = ekf.filter(y_traj)
    ukf_out = ukf.filter(y_traj)

    x_true = x_traj
    x_pf = weighted_mean(x_particles, w, axis=-2)
    x_ekf = ekf_out["m_filt"]
    x_ukf = ukf_out["m_filt"]

    assert_all_finite(x_pf, x_ekf, x_ukf)

    mse_pf = tf.reduce_mean((x_pf - x_true) ** 2)
    mse_ekf = tf.reduce_mean((x_ekf - x_true) ** 2)
    mse_ukf = tf.reduce_mean((x_ukf - x_true) ** 2)
    mse_ref = tf.minimum(mse_ekf, mse_ukf)
    tf.debugging.assert_less_equal(mse_pf, mse_ref * 3.0 + 1e-3)
