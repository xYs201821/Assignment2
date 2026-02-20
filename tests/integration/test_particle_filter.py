import pytest
import tensorflow as tf

from src.filter import BootstrapParticleFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from src.utility import weighted_mean
from tests.testhelper import (
    assert_all_finite,
    assert_weights_valid,
    assert_particles_shape,
)

pytestmark = pytest.mark.integration


def test_particle_filter_runs_sv(sv_model):
    """Bootstrap PF should run on stochastic volatility model."""
    T = 40
    batch_size = 2

    _, y_traj = sv_model.simulate(T=T, shape=(batch_size,))

    pf = BootstrapParticleFilter(sv_model, num_particles=200, ess_threshold=0.5, resample="always")
    x_particles, w, diagnostics, parent_indices = pf.filter(y_traj, resample="always")

    dx = sv_model.state_dim
    N = pf.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


def test_particle_filter_sv_compares(sv_model):
    T = 40
    batch_size = 2

    x_traj, y_traj = sv_model.simulate(T=T, shape=(batch_size,))

    pf = BootstrapParticleFilter(sv_model, num_particles=300, ess_threshold=0.5, resample="always")
    x_particles, w, _, _ = pf.filter(y_traj, resample="always")

    ekf = ExtendedKalmanFilter(sv_model, joseph=True)
    ukf = UnscentedKalmanFilter(sv_model, joseph=True)
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
    tf.debugging.assert_less_equal(mse_pf, mse_ref * 4.0 + 1e-3)


def test_pf_matches_kf_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    T = 30
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    pf = BootstrapParticleFilter(lgssm_3d, num_particles=300, ess_threshold=0.5, resample="always")
    x_particles, w, _, _ = pf.filter(y, resample="always")

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_pf = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_pf, mse_tfp * 4.0 + 1e-3)
