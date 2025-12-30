import tensorflow as tf

from src.flows.edh import EDHFlow
from src.utility import weighted_mean
from tests.testhelper import assert_all_finite


def test_edh_flow_runs(lgssm_2d):
    T = 20
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=20, num_particles=200, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = edh.filter(y_traj, reweight=True)

    dx = lgssm_2d.state_dim
    N = edh.num_particles

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


def test_edh_flow_close_to_tfp_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    # EDH should track the true state similarly to TFP on linear-Gaussian models.
    T = 20
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    edh = EDHFlow(lgssm_3d, num_lambda=20, num_particles=500, ess_threshold=0.5)
    x_particles, w, _, _ = edh.filter(y, reweight="always")

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_edh = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_edh, mse_tfp * 4.0 + 1e-4)


def test_edh_flow_nonlinear_sanity(range_bearing_ssm):
    rb = range_bearing_ssm
    rb.cov_eps_x = tf.convert_to_tensor(rb.motion_model.cov_eps, dtype=tf.float32)
    rb.m0 = tf.constant([1.0, 1.0, 1.0, 0.7], dtype=tf.float32)
    rb.P0 = tf.eye(rb.state_dim, dtype=tf.float32) * 0.1

    _, y = rb.simulate(T=100, shape=(2,))

    edh = EDHFlow(rb, num_lambda=5, num_particles=300, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = edh.filter(y, reweight=True)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess, diagnostics["step_time_s"])
    tf.debugging.assert_equal(tf.shape(parent_indices), tf.shape(w))
