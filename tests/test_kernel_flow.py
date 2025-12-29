import tensorflow as tf

from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter
from src.flows.kernel_embedded import KernelParticleFlow
from src.utility import weighted_mean
from tests.testhelper import assert_all_finite


def test_kernel_particle_flow_runs(lgssm_2d):
    T = 10
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=50,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight="never")

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert x_particles.shape == (batch_size, T, N, dx)
    assert w.shape == (batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    tf.debugging.assert_equal(tf.shape(diagnostics["step_time_s"])[0], T)

    assert_all_finite(x_particles, w, diagnostics["step_time_s"])

    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    tf.debugging.assert_greater_equal(tf.reduce_min(w), 0.0)


def test_kernel_flow_step_outputs(lgssm_2d):
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=2, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=50,
    )
    x_prev, log_w_prev, _ = flow._init_particles(y_traj, init_dist=None)
    y_t = y_traj[:, 0, :]

    x_pred, x, log_w, w, parent_indices, m_pred, P_pred = flow.step(
        x_prev,
        log_w_prev,
        y_t,
        reweight=0,
    )

    dx = lgssm_2d.state_dim
    N = flow.num_particles
    assert x_pred.shape == (batch_size, N, dx)
    assert x.shape == (batch_size, N, dx)
    assert log_w.shape == (batch_size, N)
    assert w.shape == (batch_size, N)
    assert parent_indices.shape == (batch_size, N)
    assert m_pred.shape == (batch_size, dx)
    assert P_pred.shape == (batch_size, dx, dx)

    assert_all_finite(x_pred, x, w, m_pred, P_pred)
    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_kernel_flow_sample_outputs(lgssm_2d):
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=2, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=50,
    )
    x_prev, log_w_prev, _ = flow._init_particles(y_traj, init_dist=None)
    y_t = y_traj[:, 0, :]
    w_prev = tf.exp(log_w_prev)

    x_next, log_q, m_pred, P_pred = flow.sample(x_prev, y_t, w=w_prev)

    dx = lgssm_2d.state_dim
    N = flow.num_particles
    assert x_next.shape == (batch_size, N, dx)
    assert log_q.shape == (batch_size, N)
    assert m_pred.shape == (batch_size, dx)
    assert P_pred.shape == (batch_size, dx, dx)

    assert_all_finite(x_next, log_q, m_pred, P_pred)


def test_kernel_flow_close_to_tfp_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    # Kernel flow should track the true state similarly to TFP on linear-Gaussian models.
    T = 50
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    flow = KernelParticleFlow(lgssm_3d, num_lambda=5, num_particles=200)
    x_particles, w, _, _ = flow.filter(y, reweight="never")

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_flow = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_flow, mse_tfp * 2.0 + 1e-4)


def test_kernel_flow_sv_compares(sv_model):
    T = 20
    batch_size = 2

    x_traj, y_traj = sv_model.simulate(T=T, shape=(batch_size,))

    flow = KernelParticleFlow(sv_model, num_lambda=10, num_particles=50)
    x_particles, w, _, _ = flow.filter(y_traj, reweight="never")

    ekf = ExtendedKalmanFilter(sv_model, joseph=True)
    ukf = UnscentedKalmanFilter(sv_model, joseph=True)
    ekf_out = ekf.filter(y_traj)
    ukf_out = ukf.filter(y_traj)

    x_true = x_traj
    x_flow = weighted_mean(x_particles, w, axis=-2)
    x_ekf = ekf_out["m_filt"]
    x_ukf = ukf_out["m_filt"]
    assert_all_finite(x_flow, x_ekf, x_ukf)

    mse_flow = tf.reduce_mean((x_flow - x_true) ** 2)
    mse_ekf = tf.reduce_mean((x_ekf - x_true) ** 2)
    mse_ukf = tf.reduce_mean((x_ukf - x_true) ** 2)
    mse_ref = tf.minimum(mse_ekf, mse_ukf)
    tf.debugging.assert_less_equal(mse_flow, mse_ref * 2.0 + 1e-4)
