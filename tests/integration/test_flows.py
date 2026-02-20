import pytest
import tensorflow as tf

from src.flows.edh import EDHFlow
from src.flows.kernel_embedded import KernelParticleFlow
from src.flows.ledh import LEDHFlow
from src.flows.stochastic_pf import StochasticParticleFlow
from src.flows.beta_schedule import BetaScheduleConfig
from src.utility import weighted_mean
from tests.testhelper import (
    assert_all_finite,
    assert_weights_valid,
    assert_particles_shape,
    assert_diagnostics_keys,
)

pytestmark = pytest.mark.integration


# =============================================================================
# Basic flow filter tests - verify outputs are valid
# =============================================================================

def test_edh_flow_runs_lgssm(lgssm_2d):
    T = 15
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=8, num_particles=150, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = edh.filter(y_traj, reweight=True)

    dx = lgssm_2d.state_dim
    N = edh.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_diagnostics_keys(diagnostics, ["x", "log_w", "log_z", "x_pre", "log_w_pre", "parent_index"])

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


def test_ledh_flow_runs_lgssm(lgssm_2d):
    T = 15
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    ledh = LEDHFlow(lgssm_2d, num_lambda=8, num_particles=150, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = ledh.filter(y_traj, reweight=True)

    dx = lgssm_2d.state_dim
    N = ledh.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_diagnostics_keys(diagnostics, ["x", "log_w", "log_z", "x_pre", "log_w_pre", "parent_index"])

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


@pytest.mark.parametrize("flow_cls", [EDHFlow, LEDHFlow])
def test_optimal_beta_schedule_runs_lgssm(flow_cls, lgssm_2d):
    T = 5
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))
    beta_schedule = BetaScheduleConfig(mode="optimal", mu=0.2, guard=False)
    flow = flow_cls(
        lgssm_2d,
        num_lambda=4,
        num_particles=60,
        ess_threshold=0.5,
        beta_schedule=beta_schedule,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight=True)

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


def test_kernel_flow_runs_lgssm(lgssm_2d):
    T = 10
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=80,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight="never")

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_diagnostics_keys(diagnostics, ["x", "log_w", "log_z", "x_pre", "log_w_pre", "parent_index"])

    assert_all_finite(x_particles, w)


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

    x_next, log_q = flow.sample(x_prev, y_t, w=w_prev)

    dx = lgssm_2d.state_dim
    N = flow.num_particles
    assert x_next.shape == (batch_size, N, dx)
    assert log_q.shape == (batch_size, N)

    assert_all_finite(x_next, log_q)


# =============================================================================
# Accuracy tests - verify flows achieve reasonable accuracy
# =============================================================================

def test_edh_flow_close_to_tfp_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    """EDH flow should achieve accuracy comparable to Kalman filter on LGSSM."""
    T = 20
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    edh = EDHFlow(lgssm_3d, num_lambda=8, num_particles=200, ess_threshold=0.5)
    x_particles, w, _, _ = edh.filter(y, reweight="always")

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_edh = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_edh, mse_tfp * 5.0 + 1e-3)


def test_ledh_flow_close_to_tfp_on_lgssm(lgssm_3d, sim_data_3d, tfp_ref_3d):
    """LEDH flow should achieve accuracy comparable to Kalman filter on LGSSM."""
    T = 20
    y = sim_data_3d["y_traj"][:, :T, :]
    x_true = sim_data_3d["x_traj"][:, :T, :]
    m_tfp, _ = tfp_ref_3d
    m_ref = m_tfp[:T]

    ledh = LEDHFlow(lgssm_3d, num_lambda=8, num_particles=200, ess_threshold=0.5)
    x_particles, w, _, _ = ledh.filter(y, reweight="always")

    x_mean = weighted_mean(x_particles, w, axis=-2)
    mse_ledh = tf.reduce_mean((x_mean - x_true) ** 2)
    mse_tfp = tf.reduce_mean((m_ref - x_true) ** 2)
    tf.debugging.assert_less_equal(mse_ledh, mse_tfp * 5.0 + 1e-3)


def test_edh_flow_nonlinear_sanity(range_bearing_ssm):
    """EDH flow should produce finite outputs on nonlinear SSM."""
    rb = range_bearing_ssm
    rb.cov_eps_x = tf.convert_to_tensor(rb.motion_model.cov_eps, dtype=tf.float32)
    rb.m0 = tf.constant([1.0, 1.0, 1.0, 0.7], dtype=tf.float32)
    rb.P0 = tf.eye(rb.state_dim, dtype=tf.float32) * 0.1

    _, y = rb.simulate(T=20, shape=(2,))

    edh = EDHFlow(rb, num_lambda=5, num_particles=120, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = edh.filter(y, reweight=True)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)
    tf.debugging.assert_equal(tf.shape(parent_indices), tf.shape(w))


# =============================================================================
# Edge cases and boundary tests
# =============================================================================

@pytest.mark.parametrize("num_lambda", [1, 2, 5, 20])
def test_edh_flow_various_num_lambda(lgssm_2d, num_lambda):
    """EDH should work with various num_lambda values."""
    T = 5
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=num_lambda, num_particles=50)
    x_particles, w, _, _ = edh.filter(y_traj, reweight=False)

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 50)


@pytest.mark.parametrize("num_particles", [10, 50, 200])
def test_edh_flow_various_num_particles(lgssm_2d, num_particles):
    """EDH should work with various particle counts."""
    T = 5
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=5, num_particles=num_particles)
    x_particles, w, _, _ = edh.filter(y_traj, reweight=False)

    dx = lgssm_2d.state_dim
    assert_particles_shape(x_particles, batch_size, T, num_particles, dx)
    assert_weights_valid(w, batch_size, T, num_particles)
    assert_all_finite(x_particles, w)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_edh_flow_various_batch_sizes(lgssm_2d, batch_size):
    """EDH should work with various batch sizes."""
    T = 5
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=5, num_particles=50)
    x_particles, w, _, _ = edh.filter(y_traj, reweight=False)

    dx = lgssm_2d.state_dim
    assert_particles_shape(x_particles, batch_size, T, 50, dx)
    assert_weights_valid(w, batch_size, T, 50)


@pytest.mark.parametrize("flow_cls", [EDHFlow, LEDHFlow])
def test_flow_single_time_step(flow_cls, lgssm_2d):
    """Flows should handle T=1 observation sequence."""
    T = 1
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = flow_cls(lgssm_2d, num_lambda=5, num_particles=50)
    x_particles, w, _, _ = flow.filter(y_traj, reweight=False)

    dx = lgssm_2d.state_dim
    assert_particles_shape(x_particles, batch_size, T, 50, dx)
    assert_weights_valid(w, batch_size, T, 50)


# =============================================================================
# Reweight/resample mode tests
# =============================================================================

@pytest.mark.parametrize("reweight", [True, False, "always", "never"])
def test_edh_flow_reweight_modes(lgssm_2d, reweight):
    """EDH should work with different reweight settings."""
    T = 10
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=5, num_particles=80)
    x_particles, w, _, _ = edh.filter(y_traj, reweight=reweight)

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 80)


@pytest.mark.parametrize("resample", [True, False, "auto", "never"])
def test_edh_flow_resample_modes(lgssm_2d, resample):
    """EDH should work with different resample settings."""
    T = 10
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    edh = EDHFlow(lgssm_2d, num_lambda=5, num_particles=80)
    x_particles, w, _, _ = edh.filter(y_traj, reweight=True, resample=resample)

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 80)


# =============================================================================
# Kernel flow specific tests
# =============================================================================

@pytest.mark.parametrize("kernel_type", ["scalar", "diag"])
def test_kernel_flow_kernel_types(lgssm_2d, kernel_type):
    """Kernel flow should work with different kernel types."""
    T = 5
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=50,
        kernel_type=kernel_type,
    )
    x_particles, w, _, _ = flow.filter(y_traj, reweight="never")

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 50)


@pytest.mark.parametrize("ll_grad_mode", ["linearized", "dist"])
def test_kernel_flow_grad_modes(lgssm_2d, ll_grad_mode):
    """Kernel flow should work with different gradient modes."""
    T = 5
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = KernelParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=50,
        ll_grad_mode=ll_grad_mode,
    )
    x_particles, w, _, _ = flow.filter(y_traj, reweight="never")

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 50)


# =============================================================================
# Stochastic PF tests
# =============================================================================

def test_stochastic_pf_runs_lgssm(lgssm_2d):
    """Stochastic PF should run on LGSSM."""
    T = 10
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=5,
        num_particles=80,
        diffusion=None,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight="never")

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert_all_finite(x_particles, w)
