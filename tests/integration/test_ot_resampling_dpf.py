import pytest
import tensorflow as tf

from src.filters.ot_resampling import OTResamplingDPF
from tests.testhelper import (
    assert_all_finite,
    assert_diagnostics_keys,
    assert_particles_shape,
    assert_weights_valid,
)

pytestmark = pytest.mark.integration


def test_ot_resampling_dpf_runs_lgssm(lgssm_2d):
    """OT-Resampling DPF should run end-to-end on LGSSM."""
    T = 16
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    dpf = OTResamplingDPF(
        lgssm_2d,
        num_particles=96,
        ess_threshold=0.5,
        ot_epsilon=0.25,
        ot_num_iters=60,
        resample="always",
    )
    x_particles, w, diagnostics, parent_indices = dpf.filter(y_traj, resample="always")

    dx = lgssm_2d.state_dim
    N = dpf.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_diagnostics_keys(diagnostics, ["x", "log_w", "log_z", "x_pre", "log_w_pre", "parent_index"])
    tf.debugging.assert_equal(diagnostics["parent_index"].shape, (batch_size, T, N))

    expected_w = tf.fill([batch_size, T, N], 1.0 / float(N))
    tf.debugging.assert_near(w, expected_w, atol=1e-6, rtol=1e-6)
    assert_all_finite(x_particles, w, diagnostics["log_z"], diagnostics["log_w_pre"])


def test_ot_resampling_dpf_auto_mode_emits_valid_diagnostics(lgssm_1d):
    """Auto-resample mode should produce valid outputs and diagnostics."""
    T = 12
    batch_size = 1
    _, y_traj = lgssm_1d.simulate(T=T, shape=(batch_size,))

    dpf = OTResamplingDPF(
        lgssm_1d,
        num_particles=64,
        ess_threshold=0.5,
        ot_epsilon=0.2,
        ot_num_iters=40,
        resample="auto",
    )
    x_particles, w, diagnostics, parent_indices = dpf.filter(y_traj, resample="auto")

    assert x_particles.shape == (batch_size, T, 64, 1)
    assert w.shape == (batch_size, T, 64)
    assert parent_indices.shape == (batch_size, T, 64)
    assert_diagnostics_keys(diagnostics, ["x", "log_w", "log_z", "x_pre", "log_w_pre", "parent_index"])
    assert diagnostics["parent_index"].shape == (batch_size, T, 64)

    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=tf.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_all_finite(diagnostics["log_z"], diagnostics["log_w_pre"])


def test_ot_resampling_dpf_supports_runtime_proposal(lgssm_1d):
    """OT-Resampling DPF should accept proposal via filter interface."""

    class ObservationProposal:
        def sample(self, ssm, x_prev, y_t, seed=None):
            del ssm, seed
            return tf.broadcast_to(y_t[:, tf.newaxis, :1], tf.shape(x_prev))

    T = 10
    batch_size = 2
    _, y_traj = lgssm_1d.simulate(T=T, shape=(batch_size,))

    dpf = OTResamplingDPF(
        lgssm_1d,
        num_particles=48,
        ess_threshold=0.5,
        ot_epsilon=0.2,
        ot_num_iters=40,
        resample="auto",
    )
    x_particles, w, diagnostics, parent_indices = dpf.filter(
        y_traj,
        resample="auto",
        proposal=ObservationProposal(),
    )

    assert x_particles.shape == (batch_size, T, 48, 1)
    assert w.shape == (batch_size, T, 48)
    assert parent_indices.shape == (batch_size, T, 48)
    assert diagnostics["x_pre"].shape == (batch_size, T, 48, 1)

    first_step_particles = diagnostics["x_pre"][:, 0, :, 0]
    first_obs = y_traj[:, 0, 0][:, tf.newaxis]
    tf.debugging.assert_near(first_step_particles, first_obs, atol=1e-6, rtol=1e-6)
    assert_all_finite(x_particles, w, diagnostics["log_z"], diagnostics["log_w_pre"])
