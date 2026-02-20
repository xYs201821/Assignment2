import pytest
import tensorflow as tf

from src.filters.diffusion_resampling import DiffusionResamplingDPF
from tests.testhelper import (
    assert_all_finite,
    assert_diagnostics_keys,
    assert_particles_shape,
    assert_weights_valid,
)

pytestmark = pytest.mark.integration


def test_diffusion_resampling_dpf_runs_lgssm(lgssm_2d):
    T = 12
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    dpf = DiffusionResamplingDPF(
        lgssm_2d,
        num_particles=64,
        ess_threshold=0.5,
        diff_a=-1.0,
        diff_T=1.0,
        diff_steps=6,
        diff_ode=True,
        diff_eps=1e-6,
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

    # resample="always" resets output weights to uniform at each step.
    expected_w = tf.fill([batch_size, T, N], 1.0 / float(N))
    tf.debugging.assert_near(w, expected_w, atol=1e-6, rtol=1e-6)

    assert_all_finite(
        x_particles,
        w,
        diagnostics["log_z"],
        diagnostics["log_w_pre"],
    )
