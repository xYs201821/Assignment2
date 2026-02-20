import numpy as np
import pytest
import tensorflow as tf

from src.flows.stochastic_pf import StochasticParticleFlow
from src.flows.beta_schedule import BetaScheduleConfig
from tests.testhelper import (
    assert_all_finite,
    assert_weights_valid,
    assert_particles_shape,
)

pytestmark = pytest.mark.integration


def test_stochastic_pf_runs_lgssm_no_diffusion(lgssm_2d):
    """Stochastic PF should run without diffusion."""
    T = 20
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=6,
        num_particles=120,
        diffusion=None,
        ess_threshold=0.5,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight="never")

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


def test_stochastic_pf_sample_outputs(lgssm_2d):
    """Stochastic PF sample() should return correct shapes."""
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=2, shape=(batch_size,))

    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=4,
        num_particles=50,
        diffusion=None,
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


def test_stochastic_pf_runs_lgssm_with_diffusion(lgssm_2d):
    """Stochastic PF should run with diffusion matrix."""
    T = 20
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    dx = lgssm_2d.state_dim
    diffusion = 0.05 * np.eye(dx, dtype=np.float32)

    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=6,
        num_particles=120,
        diffusion=diffusion,
        ess_threshold=0.5,
    )
    x_particles, w, diagnostics, parent_indices = flow.filter(y_traj, reweight="never")

    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess)


def test_stochastic_pf_with_optimal_beta(lgssm_2d):
    """Stochastic PF should work with optimal beta schedule."""
    T = 10
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    beta_schedule = BetaScheduleConfig(mode="optimal", mu=0.2, guard=False)
    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=6,
        num_particles=80,
        diffusion=None,
        beta_schedule=beta_schedule,
    )
    x_particles, w, _, _ = flow.filter(y_traj, reweight="never")

    dx = lgssm_2d.state_dim
    N = flow.num_particles

    assert_particles_shape(x_particles, batch_size, T, N, dx)
    assert_weights_valid(w, batch_size, T, N)
    assert_all_finite(x_particles, w)


@pytest.mark.parametrize("num_lambda", [3, 5, 20])
def test_stochastic_pf_various_num_lambda(lgssm_2d, num_lambda):
    """Stochastic PF should work with various num_lambda values."""
    T = 5
    batch_size = 2
    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    flow = StochasticParticleFlow(
        lgssm_2d,
        num_lambda=num_lambda,
        num_particles=50,
        diffusion=None,
    )
    x_particles, w, _, _ = flow.filter(y_traj, reweight="never")

    assert_all_finite(x_particles, w)
    assert_weights_valid(w, batch_size, T, 50)
