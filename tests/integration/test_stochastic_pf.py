import numpy as np
import pytest
import tensorflow as tf

from src.flows.stochastic_pf import StochasticParticleFlow
from tests.testhelper import assert_all_finite, assert_step_time_shape

pytestmark = pytest.mark.integration


def test_stochastic_pf_runs_lgssm_no_diffusion(lgssm_2d):
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

    assert x_particles.shape == (batch_size, T, N, dx)
    assert w.shape == (batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_step_time_shape(diagnostics["step_time_s"], T)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess, diagnostics["step_time_s"])
    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    tf.debugging.assert_greater_equal(tf.reduce_min(w), 0.0)


def test_stochastic_pf_sample_outputs(lgssm_2d):
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

    x_next, log_q, m_pred, P_pred = flow.sample(x_prev, y_t, w=w_prev)

    dx = lgssm_2d.state_dim
    N = flow.num_particles
    assert x_next.shape == (batch_size, N, dx)
    assert log_q.shape == (batch_size, N)
    assert m_pred.shape == (batch_size, dx)
    assert P_pred.shape == (batch_size, dx, dx)
    assert_all_finite(x_next, log_q, m_pred, P_pred)


def test_stochastic_pf_runs_lgssm_with_diffusion(lgssm_2d):
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
    assert x_particles.shape == (batch_size, T, N, dx)
    assert w.shape == (batch_size, T, N)
    assert parent_indices.shape == (batch_size, T, N)
    assert_step_time_shape(diagnostics["step_time_s"], T)

    ess = 1.0 / tf.reduce_sum(tf.square(w), axis=-1)
    assert_all_finite(x_particles, w, ess, diagnostics["step_time_s"])
    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    tf.debugging.assert_greater_equal(tf.reduce_min(w), 0.0)
