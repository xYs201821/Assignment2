import tensorflow as tf

from src.flows.ledh import LEDHFlow
from tests.testhelper import assert_all_finite


def test_ledh_flow_runs(lgssm_2d):
    T = 20
    batch_size = 2

    _, y_traj = lgssm_2d.simulate(T=T, shape=(batch_size,))

    ledh = LEDHFlow(lgssm_2d, num_lambda=10, num_particles=200, ess_threshold=0.5)
    x_particles, w, diagnostics, parent_indices = ledh.filter(y_traj, reweight=True)

    dx = lgssm_2d.state_dim
    N = ledh.num_particles

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
