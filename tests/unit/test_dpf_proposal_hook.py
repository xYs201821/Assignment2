import pytest
import tensorflow as tf

from src.filters.soft_resampling import SoftResamplingDPF
from tests.testhelper import assert_all_finite

pytestmark = pytest.mark.unit


def test_dpf_step_uses_default_bootstrap_proposal(lgssm_1d):
    dpf = SoftResamplingDPF(lgssm_1d, num_particles=16, resample="never")
    x_prev = tf.random.normal([2, 16, 1], dtype=tf.float32)
    log_w_prev = tf.fill([2, 16], -tf.math.log(tf.constant(16.0, dtype=tf.float32)))
    y_t = tf.random.normal([2, 1], dtype=tf.float32)

    x_pre, x_t, log_w, w, parent, log_w_pre, logz_t = dpf.step(
        x_prev,
        log_w_prev,
        y_t,
        resample="never",
    )

    assert x_pre.shape == (2, 16, 1)
    assert x_t.shape == (2, 16, 1)
    assert log_w.shape == (2, 16)
    assert w.shape == (2, 16)
    assert parent.shape == (2, 16)
    assert log_w_pre.shape == (2, 16)
    assert logz_t.shape == (2,)
    assert_all_finite(x_pre, x_t, log_w, w, log_w_pre, logz_t)


def test_dpf_step_supports_callable_proposal_with_log_q(lgssm_1d):
    def proposal_callable(ssm, x_prev, y_t, seed=None):
        del seed
        x_prop = x_prev + 0.2 * y_t[:, tf.newaxis, :1]
        log_q = ssm.transition_dist(x_prev).log_prob(x_prop)
        return x_prop, log_q

    dpf = SoftResamplingDPF(
        lgssm_1d,
        num_particles=12,
        resample="never",
        proposal=proposal_callable,
    )

    x_prev = tf.random.normal([1, 12, 1], dtype=tf.float32)
    log_w_prev = tf.fill([1, 12], -tf.math.log(tf.constant(12.0, dtype=tf.float32)))
    y_t = tf.constant([[1.5]], dtype=tf.float32)

    x_pre, *_ = dpf.step(x_prev, log_w_prev, y_t, resample="never")
    expected = x_prev + 0.2 * y_t[:, tf.newaxis, :1]
    tf.debugging.assert_near(x_pre, expected, atol=1e-6, rtol=1e-6)


def test_dpf_filter_supports_object_proposal(lgssm_1d):
    class ObservationProposal:
        def sample(self, ssm, x_prev, y_t, seed=None):
            del ssm, seed
            return tf.broadcast_to(y_t[:, tf.newaxis, :1], tf.shape(x_prev))

    dpf = SoftResamplingDPF(lgssm_1d, num_particles=10, resample="never")
    _, y_traj = lgssm_1d.simulate(T=5, shape=(2,))
    x_seq, w_seq, diagnostics, _ = dpf.filter(
        y_traj,
        resample="never",
        proposal=ObservationProposal(),
    )

    first_step_particles = x_seq[:, 0, :, 0]
    first_obs = y_traj[:, 0, 0][:, tf.newaxis]
    tf.debugging.assert_near(first_step_particles, first_obs, atol=1e-6, rtol=1e-6)
    assert_all_finite(x_seq, w_seq, diagnostics["log_z"], diagnostics["log_w_pre"])
