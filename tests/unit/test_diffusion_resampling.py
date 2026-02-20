import pytest
import tensorflow as tf

from src.filters.diffusion_resampling import DiffusionResamplingDPF
from tests.testhelper import assert_all_finite

pytestmark = pytest.mark.unit


def test_diffusion_resampling_step_outputs_uniform_weights_and_valid_shapes(lgssm_1d):
    dpf = DiffusionResamplingDPF(
        lgssm_1d,
        num_particles=5,
        ess_threshold=0.5,
        diff_a=-1.0,
        diff_T=1.0,
        diff_steps=4,
        diff_ode=True,
        resample="always",
    )
    x = tf.constant(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[4.0], [3.0], [2.0], [1.0], [0.0]],
        ],
        dtype=tf.float32,
    )
    log_w = tf.math.log(
        tf.constant(
            [
                [0.70, 0.10, 0.10, 0.05, 0.05],
                [0.10, 0.10, 0.20, 0.30, 0.30],
            ],
            dtype=tf.float32,
        )
    )
    x_new, log_w_new, parent_idx = dpf.resample_step(x, log_w)

    assert x_new.shape == x.shape
    assert log_w_new.shape == log_w.shape
    assert parent_idx.shape == (2, 5)
    tf.debugging.assert_near(
        tf.exp(log_w_new),
        tf.fill([2, 5], tf.constant(1.0 / 5.0, dtype=tf.float32)),
        atol=1e-6,
        rtol=1e-6,
    )
    tf.debugging.assert_greater_equal(parent_idx, 0)
    tf.debugging.assert_less(parent_idx, 5)
    assert_all_finite(x_new, log_w_new)


def test_diffusion_resampling_step_supports_pathwise_gradients(lgssm_1d):
    dpf = DiffusionResamplingDPF(
        lgssm_1d,
        num_particles=6,
        ess_threshold=0.5,
        diff_a=-1.0,
        diff_T=1.0,
        diff_steps=5,
        diff_ode=True,
        resample="always",
    )
    x = tf.Variable(tf.random.normal([2, 6, 3], dtype=tf.float32))
    log_w_raw = tf.Variable(tf.random.normal([2, 6], dtype=tf.float32))

    with tf.GradientTape() as tape:
        x_new, _, _ = dpf.resample_step(x, log_w_raw)
        loss = tf.reduce_sum(tf.square(x_new))

    grad_x, grad_log_w = tape.gradient(loss, [x, log_w_raw])
    assert grad_x is not None
    assert grad_log_w is not None
    assert_all_finite(grad_x, grad_log_w)
    assert tf.reduce_max(tf.abs(grad_x)).numpy() > 0.0
    assert tf.reduce_max(tf.abs(grad_log_w)).numpy() > 0.0


def test_diffusion_resampling_dpf_resample_step_shapes(lgssm_1d):
    dpf = DiffusionResamplingDPF(
        lgssm_1d,
        num_particles=5,
        ess_threshold=0.5,
        diff_a=-1.0,
        diff_T=1.0,
        diff_steps=4,
        diff_ode=True,
        resample="always",
    )
    x = tf.constant(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[4.0], [3.0], [2.0], [1.0], [0.0]],
        ],
        dtype=tf.float32,
    )
    log_w = tf.math.log(
        tf.constant(
            [
                [0.70, 0.10, 0.10, 0.05, 0.05],
                [0.10, 0.10, 0.20, 0.30, 0.30],
            ],
            dtype=tf.float32,
        )
    )

    x_new, log_w_new, parent_idx = dpf.resample_step(x, log_w)

    assert x_new.shape == x.shape
    assert log_w_new.shape == (2, 5)
    assert parent_idx.shape == (2, 5)
    tf.debugging.assert_near(
        tf.exp(log_w_new),
        tf.fill([2, 5], tf.constant(1.0 / 5.0, dtype=tf.float32)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert_all_finite(x_new, log_w_new)
