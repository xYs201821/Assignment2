import numpy as np
import pytest
import tensorflow as tf

from src.filters.ot_resampling import (
    OTResamplingDPF,
)

pytestmark = pytest.mark.unit


def test_pairwise_distance_default_is_nonnegative():
    x = tf.constant([[[0.0], [1.0], [3.0]]], dtype=tf.float32)
    dist = OTResamplingDPF.pairwise_distance(x)

    assert dist.shape == (1, 3, 3)
    tf.debugging.assert_near(tf.linalg.diag_part(dist), tf.zeros([1, 3], dtype=tf.float32), atol=1e-6, rtol=1e-6)
    tf.debugging.assert_greater_equal(dist, tf.zeros_like(dist))


def test_pairwise_distance_supports_custom_metric():
    x = tf.constant([[[0.0], [1.0], [3.0]]], dtype=tf.float32)
    y = tf.constant([[[2.0], [5.0]]], dtype=tf.float32)

    dist_l1 = OTResamplingDPF.pairwise_distance(
        x,
        y,
        metric_fn=lambda x_exp, y_exp: tf.reduce_sum(tf.abs(x_exp - y_exp), axis=-1),
    )
    expected = tf.constant([[[2.0, 5.0], [1.0, 4.0], [1.0, 2.0]]], dtype=tf.float32)
    tf.debugging.assert_near(dist_l1, expected, atol=1e-6, rtol=1e-6)


def test_sinkhorn_log_plan_matches_marginals():
    a = tf.constant([[0.6, 0.4]], dtype=tf.float32)
    b = tf.constant([[0.5, 0.5]], dtype=tf.float32)
    log_a = tf.math.log(a)
    log_b = tf.math.log(b)

    x = tf.constant([[[0.0], [1.0]]], dtype=tf.float32)
    cost = OTResamplingDPF.pairwise_distance(x)
    log_plan = OTResamplingDPF.sinkhorn_log_plan(log_a, log_b, cost, epsilon=0.2, num_iters=120)
    plan = tf.exp(log_plan)

    row_mass = tf.reduce_sum(plan, axis=-1)
    col_mass = tf.reduce_sum(plan, axis=-2)
    tf.debugging.assert_near(row_mass, a, atol=5e-4, rtol=5e-4)
    tf.debugging.assert_near(col_mass, b, atol=5e-4, rtol=5e-4)


def test_ot_resample_barycentric_outputs_uniform_weights_and_preserves_mean(lgssm_1d):
    dpf = OTResamplingDPF(lgssm_1d, num_particles=3, resample="always")
    x = tf.constant([[[0.0], [2.0], [6.0]]], dtype=tf.float32)
    w = tf.constant([[0.8, 0.1, 0.1]], dtype=tf.float32)
    log_w = tf.math.log(w)

    x_new, log_w_new, parent_idx = dpf.ot_resample_barycentric(
        x,
        log_w,
        epsilon=0.5,
        num_iters=120,
    )
    w_new = tf.exp(log_w_new)

    assert x_new.shape == x.shape
    assert parent_idx.shape == (1, 3)
    tf.debugging.assert_near(
        w_new,
        tf.constant([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=tf.float32),
        atol=1e-6,
        rtol=1e-6,
    )

    mean_in = tf.einsum("bn,bnd->bd", w, x)
    mean_out = tf.einsum("bn,bnd->bd", w_new, x_new)
    tf.debugging.assert_near(mean_out, mean_in, atol=2e-3, rtol=2e-3)


def test_ot_resample_barycentric_outputs_valid_for_batched_inputs(lgssm_1d):
    dpf = OTResamplingDPF(lgssm_1d, num_particles=3, resample="always")
    x = tf.constant([[[0.0], [1.0], [4.0]]], dtype=tf.float32)
    log_w = tf.math.log(tf.constant([[0.7, 0.2, 0.1]], dtype=tf.float32))

    x_new, log_w_new, parent = dpf.ot_resample_barycentric(x, log_w)

    assert x_new.shape == x.shape
    assert log_w_new.shape == log_w.shape
    assert parent.shape == (1, 3)
    assert np.isfinite(np.asarray(x_new)).all()
