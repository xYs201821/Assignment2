import pytest
import tensorflow as tf

from src.filters.particle_transformer import (
    ParticleTransformerDPF,
    ParticleTransformerResampler,
)

pytestmark = pytest.mark.unit


def test_simple_particle_transformer_resampler_outputs_row_stochastic_attention():
    layer = ParticleTransformerResampler(num_particles=4, d_model=16, hidden=16)
    x = tf.constant([[[0.0], [1.0], [2.0], [3.0]]], dtype=tf.float32)
    log_w = tf.math.log(tf.constant([[0.6, 0.2, 0.1, 0.1]], dtype=tf.float32))

    x_new, attn = layer(x, log_w)

    assert x_new.shape == x.shape
    assert attn.shape == (1, 4, 4)
    tf.debugging.assert_near(
        tf.reduce_sum(attn, axis=-1),
        tf.ones([1, 4], dtype=tf.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_simple_particle_transformer_resampler_is_scale_equivariant():
    layer = ParticleTransformerResampler(
        num_particles=5,
        d_model=16,
        hidden=16,
    )
    x = tf.random.normal([2, 5, 3], dtype=tf.float32)
    w = tf.constant(
        [
            [0.50, 0.20, 0.10, 0.10, 0.10],
            [0.10, 0.15, 0.25, 0.20, 0.30],
        ],
        dtype=tf.float32,
    )
    log_w = tf.math.log(w)
    scale = tf.constant(2.5, dtype=tf.float32)
    shift = tf.constant(1.2, dtype=tf.float32)

    y, _ = layer(x, log_w, training=False)
    y_scaled, _ = layer(scale * x + shift, log_w, training=False)

    tf.debugging.assert_near(y_scaled, scale * y + shift, atol=2e-4, rtol=2e-4)


def test_simple_particle_transformer_resampler_supports_gradients():
    layer = ParticleTransformerResampler(num_particles=5, d_model=16, hidden=16)
    x = tf.Variable(tf.random.normal([2, 5, 3], dtype=tf.float32))
    log_w = tf.Variable(
        tf.math.log(
            tf.constant(
                [
                    [0.50, 0.20, 0.10, 0.10, 0.10],
                    [0.10, 0.15, 0.25, 0.20, 0.30],
                ],
                dtype=tf.float32,
            )
        )
    )
    with tf.GradientTape() as tape:
        x_new, _ = layer(x, log_w)
        loss = tf.reduce_sum(tf.square(x_new))

    grads = tape.gradient(loss, [x, log_w] + layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        tf.debugging.assert_all_finite(g, "gradient contains NaN/Inf")


def test_particle_transformer_dpf_resample_step_outputs_uniform_weights(lgssm_1d):
    dpf = ParticleTransformerDPF(
        lgssm_1d,
        num_particles=5,
        ess_threshold=0.5,
        d_model=32,
        hidden=32,
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


def test_particle_transformer_dpf_rejects_runtime_num_particle_change(lgssm_1d):
    dpf = ParticleTransformerDPF(lgssm_1d, num_particles=8, d_model=16, hidden=16)
    y = tf.zeros([1, 3, 1], dtype=tf.float32)

    with pytest.raises(ValueError):
        _ = dpf.filter(y, num_particles=16)


def test_particle_transformer_dpf_stop_grad_through_time_switch(lgssm_1d):
    y = tf.zeros([1, 2, 1], dtype=tf.float32)
    init_particles = tf.Variable(tf.random.normal([1, 5, 1], dtype=tf.float32))

    dpf_stop = ParticleTransformerDPF(
        lgssm_1d,
        num_particles=5,
        d_model=16,
        hidden=16,
        stop_grad_through_time=True,
    )
    with tf.GradientTape() as tape_stop:
        x_seq_stop, _, _, _ = dpf_stop.filter(
            y,
            resample="never",
            init_particles=init_particles,
        )
        loss_stop = tf.reduce_sum(x_seq_stop[:, 1, :, :])
    grad_stop = tape_stop.gradient(loss_stop, init_particles)
    if grad_stop is not None:
        tf.debugging.assert_near(
            grad_stop,
            tf.zeros_like(init_particles),
            atol=1e-7,
            rtol=1e-7,
        )

    dpf_full = ParticleTransformerDPF(
        lgssm_1d,
        num_particles=5,
        d_model=16,
        hidden=16,
        stop_grad_through_time=False,
    )
    with tf.GradientTape() as tape_full:
        x_seq_full, _, _, _ = dpf_full.filter(
            y,
            resample="never",
            init_particles=init_particles,
        )
        loss_full = tf.reduce_sum(x_seq_full[:, 1, :, :])
    grad_full = tape_full.gradient(loss_full, init_particles)
    assert grad_full is not None
    tf.debugging.assert_all_finite(grad_full, "gradient contains NaN/Inf")
    assert tf.reduce_max(tf.abs(grad_full)).numpy() > 1e-8
