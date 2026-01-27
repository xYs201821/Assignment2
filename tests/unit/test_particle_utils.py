import numpy as np
import pytest
import tensorflow as tf

from src.filters.particle import ParticleFilter

pytestmark = pytest.mark.unit


def test_log_normalize_outputs():
    log_w = tf.math.log(tf.constant([[0.1, 0.2, 0.7]], dtype=tf.float32))
    log_w_norm, w, logZ = ParticleFilter._log_normalize(log_w)

    tf.debugging.assert_near(tf.reduce_sum(w, axis=-1), tf.ones([1], dtype=w.dtype), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(w.numpy(), np.array([[0.1, 0.2, 0.7]], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(logZ.numpy(), np.array([0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    tf.debugging.assert_near(tf.exp(log_w_norm), w, atol=1e-6, rtol=1e-6)


def test_ess_uniform():
    w = tf.constant([[0.25, 0.25, 0.25, 0.25]], dtype=tf.float32)
    ess = ParticleFilter.ess(w)
    tf.debugging.assert_near(ess, tf.constant([4.0], dtype=tf.float32), atol=1e-6, rtol=1e-6)


def test_normalize_reweight_modes():
    assert ParticleFilter._normalize_reweight(True) == 1
    assert ParticleFilter._normalize_reweight(False) == 0
    assert ParticleFilter._normalize_reweight("never") == 0
    assert ParticleFilter._normalize_reweight("auto") == 1
    assert ParticleFilter._normalize_reweight("always") == 2
    assert ParticleFilter._normalize_reweight(2) == 2
    with pytest.raises(ValueError):
        ParticleFilter._normalize_reweight("bad")
    with pytest.raises(ValueError):
        ParticleFilter._normalize_reweight(3)


def test_normalize_init_seed():
    seed = ParticleFilter._normalize_init_seed(5)
    assert seed.shape == (2,)
    np.testing.assert_array_equal(seed.numpy(), np.array([5, 0], dtype=np.int32))

    seed = ParticleFilter._normalize_init_seed((3, 7))
    np.testing.assert_array_equal(seed.numpy(), np.array([3, 7], dtype=np.int32))

    seed = ParticleFilter._normalize_init_seed(tf.constant([9, 2], dtype=tf.int32))
    np.testing.assert_array_equal(seed.numpy(), np.array([9, 2], dtype=np.int32))


def test_systematic_resample_degenerate_weights():
    w = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
    rng = tf.random.Generator.from_seed(0)
    idx = ParticleFilter.systematic_resample(w, rng)
    tf.debugging.assert_equal(idx, tf.constant([[0, 0, 0]], dtype=tf.int32))


def test_resample_particles_gathers():
    x = tf.constant([[[0.0], [1.0], [2.0]]], dtype=tf.float32)
    idx = tf.constant([[2, 0, 2]], dtype=tf.int32)
    out = ParticleFilter.resample_particles(x, idx)
    expected = tf.constant([[[2.0], [0.0], [2.0]]], dtype=tf.float32)
    tf.debugging.assert_near(out, expected, atol=1e-6, rtol=1e-6)
