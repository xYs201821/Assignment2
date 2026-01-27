import numpy as np
import pytest
import tensorflow as tf

from src.utility import (
    block_diag,
    cholesky_solve,
    is_psd,
    max_asym,
    quadratic_matmul,
    tf_cond,
    weighted_mean,
)

pytestmark = pytest.mark.unit


def test_quadratic_matmul_matches_manual():
    A = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    B = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    C = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    out = quadratic_matmul(A, B, C)
    expected = A.numpy() @ B.numpy() @ C.numpy().T
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_cholesky_solve_vector_and_matrix():
    A = tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float32)
    b = tf.constant([1.0, 2.0], dtype=tf.float32)
    x = cholesky_solve(A, b)
    expected = np.linalg.solve(A.numpy(), b.numpy())
    np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert x.shape == (2,)

    B = tf.eye(2, dtype=tf.float32)
    X = cholesky_solve(A, B)
    expected_mat = np.linalg.solve(A.numpy(), B.numpy())
    np.testing.assert_allclose(X.numpy(), expected_mat, rtol=1e-6, atol=1e-6)


def test_block_diag_values():
    A = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    B = tf.constant([[5.0]], dtype=tf.float32)
    out = block_diag(A, B)
    expected = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(out.numpy()[0], expected, rtol=1e-6, atol=1e-6)


def test_weighted_mean_rank_handling():
    X = tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4])
    W = tf.constant([0.2, 0.3, 0.5], dtype=tf.float32)
    mean = weighted_mean(X, W, axis=1)
    manual = tf.reduce_sum(X * W[tf.newaxis, :, tf.newaxis], axis=1) / tf.reduce_sum(W)
    tf.debugging.assert_near(mean, manual, atol=1e-6, rtol=1e-6)

    W_batch = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=tf.float32)
    mean_batch = weighted_mean(X, W_batch, axis=1)
    manual_batch = tf.reduce_sum(X * W_batch[:, :, tf.newaxis], axis=1) / tf.reduce_sum(
        W_batch, axis=1, keepdims=True
    )
    tf.debugging.assert_near(mean_batch, manual_batch, atol=1e-6, rtol=1e-6)

    unnormalized = weighted_mean(X, W, axis=1, normalize=False)
    manual_sum = tf.reduce_sum(X * W[tf.newaxis, :, tf.newaxis], axis=1)
    tf.debugging.assert_near(unnormalized, manual_sum, atol=1e-6, rtol=1e-6)


def test_tf_cond_and_symmetry_helpers():
    M = tf.constant([[1.0, 0.0], [0.0, 4.0]], dtype=tf.float32)
    cond = tf_cond(M)
    np.testing.assert_allclose(cond.numpy(), np.array(4.0, dtype=np.float32), rtol=1e-3, atol=1e-3)

    sym = tf.constant([[2.0, -1.0], [-1.0, 2.0]], dtype=tf.float32)
    asym = tf.constant([[0.0, 1.0], [-2.0, 0.0]], dtype=tf.float32)
    assert max_asym(sym).numpy() < 1e-6
    assert max_asym(asym).numpy() > 0.0

    sym_bad = tf.constant([[1.0, 2.0], [2.0, -1.0]], dtype=tf.float32)
    assert bool(is_psd(sym).numpy())
    assert not bool(is_psd(sym_bad).numpy())
