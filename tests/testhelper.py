import tensorflow as tf

def make_spd(batch, n, eps=1e-5):
    """Generate a batch of symmetric positive definite matrices."""
    A = tf.random.normal([batch, n, n], dtype=tf.float32)
    P = tf.einsum("bij,bkj->bik", A, A) 
    I = tf.eye(n, batch_shape=[batch], dtype=tf.float32)
    return P + eps * I


def assert_all_finite(*tensors):
    """Assert that all tensors contain only finite values (no NaN or Inf)."""
    for t in tensors:
        tf.debugging.assert_all_finite(t, "Found NaN/Inf.")


def assert_symmetric(P, tol=1e-5):
    """Assert that a [batch, T, dx, dx] tensor is symmetric."""
    PT = tf.transpose(P, perm=[0, 1, 3, 2])
    tf.debugging.assert_less(tf.reduce_max(tf.abs(P - PT)), tol)


def assert_psd(P, eps=-1e-5):
    """Assert that a [batch, T, dx, dx] tensor is positive semi-definite."""
    B = tf.shape(P)[0]
    T = tf.shape(P)[1]
    dx = tf.shape(P)[2]
    P_flat = tf.reshape(P, [B * T, dx, dx])
    eigvals = tf.linalg.eigvalsh(0.5 * (P_flat + tf.transpose(P_flat, [0, 2, 1])))
    tf.debugging.assert_greater_equal(tf.reduce_min(eigvals), eps)


def assert_weights_valid(w, batch_size, T, num_particles):
    """Assert that particle weights are valid (non-negative, sum to 1)."""
    assert w.shape == (batch_size, T, num_particles)
    tf.debugging.assert_greater_equal(tf.reduce_min(w), 0.0)
    tf.debugging.assert_near(
        tf.reduce_sum(w, axis=-1),
        tf.ones([batch_size, T], dtype=w.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def assert_particles_shape(x, batch_size, T, num_particles, state_dim):
    """Assert that particle tensor has correct shape."""
    assert x.shape == (batch_size, T, num_particles, state_dim)


def assert_diagnostics_keys(diagnostics, required_keys):
    """Assert that diagnostics dict contains required keys."""
    for key in required_keys:
        assert key in diagnostics, f"Missing diagnostics key: {key}"


def assert_covariance_valid(P, eps=1e-6):
    """Assert that a covariance matrix is symmetric and PSD."""
    # Symmetry
    P_sym = 0.5 * (P + tf.linalg.matrix_transpose(P))
    tf.debugging.assert_near(P, P_sym, atol=1e-5, rtol=1e-5)
    # PSD: eigenvalues >= -eps
    eigvals = tf.linalg.eigvalsh(P_sym)
    tf.debugging.assert_greater_equal(tf.reduce_min(eigvals), -eps)
