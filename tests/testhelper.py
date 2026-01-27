import tensorflow as tf

def make_spd(batch, n, eps=1e-5):
    A = tf.random.normal([batch, n, n], dtype=tf.float32)
    P = tf.einsum("bij,bkj->bik", A, A) 
    I = tf.eye(n, batch_shape=[batch], dtype=tf.float32)
    return P + eps * I

def assert_all_finite(*tensors):
    for t in tensors:
        tf.debugging.assert_all_finite(t, "Found NaN/Inf.")


def assert_symmetric(P, tol=1e-5):
    PT = tf.transpose(P, perm=[0, 1, 3, 2])
    tf.debugging.assert_less(tf.reduce_max(tf.abs(P - PT)), tol)


def assert_psd(P, eps=-1e-5):
    # P: [batch, T, dx, dx]
    B = tf.shape(P)[0]
    T = tf.shape(P)[1]
    dx = tf.shape(P)[2]
    P_flat = tf.reshape(P, [B * T, dx, dx])
    eigvals = tf.linalg.eigvalsh(0.5 * (P_flat + tf.transpose(P_flat, [0, 2, 1])))
    tf.debugging.assert_greater_equal(tf.reduce_min(eigvals), eps)


def assert_step_time_shape(step_time_s, T):
    step_time_s = tf.convert_to_tensor(step_time_s)
    if step_time_s.shape.rank == 2:
        tf.debugging.assert_equal(tf.shape(step_time_s)[1], T)
    else:
        tf.debugging.assert_equal(tf.shape(step_time_s)[0], T)
