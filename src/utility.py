import tensorflow as tf

def tf_cond(M): # find condition number of M in tensorflow, [batch, n, n] -> [batch]
    s = tf.linalg.svd(M, compute_uv=False)
    s_max = tf.reduce_max(s, axis=-1)
    s_min = tf.reduce_min(s, axis=-1)
    eps = 1e-20
    return s_max / (s_min + eps)

def cholesky_solve(A, B): # solve AX = B, [batch, n, m] x [batch, m, k] -> [batch, n, k]
    L = tf.linalg.cholesky(A)
    U = tf.linalg.triangular_solve(L, B, lower=True)
    X = tf.linalg.triangular_solve(tf.transpose(L, perm=[0, 2, 1]), U, lower=False)
    return X