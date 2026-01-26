"""Utility math helpers and plotting routines."""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_probability as tfp


def tf_cond(M, jitter=1e-6):
    """Estimate condition number for symmetric matrices.

    Shapes:
      M: [..., n, n]
    Returns:
      cond: [...]
    """
    M = tf.convert_to_tensor(M)
    tf.debugging.assert_all_finite(M, "tf_cond received NaN/Inf in input matrix")
    M_sym = (M + tf.linalg.matrix_transpose(M)) / 2.0
    eye = tf.eye(tf.shape(M_sym)[-1], batch_shape=tf.shape(M_sym)[:-2], dtype=M_sym.dtype)
    M_sym = M_sym + eye * jitter
    M_sym = tf.where(tf.math.is_finite(M_sym), M_sym, tf.zeros_like(M_sym))
    evals = tf.linalg.eigvalsh(M_sym)
    e_max = tf.reduce_max(evals, axis=-1)
    e_min = tf.reduce_min(evals, axis=-1)
    return e_max / (e_min + jitter)

def quadratic_matmul(A, B, C):
    """
    Computes ABC^T 
    """
    AB = tf.linalg.matmul(A, B)
    return tf.linalg.matmul(AB, C, transpose_b=True)

def cholesky_solve(A, B, jitter=1e-6):
    """
    Solves AX = B using Cholesky decomposition.
    """
    A = tf.convert_to_tensor(A)
    B = tf.convert_to_tensor(B)
    need_rhs = tf.equal(tf.rank(B), tf.rank(A) - 1)
    B = tf.cond(need_rhs, lambda: B[..., tf.newaxis], lambda: B)

    A_sym = (A + tf.linalg.matrix_transpose(A)) / 2.0
    eye = tf.eye(tf.shape(A_sym)[-1], batch_shape=tf.shape(A_sym)[:-2], dtype=A_sym.dtype)
    A_robust = A_sym + eye * jitter
    L = tf.linalg.cholesky(A_robust)
    Y = tf.linalg.triangular_solve(L, B, lower=True)
    X = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(L), Y, lower=False)
    return tf.cond(need_rhs, lambda: X[..., 0], lambda: X)

def block_diag(A, B):
    """
    Create a block diagonal matrix from two matrices.
    A: [batch, n, n] or [n, n]
    B: [batch, m, m] or [m, m]

    returns: [batch, n+m, n+m]
    """

    A = tf.convert_to_tensor(A, dtype=tf.float32)
    B = tf.convert_to_tensor(B, dtype=tf.float32)
    if A.shape.rank == 2:
        A = A[tf.newaxis, :, :]
    if B.shape.rank == 2:
        B = B[tf.newaxis, :, :]
    batch = tf.shape(A)[0]
    if tf.shape(B)[0] != batch:
        B = tf.broadcast_to(B, [batch, tf.shape(B)[1], tf.shape(B)[2]])

    a = tf.shape(A)[1]
    b = tf.shape(B)[1]
    Z_ab = tf.zeros([batch, a, b], dtype=tf.float32)
    Z_ba = tf.zeros([batch, b, a], dtype=tf.float32)
    top = tf.concat([A, Z_ab], axis=2)
    bot = tf.concat([Z_ba, B], axis=2)
    return tf.concat([top, bot], axis=1)

def plot_ssm_trajectory(ssm, T=200, seed=None, title=None):
    """Simulate and plot one trajectory for a state-space model."""
    # ---- simulate ----
    x_traj, y_traj = ssm.simulate(T=T, batch_size=1, seed=seed)

    # remove batch dimension → shapes:
    # x: [T, dx], y: [T, dy]
    x_traj = x_traj[0].numpy()
    y_traj = y_traj[0].numpy()

    dx = ssm.state_dim
    dy = ssm.obs_dim
    t = np.arange(T)

    # ---- total number of subplots ----
    n_plots = dx + dy
    plt.figure(figsize=(12, 3 * n_plots))

    # ---- plot state components ----
    for i in range(dx):
        plt.subplot(n_plots, 1, i + 1)
        plt.plot(t, x_traj[:, i], label=f"x[{i}]")
        plt.xlabel("time")
        plt.ylabel(f"x[{i}]")
        plt.title(f"State dimension x[{i}]")
        plt.grid(True)

    # ---- plot observation components ----
    for j in range(dy):
        plt.subplot(n_plots, 1, dx + j + 1)
        plt.plot(t, y_traj[:, j], color="black", label=f"y[{j}]")
        plt.xlabel("time")
        plt.ylabel(f"y[{j}]")
        plt.title(f"Observation dimension y[{j}]")
        plt.grid(True)

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def max_asym(P):
    """Return the maximum absolute skew-symmetric entry.

    Shapes:
      P: [..., n, n]
    Returns:
      max_asym: [...]
    """
    skew = P - tf.linalg.matrix_transpose(P)
    return tf.reduce_max(tf.abs(skew))

def is_psd(P, tol=1e-7):
    """Check if a symmetric matrix is positive semidefinite.

    Shapes:
      P: [..., n, n]
    Returns:
      is_psd: bool
    """
    eigvals = tf.linalg.eigvalsh(P)
    return tf.reduce_min(eigvals) >= -tol

def tfp_lgssm(observations, ssm, mode="filter"):
    """Ground Truth for kalman filter"""

    lgssm = tfp.distributions.LinearGaussianStateSpaceModel(
        num_timesteps=observations.shape[0],
        transition_matrix=ssm.A,
        transition_noise=tfp.distributions.MultivariateNormalTriL(
            loc=tf.zeros(ssm.state_dim, dtype=tf.float32),
            scale_tril=tf.linalg.cholesky(ssm.cov_eps_x),
        ),
        observation_matrix=ssm.C,
        observation_noise=tfp.distributions.MultivariateNormalTriL(
            loc=tf.zeros(ssm.obs_dim, dtype=tf.float32),
            scale_tril=tf.linalg.cholesky(ssm.cov_eps_y),
        ),
        initial_state_prior=tfp.distributions.MultivariateNormalTriL(
            loc=ssm.m0,
            scale_tril=tf.linalg.cholesky(ssm.P0),
        )
    )
    if mode == "smooth":
        # smoother: p(x_t | y_1:T)
        means, covs = lgssm.posterior_marginals(observations)
        return means, covs

    elif mode == "filter":
        # Kalman filter: p(x_t | y_1:t)
        (_,
         means, covs,
         _, _,
         _, _) = lgssm.forward_filter(observations)
        return means, covs

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'smooth' or 'filter'.")


def weighted_mean(X, W=None, axis=-2, normalize=True):
    """Compute weighted mean."""
    X = tf.convert_to_tensor(X)
    if W is None:
        return tf.reduce_mean(X, axis=axis)

    W = tf.convert_to_tensor(W, dtype=X.dtype)

    x_rank = tf.rank(X)
    axis_ = axis if axis >= 0 else axis + x_rank
    w_rank = tf.rank(W)

    def _rank1():
        shape = tf.ones([x_rank], dtype=tf.int32)
        shape = tf.tensor_scatter_nd_update(shape, [[axis_]], [tf.shape(W)[0]])
        return tf.reshape(W, shape)

    def _rank_xminus1():
        return W[..., tf.newaxis]

    def _default():
        return W

    Wb = tf.cond(
        tf.equal(w_rank, 1),
        _rank1,
        lambda: tf.cond(tf.equal(w_rank, x_rank - 1), _rank_xminus1, _default),
    )

    num = tf.reduce_sum(X * Wb, axis=axis_)
    if not normalize:
        return num

    den = tf.reduce_sum(Wb, axis=axis_)
    return tf.math.divide_no_nan(num, den)
