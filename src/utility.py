import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_probability as tfp

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

def plot_ssm_trajectory(ssm, T=200, seed=None, title=None):
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
    # check the degree of asymmetry of P, [batch, n, n] -> [batch]
    skew = P - tf.linalg.matrix_transpose(P)
    return tf.reduce_max(tf.abs(skew))

def is_psd(P, tol=1e-7):
    # check the minimum eigenvalue of P
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
        # Kalman smoother: p(x_t | y_1:T)
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


def weighted_mean(X, W=None, axis=1, normalize=True):
    X = tf.convert_to_tensor(X)
    if W is None:
        return tf.reduce_mean(X, axis=axis)

    W = tf.convert_to_tensor(W, dtype=X.dtype)

    x_rank = tf.rank(X)
    axis_ = axis if axis >= 0 else axis + x_rank

    if W.shape.rank == 1:
        pre = tf.ones([axis_], dtype=tf.int32)
        post = tf.ones([x_rank - axis_ - 1], dtype=tf.int32)
        Wb = tf.reshape(W, tf.concat([pre, tf.shape(W), post], axis=0))
    elif W.shape.rank == 2 and (axis_ == 1):
        # W [B,n] with X [B,n,...]
        post = tf.ones([x_rank - 2], dtype=tf.int32)
        Wb = tf.reshape(W, tf.concat([tf.shape(W), post], axis=0))
    else:
        Wb = W

    num = tf.reduce_sum(X * Wb, axis=axis_)
    if not normalize:
        return num

    den = tf.reduce_sum(Wb, axis=axis_)
    return tf.math.divide_no_nan(num, den)