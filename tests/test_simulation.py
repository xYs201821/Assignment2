import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.ssm import SSM
import tensorflow as tf
import tensorflow_probability as tfp
from src.filter import Filter

def max_asym(P):
    skew = P - tf.linalg.matrix_transpose(P)
    return tf.reduce_max(tf.abs(skew))

def test_tfp(observations, ssm, mode="filter"):
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

def test_kalman_filter():
    """Test Kalman filter"""
    dx, dy = 3, 2
    dtype = tf.float32

    A = tf.constant([[0.9, 0.1, 0.0],
                     [0.0, 0.8, 0.1],
                     [0.0, 0.0, 0.9]], dtype=dtype)
    B = tf.eye(dx, dtype=dtype)
    C = tf.constant([[1.0, 0.5, 0.0],
                     [0.0, 0.9, 0.0]], dtype=dtype)
    D = 0.3 * tf.eye(dy, dtype=dtype)

    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.eye(dx, dtype=np.float32)

    ssm = SSM(A=A, B=B, C=C, D=D, m0=m0, P0=P0)

    T = 100
    batch_size = 1
    seed = 42

    x_traj, y_traj = ssm.simulate(T=T, batch_size=batch_size, seed=seed)
    m_tfp, P_tfp = test_tfp(y_traj[0], ssm)
    # run filter with Joseph and with standard covariance updates
    kalman = Filter(ssm)
    result_joseph = kalman.filter(y_traj, joseph=True)
    result_std    = kalman.filter(y_traj, joseph=False)
    
    x_true         = x_traj[0]  # [T, dx]
    m_filt_joseph  = result_joseph["m_filt"][0] # [T, dx]
    m_filt_std     = result_std["m_filt"][0]    # [T, dx]
    P_filt_joseph  = result_joseph["P_filt"][0] # [T, dx, dx]
    P_filt_std     = result_std["P_filt"][0]    # [T, dx, dx]
    cond_P_joseph  = result_joseph["cond_P"][0] # [T]
    cond_P_std     = result_std["cond_P"][0]    # [T]
    cond_S_joseph  = result_joseph["cond_S"][0] # [T]
    cond_S_std     = result_std["cond_S"][0]    # [T]

    # MSE between filtered mean and true state
    mse_joseph = tf.reduce_mean((m_filt_joseph - x_true) ** 2, axis=0)  # [dx]
    mse_std = tf.reduce_mean((m_filt_std    - x_true) ** 2, axis=0)
    mse_tfp = tf.reduce_mean((m_tfp - x_true) ** 2, axis=0)
    print("MSE per state dim (Joseph):", mse_joseph.numpy())
    print("MSE per state dim (Std)   :", mse_std.numpy())
    print("MSE per state dim (TFP)   :", mse_tfp.numpy())
    print("================================================")
    # Compare average trace of covariance with average squared error
    trace_P_joseph = tf.reduce_mean(tf.linalg.trace(P_filt_joseph))
    trace_P_std = tf.reduce_mean(tf.linalg.trace(P_filt_std))
    trace_P_tfp = tf.reduce_mean(tf.linalg.trace(P_tfp))
    sq_err_joseph  = tf.reduce_mean(tf.reduce_sum((m_filt_joseph - x_true) ** 2, axis=1))
    sq_err_std  = tf.reduce_mean(tf.reduce_sum((m_filt_std    - x_true) ** 2, axis=1))
    sq_err_tfp  = tf.reduce_mean(tf.reduce_sum((m_tfp - x_true) ** 2, axis=1))
    print("Avg trace(P) TFP    :", trace_P_tfp.numpy())
    print("Avg trace(P) Joseph :", trace_P_joseph.numpy())
    print("Avg trace(P) Std    :", trace_P_std.numpy())
    print("================================================")
    print("Avg squared error TFP   :", sq_err_tfp.numpy())
    print("Avg squared error Joseph:", sq_err_joseph.numpy())
    print("Avg squared error Std   :", sq_err_std.numpy())
    print("================================================")
    max_diff_m = tf.reduce_max(tf.abs(m_filt_joseph - m_filt_std))
    max_diff_P = tf.reduce_max(tf.abs(P_filt_joseph - P_filt_std))
    max_diff_P_tfp = tf.reduce_max(tf.abs(P_tfp - P_filt_joseph))
    max_diff_m_tfp = tf.reduce_max(tf.abs(m_tfp - m_filt_joseph))
    print("Max |Δm| between Joseph and Std  :", max_diff_m.numpy())
    print("Max |ΔP| between Joseph and Std  :", max_diff_P.numpy())
    print("Max |Δm| between TFP and Joseph :", max_diff_m_tfp.numpy())
    print("Max |ΔP| between TFP and Joseph :", max_diff_P_tfp.numpy())
    print("================================================")
    # compare symmetry of P_filt_joseph and P_filt_std
    print("max asymmetry in P_tfp   :", max_asym(P_tfp).numpy())
    print("max asymmetry in P_filt_joseph:", max_asym(P_filt_joseph).numpy())
    print("max asymmetry in P_filt_std   :", max_asym(P_filt_std).numpy())
    print("================================================")
    # PSD check via eigenvals
    eig_joseph = tf.linalg.eigvalsh(P_filt_joseph)  # [T, dx, dx]
    eig_std = tf.linalg.eigvalsh(P_filt_std)
    eig_tfp = tf.linalg.eigvalsh(P_tfp)
    print("min eigenvalue (Joseph):", tf.reduce_min(eig_joseph).numpy())
    print("min eigenvalue (Std)   :", tf.reduce_min(eig_std).numpy())
    print("min eigenvalue (TFP)   :", tf.reduce_min(eig_tfp).numpy())
    print("================================================")
    print("cond(P) Joseph min/max:", tf.reduce_min(cond_P_joseph).numpy(),
                                    tf.reduce_max(cond_P_joseph).numpy())
    print("cond(P) Std    min/max:", tf.reduce_min(cond_P_std).numpy(),
                                    tf.reduce_max(cond_P_std).numpy())
    print("================================================")
    print("cond(S) Joseph min/max:", tf.reduce_min(cond_S_joseph).numpy(),
                                    tf.reduce_max(cond_S_joseph).numpy())
    print("cond(S) Std    min/max:", tf.reduce_min(cond_S_std).numpy(),
                                    tf.reduce_max(cond_S_std).numpy())



if __name__ == "__main__":
    test_kalman_filter()
