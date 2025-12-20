import os
import sys
import numpy as np
import pytest
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ssm import LinearGaussianSSM, StochasticVolatilitySSM, RangeBearingSSM
from src.motion_model import ConstantVelocityMotionModel
from src.utility import tfp_lgssm  

@pytest.fixture
def lgssm_3d():
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

    model = LinearGaussianSSM(A, B, C, D, m0, P0)
    return model

@pytest.fixture
def lgssm_2d():
    dx, dy = 2, 1
    A = np.eye(dx, dtype=np.float32)
    B = 0.49 * np.eye(dx, dtype=np.float32)
    C = np.ones((dy, dx), dtype=np.float32)
    D = 0.49 * np.eye(dy, dtype=np.float32)

    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.eye(dx, dtype=np.float32)
    model = LinearGaussianSSM(A, B, C, D, m0, P0)
    return model

@pytest.fixture
def sim_data_3d(lgssm_3d):
    T = 80
    batch_size = 1

    x_traj, y_traj = lgssm_3d.simulate(T=T, shape=(batch_size, ))
    return {
        "T": T,
        "batch_size": batch_size,
        "x_traj": x_traj,
        "y_traj": y_traj,
    }

@pytest.fixture
def tfp_ref_3d(lgssm_3d, sim_data_3d):
    """Ground Truth for kalman filter"""
    y_traj = sim_data_3d["y_traj"]
    m_tfp, P_tfp = tfp_lgssm(y_traj[0], lgssm_3d, mode="filter")
    return m_tfp, P_tfp

@pytest.fixture
def sv_model():
    alpha = 0.98
    sigma = 1.0
    beta  = 0.5
    return StochasticVolatilitySSM(alpha=alpha, sigma=sigma, beta=beta)

@pytest.fixture
def sv_model_logy2():
    alpha = 0.98
    sigma = 1.0
    beta  = 1.0
    return StochasticVolatilitySSM(alpha=alpha, sigma=sigma, beta=beta, obs_mode="logy2", obs_eps=1e-16)

@pytest.fixture
def constant_velocity_motion_model():
    """Constant Velocity Motion Model"""
    v = tf.constant([1.0, 0.7], dtype=tf.float32)
    dt = 0.1
    cov_eps = 0.0001*np.eye(2, dtype=np.float32)  # small perturabtion of velocity
    return ConstantVelocityMotionModel(v=v, dt=dt, cov_eps=cov_eps)

@pytest.fixture
def range_bearing_ssm(constant_velocity_motion_model):
    """Range Bearing SSM"""
    cov_eps_y = 0.09*np.eye(2, dtype=np.float32)
    return RangeBearingSSM(motion_model=constant_velocity_motion_model, cov_eps_y=cov_eps_y)