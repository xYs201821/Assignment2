import numpy as np
import tensorflow as tf

from tests.testhelper import assert_all_finite

def test_sv_simulate_shapes_and_nan(sv_model):
    T = 50
    batch_size = 8

    x_traj, y_traj = sv_model.simulate(T=T, shape=(batch_size, ))

    dx = sv_model.state_dim  
    dy = sv_model.obs_dim    

    assert x_traj.shape == (batch_size, T, dx)
    assert y_traj.shape == (batch_size, T, dy)

    assert_all_finite(x_traj, y_traj)

def test_sv_logy2_observation_mode(sv_model_logy2):
    T = 30
    batch_size = 4

    x_traj, y_traj = sv_model_logy2.simulate(T=T, shape=(batch_size, ))

    assert x_traj.shape == (batch_size, T, sv_model_logy2.state_dim)
    assert y_traj.shape == (batch_size, T, sv_model_logy2.obs_dim)
    assert_all_finite(x_traj, y_traj)
