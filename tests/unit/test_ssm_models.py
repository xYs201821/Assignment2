import numpy as np
import pytest
import tensorflow as tf

from src.ssm import StochasticVolatilitySSM

pytestmark = pytest.mark.unit


def test_lgssm_linear_maps(lgssm_3d):
    x = tf.constant([[1.0, -2.0, 0.5]], dtype=tf.float32)
    f_out = lgssm_3d.f(x)
    h_out = lgssm_3d.h(x)

    f_expected = tf.einsum("ij,bj->bi", lgssm_3d.A, x)
    h_expected = tf.einsum("ij,bj->bi", lgssm_3d.C, x)

    tf.debugging.assert_near(f_out, f_expected, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(h_out, h_expected, atol=1e-6, rtol=1e-6)


def test_range_bearing_innovation_wraps(range_bearing_ssm):
    x = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    y_pred = range_bearing_ssm.h(x)
    y = tf.stack([y_pred[..., 0], y_pred[..., 1] + 2.0 * np.pi], axis=-1)
    innov = range_bearing_ssm.innovation(y, y_pred)
    tf.debugging.assert_near(innov[..., 1], tf.zeros_like(innov[..., 1]), atol=1e-6, rtol=1e-6)
    assert tf.reduce_max(tf.abs(innov[..., 1])).numpy() <= np.pi + 1e-6


def test_range_bearing_measurement_mean_wrap(range_bearing_ssm):
    bearings = tf.constant([np.pi - 0.1, -np.pi + 0.1], dtype=tf.float32)
    ranges = tf.constant([1.0, 1.0], dtype=tf.float32)
    y = tf.stack([ranges, bearings], axis=-1)[tf.newaxis, ...]  # [1, 2, 2]
    mean = range_bearing_ssm.measurement_mean(y)
    mean_bearing = mean[0, 1]

    tf.debugging.assert_less(tf.abs(tf.abs(mean_bearing) - np.pi), 0.2)
    residual = range_bearing_ssm.measurement_residual(y, mean)
    assert tf.reduce_max(tf.abs(residual[..., 1])).numpy() <= np.pi + 1e-6


def test_stochastic_volatility_observation_modes():
    x = tf.constant([[0.3]], dtype=tf.float32)
    r = tf.constant([[1.0]], dtype=tf.float32)

    sv = StochasticVolatilitySSM(alpha=0.9, sigma=0.2, beta=0.5, obs_mode="y")
    y = sv.h_with_noise(x, r)
    expected_y = sv.beta * tf.exp(0.5 * x) * r
    tf.debugging.assert_near(y, expected_y, atol=1e-6, rtol=1e-6)

    q = tf.constant([[2.0]], dtype=tf.float32)
    f_with_noise = sv.f_with_noise(x, q)
    expected_f = sv.f(x) + sv.sigma * q
    tf.debugging.assert_near(f_with_noise, expected_f, atol=1e-6, rtol=1e-6)

    sv_log = StochasticVolatilitySSM(alpha=0.9, sigma=0.2, beta=0.5, obs_mode="logy2", obs_eps=1e-6)
    const = tf.math.digamma(0.5) + tf.math.log(2.0) + tf.math.log(sv_log.beta ** 2)
    tf.debugging.assert_near(sv_log.h(x), x + const, atol=1e-6, rtol=1e-6)
    y_log = sv_log.h_with_noise(x, r)
    tf.debugging.assert_near(y_log, sv_log.h(x) + r, atol=1e-6, rtol=1e-6)


def test_state_cov_matches_manual(lgssm_2d):
    x = tf.constant([[[1.0, 0.0], [3.0, 0.0]]], dtype=tf.float32)
    w = tf.constant([[0.25, 0.75]], dtype=tf.float32)
    cov = lgssm_2d.state_cov(x, w)

    mean = 0.25 * x[:, 0, :] + 0.75 * x[:, 1, :]
    resid = x - mean[:, tf.newaxis, :]
    manual = tf.einsum("bn,bni,bnj->bij", w, resid, resid)
    tf.debugging.assert_near(cov, manual, atol=1e-6, rtol=1e-6)
