"""Unit tests for filter base classes and core functionality."""
import numpy as np
import pytest
import tensorflow as tf

from src.filters.base import GaussianFilter
from src.filters.kalman import KalmanFilter
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.pf_bootstrap import BootstrapParticleFilter
from tests.testhelper import assert_all_finite, assert_covariance_valid

pytestmark = pytest.mark.unit


# =============================================================================
# Kalman Filter Core Tests
# =============================================================================

class TestKalmanFilterPredict:
    """Tests for Kalman filter predict step."""

    def test_predict_shapes(self, lgssm_3d):
        """Predict should return correct shapes."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        m_pred, P_pred = kf.predict(m, P)
        
        assert m_pred.shape == (batch_size, dx)
        assert P_pred.shape == (batch_size, dx, dx)

    def test_predict_finite(self, lgssm_3d):
        """Predict should produce finite outputs."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        m_pred, P_pred = kf.predict(m, P)
        
        assert_all_finite(m_pred, P_pred)

    def test_predict_covariance_psd(self, lgssm_3d):
        """Predicted covariance should be positive semi-definite."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        _, P_pred = kf.predict(m, P)
        
        # Check each batch element
        for i in range(batch_size):
            assert_covariance_valid(P_pred[i:i+1])


class TestKalmanFilterUpdate:
    """Tests for Kalman filter update step."""

    def test_update_shapes(self, lgssm_3d):
        """Update should return correct shapes."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        dy = lgssm_3d.obs_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        y = tf.random.normal([batch_size, dy])
        
        m_upd, P_upd = kf.update(m, P, y)
        
        assert m_upd.shape == (batch_size, dx)
        assert P_upd.shape == (batch_size, dx, dx)

    def test_joseph_vs_naive_update(self, lgssm_3d):
        """Joseph and naive updates should give same results."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        dy = lgssm_3d.obs_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        y = tf.random.normal([batch_size, dy])
        
        m_j, P_j = kf.update_joseph(m, P, y)
        m_n, P_n = kf.update_naive(m, P, y)
        
        tf.debugging.assert_near(m_j, m_n, atol=1e-5, rtol=1e-5)
        tf.debugging.assert_near(P_j, P_n, atol=1e-4, rtol=1e-4)

    def test_update_reduces_uncertainty(self, lgssm_3d):
        """Update should reduce uncertainty (in most cases)."""
        kf = KalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 2.0  # Large prior uncertainty
        
        # Simulate an observation
        y = lgssm_3d.h(m) + tf.random.normal([batch_size, lgssm_3d.obs_dim]) * 0.1
        
        _, P_upd = kf.update(m, P, y)
        
        # Updated covariance trace should be smaller than prior
        trace_prior = tf.linalg.trace(P)
        trace_upd = tf.linalg.trace(P_upd)
        assert tf.reduce_all(trace_upd < trace_prior)


# =============================================================================
# EKF Tests
# =============================================================================

class TestEKFJacobian:
    """Tests for EKF Jacobian computation."""

    def test_jacobian_linear_system(self, lgssm_3d):
        """EKF Jacobian on linear system should match system matrices."""
        ekf = ExtendedKalmanFilter(lgssm_3d)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        x = tf.random.normal([batch_size, dx])
        
        F, _ = ekf._jacobian(lgssm_3d.f, x)
        H, _ = ekf._jacobian(lgssm_3d.h, x)
        
        # For linear system, Jacobians should be constant matrices
        expected_F = tf.broadcast_to(lgssm_3d.A[tf.newaxis, :, :], [batch_size, dx, dx])
        expected_H = tf.broadcast_to(lgssm_3d.C[tf.newaxis, :, :], [batch_size, lgssm_3d.obs_dim, dx])
        
        tf.debugging.assert_near(F, expected_F, atol=1e-5, rtol=1e-5)
        tf.debugging.assert_near(H, expected_H, atol=1e-5, rtol=1e-5)


# =============================================================================
# UKF Tests
# =============================================================================

class TestUKFCore:
    """Tests for UKF core functionality."""

    def test_ukf_predict_shapes(self, lgssm_3d):
        """UKF predict should return correct shapes."""
        ukf = UnscentedKalmanFilter(lgssm_3d, alpha=1e-3, beta=2.0, kappa=0.0)
        batch_size = 4
        dx = lgssm_3d.state_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        m_pred, P_pred = ukf.predict(m, P)
        
        assert m_pred.shape == (batch_size, dx)
        assert P_pred.shape == (batch_size, dx, dx)

    def test_ukf_update_shapes(self, lgssm_3d):
        """UKF update should return correct shapes."""
        ukf = UnscentedKalmanFilter(lgssm_3d, alpha=1e-3, beta=2.0, kappa=0.0)
        batch_size = 4
        dx = lgssm_3d.state_dim
        dy = lgssm_3d.obs_dim
        
        m = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        y = tf.random.normal([batch_size, dy])
        
        m_upd, P_upd = ukf.update(m, P, y)
        
        assert m_upd.shape == (batch_size, dx)
        assert P_upd.shape == (batch_size, dx, dx)


# =============================================================================
# Bootstrap Particle Filter Tests
# =============================================================================

class TestBootstrapPF:
    """Tests for Bootstrap Particle Filter."""

    def test_init_particles_shapes(self, lgssm_2d):
        """_init_particles should return correct shapes."""
        pf = BootstrapParticleFilter(lgssm_2d, num_particles=100)
        
        T = 10
        batch_size = 2
        _, y = lgssm_2d.simulate(T=T, shape=(batch_size,))
        
        x, log_w, parent = pf._init_particles(y, init_dist=None)
        
        dx = lgssm_2d.state_dim
        N = pf.num_particles
        
        assert x.shape == (batch_size, N, dx)
        assert log_w.shape == (batch_size, N)
        assert parent.shape == (batch_size, N)

    def test_step_shapes(self, lgssm_2d):
        """Single step should return correct shapes."""
        pf = BootstrapParticleFilter(lgssm_2d, num_particles=100)
        
        batch_size = 2
        N = 100
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        x_prev = tf.random.normal([batch_size, N, dx])
        log_w_prev = tf.fill([batch_size, N], -tf.math.log(tf.cast(N, tf.float32)))
        y_t = tf.random.normal([batch_size, dy])
        
        # step returns: x_pre, x_t, log_w_final, w_final, parent_indices, log_w_pre, logz_t
        x_pre, x_t, log_w_next, w_next, parent, log_w_pre, logz_t = pf.step(
            x_prev, log_w_prev, y_t, resample=1
        )

        assert x_pre.shape == (batch_size, N, dx)
        assert x_t.shape == (batch_size, N, dx)
        assert log_w_next.shape == (batch_size, N)
        assert w_next.shape == (batch_size, N)
        assert parent.shape == (batch_size, N)
        assert log_w_pre.shape == (batch_size, N)
        assert logz_t.shape == (batch_size,)

    def test_step_finite_outputs(self, lgssm_2d):
        """Step should produce finite outputs."""
        pf = BootstrapParticleFilter(lgssm_2d, num_particles=100)
        
        batch_size = 2
        N = 100
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        x_prev = tf.random.normal([batch_size, N, dx])
        log_w_prev = tf.fill([batch_size, N], -tf.math.log(tf.cast(N, tf.float32)))
        y_t = tf.random.normal([batch_size, dy])
        
        x_pre, x_t, log_w, w, parent, log_w_pre, logz_t = pf.step(
            x_prev, log_w_prev, y_t, resample=1
        )

        assert_all_finite(x_pre, x_t, log_w, w, log_w_pre, logz_t)


# =============================================================================
# Filter Warmup Tests
# =============================================================================

class TestFilterWarmup:
    """Tests for filter warmup/compilation."""

    def test_kalman_warmup_no_error(self, lgssm_3d):
        """KalmanFilter warmup should complete without error."""
        kf = KalmanFilter(lgssm_3d)
        kf.warmup(batch_size=2, T=5)

    def test_ekf_warmup_no_error(self, lgssm_3d):
        """EKF warmup should complete without error."""
        ekf = ExtendedKalmanFilter(lgssm_3d)
        ekf.warmup(batch_size=2, T=5)

    def test_ukf_warmup_no_error(self, lgssm_3d):
        """UKF warmup should complete without error."""
        ukf = UnscentedKalmanFilter(lgssm_3d)
        ukf.warmup(batch_size=2, T=5)

    def test_pf_warmup_no_error(self, lgssm_2d):
        """Bootstrap PF warmup should complete without error."""
        pf = BootstrapParticleFilter(lgssm_2d, num_particles=50)
        pf.warmup(batch_size=2, T=5, resample=1)
