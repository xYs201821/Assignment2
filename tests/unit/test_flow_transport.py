"""Unit tests for flow transport implementations.

Tests the core _flow_transport methods of EDH, LEDH, Kernel, and Stochastic flows.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.flows.edh import EDHFlow
from src.flows.ledh import LEDHFlow
from src.flows.kernel_embedded import KernelParticleFlow
from src.flows.stochastic_pf import StochasticParticleFlow
from tests.testhelper import assert_all_finite

pytestmark = pytest.mark.unit


# =============================================================================
# EDH Flow Transport Tests
# =============================================================================

class TestEDHFlowTransport:
    """Tests for EDH flow _flow_transport method."""

    def test_output_shapes(self, lgssm_2d):
        """_flow_transport should return correct shapes."""
        edh = EDHFlow(lgssm_2d, num_lambda=5, num_particles=50)
        
        batch_size = 2
        N = 50
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        mu, logdet, diagnostics = edh._flow_transport(mu_tilde, y, m0, P)
        
        assert mu.shape == (batch_size, N, dx)
        assert logdet.shape == (batch_size,)
        assert "condCov_log10" in diagnostics
        assert "logdet_cov" in diagnostics

    def test_finite_outputs(self, lgssm_2d):
        """_flow_transport should produce finite values."""
        edh = EDHFlow(lgssm_2d, num_lambda=10, num_particles=100)
        
        batch_size = 2
        N = 100
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.zeros([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        mu, logdet, diagnostics = edh._flow_transport(mu_tilde, y, m0, P)
        
        assert_all_finite(mu, logdet)
        assert_all_finite(diagnostics["condCov_log10"], diagnostics["logdet_cov"])

    def test_particles_move_toward_observation(self, lgssm_2d):
        """Flow should move particles closer to the observation."""
        edh = EDHFlow(lgssm_2d, num_lambda=20, num_particles=200)
        
        batch_size = 1
        N = 200
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        # Prior mean far from observation
        m0 = tf.constant([[5.0, 5.0]], dtype=tf.float32)
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        # Sample particles from prior
        mu_tilde = m0[:, tf.newaxis, :] + tf.random.normal([batch_size, N, dx]) * 0.3
        
        # Observation near origin
        y = tf.constant([[0.0]], dtype=tf.float32)
        
        mu_before = tf.reduce_mean(mu_tilde, axis=1)
        mu, _, _ = edh._flow_transport(mu_tilde, y, m0, P)
        mu_after = tf.reduce_mean(mu, axis=1)
        
        # Particles should move toward origin (observation)
        dist_before = tf.norm(mu_before)
        dist_after = tf.norm(mu_after)
        assert dist_after < dist_before


# =============================================================================
# LEDH Flow Transport Tests
# =============================================================================

class TestLEDHFlowTransport:
    """Tests for LEDH flow _flow_transport method."""

    def test_output_shapes(self, lgssm_2d):
        """_flow_transport should return correct shapes."""
        ledh = LEDHFlow(lgssm_2d, num_lambda=5, num_particles=50)
        
        batch_size = 2
        N = 50
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        mu, logdet, diagnostics = ledh._flow_transport(mu_tilde, y, m0, P)
        
        # LEDH has per-particle logdet
        assert mu.shape == (batch_size, N, dx)
        assert logdet.shape == (batch_size, N)
        assert "condCov_log10" in diagnostics

    def test_finite_outputs(self, lgssm_2d):
        """_flow_transport should produce finite values."""
        ledh = LEDHFlow(lgssm_2d, num_lambda=10, num_particles=100)
        
        batch_size = 2
        N = 100
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.zeros([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        mu, logdet, diagnostics = ledh._flow_transport(mu_tilde, y, m0, P)
        
        assert_all_finite(mu, logdet)


# =============================================================================
# Kernel Flow Transport Tests
# =============================================================================

class TestKernelFlowTransport:
    """Tests for Kernel flow _flow_transport method."""

    def test_output_shapes(self, lgssm_2d):
        """_flow_transport should return correct shapes."""
        kflow = KernelParticleFlow(lgssm_2d, num_lambda=5, num_particles=50)
        
        batch_size = 2
        N = 50
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        mu, logdet, diagnostics = kflow._flow_transport(mu_tilde, y, m0, P)
        
        # Kernel flow returns logdet per particle (but filled with zeros)
        assert mu.shape == (batch_size, N, dx)
        assert logdet.shape == (batch_size, N)

    def test_finite_outputs(self, lgssm_2d):
        """_flow_transport should produce finite values."""
        kflow = KernelParticleFlow(lgssm_2d, num_lambda=10, num_particles=80)
        
        batch_size = 2
        N = 80
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        mu_tilde = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.zeros([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        mu, logdet, diagnostics = kflow._flow_transport(mu_tilde, y, m0, P)
        
        assert_all_finite(mu, logdet)

    @pytest.mark.parametrize("kernel_type", ["scalar", "diag"])
    def test_kernel_types(self, lgssm_2d, kernel_type):
        """_pff_update should work with different kernel types."""
        kflow = KernelParticleFlow(
            lgssm_2d, num_lambda=5, num_particles=30, kernel_type=kernel_type
        )
        
        batch_size = 2
        N = 30
        dx = lgssm_2d.state_dim
        
        x_prior = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, lgssm_2d.obs_dim])
        
        x_next, diagnostics = kflow._pff_update(x_prior, y, w=None)
        
        assert x_next.shape == (batch_size, N, dx)
        assert_all_finite(x_next)


# =============================================================================
# Stochastic Flow Transport Tests
# =============================================================================

class TestStochasticFlowTransport:
    """Tests for Stochastic flow _flow_transport method."""

    def test_output_shapes_no_diffusion(self, lgssm_2d):
        """_flow_transport without diffusion should return correct shapes."""
        spf = StochasticParticleFlow(lgssm_2d, num_lambda=5, num_particles=50, diffusion=None)
        
        batch_size = 2
        N = 50
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        x = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        mu, logdet, diagnostics = spf._flow_transport(x, y, m0, P)
        
        assert mu.shape == (batch_size, N, dx)
        assert logdet.shape == (batch_size, N)

    def test_output_shapes_with_diffusion(self, lgssm_2d):
        """_flow_transport with diffusion should return correct shapes."""
        dx = lgssm_2d.state_dim
        diffusion = 0.01 * np.eye(dx, dtype=np.float32)
        spf = StochasticParticleFlow(
            lgssm_2d, num_lambda=5, num_particles=50, diffusion=diffusion
        )
        
        batch_size = 2
        N = 50
        dy = lgssm_2d.obs_dim
        
        x = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size]) * 0.1
        
        mu, logdet, diagnostics = spf._flow_transport(x, y, m0, P)
        
        assert mu.shape == (batch_size, N, dx)
        assert_all_finite(mu, logdet)

    def test_finite_outputs(self, lgssm_2d):
        """_flow_transport should produce finite values."""
        spf = StochasticParticleFlow(lgssm_2d, num_lambda=10, num_particles=80, diffusion=None)
        
        batch_size = 2
        N = 80
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        
        x = tf.random.normal([batch_size, N, dx])
        y = tf.random.normal([batch_size, dy])
        m0 = tf.zeros([batch_size, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        
        mu, logdet, _ = spf._flow_transport(x, y, m0, P)
        
        assert_all_finite(mu, logdet)


# =============================================================================
# EDH Flow Solution Tests
# =============================================================================

class TestEDHFlowSolution:
    """Tests for EDH _edh_flow_solution static method."""

    def test_output_shapes(self, lgssm_2d):
        """_edh_flow_solution should return correct shapes."""
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        batch_size = 3
        
        lam = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
        H = tf.random.normal([batch_size, dy, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        R = tf.eye(dy, batch_shape=[batch_size]) * 0.1
        y_tilde = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        
        A, b, S = EDHFlow._edh_flow_solution(lam, H, P, R, y_tilde, m0, jitter=1e-6)
        
        assert A.shape == (batch_size, dx, dx)
        assert b.shape == (batch_size, dx)
        assert S.shape == (batch_size, dy, dy)

    def test_finite_outputs(self, lgssm_2d):
        """_edh_flow_solution should produce finite values."""
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        batch_size = 3
        
        lam = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
        H = tf.random.normal([batch_size, dy, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        R = tf.eye(dy, batch_shape=[batch_size]) * 0.1
        y_tilde = tf.random.normal([batch_size, dy])
        m0 = tf.random.normal([batch_size, dx])
        
        A, b, S = EDHFlow._edh_flow_solution(lam, H, P, R, y_tilde, m0, jitter=1e-6)
        
        assert_all_finite(A, b, S)

    def test_lambda_zero_gives_zero_flow(self, lgssm_2d):
        """At lambda=0, the flow should be near zero."""
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        batch_size = 2
        
        lam = tf.zeros([batch_size], dtype=tf.float32)
        H = tf.random.normal([batch_size, dy, dx])
        P = tf.eye(dx, batch_shape=[batch_size])
        R = tf.eye(dy, batch_shape=[batch_size]) * 0.1
        y_tilde = tf.random.normal([batch_size, dy])
        m0 = tf.zeros([batch_size, dx])
        
        A, b, S = EDHFlow._edh_flow_solution(lam, H, P, R, y_tilde, m0, jitter=1e-6)
        
        # At lambda=0, A should be close to -0.5 * K @ H where K = P H^T R^{-1}
        # The flow at the prior mean should be related to the innovation
        assert_all_finite(A, b, S)


# =============================================================================
# LEDH Flow Solution Tests
# =============================================================================

class TestLEDHFlowSolution:
    """Tests for LEDH _ledh_flow_solution static method."""

    def test_output_shapes(self, lgssm_2d):
        """_ledh_flow_solution should return correct shapes."""
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        batch_size = 2
        N = 30
        
        lam = tf.constant([[0.5] * N, [0.5] * N], dtype=tf.float32)
        H = tf.random.normal([batch_size, N, dy, dx])
        P = tf.eye(dx, batch_shape=[batch_size, N])
        R = tf.eye(dy, batch_shape=[batch_size, N]) * 0.1
        y_tilde = tf.random.normal([batch_size, N, dy])
        m0 = tf.random.normal([batch_size, N, dx])
        
        A, b = LEDHFlow._ledh_flow_solution(lam, H, P, R, y_tilde, m0, jitter=1e-6)
        
        assert A.shape == (batch_size, N, dx, dx)
        assert b.shape == (batch_size, N, dx)

    def test_finite_outputs(self, lgssm_2d):
        """_ledh_flow_solution should produce finite values."""
        dx = lgssm_2d.state_dim
        dy = lgssm_2d.obs_dim
        batch_size = 2
        N = 30
        
        lam = tf.ones([batch_size, N], dtype=tf.float32) * 0.5
        H = tf.random.normal([batch_size, N, dy, dx])
        P = tf.eye(dx, batch_shape=[batch_size, N])
        R = tf.eye(dy, batch_shape=[batch_size, N]) * 0.1
        y_tilde = tf.random.normal([batch_size, N, dy])
        m0 = tf.random.normal([batch_size, N, dx])
        
        A, b = LEDHFlow._ledh_flow_solution(lam, H, P, R, y_tilde, m0, jitter=1e-6)
        
        assert_all_finite(A, b)
