import tensorflow as tf

from src.ssm import LinearGaussianSSM

from src.flows.stochastic_pf import StochasticParticleFlow


def _make_flow(lgssm_2d, num_lambda=8):
    return StochasticParticleFlow(
        lgssm_2d,
        num_lambda=num_lambda,
        num_particles=10,
        diffusion=None,
    )


def test_optimal_beta_linear_schedule_info_zero():
    dx = 2
    dy = 1
    A = tf.eye(dx, dtype=tf.float32)
    B = tf.eye(dx, dtype=tf.float32)
    C = tf.zeros([dy, dx], dtype=tf.float32)
    D = tf.eye(dy, dtype=tf.float32)
    m0 = tf.zeros([dx], dtype=tf.float32)
    P0 = tf.eye(dx, dtype=tf.float32)
    ssm = LinearGaussianSSM(A, B, C, D, m0, P0)
    flow = _make_flow(ssm, num_lambda=8)

    x = tf.zeros([1, 1, dx], dtype=tf.float32)
    y = tf.zeros([1, dy], dtype=tf.float32)
    _, Info = flow._likelihood_terms(x, y)
    P0_inv = flow._inverse_from_cov(P0)[tf.newaxis, ...]

    beta, beta_dot, v0_star = flow.solve_optimal_beta_schedule(
        P0_inv,
        Info,
        mu=1.0,
        tol=1e-8,
    )

    expected_beta = tf.linspace(0.0, 1.0, flow.num_lambda + 1)[:-1][tf.newaxis, :]
    tf.debugging.assert_near(beta, expected_beta, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(beta_dot, tf.ones_like(beta_dot), atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(v0_star, tf.ones_like(v0_star), atol=1e-5, rtol=1e-5)
    assert beta.dtype == tf.float32
    assert beta_dot.dtype == tf.float32


def test_optimal_beta_hits_target_nonzero_info(lgssm_2d):
    flow = _make_flow(lgssm_2d, num_lambda=32)

    dx = lgssm_2d.state_dim
    dy = lgssm_2d.obs_dim
    x = tf.constant([[[0.4, -0.2]]], dtype=tf.float32)
    y = lgssm_2d.h(x[:, 0, :])
    y = tf.reshape(y, [1, dy])

    _, Info = flow._likelihood_terms(x, y)
    P0_inv = flow._inverse_from_cov(lgssm_2d.P0)[tf.newaxis, ...]
    target_beta = 1.0
    mu = 0.4

    _, _, v0_star = flow.solve_optimal_beta_schedule(
        P0_inv,
        Info,
        mu=mu,
        tol=1e-7,
        target_beta=target_beta,
    )

    def integrate_endpoint(v0):
        dtype = tf.float64
        P0_inv64 = tf.cast(P0_inv, dtype)
        Info64 = tf.cast(Info, dtype)
        mu64 = tf.cast(mu, dtype)
        beta = tf.zeros_like(tf.cast(v0, dtype))
        v = tf.cast(v0, dtype)
        h = tf.cast(1.0, dtype) / tf.cast(flow.num_lambda, dtype)

        trace_info = tf.linalg.trace(Info64)
        eye = tf.eye(tf.shape(P0_inv64)[-1], dtype=dtype)
        jitter = tf.cast(1e-6, dtype)

        def dkappa_dbeta(beta_val):
            M = P0_inv64 + beta_val * Info64
            M = 0.5 * (M + tf.linalg.matrix_transpose(M)) + jitter * eye
            M_inv = tf.linalg.inv(M)
            tr_M = tf.linalg.trace(M)
            tr_M_inv = tf.linalg.trace(M_inv)
            Minv_Info = tf.linalg.matmul(M_inv, Info64)
            tr_minv_info_minv = tf.linalg.trace(tf.linalg.matmul(Minv_Info, M_inv))
            return trace_info * tr_M_inv - tr_M * tr_minv_info_minv

        def rhs(beta_val, v_val):
            return v_val, mu64 * dkappa_dbeta(beta_val)

        for _ in range(flow.num_lambda):
            k1b, k1v = rhs(beta, v)
            k2b, k2v = rhs(beta + 0.5 * h * k1b, v + 0.5 * h * k1v)
            k3b, k3v = rhs(beta + 0.5 * h * k2b, v + 0.5 * h * k2v)
            k4b, k4v = rhs(beta + h * k3b, v + h * k3v)
            beta = beta + (h / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
            v = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return beta

    beta_end = integrate_endpoint(v0_star)
    tf.debugging.assert_near(
        beta_end,
        tf.cast(target_beta, beta_end.dtype),
        atol=5e-4,
        rtol=5e-4,
    )


def test_optimal_beta_matches_linear_when_mu_zero(lgssm_2d):
    flow = _make_flow(lgssm_2d, num_lambda=8)

    dx = lgssm_2d.state_dim
    dy = lgssm_2d.obs_dim
    x = tf.constant([[[0.4, -0.2]]], dtype=tf.float32)
    y = lgssm_2d.h(x[:, 0, :])
    y = tf.reshape(y, [1, dy])

    _, Info = flow._likelihood_terms(x, y)
    P0_inv = flow._inverse_from_cov(lgssm_2d.P0)[tf.newaxis, ...]

    beta, beta_dot, v0_star = flow.solve_optimal_beta_schedule(
        P0_inv,
        Info,
        mu=0.0,
        tol=1e-8,
    )

    expected_beta = tf.linspace(0.0, 1.0, flow.num_lambda + 1)[:-1][tf.newaxis, :]
    tf.debugging.assert_near(beta, expected_beta, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(beta_dot, tf.ones_like(beta_dot), atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(v0_star, tf.ones_like(v0_star), atol=1e-5, rtol=1e-5)
