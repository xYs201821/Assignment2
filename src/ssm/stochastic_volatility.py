import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM

tfd = tfp.distributions
tfb = tfp.bijectors


class StochasticVolatilitySSM(SSM):
    def __init__(self, alpha, sigma, beta, mu=0.01, noise_scale_func=False,
                 obs_mode="y", obs_eps=1e-6, seed=42):
        super().__init__(seed)
        self.mu = tf.convert_to_tensor(mu, dtype=tf.float32)
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        self.sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)
        self.obs_mode = obs_mode
        self.obs_eps = tf.convert_to_tensor(obs_eps, dtype=tf.float32)
        if noise_scale_func:
            self.g = lambda x: tf.exp(0.01 * tf.clip_by_value(tf.abs(x), 0.0, 10.0))
        else:
            self.g = lambda x: tf.ones_like(x)
        self.m0 = tf.constant([0.0], dtype=tf.float32)
        self.P0 = tf.reshape(self.sigma**2 / (1.0 - self.alpha**2), [1, 1])
        self.cov_eps_x = tf.eye(1, dtype=tf.float32)
        self.cov_eps_y = tf.eye(1, dtype=tf.float32)

    @property
    def state_dim(self):
        return 1

    @property
    def obs_dim(self):
        return 1

    @property
    def q_dim(self):
        return 1

    @property
    def r_dim(self):
        return 1

    def initial_state_dist(self, shape, **kwargs):
        shape = tf.convert_to_tensor(shape, tf.int32)
        L0 = tf.linalg.cholesky(self.P0)
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=L0)

    def transition_dist(self, x_prev, **kwargs):
        loc = self.f(x_prev)
        scale = self.sigma * self.g(x_prev)
        scale = tf.broadcast_to(scale, tf.shape(loc))
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def observation_dist(self, x, **kwargs):
        if self.obs_mode == "y":
            loc = tf.zeros(tf.concat([tf.shape(x)[:-1], [1]], axis=0), tf.float32)
            scale = self.beta * tf.exp(0.5 * x)
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        shift = tf.math.log(self.beta ** 2) + x
        base = tfd.Gamma(concentration=0.5, rate=0.5)
        bij = tfb.Chain([tfb.Shift(shift), tfb.Log(), tfb.Shift(self.obs_eps)])
        dist = tfd.TransformedDistribution(base, bijector=bij)
        return tfd.Independent(dist, reinterpreted_batch_ndims=1)

    def f(self, x):
        return self.mu + self.alpha * x

    def h(self, x):
        if self.obs_mode == "y":
            return tf.zeros(tf.concat([tf.shape(x)[:-1], [self.obs_dim]], axis=0), dtype=tf.float32)
        const = tf.math.digamma(0.5) + tf.math.log(2.0) + tf.math.log(self.beta ** 2)
        return x + const

    def f_with_noise(self, x, q):
        x_det = self.f(x)
        scale = self.sigma * self.g(x)
        return x_det + scale * q

    def h_with_noise(self, x, r):
        if self.obs_mode == "y":
            return self.beta * tf.exp(0.5 * x) * r
        return self.h(x) + r
