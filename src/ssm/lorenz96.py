import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM

tfd = tfp.distributions


class Lorenz96SSM(SSM):
    """
    High-dimensional SSM.

    State model: dx_a/dt = (x_{a+1} - x_{a-2}) * x_{a-1} - x_a + F
    Discrete time: x_{t+1} = x_t + dt * f(x_t) + eps_t

    Observation operators:
      linear:  y = [x_4, x_8, ..., x_{n_x}]
      abs:     y = [|x_4|, |x_8|, ..., |x_{n_x}|]
      exp:     y = [exp(x_4), exp(x_8), ..., x_{n_x}]
      square:  y = [x_4^2, x_8^2, ..., x_{n_x}^2]
    """

    def __init__(
        self,
        state_dim=40,
        obs_stride=4,
        dt=0.05,
        F=8.0,
        obs_op="linear",
        q_scale=0.1,
        r_scale=0.5,
        m0=None,
        P0=None,
        seed=None,
    ):
        super().__init__(seed=seed)
        self._state_dim = int(state_dim)
        self.obs_stride = int(obs_stride)
        if self.obs_stride <= 0:
            raise ValueError("obs_stride must be positive")
        if self._state_dim % self.obs_stride != 0:
            raise ValueError("state_dim must be divisible by obs_stride")

        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.F = tf.convert_to_tensor(F, dtype=tf.float32)
        self.obs_op = self._validate_obs_op(obs_op)

        indices = np.arange(self.obs_stride - 1, self._state_dim, self.obs_stride, dtype=np.int32)
        self.obs_indices = tf.convert_to_tensor(indices, dtype=tf.int32)
        self._obs_dim = int(indices.size)

        self.cov_eps_x = tf.eye(self._state_dim, dtype=tf.float32) * (float(q_scale) ** 2)
        self.cov_eps_y = tf.eye(self._obs_dim, dtype=tf.float32) * (float(r_scale) ** 2)
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

        if m0 is None:
            m0 = tf.ones([self._state_dim], dtype=tf.float32) * self.F
        if P0 is None:
            P0 = tf.eye(self._state_dim, dtype=tf.float32)
        self.m0 = tf.convert_to_tensor(m0, dtype=tf.float32)
        self.P0 = tf.convert_to_tensor(P0, dtype=tf.float32)
        self.L0 = tf.linalg.cholesky(self.P0)

    @staticmethod
    def _validate_obs_op(value):
        mode = str(value).lower()
        valid = {"linear", "abs", "exp", "square"}
        if mode not in valid:
            raise ValueError("obs_op must be one of: linear, abs, exp, square")
        return mode

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def q_dim(self):
        return self._state_dim

    @property
    def r_dim(self):
        return self._obs_dim

    def _l96_rhs(self, x):
        xp1 = tf.roll(x, shift=-1, axis=-1)
        xm2 = tf.roll(x, shift=2, axis=-1)
        xm1 = tf.roll(x, shift=1, axis=-1)
        return (xp1 - xm2) * xm1 - x + self.F

    def f(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x + self.dt * self._l96_rhs(x)

    def h(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_obs = tf.gather(x, self.obs_indices, axis=-1)
        if self.obs_op == "linear":
            return x_obs
        if self.obs_op == "abs":
            return tf.abs(x_obs)
        if self.obs_op == "exp":
            return tf.exp(x_obs)
        if self.obs_op == "square":
            return tf.square(x_obs)
        raise ValueError("Invalid obs_op")

    def initial_state_dist(self, shape, **kwargs):
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.L0)

    def transition_dist(self, x_prev, **kwargs):
        loc = self.f(x_prev)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Lq)

    def observation_dist(self, x, **kwargs):
        loc = self.h(x)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Lr)