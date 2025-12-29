import tensorflow as tf
import tensorflow_probability as tfp

from src.motion_model import MotionModel
from src.ssm.base import SSM

tfd = tfp.distributions


class _RangeBearingObservationDist:
    def __init__(self, ssm, x):
        self._ssm = ssm
        self._x = tf.convert_to_tensor(x, dtype=tf.float32)

    def sample(self, seed=None):
        loc = self._ssm.h(self._x)
        dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=self._ssm.Lr)
        y = dist.sample(seed=seed)
        return tf.stack([y[..., 0], self._ssm._wrap_angle(y[..., 1])], axis=-1)

    def log_prob(self, y):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        loc = self._ssm.h(self._x)
        v = self._ssm.innovation(y, loc)
        return self._ssm._obs_noise_dist.log_prob(v)


class RangeBearingSSM(SSM):
    def __init__(self, motion_model, cov_eps_y, jitter=1e-12, seed=42):
        super().__init__(seed)
        assert isinstance(motion_model, MotionModel)
        self.motion_model = motion_model
        self.jitter = tf.convert_to_tensor(jitter, dtype=tf.float32)
        self.m0 = tf.zeros([self.motion_model.state_dim], dtype=tf.float32)
        self.P0 = tf.eye(self.motion_model.state_dim, dtype=tf.float32)
        self.cov_eps_x = self.motion_model.cov_eps
        self.cov_eps_y = tf.convert_to_tensor(cov_eps_y, dtype=tf.float32)
        self.angle_indices = (1,)
        self.L0 = tf.linalg.cholesky(self.P0)
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)
        self._obs_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros([self.obs_dim], dtype=tf.float32),
            scale_tril=self.Lr,
        )

    @staticmethod
    def _wrap_angle(bearing):
        return tf.math.atan2(tf.sin(bearing), tf.cos(bearing))

    @property
    def state_dim(self):
        return self.motion_model.state_dim

    @property
    def obs_dim(self):
        return int(self.cov_eps_y.shape[-1])

    @property
    def q_dim(self):
        return int(self.cov_eps_x.shape[-1])

    @property
    def r_dim(self):
        return int(self.cov_eps_y.shape[-1])

    def initial_state_dist(self, shape, **kwargs):
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.L0)

    def transition_dist(self, x_prev, **kwargs):
        loc = self.motion_model.f(x_prev)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Lq)

    def observation_dist(self, x, **kwargs):
        return _RangeBearingObservationDist(self, x)

    def f(self, x):
        return self.motion_model.f(x)

    def h(self, x):
        px = x[..., 0]
        py = x[..., 1]

        rng_sq = px**2 + py**2
        rng = tf.sqrt(rng_sq + self.jitter)
        px_safe = tf.where(rng_sq < self.jitter, self.jitter, px)
        py_safe = tf.where(rng_sq < self.jitter, tf.zeros_like(py), py)
        bearing = tf.atan2(py_safe, px_safe)
        y = tf.stack([rng, bearing], axis=-1)
        return y

    def innovation(self, y, y_pred):
        v = y - y_pred
        v_bearing = self._wrap_angle(v[..., 1])
        return tf.stack([v[..., 0], v_bearing], axis=-1)

    def measurement_mean(self, x, W=None):
        if W is None:
            W = tf.ones([tf.shape(x)[-2]], dtype=tf.float32)
        mean_range = self._weighted_mean(x[..., 0], W, axis=-1)
        mean_sin_bearing = self._weighted_mean(tf.sin(x[..., 1]), W, axis=-1)
        mean_cos_bearing = self._weighted_mean(tf.cos(x[..., 1]), W, axis=-1)
        mean_bearing = tf.math.atan2(mean_sin_bearing, mean_cos_bearing)
        return tf.stack([mean_range, mean_bearing], axis=-1)

    def measurement_residual(self, y, y_mean):
        dr = y[..., 0] - y_mean[..., 0][..., None]
        dth = self._wrap_angle(y[..., 1] - y_mean[..., 1][..., None])
        return tf.stack([dr, dth], axis=-1)
