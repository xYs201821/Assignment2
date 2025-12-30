import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from src.utility import weighted_mean

tfd = tfp.distributions


class SSM(tf.Module):
    def __init__(self, seed=None, name=None):
        super().__init__(name=name)
        if seed is not None:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = tf.random.Generator.from_non_deterministic_state()

    def _tfp_seed(self):
        return tf.cast(self.rng.make_seeds(2)[0], dtype=tf.int32)

    def initial_state_dist(self, shape, **kwargs):
        raise NotImplementedError

    def transition_dist(self, x_prev, **kwargs):
        raise NotImplementedError

    def observation_dist(self, x, **kwargs):
        raise NotImplementedError

    def set_seed(self, seed):
        self.rng = tf.random.Generator.from_seed(seed)
        print(f"{self.__class__.__name__} set seed to {seed}.")

    @property
    def state_dim(self):
        raise NotImplementedError

    @property
    def obs_dim(self):
        raise NotImplementedError

    @property
    def q_dim(self):
        raise NotImplementedError

    @property
    def r_dim(self):
        raise NotImplementedError

    @staticmethod
    def _weighted_mean(X, W=None, axis=1, normalize=True):
        return weighted_mean(X, W, axis=axis, normalize=normalize)

    @staticmethod
    def weighted_cov(X, W=None, axis=-2, mean_fn=None, residual_fn=None):
        """
        Weighted covariance over particles.
        X: [..., num, dim]
        W: [..., num] or [num]
        returns: [..., dim, dim]
        """
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        if W is None:
            W = tf.ones(tf.shape(X)[:-1], dtype=X.dtype)

        W = tf.convert_to_tensor(W, dtype=X.dtype)
        w_sum = tf.reduce_sum(W, axis=-1, keepdims=True)
        Wn = tf.math.divide_no_nan(W, w_sum)

        mean = mean_fn(X, Wn) if mean_fn is not None else weighted_mean(X, Wn, axis=axis)
        resid = residual_fn(X, mean) if residual_fn is not None else X - mean[:, tf.newaxis, :]
        cov = tf.einsum("...n,...ni,...nj->...ij", Wn, resid, resid)
        return mean, cov

    def sample_initial_state(self, shape, return_log_prob=False, **kwargs):
        dist = self.initial_state_dist(shape, **kwargs)
        if dist is None:
            raise NotImplementedError("initial state distribution not implemented")

        x0 = tf.cast(dist.sample(seed=self._tfp_seed()), dtype=tf.float32)
        if return_log_prob:
            return x0, tf.cast(dist.log_prob(x0), dtype=tf.float32)
        return x0

    def sample_transition(self, x_prev, return_log_prob=False, **kwargs):
        dist = self.transition_dist(x_prev, **kwargs)
        if dist is not None:
            x_next = tf.cast(dist.sample(seed=self._tfp_seed()), dtype=tf.float32)
            if return_log_prob:
                return x_next, tf.cast(dist.log_prob(x_next), dtype=tf.float32)
            return x_next
        else:
            raise NotImplementedError("transition distribution not implemented")

    def sample_observation(self, x, return_log_prob=False, **kwargs):
        dist = self.observation_dist(x, **kwargs)
        if dist is not None:
            y = tf.cast(dist.sample(seed=self._tfp_seed()), dtype=tf.float32)
            if return_log_prob:
                return y, tf.cast(dist.log_prob(y), dtype=tf.float32)
            return y
        else:
            raise NotImplementedError("observation distribution not implemented")

    def f(self, x, **kwargs):
        """Transition function"""
        raise NotImplementedError

    def h(self, x, **kwargs):
        """Observation function"""
        raise NotImplementedError

    def f_with_noise(self, x, q):
        return self.f(x) + q

    def h_with_noise(self, x, r):
        return self.h(x) + r

    def step(self, x_prev, **kwargs):
        x_next = self.sample_transition(x_prev)
        y_next = self.sample_observation(x_next)
        return x_next, y_next

    def simulate(self, T, shape, x0=None, **kwargs):
        if x0 is None:
            x = self.sample_initial_state(shape)
        else:
            x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
            shape = tf.convert_to_tensor(shape, dtype=tf.int32)
            x = tf.broadcast_to(
                tf.reshape(x0, tf.concat([tf.ones_like(shape), [self.state_dim]], axis=0)),
                tf.concat([shape, [self.state_dim]], axis=0)
            )

        x_traj = []
        y_traj = []
        for _ in range(T):
            x, y = self.step(x)
            x_traj.append(x)
            y_traj.append(y)

        x_traj = tf.stack(x_traj, axis=1)
        y_traj = tf.stack(y_traj, axis=1)
        return x_traj, y_traj

    def innovation(self, y, y_pred):
        return y - y_pred

    def measurement_mean(self, y, W=None, axis=-2):
        return self._weighted_mean(y, W, axis=axis)

    def measurement_residual(self, y, y_mean):
        return y - y_mean[..., tf.newaxis, :]

    def state_mean(self, x, W=None, axis=-2):
        return self._weighted_mean(x, W, axis=axis)

    def state_residual(self, x, x_mean):
        return x - x_mean[..., tf.newaxis, :]

    def state_cov(self, x, W=None, axis=-2):
        _, cov = self.weighted_cov(x, W, axis=axis, mean_fn=self.state_mean, residual_fn=self.state_residual)
        return cov

    def measurement_cov(self, y, W=None, axis=-2):
        _, cov = self.weighted_cov(y, W, axis=axis, mean_fn=self.measurement_mean, residual_fn=self.measurement_residual)
        return cov


class LinearGaussianSSM(SSM):
    def __init__(self, A, B, C, D, m0, P0, jitter=1e-6, seed=42):
        super().__init__(seed)
        self.jitter = tf.convert_to_tensor(jitter, dtype=tf.float32)
        self.A = tf.convert_to_tensor(A, dtype=tf.float32)
        self.B = tf.convert_to_tensor(B, dtype=tf.float32)
        self.C = tf.convert_to_tensor(C, dtype=tf.float32)
        self.D = tf.convert_to_tensor(D, dtype=tf.float32)

        self.m0 = tf.convert_to_tensor(m0, dtype=tf.float32)
        self.P0 = tf.convert_to_tensor(P0, dtype=tf.float32)

        self.cov_eps_x = tf.linalg.matmul(self.B, self.B, adjoint_b=True) 
        self.cov_eps_x = self.cov_eps_x + self.jitter * tf.eye(self.state_dim, dtype=tf.float32)
        self.cov_eps_y = tf.linalg.matmul(self.D, self.D, adjoint_b=True) 
        self.cov_eps_y = self.cov_eps_y + self.jitter * tf.eye(self.obs_dim, dtype=tf.float32)
        self.L0 = tf.linalg.cholesky(self.P0 + self.jitter * tf.eye(self.state_dim, dtype=tf.float32))
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

    @property
    def state_dim(self):
        return int(self.P0.shape[-1])

    @property
    def obs_dim(self):
        return int(self.cov_eps_y.shape[-1])

    @property
    def q_dim(self):
        return int(self.B.shape[-1])

    @property
    def r_dim(self):
        return int(self.D.shape[-1])

    def initial_state_dist(self, shape, **kwargs):
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.L0)

    def transition_dist(self, x_prev, **kwargs):
        loc = self.f(x_prev)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Lq)

    def observation_dist(self, x, **kwargs):
        loc = self.h(x)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Lr)

    def f(self, x):
        return tf.einsum("ij,...j->...i", self.A, x)

    def h(self, x):
        return tf.einsum("ij,...j->...i", self.C, x)
