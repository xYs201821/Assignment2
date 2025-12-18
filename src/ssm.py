import tensorflow as tf
import numpy as np
from src.motion_model import MotionModel
from src.utility import weighted_mean

class SSM(tf.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = tf.random.Generator.from_non_deterministic_state()

    def set_seed(self, seed):
        self.rng = tf.random.Generator.from_seed(seed)
        print(f"{self.__class__.__name__} set seed to {seed}.")    
    @property 
    def state_dim(self):
        raise NotImplementedError

    @property
    def obs_dim(self):
        raise NotImplementedError

    @staticmethod
    def _weighted_mean(X, W=None, axis=1, normalize=True):
        return weighted_mean(X, W, axis=axis, normalize=normalize)
    
    def sample_initial_state(self, shape):
        raise NotImplementedError

    def sample_transition_noise(self, shape):
        raise NotImplementedError
    
    def sample_observation_noise(self, shape):
        raise NotImplementedError
    
    def f(self, x):
        """Transition function"""
        raise NotImplementedError
    
    def h(self, x):
        """Observation function"""
        raise NotImplementedError
    
    def innovation(self, y, y_pred):
        return y - y_pred

    def measurement_mean(self, y, W=None, axis=1): # [batch, n, dy], [n] -> [batch, dy]
        """ compute weighted mean of observations """
        return self._weighted_mean(y, W, axis=axis)

    def measurement_residual(self, y, y_mean): # [batch, n, dy], [batch, dy] -> [batch, n, dy]
        return y - y_mean[:, tf.newaxis, :]

    def state_mean(self, x, W=None, axis=1): # [batch, n, dx], [n] -> [batch, dx]
        return self._weighted_mean(x, W, axis=axis)

    def state_residual(self, x, x_mean): # [batch, n, dx], [batch, dx] -> [batch, n, dx]
        return x - x_mean[:, tf.newaxis, :]

    def f_with_noise(self, x, q): # additive noise model
        return self.f(x) + q

    def h_with_noise(self, x, r): # additive noise model
        return self.h(x) + r

    def log_likelihood(self, y, x):
        '''
        Placeholder. Must return log p(y|x).
        y: [B, dy] or [..., dy]
        x: [B, N, dx] or [..., dx]
            return: [..., N] or [...]
        '''
        raise NotImplementedError
    
    def sample_transition(self, x_prev, q=None):
        if q is None:
            q = self.sample_transition_noise(tf.shape(x_prev)[:-1])
        return self.f_with_noise(x_prev, q) # [batch, dx]
    
    def sample_observation(self, x, r=None):
        if r is None:
            r = self.sample_observation_noise(tf.shape(x)[:-1])
        return self.h_with_noise(x, r)

    def step(self, x_prev): # by default, additive noise model
        x_next = self.sample_transition(x_prev)
        y_next = self.sample_observation(x_next)
        return x_next, y_next
    
    def simulate(self, T, shape, x0=None):
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

        x_traj = tf.stack(x_traj, axis=1)  # [batch, T, dx]
        y_traj = tf.stack(y_traj, axis=1)  # [batch, T, dy]
        return x_traj, y_traj

class LinearGaussianSSM(SSM):
    def __init__(self, A, B, C, D, m0, P0, seed=42):
        super().__init__(seed)

        self.A = tf.convert_to_tensor(A, dtype=tf.float32)
        self.B = tf.convert_to_tensor(B, dtype=tf.float32)
        self.C = tf.convert_to_tensor(C, dtype=tf.float32)
        self.D = tf.convert_to_tensor(D, dtype=tf.float32)

        self.m0 = tf.convert_to_tensor(m0, dtype=tf.float32)
        self.P0 = tf.convert_to_tensor(P0, dtype=tf.float32)

        self.cov_eps_x = tf.linalg.matmul(self.B, self.B, adjoint_b=True)
        self.cov_eps_y = tf.linalg.matmul(self.D, self.D, adjoint_b=True)

    @property
    def state_dim(self):
        return int(self.P0.shape[0])

    @property
    def obs_dim(self):
        return int(self.cov_eps_y.shape[0])

    def sample_initial_state(self, shape):
        L0 = tf.linalg.cholesky(self.P0)
        z0 = self.rng.normal(tf.concat([shape, [self.state_dim]], axis=0), mean=0.0, stddev=1.0)
        x0 = tf.einsum('ij,...j->...i', L0, z0) + self.m0
        return x0

    def sample_transition_noise(self, shape):
        q_dim = int(self.B.shape[-1])
        q = self.rng.normal(tf.concat([shape, [q_dim]], axis=0), mean=0.0, stddev=1.0)
        return tf.einsum('ij,...j->...i', self.B, q) # [batch, q_dim] -> [batch, dx]

    def sample_observation_noise(self, shape):
        r_dim = int(self.D.shape[-1])
        r = self.rng.normal(tf.concat([shape, [r_dim]], axis=0), mean=0.0, stddev=1.0)
        return tf.einsum('ij,...j->...i', self.D, r) # [batch, r_dim] -> [batch, dy]

    def f(self, x):
        return tf.einsum('ij,...j->...i', self.A, x) 

    def h(self, x):
        return tf.einsum('ij,...j->...i', self.C, x)

class StochasticVolatilitySSM(SSM):

    def __init__(self, alpha, sigma, beta, seed=42):
        super().__init__(seed)
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        self.sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)

        self.m0 = tf.constant([0.0], dtype=tf.float32)
        self.P0 = tf.reshape(self.sigma**2 / (1.0 - self.alpha**2), [1, 1])
        self.cov_eps_x = self.sigma**2 * tf.eye(1, dtype=tf.float32)
        self.cov_eps_y = tf.eye(1, dtype=tf.float32)

    @property
    def state_dim(self): # [x_t]
        return 1
        
    @property
    def obs_dim(self):
        return 1

    def sample_initial_state(self, shape):
        L0 = tf.linalg.cholesky(self.P0)
        z0 = self.rng.normal(tf.concat([shape, [self.state_dim]], axis=0), mean=0.0, stddev=1.0)
        x0 = self.m0 + tf.einsum('ij,...j->...i', L0, z0)
        return x0
    
    def sample_transition_noise(self, shape):
        q = self.rng.normal(tf.concat([shape, [self.state_dim]], axis=0), mean=0.0, stddev=1.0)
        return self.sigma * q


    def sample_observation_noise(self, shape):
        return self.rng.normal(tf.concat([shape, [self.obs_dim]], axis=0), mean=0.0, stddev=1.0)
    
    def f(self, x):
        return self.alpha * x
    
    def h(self, x):
        return tf.zeros(tf.concat([tf.shape(x)[:-1], [self.obs_dim]], axis=0), dtype=tf.float32)

    def f_with_noise(self, x, q):
        x_det = self.f(x)
        return x_det + q
    
    def h_with_noise(self, x, r):
        return self.beta * tf.exp(0.5 * x) * r

class RangeBearingSSM(SSM):
    
    def __init__(self, motion_model, cov_eps_y, seed=42):
        super().__init__(seed)
        assert isinstance(motion_model, MotionModel)
        self.motion_model = motion_model

        self.m0 = tf.zeros([self.motion_model.state_dim], dtype=tf.float32)
        self.P0 = tf.eye(self.motion_model.state_dim, dtype=tf.float32)
        self.cov_eps_x = self.motion_model.cov_eps
        self.cov_eps_y = tf.convert_to_tensor(cov_eps_y, dtype=tf.float32)
        self.angle_indices = (1, )
        self.L0 = tf.linalg.cholesky(self.P0)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

    @staticmethod
    def _wrap_angle(bearing):
        return tf.math.atan2(tf.sin(bearing), tf.cos(bearing))

    @property
    def state_dim(self):
        return self.motion_model.state_dim

    @property
    def obs_dim(self):
        return int(self.cov_eps_y.shape[0])
    
    def sample_initial_state(self, shape):
        z0 = self.rng.normal(tf.concat([shape, [self.state_dim]], axis=0), mean=0.0, stddev=1.0)
        x0 = self.m0 + tf.einsum('ij,...j->...i', self.L0, z0)
        return x0

    def sample_transition_noise(self, shape):
        return self.motion_model.sample_transition_noise(shape)

    def sample_observation_noise(self, shape):
        return tf.einsum('ij,...j->...i', self.Lr, self.rng.normal(tf.concat([shape, [self.obs_dim]], axis=0), mean=0.0, stddev=1.0))

    def f(self, x):
        return self.motion_model.f(x)  

    def h(self, x):
        px = x[..., 0]
        py = x[..., 1]

        rng = tf.sqrt(px**2 + py**2 + 1e-20)
        bearing = self._wrap_angle(tf.atan2(py, px))
        y = tf.stack([rng, bearing], axis=-1)

        return y

    def innovation(self, y, y_pred):
        v = y - y_pred
        v_bearing = self._wrap_angle(v[..., 1])
        return tf.stack([v[..., 0], v_bearing], axis=-1)

    def measurement_mean(self, x, W=None): # [batch, n, 2], [n] -> [batch, 2]
        if W is None:
            W = tf.ones([tf.shape(x)[-1]], dtype=tf.float32)
        mean_range = self._weighted_mean(x[..., 0], W, axis=-1)
        mean_sin_bearing = self._weighted_mean(tf.sin(x[..., 1]), W, axis=-1)
        mean_cos_bearing = self._weighted_mean(tf.cos(x[..., 1]), W, axis=-1)
        mean_bearing = tf.math.atan2(mean_sin_bearing, mean_cos_bearing) # [batch]
        return tf.stack([mean_range, mean_bearing], axis=-1) # [batch, 2]

    def measurement_residual(self, y, y_mean): # [batch, n, 2], [batch, 2] -> [batch, n, 2]
        dr  = y[..., 0] - y_mean[..., 0][..., None]
        dth = self._wrap_angle(y[..., 1] - y_mean[..., 1][..., None])
        return tf.stack([dr, dth], axis=-1)  