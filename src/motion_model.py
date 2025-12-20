import tensorflow as tf
import numpy as np

class MotionModel(tf.Module):

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

    def f(self, x):
        """Transition function"""
        raise NotImplementedError
    
    def sample_transition_noise(self, shape):
        raise NotImplementedError

class ConstantVelocityMotionModel(MotionModel):

    def __init__(self, v, dt, cov_eps, seed=42):
        super().__init__(seed)
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.v = tf.convert_to_tensor(v, dtype=tf.float32)
        self.A = tf.stack(
            [
                tf.stack([1.0, 0.0, self.dt, 0.0]),
                tf.stack([0.0, 1.0, 0.0, self.dt]),
                tf.stack([0.0, 0.0, 1.0, 0.0]),
                tf.stack([0.0, 0.0, 0.0, 1.0]),
            ],
            axis=0,
        )
        if cov_eps.shape[0] == 2 * v.shape[-1]:
            self.cov_eps = cov_eps
            self.Lq = tf.linalg.cholesky(self.cov_eps)
        elif cov_eps.shape[0] == v.shape[-1]:
            Q_v = cov_eps  # only velocity noise
            zeros = tf.zeros((tf.shape(self.v)[-1], tf.shape(self.v)[-1]), tf.float32)
            self.cov_eps = tf.concat([
                tf.concat([1e-32 * tf.eye(tf.shape(self.v)[-1], dtype=tf.float32), zeros], axis=1),
                tf.concat([zeros, Q_v], axis=1)
            ], axis=0)
            self.Lq = tf.linalg.cholesky(self.cov_eps)
        else:
            raise ValueError("cov_eps must be [2*dim,2*dim] or [dim,dim] for ConstantVelocityMotionModel")
        
    @property
    def state_dim(self):
        return int(self.cov_eps.shape[0])

    def f(self, x):
        return tf.einsum("ij,...j->...i", self.A, x)


class ConstantTurnRateMotionModel(MotionModel):
    """State: [x, y, v, heading, omega]."""

    def __init__(self, dt, cov_eps, seed=42, eps=1e-6):
        super().__init__(seed)
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.eps = tf.convert_to_tensor(eps, dtype=tf.float32)
        cov_eps = tf.convert_to_tensor(cov_eps, dtype=tf.float32)
        if cov_eps.shape[0] == 5:
            self.cov_eps = cov_eps
        elif cov_eps.shape[0] == 2:
            zeros = tf.zeros((5, 5), tf.float32)
            indices = tf.constant([[2, 2], [2, 4], [4, 2], [4, 4]], dtype=tf.int32)
            updates = tf.stack([cov_eps[0, 0], cov_eps[0, 1], cov_eps[1, 0], cov_eps[1, 1]], axis=0)
            self.cov_eps = tf.tensor_scatter_nd_update(zeros, indices, updates)
        else:
            raise ValueError("cov_eps must be [5,5] or [2,2] for ConstantTurnRateMotionModel")
        self.cov_eps = self.cov_eps + self.eps * tf.eye(5, dtype=tf.float32)
        self.Lq = tf.linalg.cholesky(self.cov_eps)

    @property
    def state_dim(self):
        return 5

    def f(self, x):
        px = x[..., 0]
        py = x[..., 1]
        v = x[..., 2]
        heading = x[..., 3]
        omega = x[..., 4]

        dt = self.dt
        omega_dt = omega * dt
        sin_h = tf.sin(heading)
        cos_h = tf.cos(heading)
        sin_h_dt = tf.sin(heading + omega_dt)
        cos_h_dt = tf.cos(heading + omega_dt)

        omega_safe = tf.where(tf.abs(omega) < self.eps, tf.ones_like(omega), omega)
        px_turn = px + v / omega_safe * (sin_h_dt - sin_h)
        py_turn = py + v / omega_safe * (-cos_h_dt + cos_h)
        px_lin = px + v * dt * cos_h
        py_lin = py + v * dt * sin_h

        use_lin = tf.abs(omega) < self.eps
        px_next = tf.where(use_lin, px_lin, px_turn)
        py_next = tf.where(use_lin, py_lin, py_turn)
        heading_next = heading + omega_dt

        return tf.stack([px_next, py_next, v, heading_next, omega], axis=-1)
