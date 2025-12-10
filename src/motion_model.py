import tensorflow as tf
import numpy as np

class MotionModel(tf.Module):

    @property
    def state_dim(self):
        raise NotImplementedError

    def f(self, x):
        """Transition function"""
        raise NotImplementedError
    
    def sample_transition_noise(self, batch_size = 1, seed = None):
        raise NotImplementedError

class ConstantVelocityMotionModel(MotionModel):

    def __init__(self, v, dt, cov_eps):
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.v = tf.convert_to_tensor(v, dtype=tf.float32)
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
        
    @property
    def state_dim(self):
        return int(self.cov_eps.shape[0])

    def f(self, x):
        px, py, vx, vy = tf.unstack(x, axis=1)
        return tf.stack([px + vx * self.dt, py + vy * self.dt, vx, vy], axis=1)
    
    def sample_transition_noise(self, batch_size=1, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        q = tf.random.normal(shape=(batch_size, self.state_dim), mean=0.0, stddev=1.0)
        return tf.einsum('ij,bj->bi', self.Lq, q)