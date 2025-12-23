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

    def __init__(self, v, dt, cov_eps, seed=42, jitter=1e-12):
        super().__init__(seed)
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.v = tf.convert_to_tensor(v, dtype=tf.float32)
        self.jitter = tf.convert_to_tensor(jitter, dtype=tf.float32)
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
        elif cov_eps.shape[0] == v.shape[-1]:
            Q_v = cov_eps  # only velocity noise
            zeros = tf.zeros((tf.shape(self.v)[-1], tf.shape(self.v)[-1]), tf.float32)
            self.cov_eps = tf.concat([
                tf.concat([self.jitter * tf.eye(tf.shape(self.v)[-1], dtype=tf.float32), zeros], axis=1),
                tf.concat([zeros, Q_v], axis=1)
            ], axis=0)
        else:
            raise ValueError("cov_eps must be [2*dim,2*dim] or [dim,dim] for ConstantVelocityMotionModel")
        self.cov_eps = self.cov_eps + self.jitter * tf.eye(self.state_dim, dtype=tf.float32)
        self.Lq = tf.linalg.cholesky(self.cov_eps)
        
    @property
    def state_dim(self):
        return int(self.cov_eps.shape[0])

    def f(self, x):
        return tf.einsum("ij,...j->...i", self.A, x)


class ConstantTurnRateMotionModel(MotionModel):
    """Standard CTRV state: [x, y, v, psi, omega]."""

    def __init__(self, dt, cov_eps, seed=42, jitter=1e-12):
        super().__init__(seed)
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.jitter = tf.convert_to_tensor(jitter, dtype=tf.float32)
        cov_eps = tf.convert_to_tensor(cov_eps, dtype=tf.float32)
        if cov_eps.shape[0] == 5:
            self.cov_eps = cov_eps
        elif cov_eps.shape[0] == 3:
            zeros = tf.zeros((5, 5), tf.float32)
            indices = tf.constant(
                [
                    [2, 2], [2, 3], [2, 4],
                    [3, 2], [3, 3], [3, 4],
                    [4, 2], [4, 3], [4, 4],
                ],
                dtype=tf.int32,
            )
            updates = tf.reshape(cov_eps, [-1])
            self.cov_eps = tf.tensor_scatter_nd_update(zeros, indices, updates)
        else:
            raise ValueError("cov_eps must be [5,5] or [3,3] for ConstantTurnRateMotionModel")
        self.cov_eps = self.cov_eps + self.jitter * tf.eye(5, dtype=tf.float32)
        self.Lq = tf.linalg.cholesky(self.cov_eps)

    @property
    def state_dim(self):
        return 5

    def f(self, x):
        px = x[..., 0]
        py = x[..., 1]
        v = x[..., 2]
        psi = x[..., 3]
        omega = x[..., 4]

        dt = self.dt
        omega_dt = omega * dt
        psi_next = psi + omega_dt

        small_turn = tf.abs(omega) < self.jitter
        cos_psi = tf.cos(psi)
        sin_psi = tf.sin(psi)
        cos_psi_next = tf.cos(psi_next)
        sin_psi_next = tf.sin(psi_next)

        v_dt = v * dt
        px_straight = px + v_dt * cos_psi
        py_straight = py + v_dt * sin_psi

        omega_safe = tf.where(small_turn, tf.ones_like(omega), omega)
        radius = v / omega_safe
        px_turn = px + radius * (sin_psi_next - sin_psi)
        py_turn = py + radius * (-cos_psi_next + cos_psi)

        px_next = tf.where(small_turn, px_straight, px_turn)
        py_next = tf.where(small_turn, py_straight, py_turn)

        return tf.stack([px_next, py_next, v, psi_next, omega], axis=-1)
