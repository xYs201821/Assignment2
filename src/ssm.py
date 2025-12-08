import tensorflow as tf
import numpy as np

class SSM(tf.Module):
    def __init__(self, A, B, C, D, m0, P0):
        super().__init__()

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
        return int(self.A.shape[0])

    @property
    def obs_dim(self):
        return int(self.C.shape[0])

    def initialize(self, batch_size = 1, seed = None):
        if seed is not None:
            tf.random.set_seed(seed)

        dx = self.state_dim
        L0 = tf.linalg.cholesky(self.P0)
        z0 = tf.random.normal(shape=(batch_size, dx), mean=0.0, stddev=1.0)
        x0 = tf.einsum('ij,bj->bi', L0, z0) + self.m0[tf.newaxis, :]

        return x0

    def step(self, x_prev):
        batch_size = tf.shape(x_prev)[0]
        dx = self.state_dim
        dy = self.obs_dim

        eps_q = tf.random.normal(shape=(batch_size, dx), mean=0.0, stddev=1.0)
        x_next = tf.einsum('ij,bj->bi', self.A, x_prev) + tf.einsum('ij,bj->bi', self.B, eps_q)

        eps_r = tf.random.normal(shape=(batch_size, dy), mean=0.0, stddev=1.0)
        y_next = tf.einsum('ij,bj->bi', self.C, x_next) + tf.einsum('ij,bj->bi', self.D, eps_r)
        return x_next, y_next
    
    def simulate(self, T, batch_size=1, x0=None, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        # ----- handle initial state -----
        if x0 is None:
            # sample x0 ~ N(m0, P0), shape = [batch_size, dx]
            x = self.initialize(batch_size=batch_size)
        else:
            # convert x0 to tensor
            x = tf.convert_to_tensor(x0, dtype=tf.float32)
            # [dx] → [1, dx]
            if len(x.shape) == 1:
                x = x[tf.newaxis, :]   
            # [batch, dx] → use its batch size
            batch_size = tf.shape(x)[0]

        x_traj = []
        y_traj = []
        for _ in range(T):
            x, y = self.step(x)
            x_traj.append(x)
            y_traj.append(y)

        x_traj = tf.stack(x_traj, axis=1)    # [batch, T, dx]
        y_traj = tf.stack(y_traj, axis=1)    # [batch, T, dy]

        return x_traj, y_traj
