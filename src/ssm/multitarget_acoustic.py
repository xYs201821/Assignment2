import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM

tfd = tfp.distributions


class MultiTargetAcousticSSM(SSM):
    """
    Multi-target acoustic tracking SSM (Li & Coates 2017, Eq.(38)).

    State: [x1,y1,vx1,vy1, ..., xC,yC,vxC,vyC]  in R^{4C}
    Transition: independent constant-velocity per target (linear Gaussian)
    Observation: Ns sensors, each measures sum_c Psi / (||pos_c - R_s||^2 + d0) + N(0, sigma_w^2)
    """

    def __init__(
        self,
        num_targets=4,
        sensor_xy=None,
        area_size=40.0,
        grid_size=5,
        dt=1.0,
        Psi=10.0,
        d0=0.1,
        cov_eps_x=None,
        sigma_w=0.1,
        m0=None,
        P0=None,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.C = int(num_targets)
        self.dt = tf.convert_to_tensor(dt, tf.float32)
        self.Psi = tf.convert_to_tensor(Psi, tf.float32)
        self.d0 = tf.convert_to_tensor(d0, tf.float32)
        self.sigma_w = tf.convert_to_tensor(sigma_w, tf.float32)

        if sensor_xy is None:
            xs = tf.linspace(0.0, float(area_size), int(grid_size))
            ys = tf.linspace(0.0, float(area_size), int(grid_size))
            X, Y = tf.meshgrid(xs, ys, indexing="xy")
            sensor_xy = tf.stack(
                [tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=-1
            )
        self.sensor_xy = tf.convert_to_tensor(sensor_xy, tf.float32)
        if self.sensor_xy.shape[0] is None:
            raise ValueError("sensor_xy must have a static first dimension")
        self.Ns = int(self.sensor_xy.shape[0])

        dt = self.dt
        F1 = tf.stack(
            [
                tf.stack([1.0, 0.0, dt, 0.0]),
                tf.stack([0.0, 1.0, 0.0, dt]),
                tf.stack([0.0, 0.0, 1.0, 0.0]),
                tf.stack([0.0, 0.0, 0.0, 1.0]),
            ],
            axis=0,
        )
        self.F1 = tf.cast(F1, tf.float32)

        if cov_eps_x is None:
            q = 1.0 / 20.0
            Q = q * tf.constant(
                [
                    [1.0 / 3.0, 0.0, 0.5, 0.0],
                    [0.0, 1.0 / 3.0, 0.0, 0.5],
                    [0.5, 0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.0, 1.0],
                ],
                dtype=tf.float32,
            )
        else:
            Q = tf.convert_to_tensor(cov_eps_x, tf.float32)
        self.Q = Q

        self.F = tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorIdentity(self.C, dtype=tf.float32),
                tf.linalg.LinearOperatorFullMatrix(self.F1),
            ]
        ).to_dense()

        self.cov_eps_x = tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorIdentity(self.C, dtype=tf.float32),
                tf.linalg.LinearOperatorFullMatrix(self.Q),
            ]
        ).to_dense()

        dx = 4 * self.C
        if m0 is None:
            m0 = tf.zeros([dx], tf.float32)
        if P0 is None:
            P0 = tf.eye(dx, dtype=tf.float32) * 10.0
        self.m0 = tf.convert_to_tensor(m0, tf.float32)
        self.P0 = tf.convert_to_tensor(P0, tf.float32)

        self.L0 = tf.linalg.cholesky(self.P0)
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.cov_eps_y = tf.eye(self.Ns, dtype=tf.float32) * (self.sigma_w ** 2)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

    @property
    def state_dim(self):
        return 4 * self.C

    @property
    def obs_dim(self):
        return self.Ns

    @property
    def q_dim(self):
        return self.state_dim

    @property
    def r_dim(self):
        return self.obs_dim

    def f(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        return tf.linalg.matvec(self.F, x)

    def h(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        shape = tf.shape(x)[:-1]
        x_reshaped = tf.reshape(x, tf.concat([shape, [self.C, 4]], axis=0))
        pos = x_reshaped[..., :, 0:2]

        sensor = self.sensor_xy[tf.newaxis, ...]
        pos_e = pos[..., :, tf.newaxis, :]

        diff = pos_e - sensor
        dist2 = tf.reduce_sum(diff * diff, axis=-1)
        contrib = self.Psi / (dist2 + self.d0)
        zbar = tf.reduce_sum(contrib, axis=-2)
        return zbar

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
