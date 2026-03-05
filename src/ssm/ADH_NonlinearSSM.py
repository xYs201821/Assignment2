"""SSM definitions for HMC-oriented experiments."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM
import src.dtype_config as _dc

tfd = tfp.distributions


class ADHNonlinearSSM(SSM):
    """Andrieu-Doucet-Holenstein model with augmented time state.

    Augmented state z_n = [x_n, t_n]:
      x_n = 0.5 * x_{n-1} + 25 * x_{n-1} / (1 + x_{n-1}^2) + 8 * cos(1.2 * t_n) + v_n
      t_n = t_{n-1} + 1 + eta_n
      y_n = x_n^2 / 20 + w_n

    where:
      v_n ~ N(0, sigma_v^2),
      w_n ~ N(0, sigma_w^2),
      eta_n ~ N(0, t_process_var).

    Small time-noise variance keeps covariance matrices strictly positive-definite
    for downstream filters.
    """

    def __init__(
        self,
        sigma_v: float = 1.0,
        sigma_w: float = 1.0,
        x0_mean: float = 0.0,
        x0_var: float = 5.0,
        t0: float = 0.0,
        t0_var: float = 1e-9,
        t_process_var: float = 1e-9,
        seed: int | None = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.sigma_v = tf.Variable(
            tf.reshape(tf.convert_to_tensor(sigma_v, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="sigma_v",
        )
        self.sigma_w = tf.Variable(
            tf.reshape(tf.convert_to_tensor(sigma_w, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="sigma_w",
        )
        self.t_process_var = tf.reshape(
            tf.convert_to_tensor(t_process_var, dtype=_dc.DTYPE),
            [],
        )

        self.m0 = tf.convert_to_tensor([x0_mean, t0], dtype=_dc.DTYPE)
        self.P0 = tf.linalg.diag(tf.convert_to_tensor([x0_var, t0_var], dtype=_dc.DTYPE))

        q_x = self.sigma_v ** 2
        q_t = self.t_process_var
        self.cov_eps_x = tf.linalg.diag(tf.convert_to_tensor([q_x, q_t], dtype=_dc.DTYPE))
        self.cov_eps_y = tf.reshape(self.sigma_w ** 2, [1, 1])
        self.L0 = tf.linalg.cholesky(self.P0)
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

    def update_params(
        self,
        sigma_v: float | None = None,
        sigma_w: float | None = None,
        x0_mean: float | None = None,
        x0_var: float | None = None,
        t0: float | None = None,
        t0_var: float | None = None,
    ) -> None:
        """Update model parameters."""
        if sigma_v is not None:
            self.sigma_v.assign(tf.reshape(tf.convert_to_tensor(sigma_v, dtype=_dc.DTYPE), []))
        if sigma_w is not None:
            self.sigma_w.assign(tf.reshape(tf.convert_to_tensor(sigma_w, dtype=_dc.DTYPE), []))

        if x0_mean is not None or t0 is not None:
            x0_cur = self.m0[0]
            t0_cur = self.m0[1]
            x0_new = x0_cur if x0_mean is None else tf.cast(x0_mean, _dc.DTYPE)
            t0_new = t0_cur if t0 is None else tf.cast(t0, _dc.DTYPE)
            self.m0 = tf.stack([x0_new, t0_new], axis=0)

        if x0_var is not None or t0_var is not None:
            x0v_cur = self.P0[0, 0]
            t0v_cur = self.P0[1, 1]
            x0v_new = x0v_cur if x0_var is None else tf.cast(x0_var, _dc.DTYPE)
            t0v_new = t0v_cur if t0_var is None else tf.cast(t0_var, _dc.DTYPE)
            self.P0 = tf.linalg.diag(tf.stack([x0v_new, t0v_new], axis=0))

        self.cov_eps_x = tf.linalg.diag(
            tf.convert_to_tensor([self.sigma_v ** 2, self.t_process_var], dtype=_dc.DTYPE)
        )
        self.cov_eps_y = tf.reshape(self.sigma_w ** 2, [1, 1])
        self.L0 = tf.linalg.cholesky(self.P0)
        self.Lq = tf.linalg.cholesky(self.cov_eps_x)
        self.Lr = tf.linalg.cholesky(self.cov_eps_y)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def q_dim(self) -> int:
        return 2

    @property
    def r_dim(self) -> int:
        return 1

    def _x_drift(self, x_prev: tf.Tensor, t_next: tf.Tensor) -> tf.Tensor:
        return (
            0.5 * x_prev
            + 25.0 * x_prev / (1.0 + tf.square(x_prev))
            + 8.0 * tf.cos(1.2 * t_next)
        )

    def f(self, z, **kwargs):
        """Deterministic transition mean over augmented state."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        x_prev = z[..., 0]
        t_prev = z[..., 1]
        t_next = t_prev + 1.0
        x_next = self._x_drift(x_prev, t_next)
        return tf.stack([x_next, t_next], axis=-1)

    def h(self, z, **kwargs):
        """Observation mean (depends on x component only)."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        x = z[..., 0]
        return (tf.square(x) / 20.0)[..., tf.newaxis]

    def initial_state_dist(self, shape, **kwargs):
        """Initial state distribution p(z_0)."""
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.L0)

    def transition_dist(self, z_prev, **kwargs):
        """Transition distribution p(z_n | z_{n-1})."""
        loc = self.f(z_prev)
        scale_diag = tf.stack([self.sigma_v, tf.sqrt(self.t_process_var)], axis=0)
        scale_diag = tf.broadcast_to(scale_diag, tf.shape(loc))
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def observation_dist(self, z, **kwargs):
        """Observation distribution p(y_n | z_n)."""
        loc = self.h(z)
        scale = tf.ones_like(loc) * self.sigma_w
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def f_with_noise(self, z, q, **kwargs):
        """Transition with additive process noise."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        return self.f(z) + q

    def h_with_noise(self, z, r, **kwargs):
        """Observation with additive measurement noise."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        return self.h(z) + r

    def jacobian_f_x(self, z, q):
        """Analytic Jacobian of f_with_noise wrt state z."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        x_prev = z[..., 0]
        t_prev = z[..., 1]
        t_next = t_prev + 1.0

        dfdx = 0.5 + 25.0 * (1.0 - tf.square(x_prev)) / tf.square(1.0 + tf.square(x_prev))
        dfdt = -8.0 * 1.2 * tf.sin(1.2 * t_next)

        row1 = tf.stack([dfdx, dfdt], axis=-1)
        row2 = tf.broadcast_to(
            tf.convert_to_tensor([0.0, 1.0], dtype=_dc.DTYPE),
            tf.shape(row1),
        )
        J = tf.stack([row1, row2], axis=-2)
        return J, self.f_with_noise(z, q)

    def jacobian_f_q(self, z, q):
        """Analytic Jacobian of f_with_noise wrt process noise q (identity)."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        batch_shape = tf.shape(z)[:-1]
        eye = tf.eye(self.q_dim, batch_shape=batch_shape, dtype=_dc.DTYPE)
        return eye, self.f_with_noise(z, q)

    def jacobian_h_x(self, z, r):
        """Analytic Jacobian of h_with_noise wrt state z."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        x = z[..., 0]
        row = tf.stack([x / 10.0, tf.zeros_like(x)], axis=-1)
        J = row[..., tf.newaxis, :]
        return J, self.h_with_noise(z, r)

    def jacobian_h_r(self, z, r):
        """Analytic Jacobian of h_with_noise wrt measurement noise r (identity)."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        batch_shape = tf.shape(z)[:-1]
        eye = tf.eye(self.r_dim, batch_shape=batch_shape, dtype=_dc.DTYPE)
        return eye, self.h_with_noise(z, r)
