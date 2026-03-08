"""SSM definitions for HMC-oriented experiments."""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM
import src.dtype_config as _dc

tfd = tfp.distributions


class ADHNonlinearSSM(SSM):
    """Andrieu-Doucet-Holenstein model with deterministic external time."""

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
        self.x0_mean = tf.Variable(
            tf.reshape(tf.convert_to_tensor(x0_mean, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="x0_mean",
        )
        self.x0_var = tf.Variable(
            tf.reshape(tf.convert_to_tensor(x0_var, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="x0_var",
        )
        self.t0_mean = tf.Variable(
            tf.reshape(tf.convert_to_tensor(t0, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="t0_mean",
        )
        self.t0_var = tf.Variable(
            tf.reshape(tf.convert_to_tensor(t0_var, dtype=_dc.DTYPE), []),
            trainable=False,
            dtype=_dc.DTYPE,
            name="t0_var",
        )
        self.t_process_var = tf.reshape(
            tf.convert_to_tensor(t_process_var, dtype=_dc.DTYPE),
            [],
        )

        self._refresh_initial_state_moments()
        self.L0 = tf.linalg.cholesky(self.P0)
        self._refresh_cached_covariances()

    @staticmethod
    def _to_scalar(value) -> tf.Tensor:
        return tf.reshape(tf.convert_to_tensor(value, dtype=_dc.DTYPE), [])

    def _resolve_runtime_params(
        self,
        params: Mapping[str, tf.Tensor | float] | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if params is None:
            return (
                self._to_scalar(self.sigma_v),
                self._to_scalar(self.sigma_w),
            )
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None.")
        sigma_v = params.get("sigma_v", self.sigma_v)
        sigma_w = params.get("sigma_w", self.sigma_w)
        return (
            self._to_scalar(sigma_v),
            self._to_scalar(sigma_w),
        )

    def _refresh_initial_state_moments(self) -> None:
        self.m0 = tf.reshape(self.x0_mean.read_value(), [1])
        self.P0 = tf.reshape(self.x0_var.read_value(), [1, 1])

    def process_cov(self, params: Mapping[str, tf.Tensor | float] | None = None) -> tf.Tensor:
        sigma_v, _ = self._resolve_runtime_params(params)
        return tf.reshape(sigma_v**2, [1, 1])

    def observation_cov(self, params: Mapping[str, tf.Tensor | float] | None = None) -> tf.Tensor:
        _, sigma_w = self._resolve_runtime_params(params)
        return tf.reshape(sigma_w**2, [1, 1])

    def current_params(self) -> dict[str, tf.Tensor]:
        sigma_v, sigma_w = self._resolve_runtime_params(None)
        return {
            "sigma_v": sigma_v,
            "sigma_w": sigma_w,
        }

    def _refresh_cached_covariances(self) -> None:
        self.cov_eps_x = self.process_cov()
        self.cov_eps_y = self.observation_cov()
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

        if x0_mean is not None:
            self.x0_mean.assign(tf.reshape(tf.convert_to_tensor(x0_mean, dtype=_dc.DTYPE), []))
        if x0_var is not None:
            self.x0_var.assign(tf.reshape(tf.convert_to_tensor(x0_var, dtype=_dc.DTYPE), []))
        if t0 is not None:
            self.t0_mean.assign(tf.reshape(tf.convert_to_tensor(t0, dtype=_dc.DTYPE), []))
        if t0_var is not None:
            self.t0_var.assign(tf.reshape(tf.convert_to_tensor(t0_var, dtype=_dc.DTYPE), []))

        self._refresh_initial_state_moments()
        self.L0 = tf.linalg.cholesky(self.P0)
        self._refresh_cached_covariances()

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def q_dim(self) -> int:
        return 1

    @property
    def r_dim(self) -> int:
        return 1

    def _x_drift(self, x_prev: tf.Tensor, t_next: tf.Tensor) -> tf.Tensor:
        return (
            0.5 * x_prev
            + 25.0 * x_prev / (1.0 + tf.square(x_prev))
            + 8.0 * tf.cos(1.2 * t_next)
        )

    def _transition_time(self, z_prev: tf.Tensor, time_index: tf.Tensor | int | None) -> tf.Tensor:
        if time_index is None:
            raise ValueError("time_index is required for ADHNonlinearSSM transitions.")
        return self.t0_mean.read_value() + tf.cast(time_index, _dc.DTYPE) + 1.0

    def f(self, z, params=None, **kwargs):
        """Deterministic transition mean."""
        del params
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        x_prev = z[..., 0]
        t_next = self._transition_time(z, kwargs.get("time_index"))
        x_next = self._x_drift(x_prev, t_next)
        return x_next[..., tf.newaxis]

    def h(self, z, params=None, **kwargs):
        """Observation mean (depends on x component only)."""
        del params, kwargs
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
        params = kwargs.get("params")
        sigma_v, _ = self._resolve_runtime_params(params)
        loc = self.f(z_prev, params=params, time_index=kwargs.get("time_index"))
        scale_diag = tf.reshape(sigma_v, [1])
        scale_diag = tf.broadcast_to(scale_diag, tf.shape(loc))
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def observation_dist(self, z, **kwargs):
        """Observation distribution p(y_n | z_n)."""
        params = kwargs.get("params")
        _, sigma_w = self._resolve_runtime_params(params)
        loc = self.h(z, params=params)
        scale = tf.ones_like(loc) * sigma_w
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def f_with_noise(self, z, q, params=None, **kwargs):
        """Transition with additive process noise."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        return self.f(z, params=params, **kwargs) + q

    def h_with_noise(self, z, r, params=None, **kwargs):
        """Observation with additive measurement noise."""
        del kwargs
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        return self.h(z, params=params) + r

    def jacobian_f_x(self, z, q, params=None, time_index=None):
        """Analytic Jacobian of f_with_noise wrt state z."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        x_prev = z[..., 0]
        t_next = self._transition_time(z, time_index)

        dfdx = 0.5 + 25.0 * (1.0 - tf.square(x_prev)) / tf.square(1.0 + tf.square(x_prev))
        del t_next
        jac = dfdx[..., tf.newaxis, tf.newaxis]
        return jac, self.f_with_noise(z, q, params=params, time_index=time_index)

    def jacobian_f_q(self, z, q, params=None, time_index=None):
        """Analytic Jacobian of f_with_noise wrt process noise q (identity)."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        q = tf.convert_to_tensor(q, dtype=_dc.DTYPE)
        batch_shape = tf.shape(z)[:-1]
        eye = tf.eye(self.q_dim, batch_shape=batch_shape, dtype=_dc.DTYPE)
        return eye, self.f_with_noise(z, q, params=params, time_index=time_index)

    def jacobian_h_x(self, z, r, params=None):
        """Analytic Jacobian of h_with_noise wrt state z."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        x = z[..., 0]
        J = (x / 10.0)[..., tf.newaxis, tf.newaxis]
        return J, self.h_with_noise(z, r, params=params)

    def jacobian_h_r(self, z, r, params=None):
        """Analytic Jacobian of h_with_noise wrt measurement noise r (identity)."""
        z = tf.convert_to_tensor(z, dtype=_dc.DTYPE)
        r = tf.convert_to_tensor(r, dtype=_dc.DTYPE)
        batch_shape = tf.shape(z)[:-1]
        eye = tf.eye(self.r_dim, batch_shape=batch_shape, dtype=_dc.DTYPE)
        return eye, self.h_with_noise(z, r, params=params)

    def step(self, x_prev, **kwargs):
        x_next = self.sample_transition(x_prev, **kwargs)
        y_next = self.sample_observation(x_next, **kwargs)
        return x_next, y_next

    def simulate(self, T, shape, x0=None, **kwargs):
        if x0 is None:
            x = self.sample_initial_state(shape, **kwargs)
        else:
            x0 = tf.convert_to_tensor(x0, dtype=_dc.DTYPE)
            shape = tf.convert_to_tensor(shape, dtype=tf.int32)
            x = tf.broadcast_to(
                tf.reshape(x0, tf.concat([tf.ones_like(shape), [self.state_dim]], axis=0)),
                tf.concat([shape, [self.state_dim]], axis=0),
            )

        x_traj = []
        y_traj = []
        for step_index in range(T):
            x, y = self.step(x, time_index=step_index, **kwargs)
            x_traj.append(x)
            y_traj.append(y)

        return tf.stack(x_traj, axis=1), tf.stack(y_traj, axis=1)
