"""VRNN state-space model with binary observations."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM

tfd = tfp.distributions


def _as_vector(value, size: int, default: float = 0.0) -> tf.Tensor:
    if value is None:
        return tf.fill([size], tf.cast(default, tf.float32))
    x = tf.convert_to_tensor(value, dtype=tf.float32)
    if x.shape.rank == 0:
        return tf.fill([size], x)
    x = tf.reshape(x, [-1])
    if x.shape[0] == 1:
        return tf.fill([size], x[0])
    if x.shape[0] != size:
        raise ValueError(f"Expected vector of size {size}, got shape {tuple(x.shape)}.")
    return tf.cast(x, tf.float32)


class _VRNNTransitionDist:
    """Transition with deterministic R_t and Gaussian Z_t."""

    def __init__(
        self,
        r_next: tf.Tensor,
        z_mu: tf.Tensor,
        z_scale: tf.Tensor,
        *,
        deterministic_tol: float = 1e-6,
    ) -> None:
        self._r_next = tf.convert_to_tensor(r_next, dtype=tf.float32)
        self._z_dist = tfd.MultivariateNormalDiag(
            loc=tf.convert_to_tensor(z_mu, dtype=tf.float32),
            scale_diag=tf.convert_to_tensor(z_scale, dtype=tf.float32),
        )
        self._deterministic_tol = tf.convert_to_tensor(float(deterministic_tol), dtype=tf.float32)
        self._r_dim = int(self._r_next.shape[-1])

    def sample(self, seed=None):
        z = self._z_dist.sample(seed=seed)
        return tf.concat([self._r_next, z], axis=-1)

    def log_prob(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        r = x[..., : self._r_dim]
        z = x[..., self._r_dim :]
        log_pz = self._z_dist.log_prob(z)

        r_err = tf.reduce_max(tf.abs(r - self._r_next), axis=-1)
        r_match = r_err <= self._deterministic_tol
        neg_inf = tf.fill(tf.shape(log_pz), tf.cast(-np.inf, log_pz.dtype))
        return tf.where(r_match, log_pz, neg_inf)


class VRNNBinarySSM(SSM):
    """Observation-driven VRNN SSM with X_t=(R_t, Z_t) and Bernoulli Y_t."""

    def __init__(
        self,
        obs_dim: int = 88,
        latent_dim: int = 8,
        recurrent_dim: int = 16,
        embed_dim: int = 32,
        y_embed_dim: int = 32,
        transition_hidden_dim: int = 64,
        emission_hidden_dim: int = 64,
        transition_r_std: float = 0.02,
        deterministic_r: bool = True,
        deterministic_tol: float = 1e-6,
        min_scale: float = 1e-3,
        m0=None,
        P0=None,
        seed: int | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__(seed=seed)
        self._obs_dim = int(obs_dim)
        self._latent_dim = int(latent_dim)
        self._recurrent_dim = int(recurrent_dim)
        self._state_dim = self._recurrent_dim + self._latent_dim
        self._embed_dim = int(embed_dim)
        self._y_embed_dim = int(y_embed_dim)
        self._transition_hidden_dim = int(transition_hidden_dim)
        self._emission_hidden_dim = int(emission_hidden_dim)
        self.transition_r_std = tf.convert_to_tensor(float(transition_r_std), dtype=tf.float32)
        self.deterministic_r = bool(deterministic_r)
        self.deterministic_tol = float(deterministic_tol)
        self.min_scale = tf.convert_to_tensor(float(min_scale), dtype=tf.float32)

        self.m0 = _as_vector(m0, self._state_dim, default=0.0)
        if P0 is None:
            P0 = np.eye(self._state_dim, dtype=np.float32)
        self.P0 = tf.convert_to_tensor(P0, dtype=tf.float32)
        self.L0 = tf.linalg.cholesky(self.P0)

        # Compatibility fields used by filters that inspect additive-noise covariances.
        noise_floor = max(float(transition_r_std) ** 2, 1e-6)
        self.cov_eps_x = tf.eye(self._state_dim, dtype=tf.float32) * tf.cast(noise_floor, tf.float32)
        self.cov_eps_y = tf.eye(self._obs_dim, dtype=tf.float32) * 0.25

        self.z_embed = tf.keras.layers.Dense(self._embed_dim, activation="tanh", name="vrnn_z_embed")
        self.y_embed = tf.keras.layers.Dense(self._y_embed_dim, activation="tanh", name="vrnn_y_embed")
        self.transition_hidden = tf.keras.layers.Dense(
            self._transition_hidden_dim, activation="tanh", name="vrnn_transition_hidden"
        )
        self.rnn_cell = tf.keras.layers.GRUCell(self._recurrent_dim, name="vrnn_gru")
        self.prior_mu = tf.keras.layers.Dense(self._latent_dim, activation=None, name="vrnn_prior_mu")
        self.prior_scale_raw = tf.keras.layers.Dense(
            self._latent_dim, activation=None, name="vrnn_prior_scale_raw"
        )
        self.obs_hidden = tf.keras.layers.Dense(
            self._emission_hidden_dim, activation="tanh", name="vrnn_obs_hidden"
        )
        self.obs_logits = tf.keras.layers.Dense(self._obs_dim, activation=None, name="vrnn_obs_logits")
        self._layers = [
            self.z_embed,
            self.y_embed,
            self.transition_hidden,
            self.rnn_cell,
            self.prior_mu,
            self.prior_scale_raw,
            self.obs_hidden,
            self.obs_logits,
        ]

        self._build_layers()
        self._set_trainable(bool(trainable))

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def q_dim(self):
        return self._state_dim

    @property
    def r_dim(self):
        return self._obs_dim

    @property
    def trainable_variables(self):
        vars_out = []
        for layer in self._layers:
            vars_out.extend(layer.trainable_variables)
        return vars_out

    @property
    def variables(self):
        vars_out = []
        for layer in self._layers:
            vars_out.extend(layer.variables)
        return vars_out

    def _build_layers(self) -> None:
        y0 = tf.zeros([1, self._obs_dim], dtype=tf.float32)
        z0 = tf.zeros([1, self._latent_dim], dtype=tf.float32)
        r0 = tf.zeros([1, self._recurrent_dim], dtype=tf.float32)
        rnn_in = tf.concat([self.y_embed(y0), self.z_embed(z0)], axis=-1)
        h = self.transition_hidden(rnn_in)
        r1, _ = self.rnn_cell(h, [r0])
        _ = self.prior_mu(r1)
        _ = self.prior_scale_raw(r1)
        _ = self.obs_logits(self.obs_hidden(tf.concat([r1, self.z_embed(z0)], axis=-1)))

    def _set_trainable(self, trainable: bool) -> None:
        for layer in self._layers:
            layer.trainable = trainable

    def _split_state(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        r = x[..., : self._recurrent_dim]
        z = x[..., self._recurrent_dim :]
        return r, z

    def _transition_params_flat(
        self,
        r_prev: tf.Tensor,
        z_prev: tf.Tensor,
        y_prev: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        y_embed = self.y_embed(y_prev)
        z_embed = self.z_embed(z_prev)
        hidden = self.transition_hidden(tf.concat([y_embed, z_embed], axis=-1))
        r_next, _ = self.rnn_cell(hidden, [r_prev])
        z_mu = self.prior_mu(r_next)
        z_scale = tf.nn.softplus(self.prior_scale_raw(r_next)) + self.min_scale
        return r_next, z_mu, z_scale

    def _reshape_to_flat(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        lead_shape = tf.shape(x)[:-1]
        flat = tf.reshape(x, [-1, self.state_dim])
        return flat, lead_shape

    def _reshape_from_flat(self, flat: tf.Tensor, lead_shape: tf.Tensor, tail_dim: int) -> tf.Tensor:
        return tf.reshape(flat, tf.concat([lead_shape, [tail_dim]], axis=0))

    def _prepare_y_prev(self, y_prev: tf.Tensor | None, lead_shape: tf.Tensor) -> tf.Tensor:
        target_shape = tf.concat([lead_shape, [self.obs_dim]], axis=0)
        if y_prev is None:
            return tf.zeros(target_shape, dtype=tf.float32)

        y_prev = tf.convert_to_tensor(y_prev, dtype=tf.float32)
        if y_prev.shape.rank == 1:
            y_prev = y_prev[tf.newaxis, :]

        y_rank = tf.rank(y_prev)
        lead_rank = tf.shape(lead_shape)[0]

        def _insert_singleton_before_obs():
            new_shape = tf.concat([tf.shape(y_prev)[:-1], [1], tf.shape(y_prev)[-1:]], axis=0)
            return tf.reshape(y_prev, new_shape)

        y_prev = tf.cond(
            tf.equal(y_rank, lead_rank),
            _insert_singleton_before_obs,
            lambda: y_prev,
        )
        return tf.broadcast_to(y_prev, target_shape)

    def _obs_logits_from_state(self, x: tf.Tensor) -> tf.Tensor:
        x_flat, lead_shape = self._reshape_to_flat(x)
        r_flat, z_flat = self._split_state(x_flat)
        feat = tf.concat([r_flat, self.z_embed(z_flat)], axis=-1)
        logits_flat = self.obs_logits(self.obs_hidden(feat))
        return self._reshape_from_flat(logits_flat, lead_shape, self.obs_dim)

    def initial_state_dist(self, shape, **kwargs):
        del kwargs
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(self.m0, tf.concat([shape, [self.state_dim]], axis=0))
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.L0)

    def transition_dist(self, x_prev, **kwargs):
        x_prev = tf.convert_to_tensor(x_prev, dtype=tf.float32)
        x_flat, lead_shape = self._reshape_to_flat(x_prev)
        y_prev = self._prepare_y_prev(kwargs.get("y_prev", None), lead_shape)
        y_prev_flat = tf.reshape(y_prev, [-1, self.obs_dim])

        r_prev, z_prev = self._split_state(x_flat)
        r_next_flat, z_mu_flat, z_scale_flat = self._transition_params_flat(r_prev, z_prev, y_prev_flat)

        r_next = self._reshape_from_flat(r_next_flat, lead_shape, self._recurrent_dim)
        z_mu = self._reshape_from_flat(z_mu_flat, lead_shape, self._latent_dim)
        z_scale = self._reshape_from_flat(z_scale_flat, lead_shape, self._latent_dim)

        if self.deterministic_r:
            return _VRNNTransitionDist(
                r_next=r_next,
                z_mu=z_mu,
                z_scale=z_scale,
                deterministic_tol=self.deterministic_tol,
            )

        loc = tf.concat([r_next, z_mu], axis=-1)
        r_scale = tf.fill(tf.shape(r_next), self.transition_r_std)
        scale_diag = tf.concat([r_scale, z_scale], axis=-1)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def observation_dist(self, x, **kwargs):
        del kwargs
        logits = self._obs_logits_from_state(x)
        return tfd.Independent(
            tfd.Bernoulli(logits=logits, dtype=tf.float32),
            reinterpreted_batch_ndims=1,
        )

    def f(self, x, **kwargs):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_flat, lead_shape = self._reshape_to_flat(x)
        y_prev = self._prepare_y_prev(kwargs.get("y_prev", None), lead_shape)
        y_prev_flat = tf.reshape(y_prev, [-1, self.obs_dim])
        r_prev, z_prev = self._split_state(x_flat)
        r_next, z_mu, _ = self._transition_params_flat(r_prev, z_prev, y_prev_flat)
        f_flat = tf.concat([r_next, z_mu], axis=-1)
        return self._reshape_from_flat(f_flat, lead_shape, self.state_dim)

    def h(self, x, **kwargs):
        del kwargs
        logits = self._obs_logits_from_state(x)
        return tf.math.sigmoid(logits)

    def f_with_noise(self, x, q):
        del x, q
        raise NotImplementedError("VRNN transition is non-additive; use transition_dist(..., y_prev=...).")

    def h_with_noise(self, x, r):
        del x, r
        raise NotImplementedError("Bernoulli observation does not support additive observation noise.")

    def simulate(self, T, shape, x0=None, **kwargs):
        """Simulate VRNN trajectory with observation feedback into transition."""
        y0 = kwargs.pop("y0", None)
        if kwargs:
            raise TypeError(f"Unsupported kwargs for VRNN simulation: {tuple(kwargs.keys())}")

        shape = tf.convert_to_tensor(shape, tf.int32)
        if x0 is None:
            x = self.sample_initial_state(shape)
        else:
            x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
            x = tf.broadcast_to(
                tf.reshape(x0, tf.concat([tf.ones_like(shape), [self.state_dim]], axis=0)),
                tf.concat([shape, [self.state_dim]], axis=0),
            )

        if y0 is None:
            y_prev = tf.zeros(tf.concat([shape, [self.obs_dim]], axis=0), dtype=tf.float32)
        else:
            y0 = tf.convert_to_tensor(y0, dtype=tf.float32)
            y_prev = tf.broadcast_to(
                tf.reshape(y0, tf.concat([tf.ones_like(shape), [self.obs_dim]], axis=0)),
                tf.concat([shape, [self.obs_dim]], axis=0),
            )

        x_traj = []
        y_traj = []
        for _ in range(T):
            x = self.sample_transition(x, y_prev=y_prev)
            y = self.sample_observation(x)
            x_traj.append(x)
            y_traj.append(y)
            y_prev = tf.cast(y, tf.float32)

        return tf.stack(x_traj, axis=1), tf.stack(y_traj, axis=1)

