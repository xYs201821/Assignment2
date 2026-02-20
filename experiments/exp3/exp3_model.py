from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm import LinearGaussianSSM

tfd = tfp.distributions


def _build_exp3_linear_params(
    dx: int,
    dy: int,
    a_decay: float = 0.42,
    process_var: float = 1.0,
    obs_var: float = 1.0,
    p0_scale: float = 1.0,
    m0_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Build exp3 linear-Gaussian SSM parameters."""
    i = np.arange(dx, dtype=np.float32)
    # A_ij = decay^{|i-j|+1} (Toeplitz correlation-like transition matrix).
    A = np.power(np.float32(a_decay), np.abs(i[:, None] - i[None, :]) + 1.0).astype(
        np.float32
    )

    B = np.sqrt(np.float32(process_var)) * np.eye(dx, dtype=np.float32)
    C = np.zeros((dy, dx), dtype=np.float32)
    C[np.arange(dy), np.arange(dy)] = 1.0
    D = np.sqrt(np.float32(obs_var)) * np.eye(dy, dtype=np.float32)
    m0 = np.full((dx,), np.float32(m0_value), dtype=np.float32)
    P0 = np.float32(p0_scale) * np.eye(dx, dtype=np.float32)
    obs_idx = np.arange(dy, dtype=np.int32)
    return {"A": A, "B": B, "C": C, "D": D, "m0": m0, "P0": P0, "obs_idx": obs_idx}


def build_exp3_linear_ssm_pair(
    model_cfg: Dict[str, Any],
    seed: int,
    *,
    fit_trainable: bool = False,
) -> tuple[LinearGaussianSSM, LinearGaussianSSM]:
    """Build simulation and fit LGSSMs using exp3 linear-Gaussian settings."""
    state_dim = int(model_cfg.get("state_dim", 8))
    obs_dim = int(model_cfg.get("obs_dim", max(1, state_dim // 2)))
    a_decay = float(model_cfg.get("a_decay", model_cfg.get("transition_decay", 0.42)))
    process_var = float(
        model_cfg.get("process_var", model_cfg.get("q_var", model_cfg.get("process_noise_var", 1.0)))
    )
    obs_var = float(
        model_cfg.get("obs_var", model_cfg.get("r_var", model_cfg.get("obs_noise_var", 1.0)))
    )
    p0_scale = float(model_cfg.get("p0_scale", 1.0))
    m0_value = float(model_cfg.get("m0_value", 0.0))

    params = _build_exp3_linear_params(
        dx=state_dim,
        dy=obs_dim,
        a_decay=a_decay,
        process_var=process_var,
        obs_var=obs_var,
        p0_scale=p0_scale,
        m0_value=m0_value,
    )
    sim_ssm = LinearGaussianSSM(
        A=params["A"],
        B=params["B"],
        C=params["C"],
        D=params["D"],
        m0=params["m0"],
        P0=params["P0"],
        seed=seed,
        trainable=False,
    )
    sim_ssm.obs_indices = params["obs_idx"]

    fit_ssm = LinearGaussianSSM(
        A=params["A"],
        B=params["B"],
        C=params["C"],
        D=params["D"],
        m0=params["m0"],
        P0=params["P0"],
        seed=seed,
        trainable=bool(fit_trainable),
    )
    fit_ssm.obs_indices = params["obs_idx"]
    return sim_ssm, fit_ssm


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float32), np.float32(1e-6))
    return np.log(np.expm1(x)).astype(np.float32)


class LinearGaussianProposalPhi(tf.Module):
    """Proposal q_phi(x_t|x_{t-1}, y_t) with d_phi = d_x + d_y."""

    @staticmethod
    def _expand_vector(value: Any, size: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full((size,), float(arr), dtype=np.float32)
        arr = arr.reshape(-1)
        if arr.size == 1:
            return np.full((size,), float(arr[0]), dtype=np.float32)
        if arr.size != size:
            raise ValueError(f"{name} must be scalar or length {size}, got shape {arr.shape}.")
        return arr.astype(np.float32)

    def __init__(
        self,
        A: Any,
        state_dim: int,
        obs_dim: int,
        init_delta: Any = 1.0,
        init_gamma: Any = 1.0,
        init_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "proposal_phi")
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)
        if self.obs_dim > self.state_dim:
            raise ValueError("obs_dim must be <= state_dim for rectangular-diagonal Gamma_phi.")
        self._delta_eps = tf.constant(1e-4, dtype=tf.float32)
        self.A = tf.convert_to_tensor(A, dtype=tf.float32)

        delta0 = self._expand_vector(init_delta, self.state_dim, "proposal.init_delta")
        gamma0 = self._expand_vector(init_gamma, self.obs_dim, "proposal.init_gamma")
        noise_std = float(init_noise_std)
        if noise_std > 0.0:
            rng = np.random.default_rng() if rng is None else rng
            delta0 = delta0 + rng.normal(0.0, noise_std, size=delta0.shape).astype(np.float32)
            gamma0 = gamma0 + rng.normal(0.0, noise_std, size=gamma0.shape).astype(np.float32)
        delta0 = np.maximum(delta0, np.float32(self._delta_eps.numpy() * 2.0))

        rho0 = _inverse_softplus(delta0 - np.float32(self._delta_eps.numpy()))
        self.rho_delta = tf.Variable(rho0, trainable=True, dtype=tf.float32, name="rho_delta")
        self.gamma_raw = tf.Variable(gamma0, trainable=True, dtype=tf.float32, name="gamma_raw")

    @property
    def delta(self) -> tf.Tensor:
        return tf.nn.softplus(self.rho_delta) + self._delta_eps

    @property
    def gamma(self) -> tf.Tensor:
        return self.gamma_raw

    def phi_vector(self) -> tf.Tensor:
        return tf.concat([self.delta, self.gamma], axis=0)

    def rmse_phi_to_one(self) -> tf.Tensor:
        phi = self.phi_vector()
        return tf.sqrt(tf.reduce_mean(tf.square(phi - tf.ones_like(phi))))

    def _proposal_mean(self, x_prev: tf.Tensor, y_t: tf.Tensor) -> tf.Tensor:
        ax = tf.einsum("ij,bnj->bni", self.A, x_prev)
        gy_obs = self.gamma[tf.newaxis, :] * y_t  # [B, dy]
        pad_dim = self.state_dim - self.obs_dim
        gy = tf.pad(gy_obs, paddings=[[0, 0], [0, pad_dim]])  # [B, dx]
        return tf.math.divide_no_nan(
            ax + gy[:, tf.newaxis, :],
            self.delta[tf.newaxis, tf.newaxis, :],
        )

    def _proposal_dist(self, x_prev: tf.Tensor, y_t: tf.Tensor) -> tfd.Distribution:
        mu_q = self._proposal_mean(x_prev, y_t)
        scale_diag = tf.sqrt(self.delta)[tf.newaxis, tf.newaxis, :]
        return tfd.MultivariateNormalDiag(loc=mu_q, scale_diag=scale_diag)

    def sample(self, ssm, x_prev, y_t, seed=None):
        del ssm
        x_prev = tf.convert_to_tensor(x_prev, dtype=tf.float32)
        y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
        dist = self._proposal_dist(x_prev, y_t)
        x_t = dist.sample(seed=seed)
        log_q = dist.log_prob(x_t)
        return x_t, log_q

    def log_prob(self, ssm, x, x_prev, y_t):
        del ssm
        x_prev = tf.convert_to_tensor(x_prev, dtype=tf.float32)
        y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self._proposal_dist(x_prev, y_t).log_prob(x)


__all__ = [
    "LinearGaussianProposalPhi",
    "build_exp3_linear_ssm_pair",
]
