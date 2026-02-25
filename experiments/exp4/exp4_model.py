from __future__ import annotations

from typing import Any, Dict

import numpy as np

from experiments.common.exp_utils import set_seed
from src.ssm import VRNNBinarySSM


def _initial_state_params(state_dim: int, p0_diag: float) -> tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros((state_dim,), dtype=np.float32)
    P0 = np.eye(state_dim, dtype=np.float32) * float(p0_diag)
    return m0, P0


def build_exp4_vrnn_ssm(
    model_cfg: Dict[str, Any],
    seed: int,
    *,
    trainable: bool,
) -> VRNNBinarySSM:
    obs_dim = int(model_cfg.get("obs_dim", 88))
    latent_dim = int(model_cfg.get("latent_dim", 8))
    recurrent_dim = int(model_cfg.get("recurrent_dim", 16))
    embed_dim = int(model_cfg.get("embed_dim", 32))
    y_embed_dim = int(model_cfg.get("y_embed_dim", 32))
    transition_hidden_dim = int(model_cfg.get("transition_hidden_dim", 64))
    emission_hidden_dim = int(model_cfg.get("emission_hidden_dim", 64))
    transition_r_std = float(model_cfg.get("transition_r_std", 0.02))
    deterministic_r = bool(model_cfg.get("deterministic_r", True))
    deterministic_tol = float(model_cfg.get("deterministic_tol", 1e-6))
    min_scale = float(model_cfg.get("min_scale", 1e-3))
    p0_diag = float(model_cfg.get("p0_diag", 1.0))

    set_seed(int(seed))
    state_dim = recurrent_dim + latent_dim
    m0, P0 = _initial_state_params(state_dim=state_dim, p0_diag=p0_diag)
    return VRNNBinarySSM(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        recurrent_dim=recurrent_dim,
        embed_dim=embed_dim,
        y_embed_dim=y_embed_dim,
        transition_hidden_dim=transition_hidden_dim,
        emission_hidden_dim=emission_hidden_dim,
        transition_r_std=transition_r_std,
        deterministic_r=deterministic_r,
        deterministic_tol=deterministic_tol,
        min_scale=min_scale,
        m0=m0,
        P0=P0,
        seed=int(seed),
        trainable=bool(trainable),
    )


def build_exp4_vrnn_ssm_pair(
    model_cfg: Dict[str, Any],
    seed: int,
    *,
    sim_seed_offset: int = 100_000,
    fit_seed_offset: int = 200_000,
    fit_trainable: bool = True,
) -> tuple[VRNNBinarySSM, VRNNBinarySSM]:
    sim_ssm = build_exp4_vrnn_ssm(
        model_cfg=model_cfg,
        seed=int(seed) + int(sim_seed_offset),
        trainable=False,
    )
    fit_ssm = build_exp4_vrnn_ssm(
        model_cfg=model_cfg,
        seed=int(seed) + int(fit_seed_offset),
        trainable=bool(fit_trainable),
    )
    return sim_ssm, fit_ssm


__all__ = [
    "build_exp4_vrnn_ssm",
    "build_exp4_vrnn_ssm_pair",
]
