from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM
from experiments.hmc.parameterization import (
    log_abs_det_jacobian,
    sigma2_sd_to_log_rw_std,
    sigma2_to_unconstrained,
    unconstrained_to_sigma2,
)
from experiments.hmc.pure_filter import PurePFConfig, PureParticleFilter, build_resampler
from experiments.hmc.pure_proposals import build_pure_proposal


tfd = tfp.distributions
FLOW_REWEIGHT = "always"


def _log_prior_sigma2_tf(sigma2: tf.Tensor, prior: tfd.Distribution) -> tf.Tensor:
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    return tf.reduce_sum(prior.log_prob(sigma2))


def _build_ssm(cfg: Any, sigma2: tf.Tensor) -> ADHNonlinearSSM:
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    sigma_v = tf.reshape(tf.sqrt(sigma2[0]), [])
    sigma_w = tf.reshape(tf.sqrt(sigma2[1]), [])
    return ADHNonlinearSSM(
        sigma_v=sigma_v,
        sigma_w=sigma_w,
        x0_mean=cfg.x0_mean,
        x0_var=cfg.x0_var,
        t0=cfg.t0,
        t0_var=cfg.t0_var,
        seed=int(cfg.seed),
    )


def _params_from_sigma2(sigma2: tf.Tensor) -> dict[str, tf.Tensor]:
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    sigma_v = tf.reshape(tf.sqrt(sigma2[0]), [])
    sigma_w = tf.reshape(tf.sqrt(sigma2[1]), [])
    return {"sigma_v": sigma_v, "sigma_w": sigma_w}


def _params_from_unconstrained(unconstrained: tf.Tensor) -> dict[str, tf.Tensor]:
    return _params_from_sigma2(unconstrained_to_sigma2(unconstrained))


def _build_dpf_proposal(cfg: Any):
    if cfg.proposal is not None:
        return cfg.proposal
    return build_pure_proposal(cfg.proposal_kind, num_lambda=int(cfg.num_lambda))


def _build_inner_pf(
    *,
    ssm: ADHNonlinearSSM,
    y_obs: tf.Tensor,
    cfg: Any,
):
    del y_obs
    proposal = _build_dpf_proposal(cfg)
    resampler = build_resampler(
        cfg.inner_pf,
        soft_lam=float(cfg.soft_lam),
        ot_epsilon=float(cfg.ot_epsilon),
        ot_num_iters=int(cfg.ot_num_iters),
        ot_jitter=float(cfg.ot_jitter),
    )
    pf_cfg = PurePFConfig(
        num_particles=int(cfg.num_particles),
        ess_threshold=float(cfg.ess_threshold),
        resample=str(cfg.resample),
    )
    pf = PureParticleFilter(ssm=ssm, proposal=proposal, resampler=resampler, cfg=pf_cfg)
    return pf, str(cfg.inner_pf).strip().lower()

@tf.function(reduce_retracing=True)
def _sum_logz_from_filter(
    pf: Any,
    y_obs: tf.Tensor,
    resample: str,
    params: dict[str, tf.Tensor] | None,
    seed: tf.Tensor | int | None,
) -> tf.Tensor:
    _, _, diagnostics, _ = pf.filter(
        y_obs,
        params=params,
        seed=seed,
        resample=resample,
    )
    log_z = tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32)
    if log_z.shape.rank == 1:
        return tf.reduce_sum(log_z)
    return tf.reduce_sum(log_z[0])


def _window_se(values: np.ndarray) -> float:
    n = int(values.shape[0])
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _log_target_from_unconstrained(
    unconstrained: tf.Tensor,
    *,
    pf: Any,
    y_obs: tf.Tensor,
    resample: str,
    prior: tfd.Distribution,
    seed: tf.Tensor | int | None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    sigma2 = unconstrained_to_sigma2(unconstrained)
    params = _params_from_sigma2(sigma2)
    ll = _sum_logz_from_filter(
        pf=pf,
        y_obs=y_obs,
        resample=resample,
        params=params,
        seed=seed,
    )
    lp = _log_prior_sigma2_tf(sigma2, prior)
    log_jac = log_abs_det_jacobian(unconstrained)
    return ll + lp + log_jac, ll, lp, sigma2


def _log_rw_density(target: tf.Tensor, loc: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    loc = tf.convert_to_tensor(loc, dtype=tf.float32)
    scale = tf.convert_to_tensor(scale, dtype=tf.float32)
    return tf.reduce_sum(tfd.Normal(loc=loc, scale=scale).log_prob(target))


def run_pmmh(y_obs: tf.Tensor, cfg: dict) -> Dict[str, Any]:
    cfg = SimpleNamespace(**cfg)
    rng = tf.random.Generator.from_seed(int(cfg.seed))
    prior = tfd.InverseGamma(
        concentration=tf.convert_to_tensor(cfg.prior_alpha, dtype=tf.float32),
        scale=tf.convert_to_tensor(cfg.prior_beta, dtype=tf.float32),
    )

    init_sigma2 = tf.convert_to_tensor([cfg.init_sigma2_v, cfg.init_sigma2_w], dtype=tf.float32)
    if bool(tf.reduce_any(init_sigma2 <= 0.0).numpy()):
        raise ValueError("Initial sigma_v^2 and sigma_w^2 must be positive for log-parameterization.")
    ssm = _build_ssm(cfg, init_sigma2)
    pf, inner_pf = _build_inner_pf(ssm=ssm, y_obs=y_obs, cfg=cfg)

    num_steps = int(cfg.num_steps)
    prop_std_sigma2 = tf.convert_to_tensor([cfg.proposal_std_v, cfg.proposal_std_w], dtype=tf.float32)
    if bool(tf.reduce_any(prop_std_sigma2 <= 0.0).numpy()):
        raise ValueError("PMMH proposal_std_v and proposal_std_w must be positive.")
    unconstrained = sigma2_to_unconstrained(init_sigma2)
    sigma2 = tf.identity(init_sigma2)

    sigma2_chain = np.zeros((num_steps, 2), dtype=np.float32)
    log_sigma2_chain = np.zeros((num_steps, 2), dtype=np.float32)
    loglik_chain = np.zeros(num_steps, dtype=np.float32)
    logprior_chain = np.zeros(num_steps, dtype=np.float32)
    logpost_chain = np.zeros(num_steps, dtype=np.float32)
    logtarget_chain = np.zeros(num_steps, dtype=np.float32)
    accept = np.zeros(num_steps, dtype=np.int32)

    pf_calls = 0
    t_start = time.perf_counter()
    last_report_end = 0

    init_seed = rng.uniform([], minval=1, maxval=2**31 - 1, dtype=tf.int32)
    ltarget, ll, lp, sigma2 = _log_target_from_unconstrained(
        unconstrained,
        pf=pf,
        y_obs=y_obs,
        resample=cfg.resample,
        prior=prior,
        seed=init_seed,
    )
    pf_calls += 1
    lpost = ll + lp

    for i in range(num_steps):
        prop_std = sigma2_sd_to_log_rw_std(sigma2, prop_std_sigma2)
        unconstrained_prop = unconstrained + rng.normal(shape=[2], stddev=prop_std, dtype=tf.float32)
        ltarget_prop, ll_prop, lp_prop, sigma2_prop = _log_target_from_unconstrained(
            unconstrained_prop,
            pf=pf,
            y_obs=y_obs,
            resample=cfg.resample,
            prior=prior,
            seed=rng.uniform([], minval=1, maxval=2**31 - 1, dtype=tf.int32),
        )
        prop_std_rev = sigma2_sd_to_log_rw_std(sigma2_prop, prop_std_sigma2)
        log_q_fwd = _log_rw_density(unconstrained_prop, unconstrained, prop_std)
        log_q_rev = _log_rw_density(unconstrained, unconstrained_prop, prop_std_rev)
        pf_calls += 1
        lpost_prop = ll_prop + lp_prop

        log_alpha = ltarget_prop - ltarget + log_q_rev - log_q_fwd
        log_u = tf.math.log(rng.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32))
        accepted = bool((log_u < log_alpha).numpy())
        if accepted:
            unconstrained = unconstrained_prop
            sigma2 = sigma2_prop
            ll = ll_prop
            lp = lp_prop
            lpost = lpost_prop
            ltarget = ltarget_prop
            accept[i] = 1

        sigma2_chain[i] = sigma2.numpy()
        log_sigma2_chain[i] = unconstrained.numpy()
        loglik_chain[i] = float(ll.numpy())
        logprior_chain[i] = float(lp.numpy())
        logpost_chain[i] = float(lpost.numpy())
        logtarget_chain[i] = float(ltarget.numpy())

        if cfg.verbose and ((i + 1) % max(1, int(cfg.print_every)) == 0):
            window_end = i + 1
            win = sigma2_chain[last_report_end:window_end]
            se_v2 = _window_se(win[:, 0])
            se_w2 = _window_se(win[:, 1])
            acc_rate = float(np.mean(accept[:window_end]))
            print(
                f"[PMMH] step={window_end}/{num_steps} acc={acc_rate:.3f} "
                f"sigma_v2={float(sigma2[0].numpy()):.3f} sigma_w2={float(sigma2[1].numpy()):.3f} "
                f"se_v2={se_v2:.3f} se_w2={se_w2:.3f}"
            )
            last_report_end = window_end

    elapsed = time.perf_counter() - t_start
    burnin = num_steps // 2
    sigma2_chain64 = sigma2_chain.astype(np.float64)
    tail = sigma2_chain64[burnin:] if burnin < sigma2_chain64.shape[0] else sigma2_chain64

    return {
        "sigma2_chain": sigma2_chain64,
        "log_sigma2_chain": log_sigma2_chain.astype(np.float64),
        "accept": accept,
        "accept_rate": float(np.mean(accept)),
        "loglik_chain": loglik_chain,
        "logprior_chain": logprior_chain,
        "logpost_chain": logpost_chain,
        "logtarget_chain": logtarget_chain,
        "burnin": int(burnin),
        "posterior_mean_sigma2": np.mean(tail, axis=0),
        "posterior_std_sigma2": np.std(tail, axis=0),
        "runtime_sec": float(elapsed),
        "pf_calls": int(pf_calls),
        "num_steps": int(num_steps),
        "inner_pf": inner_pf,
        "proposal_kind": str(cfg.proposal_kind),
        "num_particles": int(cfg.num_particles),
        "num_lambda": int(cfg.num_lambda),
        "reweight": FLOW_REWEIGHT,
        "resample": str(cfg.resample),
        "ot_epsilon": float(cfg.ot_epsilon),
        "ot_num_iters": int(cfg.ot_num_iters),
        "ot_jitter": float(cfg.ot_jitter),
    }
