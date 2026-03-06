from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM
from experiments.hmc.parameterization import (
    log_abs_det_jacobian,
    sigma2_to_unconstrained,
    unconstrained_to_sigma2,
)
from experiments.hmc.pure_filter import PurePFConfig, PureParticleFilter, build_resampler
from experiments.hmc.pure_proposals import build_pure_proposal


tfd = tfp.distributions
FLOW_REWEIGHT = "always"


@dataclass
class HMCConfig:
    # Outer HMC chain.
    num_steps: int = 10000
    burnin: int | None = None  # If None, use num_steps // 2.
    step_size: float = 0.05
    num_leapfrog_steps: int = 5
    target_accept_prob: float = 0.6
    adaptation_rate: float = 0.01
    adaptation_steps: int | None = None  # If None, use burnin.

    # Inner PF engine.
    inner_pf: str = "ot"  # {"standard", "ot"}
    num_particles: int = 1000
    ess_threshold: float = 0.5
    resample: str = "always"  # {"never","auto","always"} via PF normalize logic.

    # Proposal q(x_t | x_{t-1}, y_t) used inside the inner PF.
    proposal_kind: str = "bootstrap"  # {"bootstrap", "ledh", "edh"}
    num_lambda: int = 20  # Flow discretization steps for LEDH/EDH.
    proposal: Any = None

    # OT resampling controls (only used when inner_pf == "ot").
    ot_epsilon: float = 0.1
    ot_num_iters: int = 25
    ot_jitter: float = 1e-6

    # Prior on static parameters sigma_v^2 and sigma_w^2.
    prior_alpha: float = 0.01
    prior_beta: float = 0.01

    # Initial point for HMC chain (provided in sigma2-space, sampled in log-sigma2 space).
    init_sigma2_v: float = 10.0
    init_sigma2_w: float = 10.0

    # RNG: outer chain seed and frozen PF randomness seed.
    seed: int = 0
    frozen_pf_seed: int | None = None

    # SSM initial-state prior parameters p(z_0).
    x0_mean: float = 0.0
    x0_var: float = 5.0
    t0: float = 0.0
    t0_var: float = 1e-9

    # Runtime and logging.
    verbose: bool = True
    print_every: int = 50


def _log_prior_sigma2_tf(sigma2: tf.Tensor, prior: tfd.Distribution) -> tf.Tensor:
    sigma2 = tf.convert_to_tensor(sigma2, dtype=tf.float32)
    return tf.reduce_sum(prior.log_prob(sigma2))


def _build_ssm(cfg: HMCConfig, sigma2: tf.Tensor) -> ADHNonlinearSSM:
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


def _build_dpf_proposal(cfg: HMCConfig):
    if cfg.proposal is not None:
        return cfg.proposal
    return build_pure_proposal(cfg.proposal_kind, num_lambda=int(cfg.num_lambda))


def _build_inner_pf(
    *,
    ssm: ADHNonlinearSSM,
    y_obs: tf.Tensor,
    cfg: HMCConfig,
):
    del y_obs
    proposal = _build_dpf_proposal(cfg)
    resampler = build_resampler(
        cfg.inner_pf,
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


def run_hmc(y_obs: tf.Tensor, cfg: HMCConfig) -> Dict[str, Any]:
    prior = tfd.InverseGamma(
        concentration=tf.convert_to_tensor(cfg.prior_alpha, dtype=tf.float32),
        scale=tf.convert_to_tensor(cfg.prior_beta, dtype=tf.float32),
    )

    init_sigma2 = tf.convert_to_tensor([cfg.init_sigma2_v, cfg.init_sigma2_w], dtype=tf.float32)
    if bool(tf.reduce_any(init_sigma2 <= 0.0).numpy()):
        raise ValueError("Initial sigma_v^2 and sigma_w^2 must be positive for log-parameterization.")
    init_unconstrained = sigma2_to_unconstrained(init_sigma2)
    ssm = _build_ssm(cfg, init_sigma2)
    pf, inner_pf = _build_inner_pf(ssm=ssm, y_obs=y_obs, cfg=cfg)

    num_steps = int(cfg.num_steps)
    burnin = num_steps // 2 if cfg.burnin is None else int(cfg.burnin)
    burnin = max(0, min(burnin, num_steps))
    adaptation_steps = burnin if cfg.adaptation_steps is None else int(cfg.adaptation_steps)
    adaptation_steps = max(0, min(adaptation_steps, num_steps))

    frozen_pf_seed = int(cfg.seed) + 12345 if cfg.frozen_pf_seed is None else int(cfg.frozen_pf_seed)
    frozen_pf_seed_t = tf.convert_to_tensor(frozen_pf_seed, dtype=tf.int32)

    target_calls = tf.Variable(0, dtype=tf.int32, trainable=False)
    progress_step = tf.Variable(-1, dtype=tf.int32, trainable=False)
    accepted_steps = tf.Variable(0, dtype=tf.int32, trainable=False)

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(unconstrained: tf.Tensor) -> tf.Tensor:
        target_calls.assign_add(1)
        unconstrained = tf.convert_to_tensor(unconstrained, dtype=tf.float32)
        sigma2 = unconstrained_to_sigma2(unconstrained)
        params = _params_from_unconstrained(unconstrained)
        ll = _sum_logz_from_filter(
            pf=pf,
            y_obs=y_obs,
            resample=cfg.resample,
            params=params,
            seed=frozen_pf_seed_t,
        )
        lp = _log_prior_sigma2_tf(sigma2, prior)
        return ll + lp + log_abs_det_jacobian(unconstrained)

    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=tf.convert_to_tensor(cfg.step_size, dtype=tf.float32),
            num_leapfrog_steps=int(cfg.num_leapfrog_steps),
        ),
        num_adaptation_steps=int(adaptation_steps),
        target_accept_prob=float(cfg.target_accept_prob),
        adaptation_rate=float(cfg.adaptation_rate),
    )

    def trace_fn(_, kr):
        inner = kr.inner_results
        step = progress_step.assign_add(1)
        is_mcmc_step = step > 0
        accepted = accepted_steps.assign_add(
            tf.where(is_mcmc_step, tf.cast(inner.is_accepted, tf.int32), tf.constant(0, dtype=tf.int32))
        )
        if cfg.verbose and int(cfg.print_every) > 0:
            every = tf.constant(int(cfg.print_every), dtype=tf.int32)
            total_steps = tf.constant(int(num_steps), dtype=tf.int32)
            should_log = tf.logical_and(
                is_mcmc_step,
                tf.logical_or(
                    tf.equal(step, 1),
                    tf.logical_or(
                        tf.equal(tf.math.floormod(step, every), 0),
                        tf.equal(step, total_steps),
                    ),
                ),
            )

            def _print_progress():
                tf.print(
                    "[HMC] progress",
                    step,
                    "/",
                    total_steps,
                    "accept_rate~",
                    tf.cast(accepted, tf.float32) / tf.cast(step, tf.float32),
                    "step_size",
                    kr.new_step_size,
                )
                return tf.constant(0, dtype=tf.int32)

            tf.cond(should_log, _print_progress, lambda: tf.constant(0, dtype=tf.int32))
        return {
            "is_accepted": inner.is_accepted,
            "log_accept_ratio": inner.log_accept_ratio,
            "step_size": kr.new_step_size,
        }

    @tf.function(reduce_retracing=True)
    def _sample_chain_graph():
        return tfp.mcmc.sample_chain(
            num_results=int(num_steps),
            current_state=init_unconstrained,
            kernel=kernel,
            num_burnin_steps=0,
            trace_fn=trace_fn,
            seed=int(cfg.seed),
        )

    if cfg.verbose:
        print(
            f"[HMC] start steps={num_steps} burnin={burnin} leapfrog={int(cfg.num_leapfrog_steps)} "
            f"T={int(tf.shape(y_obs)[1]) if y_obs.shape.rank and y_obs.shape.rank >= 2 else 'unknown'} "
            f"num_particles={int(cfg.num_particles)} inner_pf={inner_pf} proposal={cfg.proposal_kind}",
            flush=True,
        )
        print("[HMC] tracing/compiling TensorFlow graph, first update may take a while...", flush=True)
    target_calls.assign(0)
    progress_step.assign(-1)
    accepted_steps.assign(0)
    t_start = time.perf_counter()
    unconstrained_samples, trace = _sample_chain_graph()
    elapsed = time.perf_counter() - t_start

    sigma2_samples = unconstrained_to_sigma2(unconstrained_samples)
    sigma2_chain = np.asarray(sigma2_samples.numpy(), dtype=np.float64)
    log_sigma2_chain = np.asarray(unconstrained_samples.numpy(), dtype=np.float64)
    accept = np.asarray(trace["is_accepted"].numpy(), dtype=np.int32)
    num_results = int(sigma2_chain.shape[0])

    logprior_chain = np.asarray(
        tf.reduce_sum(prior.log_prob(tf.cast(sigma2_samples, tf.float32)), axis=-1).numpy(),
        dtype=np.float32,
    )
    loglik_chain = np.zeros(num_results, dtype=np.float32)
    logtarget_chain = np.zeros(num_results, dtype=np.float32)
    for i in range(num_results):
        ll_i = _sum_logz_from_filter(
            pf=pf,
            y_obs=y_obs,
            resample=cfg.resample,
            params=_params_from_sigma2(tf.cast(sigma2_samples[i], tf.float32)),
            seed=frozen_pf_seed_t,
        )
        loglik_chain[i] = float(ll_i.numpy())
        logtarget_chain[i] = float(loglik_chain[i] + logprior_chain[i] + np.sum(log_sigma2_chain[i]))

    logpost_chain = loglik_chain + logprior_chain

    tail = sigma2_chain[burnin:] if burnin < sigma2_chain.shape[0] else sigma2_chain
    if cfg.verbose:
        std = np.std(tail, axis=0) if tail.size else np.zeros(2, dtype=np.float64)
        print(
            f"[HMC] done steps={num_steps} burnin={burnin} acc={float(np.mean(accept)):.3f} "
            f"step_size_final={float(trace['step_size'][-1].numpy()):.5f} "
            f"std_v2={float(std[0]):.4f} std_w2={float(std[1]):.4f}"
        )

    return {
        "sigma2_chain": sigma2_chain,
        "log_sigma2_chain": log_sigma2_chain,
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
        "pf_calls": int(target_calls.numpy()) + num_results,
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
        "frozen_pf_seed": int(frozen_pf_seed),
        "step_size_final": float(trace["step_size"][-1].numpy()),
        "num_leapfrog_steps": int(cfg.num_leapfrog_steps),
        "target_accept_prob": float(cfg.target_accept_prob),
    }
