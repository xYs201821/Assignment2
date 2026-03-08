from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from experiments.hmc.diagnostics import compute_chain_ess
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
ProgressCallback = Callable[[int, Dict[str, float], str | None], None]


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


def _emit_progress(
    progress_callback: ProgressCallback | None,
    step: int,
    metrics: Dict[str, float],
    message: str | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(step, metrics, message)


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


def _prepare_gradient_mcmc(y_obs: tf.Tensor, cfg_dict: dict) -> dict[str, Any]:
    cfg = SimpleNamespace(**cfg_dict)
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

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(unconstrained: tf.Tensor) -> tf.Tensor:
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

    return {
        "cfg": cfg,
        "prior": prior,
        "pf": pf,
        "inner_pf": inner_pf,
        "y_obs": y_obs,
        "num_steps": num_steps,
        "burnin": burnin,
        "adaptation_steps": adaptation_steps,
        "init_unconstrained": init_unconstrained,
        "frozen_pf_seed": frozen_pf_seed,
        "frozen_pf_seed_t": frozen_pf_seed_t,
        "target_log_prob_fn": target_log_prob_fn,
    }


def _finalize_gradient_mcmc(
    ctx: dict[str, Any],
    *,
    log_sigma2_chain: np.ndarray,
    accept: np.ndarray,
    elapsed: float,
) -> Dict[str, Any]:
    cfg = ctx["cfg"]
    prior = ctx["prior"]
    pf = ctx["pf"]
    y_obs = ctx["y_obs"]
    burnin = int(ctx["burnin"])
    inner_pf = ctx["inner_pf"]
    frozen_pf_seed = int(ctx["frozen_pf_seed"])
    frozen_pf_seed_t = ctx["frozen_pf_seed_t"]

    num_results = int(log_sigma2_chain.shape[0])
    unconstrained_t = tf.convert_to_tensor(log_sigma2_chain, dtype=tf.float32)
    sigma2_samples = unconstrained_to_sigma2(unconstrained_t)
    sigma2_chain = np.asarray(sigma2_samples.numpy(), dtype=np.float64)
    log_sigma2_chain64 = np.asarray(log_sigma2_chain, dtype=np.float64)

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
        logtarget_chain[i] = float(loglik_chain[i] + logprior_chain[i] + np.sum(log_sigma2_chain64[i]))

    logpost_chain = loglik_chain + logprior_chain
    burnin = max(0, min(burnin, num_results))
    tail = sigma2_chain[burnin:] if burnin < num_results else sigma2_chain
    chain_diag = compute_chain_ess(sigma2_chain, burnin=burnin)

    return {
        "sigma2_chain": sigma2_chain,
        "log_sigma2_chain": log_sigma2_chain64,
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
        "num_steps": int(num_results),
        "inner_pf": inner_pf,
        "proposal_kind": str(cfg.proposal_kind),
        "num_particles": int(cfg.num_particles),
        "num_lambda": int(cfg.num_lambda),
        "reweight": FLOW_REWEIGHT,
        "resample": str(cfg.resample),
        "ot_epsilon": float(cfg.ot_epsilon),
        "ot_num_iters": int(cfg.ot_num_iters),
        "ot_jitter": float(cfg.ot_jitter),
        "frozen_pf_seed": frozen_pf_seed,
        **chain_diag,
    }


def run_hmc(
    y_obs: tf.Tensor,
    cfg: dict,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    ctx = _prepare_gradient_mcmc(y_obs, cfg)
    cfg = ctx["cfg"]
    num_steps = int(ctx["num_steps"])
    burnin = int(ctx["burnin"])
    adaptation_steps = int(ctx["adaptation_steps"])
    inner_pf = ctx["inner_pf"]
    target_log_prob_fn = ctx["target_log_prob_fn"]
    current_state = ctx["init_unconstrained"]

    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=tf.convert_to_tensor(cfg.step_size, dtype=tf.float32),
            num_leapfrog_steps=int(cfg.num_leapfrog_steps),
        ),
        num_adaptation_steps=adaptation_steps,
        target_accept_prob=float(cfg.target_accept_prob),
        adaptation_rate=float(cfg.adaptation_rate),
    )

    @tf.function(reduce_retracing=True)
    def _one_step(state, kernel_results):
        return kernel.one_step(state, kernel_results)

    if cfg.verbose:
        T_obs = int(tf.shape(y_obs)[1]) if y_obs.shape.rank and y_obs.shape.rank >= 2 else "unknown"
        print(
            f"[HMC] start steps={num_steps} burnin={burnin} leapfrog={int(cfg.num_leapfrog_steps)} "
            f"T={T_obs} num_particles={int(cfg.num_particles)} "
            f"inner_pf={inner_pf} proposal={cfg.proposal_kind}",
            flush=True,
        )
        print("[HMC] compiling single-step graph (first step may take a while)...", flush=True)

    kernel_results = kernel.bootstrap_results(current_state)

    log_sigma2_chain = np.empty((num_steps, 2), dtype=np.float32)
    accept = np.zeros(num_steps, dtype=np.int32)
    step_size_chain = np.empty(num_steps, dtype=np.float32)

    accepted_count = 0
    print_every = max(1, int(cfg.print_every))
    t_start = time.perf_counter()

    for i in range(num_steps):
        current_state, kernel_results = _one_step(current_state, kernel_results)
        inner = kernel_results.inner_results
        is_accepted = bool(inner.is_accepted.numpy())
        step_size = float(kernel_results.new_step_size.numpy())
        sigma2_current = np.exp(current_state.numpy().astype(np.float64))

        log_sigma2_chain[i] = current_state.numpy()
        accept[i] = int(is_accepted)
        step_size_chain[i] = step_size

        if is_accepted:
            accepted_count += 1

        if cfg.verbose:
            step_num = i + 1
            message = None
            if step_num == 1 or step_num % print_every == 0 or step_num == num_steps:
                message = (
                    f"[HMC] step {step_num}/{num_steps} "
                    f"accept_rate={accepted_count / step_num:.3f} "
                    f"sigma_v2={float(sigma2_current[0]):.3f} "
                    f"sigma_w2={float(sigma2_current[1]):.3f} "
                    f"step_size={step_size:.5f}"
                )
                print(message, flush=True)
            _emit_progress(
                progress_callback,
                step_num,
                {
                    "accept_rate": float(accepted_count / step_num),
                    "accepted": float(int(is_accepted)),
                    "sigma_v2": float(sigma2_current[0]),
                    "sigma_w2": float(sigma2_current[1]),
                    "step_size": float(step_size),
                },
                message,
            )
        else:
            step_num = i + 1
            _emit_progress(
                progress_callback,
                step_num,
                {
                    "accept_rate": float(accepted_count / step_num),
                    "accepted": float(int(is_accepted)),
                    "sigma_v2": float(sigma2_current[0]),
                    "sigma_w2": float(sigma2_current[1]),
                    "step_size": float(step_size),
                },
            )
    elapsed = time.perf_counter() - t_start
    result = _finalize_gradient_mcmc(
        ctx,
        log_sigma2_chain=log_sigma2_chain,
        accept=accept,
        elapsed=elapsed,
    )
    if cfg.verbose:
        sigma2_chain = np.asarray(result["sigma2_chain"], dtype=np.float64)
        burnin_used = int(result["burnin"])
        tail = sigma2_chain[burnin_used:] if burnin_used < sigma2_chain.shape[0] else sigma2_chain
        std = np.std(tail, axis=0) if tail.size else np.zeros(2, dtype=np.float64)
        print(
            f"[HMC] done steps={int(result['num_steps'])} burnin={burnin_used} acc={float(np.mean(accept)):.3f} "
            f"step_size_final={step_size_chain[-1]:.5f} "
            f"std_v2={float(std[0]):.4f} std_w2={float(std[1]):.4f}",
            flush=True,
        )

    result.update(
        {
            "step_size_final": float(step_size_chain[-1]),
            "num_leapfrog_steps": int(cfg.num_leapfrog_steps),
            "target_accept_prob": float(cfg.target_accept_prob),
        }
    )
    return result


def run_nuts(
    y_obs: tf.Tensor,
    cfg: dict,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    ctx = _prepare_gradient_mcmc(y_obs, cfg)
    cfg = ctx["cfg"]
    num_steps = int(ctx["num_steps"])
    burnin = int(ctx["burnin"])
    adaptation_steps = int(ctx["adaptation_steps"])
    inner_pf = ctx["inner_pf"]
    target_log_prob_fn = ctx["target_log_prob_fn"]
    current_state = ctx["init_unconstrained"]

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=tf.convert_to_tensor(cfg.step_size, dtype=tf.float32),
            max_tree_depth=int(cfg.max_tree_depth),
        ),
        num_adaptation_steps=adaptation_steps,
        target_accept_prob=float(cfg.target_accept_prob),
        exploration_shrinkage=float(cfg.adaptation_rate),
    )

    @tf.function(reduce_retracing=True)
    def _one_step(state, kernel_results):
        return kernel.one_step(state, kernel_results)

    if cfg.verbose:
        T_obs = int(tf.shape(y_obs)[1]) if y_obs.shape.rank and y_obs.shape.rank >= 2 else "unknown"
        print(
            f"[NUTS] start steps={num_steps} burnin={burnin} max_tree_depth={int(cfg.max_tree_depth)} "
            f"T={T_obs} num_particles={int(cfg.num_particles)} "
            f"inner_pf={inner_pf} proposal={cfg.proposal_kind}",
            flush=True,
        )
        print("[NUTS] compiling single-step graph (first step may take a while)...", flush=True)

    kernel_results = kernel.bootstrap_results(current_state)

    log_sigma2_chain = np.empty((num_steps, 2), dtype=np.float32)
    accept = np.zeros(num_steps, dtype=np.int32)
    step_size_chain = np.empty(num_steps, dtype=np.float32)
    leapfrogs_chain = np.empty(num_steps, dtype=np.int32)
    max_depth_hits = np.zeros(num_steps, dtype=np.int32)
    divergences = np.zeros(num_steps, dtype=np.int32)

    accepted_count = 0
    print_every = max(1, int(cfg.print_every))
    t_start = time.perf_counter()

    for i in range(num_steps):
        current_state, kernel_results = _one_step(current_state, kernel_results)
        inner = kernel_results.inner_results
        is_accepted = bool(inner.is_accepted.numpy())
        step_size = float(kernel_results.new_step_size.numpy())
        leapfrogs_taken = int(inner.leapfrogs_taken.numpy())
        reach_max_depth = bool(inner.reach_max_depth.numpy())
        has_divergence = bool(inner.has_divergence.numpy())
        sigma2_current = np.exp(current_state.numpy().astype(np.float64))

        log_sigma2_chain[i] = current_state.numpy()
        accept[i] = int(is_accepted)
        step_size_chain[i] = step_size
        leapfrogs_chain[i] = leapfrogs_taken
        max_depth_hits[i] = int(reach_max_depth)
        divergences[i] = int(has_divergence)

        if is_accepted:
            accepted_count += 1

        if cfg.verbose:
            step_num = i + 1
            message = None
            if step_num == 1 or step_num % print_every == 0 or step_num == num_steps:
                message = (
                    f"[NUTS] step {step_num}/{num_steps} "
                    f"accept_rate={accepted_count / step_num:.3f} "
                    f"sigma_v2={float(sigma2_current[0]):.3f} "
                    f"sigma_w2={float(sigma2_current[1]):.3f} "
                    f"step_size={step_size:.5f} "
                    f"leapfrogs={leapfrogs_taken} "
                    f"max_depth_hit={int(reach_max_depth)} "
                    f"divergence={int(has_divergence)}"
                )
                print(message, flush=True)
            _emit_progress(
                progress_callback,
                step_num,
                {
                    "accept_rate": float(accepted_count / step_num),
                    "accepted": float(int(is_accepted)),
                    "sigma_v2": float(sigma2_current[0]),
                    "sigma_w2": float(sigma2_current[1]),
                    "step_size": float(step_size),
                    "leapfrogs": float(leapfrogs_taken),
                    "max_depth_hit": float(int(reach_max_depth)),
                    "divergence": float(int(has_divergence)),
                },
                message,
            )
        else:
            step_num = i + 1
            _emit_progress(
                progress_callback,
                step_num,
                {
                    "accept_rate": float(accepted_count / step_num),
                    "accepted": float(int(is_accepted)),
                    "sigma_v2": float(sigma2_current[0]),
                    "sigma_w2": float(sigma2_current[1]),
                    "step_size": float(step_size),
                    "leapfrogs": float(leapfrogs_taken),
                    "max_depth_hit": float(int(reach_max_depth)),
                    "divergence": float(int(has_divergence)),
                },
            )
    elapsed = time.perf_counter() - t_start
    result = _finalize_gradient_mcmc(
        ctx,
        log_sigma2_chain=log_sigma2_chain,
        accept=accept,
        elapsed=elapsed,
    )
    if cfg.verbose:
        sigma2_chain = np.asarray(result["sigma2_chain"], dtype=np.float64)
        burnin_used = int(result["burnin"])
        tail = sigma2_chain[burnin_used:] if burnin_used < sigma2_chain.shape[0] else sigma2_chain
        std = np.std(tail, axis=0) if tail.size else np.zeros(2, dtype=np.float64)
        print(
            f"[NUTS] done steps={int(result['num_steps'])} burnin={burnin_used} acc={float(np.mean(accept)):.3f} "
            f"step_size_final={step_size_chain[-1]:.5f} "
            f"mean_leapfrogs={float(np.mean(leapfrogs_chain)):.2f} "
            f"max_depth_rate={float(np.mean(max_depth_hits)):.3f} "
            f"divergence_rate={float(np.mean(divergences)):.3f} "
            f"std_v2={float(std[0]):.4f} std_w2={float(std[1]):.4f}",
            flush=True,
        )

    result.update(
        {
            "step_size_final": float(step_size_chain[-1]),
            "max_tree_depth": int(cfg.max_tree_depth),
            "leapfrogs_chain": leapfrogs_chain,
            "mean_leapfrogs": float(np.mean(leapfrogs_chain)),
            "max_depth_hit_rate": float(np.mean(max_depth_hits)),
            "divergence_rate": float(np.mean(divergences)),
            "target_accept_prob": float(cfg.target_accept_prob),
        }
    )
    return result
