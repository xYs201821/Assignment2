from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Comment this out if you want TensorFlow INFO logs again.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def _preparse_device(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    args, _ = parser.parse_known_args(argv)
    return str(args.device).strip().lower()


_PRESELECTED_DEVICE = _preparse_device(sys.argv[1:])

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("WARNING")
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.WARNING)
    absl_logging.set_stderrthreshold("warning")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_tf_device(device: str) -> str:
    selected = str(device).strip().lower()
    gpus = tf.config.list_physical_devices("GPU")
    if selected == "auto":
        return "gpu" if gpus else "cpu"
    if selected == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to switch TensorFlow to CPU before runtime initialization."
            ) from exc
        return "cpu"
    if selected == "gpu":
        if not gpus:
            raise ValueError("Requested --device gpu, but no GPU is available.")
        return "gpu"
    raise ValueError("device must be one of {'auto', 'cpu', 'gpu'}.")


_ACTIVE_DEVICE = _configure_tf_device(_PRESELECTED_DEVICE)

from experiments.common.exp_utils import load_config
from experiments.hmc.hmc_runner import run_hmc
from experiments.hmc.pmmh_runner import run_pmmh
from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp_hmc_config.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PMMH/HMC posterior inference for ADH nonlinear SSM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--sampler", type=str, choices=["hmc", "pmmh"], default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--mcmc-seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--drop", type=int, default=None, help="Keep one sample every `drop` steps after burn-in.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default=_PRESELECTED_DEVICE,
        help="Execution device. 'auto' keeps TensorFlow default device selection.",
    )
    parser.add_argument(
        "--inner-pf",
        type=str,
        choices=["standard", "soft", "ot"],
        default=None,
    )
    parser.add_argument("--soft-lam", type=float, default=None, help="Soft resampling λ).")
    parser.add_argument("--proposal-kind", type=str, choices=["bootstrap", "ledh", "edh"], default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--num-lambda", type=int, default=None)
    parser.add_argument(
        "--proposal-std-v",
        type=float,
        default=None,
        help="PMMH proposal SD interpreted on sigma_v^2 scale; internally mapped to log-space.",
    )
    parser.add_argument(
        "--proposal-std-w",
        type=float,
        default=None,
        help="PMMH proposal SD interpreted on sigma_w^2 scale; internally mapped to log-space.",
    )
    parser.add_argument("--hmc-step-size", type=float, default=None)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=None)
    parser.add_argument("--adaptation-rate", type=float, default=None)
    parser.add_argument("--adaptation-steps", type=int, default=None)
    parser.add_argument("--frozen-pf-seed", type=int, default=None)
    parser.add_argument("--ess-threshold", type=float, default=None)
    parser.add_argument("--resample", type=str, default=None)
    parser.add_argument("--ot-epsilon", type=float, default=None)
    parser.add_argument("--ot-num-iters", type=int, default=None)
    parser.add_argument("--ot-jitter", type=float, default=None)
    parser.add_argument("--prior-alpha", type=float, default=None)
    parser.add_argument("--prior-beta", type=float, default=None)
    parser.add_argument("--init-sigma-v2", type=float, default=None)
    parser.add_argument("--init-sigma-w2", type=float, default=None)
    parser.add_argument("--true-sigma-v2", type=float, default=None)
    parser.add_argument("--true-sigma-w2", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output tag. Default: {sampler}_T{T}_N{N}_S{steps}",
    )
    parser.add_argument("--show", action="store_true", default=None)
    return parser.parse_args()


def _plot_summary(
    x_true: np.ndarray,
    y_obs: np.ndarray,
    sigma2_chain: np.ndarray,
    burnin: int,
    plot_path: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(x_true.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0, 0].plot(t, x_true, color="tab:blue", linewidth=1.1)
    axes[0, 0].set_title("latent x_t")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, linestyle=":")

    axes[0, 1].plot(t, y_obs, color="tab:orange", linewidth=1.1)
    axes[0, 1].set_title("observed y_t")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, linestyle=":")

    burnin = int(max(0, min(int(burnin), int(sigma2_chain.shape[0]))))
    sigma2_post = np.asarray(sigma2_chain[burnin:], dtype=np.float64)
    if sigma2_post.shape[0] == 0:
        sigma2_post = np.asarray(sigma2_chain, dtype=np.float64)
        trace_idx = np.arange(sigma2_post.shape[0])
    else:
        trace_idx = np.arange(burnin, burnin + sigma2_post.shape[0])

    axes[1, 0].plot(trace_idx, sigma2_post[:, 0], color="tab:green", linewidth=1.0)
    axes[1, 0].set_title(r"trace (post burn-in): $\sigma_V^2$")
    axes[1, 0].set_xlabel("MCMC iter")
    axes[1, 0].grid(True, linestyle=":")

    axes[1, 1].plot(trace_idx, sigma2_post[:, 1], color="tab:red", linewidth=1.0)
    axes[1, 1].set_title(r"trace (post burn-in): $\sigma_W^2$")
    axes[1, 1].set_xlabel("MCMC iter")
    axes[1, 1].grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_posterior_after_burnin(
    sigma_samples: np.ndarray,
    true_sigma_v: float,
    true_sigma_w: float,
    plot_path: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    samples = np.asarray(sigma_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sigma_samples must have shape [num_samples, 2].")
    if samples.shape[0] == 0:
        raise ValueError("sigma_samples is empty after burn-in/drop thinning.")
    sigma_v = samples[:, 0]
    sigma_w = samples[:, 1]
    bins = max(12, min(50, int(np.sqrt(max(20, samples.shape[0])))))

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    axes[0, 0].hist(sigma_v, bins=bins, histtype="step", color="k", linewidth=1.8)
    axes[0, 0].axvline(float(true_sigma_v), color="k", linestyle="--", linewidth=1.2)
    axes[0, 0].set_ylabel(r"$\sigma_V$")

    axes[0, 1].plot(sigma_w, sigma_v, "k+", markersize=6, alpha=0.75)
    axes[0, 1].set_ylabel(r"$\sigma_V$")

    axes[1, 0].plot(sigma_v, sigma_w, "k+", markersize=6, alpha=0.75)
    axes[1, 0].set_xlabel(r"$\sigma_V$")
    axes[1, 0].set_ylabel(r"$\sigma_W$")

    axes[1, 1].hist(sigma_w, bins=bins, histtype="step", color="k", linewidth=1.8)
    axes[1, 1].axvline(float(true_sigma_w), color="k", linestyle="--", linewidth=1.2)
    axes[1, 1].set_xlabel(r"$\sigma_W$")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _cv(cli_val, yaml_val, fallback):
    """CLI > yaml > fallback priority resolution."""
    if cli_val is not None:
        return cli_val
    if yaml_val is not None:
        return yaml_val
    return fallback


def _build_cfg(args, yaml_cfg: dict, sampler: str) -> dict:
    """Build the runner config dict from yaml + CLI overrides."""
    filter_cfg = yaml_cfg.get("filter", {})
    prior_cfg = yaml_cfg.get("prior", {})
    sampler_yaml = yaml_cfg.get(sampler, {})

    num_steps = int(_cv(args.num_steps, sampler_yaml.get("num_steps"), 10000))
    mcmc_seed = _cv(args.mcmc_seed, yaml_cfg.get("experiment", {}).get("mcmc_seed"), 0)

    shared = dict(
        num_steps      = num_steps,
        inner_pf       = str(_cv(args.inner_pf,      filter_cfg.get("inner_pf"),      "ot")),
        proposal_kind  = str(_cv(args.proposal_kind, filter_cfg.get("proposal_kind"), "bootstrap")),
        num_particles  = int(_cv(args.num_particles, filter_cfg.get("num_particles"),  1000)),
        num_lambda     = int(_cv(args.num_lambda,    filter_cfg.get("num_lambda"),     20)),
        ess_threshold  = float(_cv(args.ess_threshold, filter_cfg.get("ess_threshold"), 0.5)),
        soft_lam       = float(_cv(args.soft_lam,    filter_cfg.get("soft_lam"),      0.95)),
        ot_epsilon     = float(_cv(args.ot_epsilon,  filter_cfg.get("ot_epsilon"),    0.1)),
        ot_num_iters   = int(_cv(args.ot_num_iters,  filter_cfg.get("ot_num_iters"),  25)),
        ot_jitter      = float(_cv(args.ot_jitter,   filter_cfg.get("ot_jitter"),     1e-6)),
        prior_alpha    = float(_cv(args.prior_alpha, prior_cfg.get("alpha"),          0.01)),
        prior_beta     = float(_cv(args.prior_beta,  prior_cfg.get("beta"),           0.01)),
        init_sigma2_v  = float(_cv(args.init_sigma_v2, sampler_yaml.get("init_sigma_v2"), 10.0)),
        init_sigma2_w  = float(_cv(args.init_sigma_w2, sampler_yaml.get("init_sigma_w2"), 10.0)),
        seed           = int(mcmc_seed),
        proposal       = None,
        x0_mean        = 0.0,
        x0_var         = 5.0,
        t0             = 0.0,
        t0_var         = 1e-9,
        verbose        = True,
        print_every    = max(1, int(_cv(args.print_every, yaml_cfg.get("experiment", {}).get("print_every"), num_steps // 10))),
    )

    if sampler == "hmc":
        resample = str(_cv(args.resample, filter_cfg.get("resample"), "always"))
        burnin_raw = _cv(args.burnin, sampler_yaml.get("burnin"), None)
        return dict(
            **shared,
            resample            = resample,
            burnin              = None if burnin_raw is None else int(burnin_raw),
            step_size           = float(_cv(args.hmc_step_size,      sampler_yaml.get("step_size"),          0.05)),
            num_leapfrog_steps  = int(_cv(args.hmc_leapfrog_steps,   sampler_yaml.get("num_leapfrog_steps"), 5)),
            target_accept_prob  = float(_cv(args.target_accept_prob, sampler_yaml.get("target_accept_prob"), 0.6)),
            adaptation_rate     = float(_cv(args.adaptation_rate,    sampler_yaml.get("adaptation_rate"),    0.01)),
            adaptation_steps    = _cv(args.adaptation_steps, sampler_yaml.get("adaptation_steps"), None),
            frozen_pf_seed      = _cv(args.frozen_pf_seed,   sampler_yaml.get("frozen_pf_seed"),   None),
        )
    else:
        resample = str(_cv(args.resample, filter_cfg.get("resample"), "auto"))
        return dict(
            **shared,
            resample       = resample,
            proposal_std_v = float(_cv(args.proposal_std_v, sampler_yaml.get("proposal_std_v"), 0.15)),
            proposal_std_w = float(_cv(args.proposal_std_w, sampler_yaml.get("proposal_std_w"), 0.08)),
        )


def main() -> None:
    args = _parse_args()
    yaml_cfg = load_config(args.config, [])
    exp_cfg = yaml_cfg.get("experiment", {})
    true_cfg = yaml_cfg.get("true_params", {})

    sampler    = str(_cv(args.sampler,    exp_cfg.get("sampler"),   "hmc")).strip().lower()
    T          = int(_cv(args.T,          exp_cfg.get("T"),         100))
    data_seed  = int(_cv(args.data_seed,  exp_cfg.get("data_seed"), 123))
    drop       = max(1, int(_cv(args.drop, exp_cfg.get("drop"),     10)))
    true_sv2   = float(_cv(args.true_sigma_v2, true_cfg.get("sigma_v2"), 10.0))
    true_sw2   = float(_cv(args.true_sigma_w2, true_cfg.get("sigma_w2"), 1.0))
    out_dir    = Path(_cv(args.output_dir, exp_cfg.get("output_dir"), "results/exp_hmc"))
    show       = bool(_cv(args.show,       exp_cfg.get("show"),      False))
    tag_override = args.tag or exp_cfg.get("tag")

    print(f"[config] {args.config}")
    print(f"[device] requested={args.device} active={_ACTIVE_DEVICE}")
    tf.random.set_seed(data_seed)

    cfg = _build_cfg(args, yaml_cfg, sampler)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag_override or f"{sampler}_T{T}_N{cfg['num_particles']}_S{cfg['num_steps']}"

    true_ssm = ADHNonlinearSSM(
        sigma_v=math.sqrt(true_sv2),
        sigma_w=math.sqrt(true_sw2),
        seed=data_seed,
    )
    x_true, y_obs = true_ssm.simulate(T=T, shape=[1])

    if sampler == "hmc":
        result = run_hmc(y_obs, cfg)
    else:
        result = run_pmmh(y_obs, cfg)
    inner_pf = str(result.get("inner_pf", cfg["inner_pf"])).strip().lower()
    if inner_pf == "standard":
        print(
            f"[inner] standard proposal={cfg['proposal_kind']} num_particles={cfg['num_particles']} "
            f"num_lambda={cfg['num_lambda']} resample={cfg['resample']}"
        )
    elif inner_pf == "ot":
        print(
            f"[inner] ot proposal={cfg['proposal_kind']} num_particles={cfg['num_particles']} "
            f"num_lambda={cfg['num_lambda']} resample={cfg['resample']} "
            f"ot_eps={cfg['ot_epsilon']} ot_iters={cfg['ot_num_iters']}"
        )

    x_np = x_true.numpy()[0, :, 0]
    y_np = y_obs.numpy()[0, :, 0]
    sigma2_chain = np.asarray(result["sigma2_chain"], dtype=np.float64)
    sigma_chain = np.sqrt(np.maximum(sigma2_chain, 0.0))
    burnin = int(result["burnin"])
    sigma_chain_post = sigma_chain[burnin:]
    sigma_chain_post_thinned = sigma_chain_post[::drop]
    sigma2_chain_post_thinned = np.square(sigma_chain_post_thinned)

    plot_path = out_dir / f"{tag}_summary.png"
    posterior_plot_path = out_dir / f"{tag}_posterior_after_burnin.png"
    data_path = out_dir / f"{tag}_result.npz"
    _plot_summary(
        x_np,
        y_np,
        sigma2_chain,
        burnin=burnin,
        plot_path=plot_path,
        show=show,
    )
    _plot_posterior_after_burnin(
        sigma_samples=sigma_chain_post_thinned,
        true_sigma_v=float(np.sqrt(true_sv2)),
        true_sigma_w=float(np.sqrt(true_sw2)),
        plot_path=posterior_plot_path,
        show=show,
    )

    save_payload = {
        "x_true": x_true.numpy(),
        "y_obs": y_obs.numpy(),
        "burnin": np.int32(burnin),
        "drop": np.int32(drop),
        "sigma2_chain": sigma2_chain,
        "sigma_chain": sigma_chain,
        "sigma_chain_post": sigma_chain_post,
        "sigma_chain_post_thinned": sigma_chain_post_thinned,
        "sigma2_chain_post_thinned": sigma2_chain_post_thinned,
        "accept": np.asarray(result["accept"]),
        "loglik_chain": np.asarray(result["loglik_chain"]),
        "logpost_chain": np.asarray(result["logpost_chain"]),
    }
    if "log_sigma2_chain" in result:
        save_payload["log_sigma2_chain"] = np.asarray(result["log_sigma2_chain"])
    if "logtarget_chain" in result:
        save_payload["logtarget_chain"] = np.asarray(result["logtarget_chain"])
    np.savez(data_path, **save_payload)

    mean_v2, mean_w2 = np.mean(sigma2_chain_post_thinned, axis=0)
    std_v2, std_w2 = np.std(sigma2_chain_post_thinned, axis=0)
    mean_v, mean_w = np.mean(sigma_chain_post_thinned, axis=0)
    std_v, std_w = np.std(sigma_chain_post_thinned, axis=0)
    print(f"[done] result saved: {data_path}")
    print(f"[done] plot saved:   {plot_path}")
    print(f"[done] plot saved:   {posterior_plot_path}")
    print(f"[post] burnin={burnin} drop={drop} kept={sigma_chain_post_thinned.shape[0]}")
    print(f"[{sampler.upper()}] accept_rate={result['accept_rate']:.3f} runtime_sec={result['runtime_sec']:.2f}")
    print(f"[post] sigma_v2 mean={mean_v2:.4f} std={std_v2:.4f}")
    print(f"[post] sigma_w2 mean={mean_w2:.4f} std={std_w2:.4f}")
    print(f"[post] sigma_v  mean={mean_v:.4f} std={std_v:.4f}")
    print(f"[post] sigma_w  mean={mean_w:.4f} std={std_w:.4f}")


if __name__ == "__main__":
    main()
