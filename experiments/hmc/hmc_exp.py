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

from experiments.hmc.hmc_runner import HMCConfig, run_hmc
from experiments.hmc.pmmh_runner import PMMHConfig, run_pmmh
from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PMMH/HMC posterior inference for ADH nonlinear SSM."
    )
    parser.add_argument("--sampler", type=str, choices=["hmc", "pmmh"], default="hmc")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--mcmc-seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--drop", type=int, default=10, help="Keep one sample every `drop` steps after burn-in.")
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
        choices=["standard", "ot"],
        default=None,
    )
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
    parser.add_argument("--true-sigma-v2", type=float, default=10.0)
    parser.add_argument("--true-sigma-w2", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hmc"))
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output tag. Default: {sampler}_T{T}_N{N}_S{steps}",
    )
    parser.add_argument("--show", action="store_true")
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


def main() -> None:
    args = _parse_args()
    print(f"[device] requested={args.device} active={_ACTIVE_DEVICE}")
    tf.random.set_seed(int(args.data_seed))
    drop = max(1, int(args.drop))
    sampler = str(args.sampler).strip().lower()
    if sampler == "hmc":
        cfg = HMCConfig()
    else:
        cfg = PMMHConfig()

    if args.num_steps is not None:
        cfg.num_steps = int(args.num_steps)
    if args.num_particles is not None:
        cfg.num_particles = int(args.num_particles)
    if args.inner_pf is not None:
        cfg.inner_pf = str(args.inner_pf)
    if args.proposal_kind is not None:
        cfg.proposal_kind = str(args.proposal_kind)
    if args.num_lambda is not None:
        cfg.num_lambda = int(args.num_lambda)
    if args.ess_threshold is not None:
        cfg.ess_threshold = float(args.ess_threshold)
    if args.resample is not None:
        cfg.resample = str(args.resample)
    if args.ot_epsilon is not None:
        cfg.ot_epsilon = float(args.ot_epsilon)
    if args.ot_num_iters is not None:
        cfg.ot_num_iters = int(args.ot_num_iters)
    if args.ot_jitter is not None:
        cfg.ot_jitter = float(args.ot_jitter)
    if args.prior_alpha is not None:
        cfg.prior_alpha = float(args.prior_alpha)
    if args.prior_beta is not None:
        cfg.prior_beta = float(args.prior_beta)
    if args.init_sigma_v2 is not None:
        cfg.init_sigma2_v = float(args.init_sigma_v2)
    if args.init_sigma_w2 is not None:
        cfg.init_sigma2_w = float(args.init_sigma_w2)
    if args.mcmc_seed is not None:
        cfg.seed = int(args.mcmc_seed)

    if sampler == "hmc":
        if args.burnin is not None:
            cfg.burnin = int(args.burnin)
        if args.hmc_step_size is not None:
            cfg.step_size = float(args.hmc_step_size)
        if args.hmc_leapfrog_steps is not None:
            cfg.num_leapfrog_steps = int(args.hmc_leapfrog_steps)
        if args.target_accept_prob is not None:
            cfg.target_accept_prob = float(args.target_accept_prob)
        if args.adaptation_rate is not None:
            cfg.adaptation_rate = float(args.adaptation_rate)
        if args.adaptation_steps is not None:
            cfg.adaptation_steps = int(args.adaptation_steps)
        if args.frozen_pf_seed is not None:
            cfg.frozen_pf_seed = int(args.frozen_pf_seed)
    else:
        if args.proposal_std_v is not None:
            cfg.proposal_std_v = float(args.proposal_std_v)
        if args.proposal_std_w is not None:
            cfg.proposal_std_w = float(args.proposal_std_w)

    if args.print_every is not None:
        cfg.print_every = max(1, int(args.print_every))
    else:
        cfg.print_every = max(1, int(cfg.num_steps) // 10)
    cfg.verbose = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{sampler}_T{int(args.T)}_N{int(cfg.num_particles)}_S{int(cfg.num_steps)}"

    true_ssm = ADHNonlinearSSM(
        sigma_v=math.sqrt(float(args.true_sigma_v2)),
        sigma_w=math.sqrt(float(args.true_sigma_w2)),
        seed=int(args.data_seed),
    )
    x_true, y_obs = true_ssm.simulate(T=int(args.T), shape=[1])

    if sampler == "hmc":
        result = run_hmc(y_obs, cfg)
    else:
        result = run_pmmh(y_obs, cfg)
    inner_pf = str(result.get("inner_pf", cfg.inner_pf)).strip().lower()
    if inner_pf == "standard":
        print(
            f"[inner] standard proposal={cfg.proposal_kind} num_particles={cfg.num_particles} "
            f"num_lambda={cfg.num_lambda} resample={cfg.resample}"
        )
    elif inner_pf == "ot":
        print(
            f"[inner] ot proposal={cfg.proposal_kind} num_particles={cfg.num_particles} "
            f"num_lambda={cfg.num_lambda} resample={cfg.resample} "
            f"ot_eps={cfg.ot_epsilon} ot_iters={cfg.ot_num_iters}"
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
        show=args.show,
    )
    _plot_posterior_after_burnin(
        sigma_samples=sigma_chain_post_thinned,
        true_sigma_v=float(np.sqrt(float(args.true_sigma_v2))),
        true_sigma_w=float(np.sqrt(float(args.true_sigma_w2))),
        plot_path=posterior_plot_path,
        show=args.show,
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
