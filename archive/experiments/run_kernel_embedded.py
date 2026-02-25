import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, ".."))
sys.path.append(this_dir)

from config import KernelFlowConfig
from experiment_helper import configure_matplotlib, ensure_dir, print_metrics, rmse_all, save_npz, set_global_seed, to_numpy
from src.flows.kernel_embedded import KernelParticleFlow
from src.ssm_hu21 import HuLorenz96SSM
from src.utility import weighted_mean


def gaussian_pdf(x, mu, sigma):
    if sigma <= 0.0:
        return np.zeros_like(x)
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return coeff * np.exp(-0.5 * z * z)


def plot_state_rmse(x_true, mean_scalar, mean_diag, save_path, show=True):
    err_scalar = np.linalg.norm(x_true - mean_scalar, axis=-1)
    err_diag = np.linalg.norm(x_true - mean_diag, axis=-1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(err_scalar, label="Scalar kernel", linewidth=1.6)
    ax.plot(err_diag, label="Diag matrix kernel", linewidth=1.6)
    ax.set_title("State RMSE per time step")
    ax.set_xlabel("t")
    ax.set_ylabel("||x_t - mean||")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_obs_variance(time, var_scalar, var_diag, obs_var, save_path, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, var_scalar, label="Scalar kernel", linewidth=1.6)
    ax.plot(time, var_diag, label="Diag matrix kernel", linewidth=1.6)
    ax.axhline(obs_var, color="k", linestyle="--", linewidth=1.0, label="obs noise var")
    ax.set_title("Mean observation variance across dimensions")
    ax.set_xlabel("t")
    ax.set_ylabel("mean var[h(x_t)]")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_obs_marginals(
    y_scalar,
    y_diag,
    y_obs,
    y_true,
    obs_sigma,
    dims,
    t_idx,
    save_path,
    show=True,
):
    fig, axes = plt.subplots(len(dims), 1, figsize=(9, 3 * len(dims)))
    if len(dims) == 1:
        axes = [axes]
    for ax, dim in zip(axes, dims):
        scalar_vals = y_scalar[t_idx, :, dim]
        diag_vals = y_diag[t_idx, :, dim]
        scalar_vals = scalar_vals[np.isfinite(scalar_vals)]
        diag_vals = diag_vals[np.isfinite(diag_vals)]
        if scalar_vals.size == 0 and diag_vals.size == 0:
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Observation marginal dim {dim} at t={t_idx}")
            ax.set_xlabel("y")
            ax.set_ylabel("density")
            ax.grid(True, linestyle=":")
            continue
        total_size = max(scalar_vals.size, diag_vals.size)
        bins = max(20, min(60, max(1, total_size // 4)))
        if scalar_vals.size > 0:
            ax.hist(scalar_vals, bins=bins, density=True, alpha=0.5, label="Scalar kernel")
        if diag_vals.size > 0:
            ax.hist(diag_vals, bins=bins, density=True, alpha=0.5, label="Diag matrix kernel")
        if np.isfinite(y_obs[t_idx, dim]):
            ax.axvline(y_obs[t_idx, dim], color="k", linestyle="--", linewidth=1.2, label="y_t")
        if np.isfinite(y_true[t_idx, dim]):
            ax.axvline(y_true[t_idx, dim], color="gray", linestyle=":", linewidth=1.2, label="h(x_t)")
        if obs_sigma > 0.0 and np.isfinite(y_true[t_idx, dim]):
            candidates = []
            if scalar_vals.size > 0:
                candidates.append(scalar_vals.min())
                candidates.append(scalar_vals.max())
            if diag_vals.size > 0:
                candidates.append(diag_vals.min())
                candidates.append(diag_vals.max())
            if np.isfinite(y_obs[t_idx, dim]):
                candidates.append(y_obs[t_idx, dim])
            if candidates:
                lo = min(candidates) - 3.0 * obs_sigma
                hi = max(candidates) + 3.0 * obs_sigma
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, gaussian_pdf(xs, y_true[t_idx, dim], obs_sigma), color="gray", linewidth=1.0)
        ax.set_title(f"Observation marginal dim {dim} at t={t_idx}")
        ax.set_xlabel("y")
        ax.set_ylabel("density")
        ax.grid(True, linestyle=":")
        ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_filter(ssm, y_obs, cfg, kernel_type):
    flow = KernelParticleFlow(
        ssm,
        num_particles=cfg.num_particles,
        num_lambda=cfg.num_lambda,
        alpha=cfg.alpha,
        kernel_type=kernel_type,
        localization_radius=cfg.localization_radius,
    )
    x_particles, w, diagnostics, parents = flow.filter(y_obs, reweight="never")
    return {
        "x": x_particles,
        "diagnostics": diagnostics,
        "parents": parents,
    }


def main():
    configure_matplotlib()
    cfg = KernelFlowConfig()
    parser = argparse.ArgumentParser(description="Kernel-embedded particle flow comparison (Hu 2021).")
    parser.add_argument("--state_dim", type=int, default=cfg.state_dim)
    parser.add_argument("--obs_stride", type=int, default=cfg.obs_stride)
    parser.add_argument("--dt", type=float, default=cfg.dt)
    parser.add_argument("--F", type=float, default=cfg.F)
    parser.add_argument("--obs_op", choices=["linear", "abs", "exp", "square"], default=cfg.obs_op)
    parser.add_argument("--q_scale", type=float, default=cfg.q_scale)
    parser.add_argument("--r_scale", type=float, default=cfg.r_scale)
    parser.add_argument("--x0_noise", type=float, default=cfg.x0_noise)
    parser.add_argument("--T", type=int, default=cfg.T)
    parser.add_argument("--batch", type=int, default=cfg.batch_size)
    parser.add_argument("--num_particles", type=int, default=cfg.num_particles)
    parser.add_argument("--num_lambda", type=int, default=cfg.num_lambda)
    parser.add_argument("--alpha", type=float, default=cfg.alpha)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--out_dir", type=str, default=cfg.out_dir)
    parser.add_argument("--plot_dims", type=int, nargs="*", default=list(cfg.plot_dims))
    parser.add_argument("--t_plot", type=int, default=cfg.t_plot)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    cfg.state_dim = args.state_dim
    cfg.obs_stride = args.obs_stride
    cfg.dt = args.dt
    cfg.F = args.F
    cfg.obs_op = args.obs_op
    cfg.q_scale = args.q_scale
    cfg.r_scale = args.r_scale
    cfg.x0_noise = args.x0_noise
    cfg.T = args.T
    cfg.batch_size = args.batch
    cfg.num_particles = args.num_particles
    cfg.num_lambda = args.num_lambda
    cfg.alpha = args.alpha
    cfg.seed = args.seed
    cfg.out_dir = args.out_dir
    cfg.plot_dims = tuple(args.plot_dims)
    cfg.t_plot = args.t_plot

    set_global_seed(cfg.seed)
    ssm = HuLorenz96SSM(
        state_dim=cfg.state_dim,
        obs_stride=cfg.obs_stride,
        dt=cfg.dt,
        F=cfg.F,
        obs_op=cfg.obs_op,
        q_scale=cfg.q_scale,
        r_scale=cfg.r_scale,
        seed=cfg.seed,
    )

    x0 = tf.ones([cfg.state_dim], dtype=tf.float32) * tf.cast(cfg.F, tf.float32)
    if cfg.x0_noise > 0.0:
        x0 = x0 + tf.random.normal([cfg.state_dim], stddev=float(cfg.x0_noise))
    x_true, y_obs = ssm.simulate(T=cfg.T, shape=(cfg.batch_size,), x0=x0)
    y_true = ssm.h(x_true)
    x_true_np = to_numpy(x_true[0])
    y_obs_np = to_numpy(y_obs[0])
    y_true_np = to_numpy(y_true[0])

    results = {}
    for kernel in ("scalar", "diag"):
        ssm.set_seed(cfg.seed)
        results[kernel] = run_filter(ssm, y_obs, cfg, kernel)

    metrics = {}
    means = {}
    obs_vars = {}
    obs_traces = {}
    for kernel, out in results.items():
        mean = tf.reduce_mean(out["x"], axis=-2)
        means[kernel] = to_numpy(mean[0])
        metrics[f"rmse_state_{kernel}"] = float(rmse_all(x_true, mean).numpy())
        y_particles = ssm.h(out["x"])
        y_particles_np = to_numpy(y_particles[0])
        obs_traces[kernel] = y_particles_np
        var_y = np.nanvar(y_particles_np, axis=1)
        obs_vars[kernel] = np.nanmean(var_y, axis=-1)

    print_metrics("Kernel Flow", metrics)

    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)
    show = not args.no_show
    obs_dim = ssm.obs_dim
    plot_dims = [d for d in cfg.plot_dims if 0 <= d < obs_dim]
    if not plot_dims:
        plot_dims = [0]
    if not args.no_save:
        save_npz(
            out_dir / "kernel_flow_results.npz",
            x_true=x_true_np,
            y_obs=y_obs_np,
            mean_scalar=means["scalar"],
            mean_diag=means["diag"],
            obs_var_scalar=obs_vars["scalar"],
            obs_var_diag=obs_vars["diag"],
            obs_indices=to_numpy(ssm.obs_indices),
        )

    t_plot = cfg.t_plot if cfg.t_plot >= 0 else (cfg.T - 1)
    t_plot = int(np.clip(t_plot, 0, cfg.T - 1))
    obs_sigma = float(cfg.r_scale)
    time = np.arange(cfg.T)

    plot_state_rmse(
        x_true_np,
        means["scalar"],
        means["diag"],
        save_path=out_dir / "kernel_flow_state_rmse.png",
        show=show,
    )
    plot_obs_variance(
        time,
        obs_vars["scalar"],
        obs_vars["diag"],
        obs_sigma**2,
        save_path=out_dir / "kernel_flow_obs_variance.png",
        show=show,
    )
    plot_obs_marginals(
        obs_traces["scalar"],
        obs_traces["diag"],
        y_obs_np,
        y_true_np,
        obs_sigma,
        plot_dims,
        t_plot,
        save_path=out_dir / "kernel_flow_obs_marginals.png",
        show=show,
    )


if __name__ == "__main__":
    main()
