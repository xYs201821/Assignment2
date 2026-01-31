"""Dai(22) bearing-only stochastic particle flow: single-step + Figure 2/Table 1 outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.ssm.base import SSM
from src.flows.stochastic_pf import StochasticParticleFlow


tfd = tfp.distributions


def wrap_angle(angle: tf.Tensor) -> tf.Tensor:
    return tf.math.atan2(tf.sin(angle), tf.cos(angle))


class BearingOnlySSM(SSM):
    """2D position with bearing-only observations from fixed sensors."""

    def __init__(self, sensors: np.ndarray, R: np.ndarray, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.sensors = tf.convert_to_tensor(sensors, dtype=tf.float32)  # [2, 2]
        self.cov_eps_y = tf.convert_to_tensor(R, dtype=tf.float32)
        self.cov_eps_x = tf.eye(2, dtype=tf.float32) * 1e-9

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    @property
    def q_dim(self) -> int:
        return 2

    @property
    def r_dim(self) -> int:
        return 2

    def initial_state_dist(self, shape, **kwargs):
        raise NotImplementedError

    def transition_dist(self, x_prev, **kwargs):
        loc = tf.convert_to_tensor(x_prev, dtype=tf.float32)
        L = tf.linalg.cholesky(self.cov_eps_x)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=L)

    def observation_dist(self, x, **kwargs):
        loc = self.h(x)
        L = tf.linalg.cholesky(self.cov_eps_y)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=L)

    def f(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32)

    def h(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        dx = x[..., 0][..., tf.newaxis] - self.sensors[:, 0]
        dy = x[..., 1][..., tf.newaxis] - self.sensors[:, 1]
        return tf.math.atan2(dy, dx)

    def innovation(self, y, y_pred):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        return wrap_angle(y - y_pred)

    def jacobian_h_x(self, x, r):
        """Analytic Jacobian of bearing-only observation wrt state."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        r = tf.convert_to_tensor(r, dtype=tf.float32)
        dx = x[..., 0][..., tf.newaxis] - self.sensors[:, 0]
        dy = x[..., 1][..., tf.newaxis] - self.sensors[:, 1]
        r2 = dx * dx + dy * dy
        r2 = tf.where(r2 > 0.0, r2, tf.ones_like(r2) * 1e-12)
        j11 = -dy / r2
        j12 = dx / r2
        J = tf.stack([j11, j12], axis=-1)
        y = tf.math.atan2(dy, dx) + r
        return J, y

    def jacobian_h_r(self, x, r):
        """Jacobian of observation wrt noise (identity for additive noise)."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        r = tf.convert_to_tensor(r, dtype=tf.float32)
        batch_shape = tf.shape(x)[:-1]
        J = tf.eye(self.r_dim, batch_shape=batch_shape, dtype=tf.float32)
        y = self.h(x) + r
        return J, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dai(22) bearing-only SPF (single step).")
    parser.add_argument("--num_particles", type=int, default=50)
    parser.add_argument("--num_lambda", type=int, default=10000)
    parser.add_argument("--beta_mode", choices=["linear", "optimal", "both"], default="linear")
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--shared-init", dest="shared_init", action="store_true")
    parser.add_argument("--no-shared-init", dest="shared_init", action="store_false")
    parser.set_defaults(shared_init=True)
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _stiffness_ratio(
    Q: np.ndarray,
    P0_inv: np.ndarray,
    Info: np.ndarray,
    beta: np.ndarray,
    beta_dot: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    P0_inv = np.asarray(P0_inv, dtype=np.float64)
    Info = np.asarray(Info, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    beta_dot = np.asarray(beta_dot, dtype=np.float64)
    out = np.zeros_like(beta, dtype=np.float64)
    for j in range(beta.shape[0]):
        M = P0_inv + beta[j] * Info
        Minv = np.linalg.inv(M)
        F = -0.5 * (Q @ M + beta_dot[j] * (Minv @ Info))
        eig = np.linalg.eigvals(F)
        real = np.abs(np.real(eig))
        denom = max(np.min(real), eps)
        out[j] = np.max(real) / denom
    return out


def main() -> None:
    args = parse_args()

    sensors = np.array([[-3.5, 0.0], [3.5, 0.0]], dtype=np.float32)
    z = np.array([0.4754, 1.1868], dtype=np.float32)
    x_true = np.array([4.0, 4.0], dtype=np.float32)

    m0 = np.array([3.0, 5.0], dtype=np.float32)
    P0 = np.diag([1000.0, 2.0]).astype(np.float32)
    R = np.diag([0.04, 0.04]).astype(np.float32)
    Q = np.diag([4.0, 0.4]).astype(np.float32)

    ssm = BearingOnlySSM(sensors=sensors, R=R, seed=args.seed)
    flow = StochasticParticleFlow(
        ssm,
        num_lambda=args.num_lambda,
        num_particles=args.num_particles,
        diffusion=Q,
        reweight="never",
        beta_mode="linear",
        optimal_beta=False,
    )

    batch_size = max(1, int(args.mc_runs))
    rng = np.random.default_rng(args.seed)
    L0 = np.linalg.cholesky(P0).astype(np.float32)
    if args.shared_init:
        eps0 = rng.standard_normal(size=(args.num_particles, 2)).astype(np.float32)
        x0 = m0 + eps0 @ L0.T
        x0 = np.broadcast_to(x0, (batch_size, args.num_particles, 2)).copy()
    else:
        eps0 = rng.standard_normal(size=(batch_size, args.num_particles, 2)).astype(np.float32)
        x0 = m0 + eps0 @ L0.T

    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float32)  # [B, N, 2]
    y_tf = tf.convert_to_tensor(np.broadcast_to(z, (batch_size, 2)), dtype=tf.float32)
    m0_tf = tf.convert_to_tensor(np.broadcast_to(m0, (batch_size, 2)), dtype=tf.float32)
    P0_tf = tf.convert_to_tensor(np.broadcast_to(P0, (batch_size, 2, 2)), dtype=tf.float32)

    _, Info = flow._likelihood_terms(x0_tf, y_tf)
    P0_inv = flow._inverse_from_cov(P0_tf)
    beta_opt, beta_dot_opt, _ = flow.solve_optimal_beta_schedule(
        P0_inv,
        Info,
        mu=args.mu,
    )
    beta_base = np.linspace(0.0, 1.0, args.num_lambda + 1, dtype=np.float32)[:-1]
    beta_dot_base = np.ones_like(beta_base)
    beta_base_tf = tf.convert_to_tensor(beta_base[tf.newaxis, :], dtype=tf.float32)
    beta_dot_base_tf = tf.convert_to_tensor(beta_dot_base[tf.newaxis, :], dtype=tf.float32)

    flow.beta = beta_base_tf
    flow.beta_dot = beta_dot_base_tf
    flow.optimal_beta = False
    flow.beta_guard = False
    ssm.rng = tf.random.Generator.from_seed(int(args.seed))
    x_post_base, _, _ = flow._flow_transport(x0_tf, y_tf, m0_tf, P0_tf, beta_mu=args.mu)

    flow.beta_guard = True
    ssm.rng = tf.random.Generator.from_seed(int(args.seed))
    x_post_opt, _, _ = flow._flow_transport(
        x0_tf,
        y_tf,
        m0_tf,
        P0_tf,
        beta_mu=args.mu,
        beta_override=beta_opt,
        beta_dot_override=beta_dot_opt,
    )

    x_post_base_np = x_post_base.numpy()
    x_post_opt_np = x_post_opt.numpy()
    mean_base_all = x_post_base_np.mean(axis=1)
    mean_opt_all = x_post_opt_np.mean(axis=1)
    mean_base = mean_base_all[0]
    mean_opt = mean_opt_all[0]
    if args.beta_mode == "optimal":
        x_post = x_post_opt_np[0]
        mean_post = mean_opt
    else:
        x_post = x_post_base_np[0]
        mean_post = mean_base

    if args.beta_mode == "both":
        print(
            "Posterior mean (single step, beta_mode=both): "
            f"baseline={mean_base}, optimal={mean_opt}"
        )
    else:
        print(f"Posterior mean (single step, beta_mode={args.beta_mode}): {mean_post}")
    print(f"Posterior mean (baseline): {mean_base}")
    print(f"Posterior mean (optimal): {mean_opt}")

    lam = beta_base.copy()
    beta_opt_np = beta_opt.numpy()
    beta_dot_opt_np = beta_dot_opt.numpy()
    P0_inv_np = P0_inv.numpy()
    Info_np = Info.numpy()
    beta_opt_plot = beta_opt_np[0]
    beta_dot_opt_plot = beta_dot_opt_np[0]
    P0_inv_plot = P0_inv_np[0]
    Info_plot = Info_np[0]
    stiff_base = _stiffness_ratio(Q, P0_inv_plot, Info_plot, beta_base, beta_dot_base)
    stiff_opt = _stiffness_ratio(Q, P0_inv_plot, Info_plot, beta_opt_plot, beta_dot_opt_plot)

    print(f"beta_opt range: [{beta_opt_plot.min():.6f}, {beta_opt_plot.max():.6f}]")
    print(f"beta_dot_opt range: [{beta_dot_opt_plot.min():.6f}, {beta_dot_opt_plot.max():.6f}]")
    print(f"beta_opt monotone: {np.all(np.diff(beta_opt_plot) >= -1e-6)}")

    if args.out is not None:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "exp5_dai22_figure2_data.npz",
            lam=lam,
            beta_base=beta_base,
            beta_opt=beta_opt_plot,
            beta_dot_opt=beta_dot_opt_plot,
            stiff_base=stiff_base,
            stiff_opt=stiff_opt,
        )
        if args.beta_mode == "both":
            np.savez_compressed(
                out_dir / "exp5_dai22_single_step.npz",
                x_post_baseline=x_post_base_np[0],
                x_post_optimal=x_post_opt_np[0],
            )
        else:
            np.savez_compressed(out_dir / "exp5_dai22_single_step.npz", x_post=x_post)

    if args.show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        axes[0, 0].plot(lam, beta_base, label="baseline beta=lambda", color="C0", linestyle="--")
        axes[0, 0].plot(lam, beta_opt_plot, label="optimal beta*", color="C1")
        axes[0, 0].set_xlabel("lambda")
        axes[0, 0].set_ylabel("beta(lambda)")
        axes[0, 0].grid(True, linestyle=":")
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(lam, beta_opt_plot - beta_base, color="C1")
        axes[0, 1].set_xlabel("lambda")
        axes[0, 1].set_ylabel("e(lambda)=beta* - lambda")
        axes[0, 1].grid(True, linestyle=":")

        axes[1, 0].plot(lam, beta_dot_opt_plot, color="C1")
        axes[1, 0].set_xlabel("lambda")
        axes[1, 0].set_ylabel("u*(lambda)=beta_dot*(lambda)")
        axes[1, 0].grid(True, linestyle=":")

        axes[1, 1].plot(lam, stiff_base, label="baseline", color="C0", linestyle="--")
        axes[1, 1].plot(lam, stiff_opt, label="optimal", color="C1")
        axes[1, 1].set_xlabel("lambda")
        axes[1, 1].set_ylabel("R_stiff")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, linestyle=":")
        axes[1, 1].legend(fontsize=8)

        fig.tight_layout()
        if args.out is not None:
            fig.savefig(out_dir / "exp5_dai22_figure2.png", dpi=150)
        if args.show:
            plt.show()
        plt.close(fig)

    if args.mc_runs > 0:
        xb = x_post_base_np[: args.mc_runs]
        xo = x_post_opt_np[: args.mc_runs]
        mean_b = xb.mean(axis=1)
        mean_o = xo.mean(axis=1)
        denom = max(args.num_particles - 1, 1)
        resid_b = xb - mean_b[:, np.newaxis, :]
        resid_o = xo - mean_o[:, np.newaxis, :]
        cov_b = np.einsum("bni,bnj->bij", resid_b, resid_b) / denom
        cov_o = np.einsum("bni,bnj->bij", resid_o, resid_o) / denom
        mse_b_vals = np.sum((mean_b - x_true) ** 2, axis=1)
        mse_o_vals = np.sum((mean_o - x_true) ** 2, axis=1)
        trp_b_vals = np.trace(cov_b, axis1=1, axis2=2)
        trp_o_vals = np.trace(cov_o, axis1=1, axis2=2)
        rows = [
            [int(i), float(mse_b_vals[i]), float(mse_o_vals[i]), float(trp_b_vals[i]), float(trp_o_vals[i])]
            for i in range(args.mc_runs)
        ]
        mse_b_mean = float(np.mean(mse_b_vals))
        mse_o_mean = float(np.mean(mse_o_vals))
        trp_b_mean = float(np.mean(trp_b_vals))
        trp_o_mean = float(np.mean(trp_o_vals))
        mse_improve = 100.0 * (mse_b_mean - mse_o_mean) / mse_b_mean if mse_b_mean > 0 else float("nan")
        trp_improve = 100.0 * (trp_b_mean - trp_o_mean) / trp_b_mean if trp_b_mean > 0 else float("nan")

        print("MC summary (Table 1):")
        print("  run_id    mse_baseline    mse_opt       trP_baseline   trP_opt")
        for row in rows:
            run_id, mse_b, mse_o, trp_b, trp_o = row
            print(f"  {run_id:>6}  {mse_b:>13.6f}  {mse_o:>12.6f}  {trp_b:>13.6f}  {trp_o:>10.6f}")
        print(
            f"  {'mean':>6}  {mse_b_mean:>13.6f}  {mse_o_mean:>12.6f}"
            f"  {trp_b_mean:>13.6f}  {trp_o_mean:>10.6f}"
        )
        print(f"  MSE baseline mean: {mse_b_mean:.6f}")
        print(f"  MSE optimal mean:  {mse_o_mean:.6f} ({mse_improve:.2f}% improvement)")
        print(f"  tr(P) baseline mean: {trp_b_mean:.6f}")
        print(f"  tr(P) optimal mean:  {trp_o_mean:.6f} ({trp_improve:.2f}% improvement)")

        if args.out is not None:
            out_dir = args.out
            with (out_dir / "exp5_dai22_table1.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "mse_baseline", "mse_opt", "trP_baseline", "trP_opt"])
                writer.writerows(rows)
                writer.writerow(["mean", mse_b_mean, mse_o_mean, trp_b_mean, trp_o_mean])


if __name__ == "__main__":
    main()
