from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate ADH nonlinear SSM data and plot one trajectory."
    )
    parser.add_argument("--T", type=int, default=500, help="Trajectory length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Simulation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sigma-v",
        type=float,
        default=math.sqrt(10.0),
        help="Process noise std for x-state.",
    )
    parser.add_argument(
        "--sigma-w",
        type=float,
        default=1.0,
        help="Observation noise std.",
    )
    parser.add_argument("--x0-mean", type=float, default=0.0, help="Initial x mean.")
    parser.add_argument("--x0-var", type=float, default=5.0, help="Initial x variance.")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial time-state mean.")
    parser.add_argument(
        "--t0-var",
        type=float,
        default=1e-9,
        help="Initial time-state variance.",
    )
    parser.add_argument(
        "--t-process-var",
        type=float,
        default=1e-9,
        help="Time-state process variance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hmc"),
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output tag. Default: adh_T{T}_seed{seed}.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively.",
    )
    return parser.parse_args()


def _plot_trajectory(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    path: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t_axis, x_axis, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("x_t")
    axes[0].set_title("ADH nonlinear SSM trajectory")
    axes[0].grid(True, linestyle=":")

    axes[1].plot(t_axis, y_axis, color="tab:orange", linewidth=1.2)
    axes[1].set_ylabel("y_t")
    axes[1].set_xlabel("time state")
    axes[1].grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    tf.random.set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"adh_T{int(args.T)}_seed{int(args.seed)}"

    ssm = ADHNonlinearSSM(
        sigma_v=float(args.sigma_v),
        sigma_w=float(args.sigma_w),
        x0_mean=float(args.x0_mean),
        x0_var=float(args.x0_var),
        t0=float(args.t0),
        t0_var=float(args.t0_var),
        t_process_var=float(args.t_process_var),
        seed=int(args.seed),
    )

    x_traj, y_traj = ssm.simulate(T=int(args.T), shape=[int(args.batch_size)])
    x_np = x_traj.numpy()
    y_np = y_traj.numpy()

    t_axis = x_np[0, :, 1]
    x_axis = x_np[0, :, 0]
    y_axis = y_np[0, :, 0]

    plot_path = out_dir / f"{tag}_trajectory.png"
    data_path = out_dir / f"{tag}_data.npz"

    _plot_trajectory(t_axis=t_axis, x_axis=x_axis, y_axis=y_axis, path=plot_path, show=args.show)
    np.savez(data_path, x_traj=x_np, y_traj=y_np)

    print(f"[done] saved data: {data_path}")
    print(f"[done] saved plot: {plot_path}")
    print(f"[shape] x_traj={x_np.shape}, y_traj={y_np.shape}")


if __name__ == "__main__":
    main()
