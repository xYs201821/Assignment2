import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataclasses import dataclass, field

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, ".."))

from experiment_helper import CommonConfig, ensure_dir, to_numpy
from src.ssm_multi_target_acoustic import MultiTargetAcousticSSM


@dataclass
class MultiTargetConfig:
    num_targets: int = 4
    area_size: float = 40.0
    grid_size: int = 5 # default 5x5 -> 25 sensors
    Psi: float = 10.0
    d0: float = 0.1
    sigma_w: float = 0.1
    seed: int = 42
    out_dir: str = "runs/multi_target"
    max_plot_sensors: int = 6
    x0: np.ndarray = field(
        default_factory=lambda: np.array(
            [12.0, 6.0, 0.001, 0.001, 32.0, 32.0, -0.001, -0.005, 20.0, 13.0, -0.1, 0.01, 15.0, 35.0, 0.002, -0.02],
            dtype=np.float32,
        )
    
    )


def sensor_index_label(index: int, grid_size: int) -> str:
    col = index % grid_size
    row = index // grid_size
    return f"({col},{row})"

def plot_multitarget(
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    sensors: np.ndarray,
    out_path: Path,
    show: bool = True,
    max_sensors: int = 6,
    area_size: float = 40.0,
    grid_size: int = 5,
) -> None:
    T = x_traj.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    ax_traj = axes[0]
    num_targets = x_traj.shape[-1] // 4
    colors = ["tab:red", "tab:green", "tab:cyan", "tab:purple"]
    for c in range(num_targets):
        idx = 4 * c
        color = colors[c % len(colors)]
        ax_traj.plot(
            x_traj[:, idx],
            x_traj[:, idx + 1],
            label=f"Target {c + 1} (true)",
            linewidth=1.8,
            color=color,
        )
        ax_traj.scatter(
            [x_traj[0, idx]],
            [x_traj[0, idx + 1]],
            marker="x",
            color="k",
            s=90,
            linewidths=2.5,
            label="start position" if c == 0 else None,
        )
    ax_traj.scatter(
        sensors[:, 0],
        sensors[:, 1],
        marker="o",
        color="tab:blue",
        label="Sensor",
        s=50,
        zorder=3,
    )
    all_x = np.concatenate([sensors[:, 0], x_traj[:, ::4].reshape(-1)])
    all_y = np.concatenate([sensors[:, 1], x_traj[:, 1::4].reshape(-1)])
    x_margin = max(2.0, 0.1 * (np.max(all_x) - np.min(all_x)))
    y_margin = max(2.0, 0.1 * (np.max(all_y) - np.min(all_y)))
    ax_traj.set_title("Multi-target acoustic: true trajectories")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_xlim(-20.0, 60.0)
    ax_traj.set_ylim(-20.0, 60.0)
    ax_traj.set_xticks(np.linspace(0.0, area_size, grid_size))
    ax_traj.set_yticks(np.linspace(0.0, area_size, grid_size))
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, linestyle=":", linewidth=0.8)
    handles, labels = ax_traj.get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        filtered.append((h, l))
        seen.add(l)
    if filtered:
        legend_handles, legend_labels = zip(*filtered)
        ax_traj.legend(
            legend_handles,
            legend_labels,
            loc="lower left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False,
        )

    ax_meas = axes[1]
    sensor_count = min(max_sensors, sensors.shape[0])
    for sensor_id in range(sensor_count):
        ax_meas.plot(
            np.arange(T),
            y_traj[:, sensor_id],
            label=f"sensor {sensor_index_label(sensor_id, grid_size)}",
        )
    ax_meas.set_title(f"Sensor readings (first {sensor_count} sensors)")
    ax_meas.set_xlabel("t")
    ax_meas.set_ylabel("observation")
    ax_meas.grid(True)
    ax_meas.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    config = MultiTargetConfig()
    parser = argparse.ArgumentParser(description="Run the multi-target acoustic SSM and plot outputs.")
    parser.add_argument("--num_targets", type=int, default=config.num_targets, help="Number of moving targets.")
    parser.add_argument("--area_size", type=float, default=config.area_size, help="Size of the square area (length).")
    parser.add_argument("--grid_size", type=int, default=config.grid_size, help="Side length of the sensor grid.")
    parser.add_argument("--T", type=int, default=60, help="Number of timesteps to simulate.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for simulation.")
    parser.add_argument("--Psi", type=float, default=config.Psi, help="Sensor signal strength.")
    parser.add_argument("--d0", type=float, default=config.d0, help="Distance offset in the observation model.")
    parser.add_argument("--sigma_w", type=float, default=config.sigma_w, help="Observation noise.")
    parser.add_argument("--seed", type=int, default=config.seed, help="Random seed.")
    parser.add_argument("--out_dir", type=str, default=config.out_dir, help="Directory for saving plots.")
    parser.add_argument("--max_plot_sensors", type=int, default=config.max_plot_sensors, help="Number of sensors to visualize.")
    parser.add_argument("--no_show", action="store_true", help="Do not display the figure interactively.")
    args = parser.parse_args()

    config.num_targets = args.num_targets
    config.area_size = args.area_size
    config.grid_size = args.grid_size
    config.Psi = args.Psi
    config.d0 = args.d0
    config.sigma_w = args.sigma_w
    config.seed = args.seed
    config.out_dir = args.out_dir
    config.max_plot_sensors = args.max_plot_sensors

    tf.random.set_seed(config.seed)

    common = CommonConfig(T=args.T, batch_size=args.batch)
    ssm = MultiTargetAcousticSSM(
        num_targets=config.num_targets,
        area_size=config.area_size,
        grid_size=config.grid_size,
        Psi=config.Psi,
        d0=config.d0,
        sigma_w=config.sigma_w,
        seed=config.seed,
    )

    x_traj, y_traj = ssm.simulate(T=common.T, shape=(common.batch_size,), x0=config.x0)
    x_traj = to_numpy(x_traj[0])
    y_traj = to_numpy(y_traj[0])
    sensors = to_numpy(ssm.sensor_xy)

    out_path = Path(config.out_dir) / "multitarget_acoustic.png"
    plot_multitarget(
        x_traj=x_traj,
        y_traj=y_traj,
        sensors=sensors,
        out_path=out_path,
        show=not args.no_show,
        max_sensors=config.max_plot_sensors,
        area_size=config.area_size,
        grid_size=config.grid_size,
    )

    stats = []
    for idx in range(min(6, sensors.shape[0])):
        stats.append(
            (idx + 1, float(np.mean(y_traj[:, idx])), float(np.std(y_traj[:, idx])))
        )
    print(f"Saved figure to {out_path}")
    print("Sensor stats (first 6 sensors):")
    for sensor_id, mean, std in stats:
        print(f"  sensor {sensor_index_label(sensor_id - 1, config.grid_size)}: mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    main()
