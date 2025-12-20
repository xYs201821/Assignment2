import os
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import sys
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(this_dir)

from src.ssm import RangeBearingSSM
from src.motion_model import ConstantVelocityMotionModel, ConstantTurnRateMotionModel
from src.utility import weighted_mean
from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter
from experiment_helper import (
    CommonConfig,
    set_global_seed,
    to_numpy,
    rmse_all,
    innovation_rmse,
    ensure_dir,
    save_npz,
    print_metrics,
    make_init_dist,
    aggregate_metrics,
)


@dataclass
class RBConfig:
    dt: float = 0.1
    motion_model: str = "cv"
    q_scale_pos: float = 0.0
    q_scale_v: float = 0.2
    q_scale_omega: float = 0.01
    r_range: float = 0.5
    r_kappa: float = 1e4

    x0_true: Tuple[float, ...] = (10.0, 10.0, 2.0, 1.5)
    m0_est: Tuple[float, ...] = (8.0, 12.0, 0.0, 0.0)
    P0_scale: float = 5.0

    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    pf_particles: int = 800
    pf_ess_threshold: float = 0.5


def run_rb_once(common: CommonConfig, cfg: RBConfig, seed: Optional[int]) -> Dict[str, Any]:
    if seed is not None:
        set_global_seed(seed)

    cov_eps_y = np.diag([cfg.r_range**2, 1.0 / cfg.r_kappa]).astype(np.float32)
    if cfg.motion_model == "cv":
        cov_eps_x = np.diag([cfg.q_scale_pos**2, cfg.q_scale_pos**2]).astype(np.float32)
        motion_model = ConstantVelocityMotionModel(
            v=tf.zeros([2], dtype=tf.float32),
            dt=cfg.dt,
            cov_eps=cov_eps_x,
        )
        expected_dim = 4
    elif cfg.motion_model == "ctrv":
        cov_eps_x = np.diag([cfg.q_scale_v**2, cfg.q_scale_omega**2]).astype(np.float32)
        motion_model = ConstantTurnRateMotionModel(dt=cfg.dt, cov_eps=cov_eps_x)
        expected_dim = 5
    else:
        raise ValueError("motion_model must be 'cv' or 'ctrv'")

    if len(cfg.x0_true) != expected_dim or len(cfg.m0_est) != expected_dim:
        raise ValueError(f"motion_model '{cfg.motion_model}' expects {expected_dim}D x0_true/m0_est")

    ssm = RangeBearingSSM(motion_model=motion_model, cov_eps_y=cov_eps_y, seed=seed)
    if seed is not None:
        ssm.set_seed(seed)

    x0_true = tf.constant(cfg.x0_true, dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant(cfg.m0_est, dtype=tf.float32)
    P0 = tf.eye(ssm.state_dim, dtype=tf.float32) * cfg.P0_scale

    ekf = ExtendedKalmanFilter(ssm)
    ukf = UnscentedKalmanFilter(ssm, alpha=cfg.ukf_alpha, beta=cfg.ukf_beta, kappa=cfg.ukf_kappa)
    pf = ParticleFilter(ssm, num_particles=cfg.pf_particles, ess_threshold=cfg.pf_ess_threshold)

    ekf_res = ekf.filter(y_obs, m0=m0, P0=P0)
    ukf_res = ukf.filter(y_obs, m0=m0, P0=P0)
    init_dist = make_init_dist(m0, P0)
    pf_x, pf_w, pf_diag, pf_parent = pf.filter(y_obs, init_dist=init_dist)

    pos_true = x_true[:, :, :2]
    pos_ekf = ekf_res["m_filt"][:, :, :2]
    pos_ukf = ukf_res["m_filt"][:, :, :2]
    pos_pf = weighted_mean(pf_x, pf_w, axis=-2)[:, :, :2]

    metrics = {
        "rmse_pos_ekf": float(rmse_all(pos_true, pos_ekf).numpy()),
        "rmse_pos_ukf": float(rmse_all(pos_true, pos_ukf).numpy()),
        "rmse_pos_pf": float(rmse_all(pos_true, pos_pf).numpy()),
        "rmse_innov_ekf": float(innovation_rmse(ssm, y_obs, ekf_res["m_pred"]).numpy()),
        "rmse_innov_ukf": float(innovation_rmse(ssm, y_obs, ukf_res["m_pred"]).numpy()),
        "mean_condP_ekf": float(tf.reduce_mean(ekf_res["cond_P"]).numpy()),
        "mean_condS_ekf": float(tf.reduce_mean(ekf_res["cond_S"]).numpy()),
        "max_condP_ekf": float(tf.reduce_max(ekf_res["cond_P"]).numpy()),
        "max_condS_ekf": float(tf.reduce_max(ekf_res["cond_S"]).numpy()),
        "mean_condP_ukf": float(tf.reduce_mean(ukf_res["cond_P"]).numpy()),
        "mean_condS_ukf": float(tf.reduce_mean(ukf_res["cond_S"]).numpy()),
        "max_condP_ukf": float(tf.reduce_max(ukf_res["cond_P"]).numpy()),
        "max_condS_ukf": float(tf.reduce_max(ukf_res["cond_S"]).numpy()),
        "mean_ess_pf": float(tf.reduce_mean(pf_diag["ess"]).numpy()),
    }

    return {
        "x_true": x_true,
        "y_obs": y_obs,
        "ekf": ekf_res,
        "ukf": ukf_res,
        "pf": {
            "x": pf_x,
            "w": pf_w,
            "diagnostics": pf_diag,
            "parents": pf_parent,
            "ess_threshold": cfg.pf_ess_threshold,
            "num_particles": cfg.pf_particles,
        },
        "metrics": metrics,
    }


def plot_rb(result: Dict[str, Any], title_suffix: str = "", save_path: Optional[Path] = None, show: bool = True) -> None:
    x_true = to_numpy(result["x_true"])[0]
    ekf_m = to_numpy(result["ekf"]["m_filt"])[0]
    ukf_m = to_numpy(result["ukf"]["m_filt"])[0]
    y_obs = to_numpy(result["y_obs"])[0]

    pos_true = x_true[:, :2]
    pos_ekf = ekf_m[:, :2]
    pos_ukf = ukf_m[:, :2]
    pf_x = to_numpy(result["pf"]["x"])[0]
    pf_w = to_numpy(result["pf"]["w"])[0]
    pos_pf = to_numpy(weighted_mean(tf.convert_to_tensor(pf_x), tf.convert_to_tensor(pf_w), axis=-2))[:, :2]
    pf_parents = to_numpy(result["pf"]["parents"])[0]
    pf_ess = to_numpy(result["pf"]["diagnostics"]["ess"])[0, :]
    ess_threshold = result["pf"].get("ess_threshold")
    num_particles = result["pf"].get("num_particles")

    obs_x = y_obs[:, 0] * np.cos(y_obs[:, 1])
    obs_y = y_obs[:, 0] * np.sin(y_obs[:, 1])

    rmse_ekf = result["metrics"]["rmse_pos_ekf"]
    rmse_ukf = result["metrics"]["rmse_pos_ukf"]
    rmse_pf = result["metrics"]["rmse_pos_pf"]
    colors = {"ekf": "tab:blue", "ukf": "tab:orange", "pf": "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
    axes[0].scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    axes[0].plot(pos_ekf[:, 0], pos_ekf[:, 1], linestyle="--", color=colors["ekf"],
                 label=f"EKF (RMSE={rmse_ekf:.3f})")
    axes[0].plot(pos_ukf[:, 0], pos_ukf[:, 1], linestyle="-.", color=colors["ukf"],
                 label=f"UKF (RMSE={rmse_ukf:.3f})")
    axes[0].plot(pos_pf[:, 0], pos_pf[:, 1], linestyle=":", color=colors["pf"],
                 label=f"PF (RMSE={rmse_pf:.3f})")
    axes[0].set_title(f"Range-Bearing: EKF vs UKF {title_suffix}".strip())
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].axis("equal")
    axes[0].legend()

    err_ekf = np.linalg.norm(pos_true - pos_ekf, axis=1)
    err_ukf = np.linalg.norm(pos_true - pos_ukf, axis=1)
    err_pf = np.linalg.norm(pos_true - pos_pf, axis=1)
    axes[1].plot(err_ekf, linestyle="--", color=colors["ekf"], label="EKF position error")
    axes[1].plot(err_ukf, linestyle="-.", color=colors["ukf"], label="UKF position error")
    axes[1].plot(err_pf, linestyle=":", color=colors["pf"], label="PF position error")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("pos error")
    axes[1].grid(True)
    axes[1].legend()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    T = pos_true.shape[0]
    t = np.arange(T)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    line_true, = ax.plot(pos_true[:, 0], pos_true[:, 1], t, color="black", linewidth=2, label="True")
    line_ekf, = ax.plot(pos_ekf[:, 0], pos_ekf[:, 1], t, linestyle="--", color=colors["ekf"], label="EKF")
    line_ukf, = ax.plot(pos_ukf[:, 0], pos_ukf[:, 1], t, linestyle="-.", color=colors["ukf"], label="UKF")
    line_pf, = ax.plot(pos_pf[:, 0], pos_pf[:, 1], t, linestyle=":", color=colors["pf"], label="PF")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(f"Range-Bearing: Trajectory over Time {title_suffix}".strip())
    ax.legend()
    if show:
        lines = [line_true, line_ekf, line_ukf, line_pf]
        labels = [line.get_label() for line in lines]
        visibility = [line.get_visible() for line in lines]
        rax = fig.add_axes([0.02, 0.55, 0.18, 0.2])
        check = CheckButtons(rax, labels, visibility)

        def _toggle(label):
            idx = labels.index(label)
            lines[idx].set_visible(not lines[idx].get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(_toggle)
    if save_path is not None:
        traj_path = save_path.parent / "traj_3d.png"
        plt.tight_layout()
        plt.savefig(str(traj_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(pf_ess, color=colors["pf"], label="PF ESS")
    if ess_threshold is not None and num_particles is not None:
        plt.axhline(ess_threshold * num_particles, color="red", linestyle="--", linewidth=1.0,
                    label="ESS threshold")
    plt.title(f"Range-Bearing: PF ESS {title_suffix}".strip())
    plt.xlabel("t")
    plt.ylabel("ESS")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        ess_path = save_path.parent / "pf_ess.png"
        plt.tight_layout()
        plt.savefig(str(ess_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    T = pos_pf.shape[0]
    N = pf_x.shape[1]
    step = 5
    times = np.arange(0, T, step, dtype=int)
    max_pts = min(200, N)

    plt.figure(figsize=(7, 7))
    for t in times:
        wt = pf_w[t]
        idx = np.argsort(wt)[::-1][:max_pts]
        pts = pf_x[t, idx, :2]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.2, color="gray")

    if len(times) > 1:
        x0 = pos_pf[times[:-1], 0]
        y0 = pos_pf[times[:-1], 1]
        u = pos_pf[times[1:], 0] - x0
        v = pos_pf[times[1:], 1] - y0
        plt.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1.0,
                   width=0.0025, color=colors["pf"], label="PF direction")

    plt.plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
    plt.plot(pos_pf[:, 0], pos_pf[:, 1], linestyle=":", color=colors["pf"], label="PF mean")
    plt.scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    plt.title(f"Range-Bearing: PF Particles (step={step}) {title_suffix}".strip())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    if save_path is not None:
        path_plot = save_path.parent / "pf_paths.png"
        plt.tight_layout()
        plt.savefig(str(path_plot), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def run(common: CommonConfig, cfg: RBConfig, seeds: List[Optional[int]]) -> None:
    out_root = Path(common.out_dir)
    ensure_dir(out_root)

    rb_runs = []
    for sd in seeds:
        r = run_rb_once(common, cfg, seed=sd)
        rb_runs.append(r)
        label = "None" if sd is None else str(sd)
        print_metrics(f"[RB seed={label}]", r["metrics"])

        if common.save:
            run_dir = out_root / f"rb_seed{sd}"
            ensure_dir(run_dir)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"common": asdict(common), "rb": asdict(cfg), "metrics": r["metrics"]}, f, indent=2)
            save_npz(
                run_dir / "results.npz",
                x_true=r["x_true"], y_obs=r["y_obs"],
                ekf_m=r["ekf"]["m_filt"], ukf_m=r["ukf"]["m_filt"],
                ekf_P=r["ekf"]["P_filt"], ukf_P=r["ukf"]["P_filt"],
                ekf_condP=r["ekf"]["cond_P"], ukf_condP=r["ukf"]["cond_P"],
                ekf_condS=r["ekf"]["cond_S"], ukf_condS=r["ukf"]["cond_S"],
                pf_x=r["pf"]["x"], pf_w=r["pf"]["w"],
                pf_ess=r["pf"]["diagnostics"]["ess"], pf_logZ=r["pf"]["diagnostics"]["logZ"],
                pf_parents=r["pf"]["parents"],
            )
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
        else:
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

    mean_rmse_ekf, std_rmse_ekf = aggregate_metrics(rb_runs, "rmse_pos_ekf")
    mean_rmse_ukf, std_rmse_ukf = aggregate_metrics(rb_runs, "rmse_pos_ukf")
    mean_rmse_pf, std_rmse_pf = aggregate_metrics(rb_runs, "rmse_pos_pf")
    print("\n[Range-Bearing Summary]")
    print(f"EKF Pos RMSE (mean±std): {mean_rmse_ekf:.6g} ± {std_rmse_ekf:.6g}")
    print(f"UKF Pos RMSE (mean±std): {mean_rmse_ukf:.6g} ± {std_rmse_ukf:.6g}")
    print(f"PF  Pos RMSE (mean±std): {mean_rmse_pf:.6g} ± {std_rmse_pf:.6g}")


def main():
    common = CommonConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rb_motion", choices=["cv", "ctrv"], default=None)
    parser.add_argument("--q_scale_pos", type=float, default=None)
    parser.add_argument("--q_scale_v", type=float, default=None)
    parser.add_argument("--q_scale_omega", type=float, default=None)
    parser.add_argument("--r_range", type=float, default=None)
    parser.add_argument("--r_kappa", type=float, default=None)
    parser.add_argument("--x0_true", type=float, nargs="+", default=None)
    parser.add_argument("--m0_est", type=float, nargs="+", default=None)
    parser.add_argument("--P0_scale", type=float, default=None)
    parser.add_argument("--ukf_alpha", type=float, default=None)
    parser.add_argument("--ukf_beta", type=float, default=None)
    parser.add_argument("--ukf_kappa", type=float, default=None)
    parser.add_argument("--pf_particles", type=int, default=None)
    parser.add_argument("--pf_ess_threshold", type=float, default=None)
    parser.add_argument("--T", type=int, default=common.T)
    parser.add_argument("--batch", type=int, default=common.batch_size)
    parser.add_argument("--seeds", type=int, nargs="*", default=common.seed)
    parser.add_argument("--no_seed", action="store_true")
    parser.add_argument("--out_dir", type=str, default=common.out_dir)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    if args.T is not None:
        common.T = args.T
    if args.batch is not None:
        common.batch_size = args.batch
    if args.out_dir is not None:
        common.out_dir = args.out_dir

    common.seed = None if args.no_seed else args.seeds
    common.save = (not args.no_save)
    common.show = (not args.no_show)

    cfg = RBConfig()
    if args.rb_motion is not None:
        cfg.motion_model = args.rb_motion
    if args.q_scale_pos is not None:
        cfg.q_scale_pos = args.q_scale_pos
    if args.q_scale_v is not None:
        cfg.q_scale_v = args.q_scale_v
    if args.q_scale_omega is not None:
        cfg.q_scale_omega = args.q_scale_omega
    if args.r_range is not None:
        cfg.r_range = args.r_range
    if args.r_kappa is not None:
        cfg.r_kappa = args.r_kappa
    if args.P0_scale is not None:
        cfg.P0_scale = args.P0_scale
    if args.ukf_alpha is not None:
        cfg.ukf_alpha = args.ukf_alpha
    if args.ukf_beta is not None:
        cfg.ukf_beta = args.ukf_beta
    if args.ukf_kappa is not None:
        cfg.ukf_kappa = args.ukf_kappa
    if args.pf_particles is not None:
        cfg.pf_particles = args.pf_particles
    if args.pf_ess_threshold is not None:
        cfg.pf_ess_threshold = args.pf_ess_threshold
    if args.x0_true is not None:
        cfg.x0_true = tuple(args.x0_true)
    elif cfg.motion_model == "ctrv":
        cfg.x0_true = (10.0, 10.0, 2.0, 1.5, 0.0)
    if args.m0_est is not None:
        cfg.m0_est = tuple(args.m0_est)
    elif cfg.motion_model == "ctrv":
        cfg.m0_est = (8.0, 12.0, 0.0, 0.0, 0.0)

    seeds = args.seeds if not args.no_seed else [None]
    run(common, cfg, seeds)


if __name__ == "__main__":
    main()
