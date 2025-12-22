import os
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import sys
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(this_dir)

from src.ssm import RangeBearingSSM
from src.motion_model import ConstantVelocityMotionModel, ConstantTurnRateMotionModel
from src.utility import weighted_mean
from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter, BootstrapParticleFilter
from src.flows.edh import EDHFlow
from src.flows.ledh import LEDHFlow
from experiment_helper import (
    CommonConfig,
    set_global_seed,
    to_numpy,
    rmse_all,
    innovation_rmse,
    ensure_dir,
    save_npz,
    print_metrics,
    print_runtime,
    print_particle_log_terms,
    make_init_dist,
    aggregate_metrics,
    timed_call,
    save_runtime,
    make_seeds,
)

def _reweight_mode(mode):
    if isinstance(mode, bool):
        return 1 if mode else 0
    if isinstance(mode, str):
        mapping = {"never": 0, "auto": 1, "always": 2}
        if mode not in mapping:
            raise ValueError(f"Invalid reweight mode: {mode}")
        return mapping[mode]
    if isinstance(mode, int):
        if mode not in (0, 1, 2):
            raise ValueError("reweight int must be 0 (never), 1 (auto), or 2 (always)")
        return mode
    raise ValueError("reweight must be bool, int, or one of 'auto', 'never', 'always'")


@dataclass
class RBConfig:
    dt: float = 1.0
    motion_model: str = "cv"
    q_scale_pos: float = 1.0 / 30.0
    q_scale_v: float = 1.0 /10.0
    q_scale_psi: float = 0.0
    q_scale_omega: float = 0.003
    r_range: float = 0.2
    r_bearing: float = 0.1

    x0_true: Tuple[float, ...] = (0.0, 0.0, 0.5, np.pi/4, 0.0)
    m0_est: Tuple[float, ...] = (0.1, 0.1, 0.4, np.pi/4, 0.0)
    P0_scale: Tuple[float, ...] = (1.0, 1.0, 0.3, 1.0, 0.1)

    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    pf_particles: int = 10000
    pf_ess_threshold: float = 0.5
    pf_reweight: str = "always"
    edh_particles: int = 100
    edh_num_lambda: int = 10
    edh_ess_threshold: float = 0.5
    edh_reweight: str = "never"


def run_rb_once(common: CommonConfig, cfg: RBConfig, seed: Optional[int]) -> Dict[str, Any]:
    if seed is not None:
        set_global_seed(seed)

    cov_eps_y = np.diag([cfg.r_range**2, cfg.r_bearing**2]).astype(np.float32)
    if cfg.motion_model == "cv":
        cov_eps_x = np.diag([cfg.q_scale_v**2, cfg.q_scale_v**2]).astype(np.float32)
        motion_model = ConstantVelocityMotionModel(
            v=tf.zeros([2], dtype=tf.float32),
            dt=cfg.dt,
            cov_eps=cov_eps_x,
        )
        expected_dim = 4
    elif cfg.motion_model == "ctrv":
        cov_eps_x = np.diag([cfg.q_scale_v**2, cfg.q_scale_psi**2, cfg.q_scale_omega**2]).astype(np.float32)
        motion_model = ConstantTurnRateMotionModel(dt=cfg.dt, cov_eps=cov_eps_x)
        expected_dim = 5
    else:
        raise ValueError("motion_model must be 'cv' or 'ctrv'")

    x0_true = cfg.x0_true
    m0_est = cfg.m0_est
    P0_scale = cfg.P0_scale
    if cfg.motion_model == "cv":
        if len(x0_true) == 5:
            vx0 = x0_true[2] * np.cos(x0_true[3])
            vy0 = x0_true[2] * np.sin(x0_true[3])
            x0_true = (x0_true[0], x0_true[1], vx0, vy0)
        if len(m0_est) == 5:
            vx0 = m0_est[2] * np.cos(m0_est[3])
            vy0 = m0_est[2] * np.sin(m0_est[3])
            m0_est = (m0_est[0], m0_est[1], vx0, vy0)
        if len(P0_scale) == 5:
            v0 = m0_est[2]
            psi0 = m0_est[3]
            sigma_v = P0_scale[2]
            sigma_psi = P0_scale[3]
            sigma_vx = np.sqrt((np.cos(psi0) * sigma_v) ** 2 + (v0 * np.sin(psi0) * sigma_psi) ** 2)
            sigma_vy = np.sqrt((np.sin(psi0) * sigma_v) ** 2 + (v0 * np.cos(psi0) * sigma_psi) ** 2)
            P0_scale = (P0_scale[0], P0_scale[1], sigma_vx, sigma_vy)

    if len(x0_true) != expected_dim or len(m0_est) != expected_dim or len(P0_scale) != expected_dim:
        raise ValueError(f"motion_model '{cfg.motion_model}' expects {expected_dim}D x0_true/m0_est")

    ssm = RangeBearingSSM(motion_model=motion_model, cov_eps_y=cov_eps_y, seed=seed)

    x0_true = tf.constant(x0_true, dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant(m0_est, dtype=tf.float32)
    P0 = tf.linalg.diag(tf.constant(P0_scale, dtype=tf.float32) ** 2)

    ekf = ExtendedKalmanFilter(ssm)
    ukf = UnscentedKalmanFilter(ssm, alpha=cfg.ukf_alpha, beta=cfg.ukf_beta, kappa=cfg.ukf_kappa)
    pf = BootstrapParticleFilter(ssm, num_particles=cfg.pf_particles, ess_threshold=cfg.pf_ess_threshold)
    edh = EDHFlow(
        ssm,
        num_lambda=cfg.edh_num_lambda,
        num_particles=cfg.edh_particles,
        ess_threshold=cfg.edh_ess_threshold,
    )
    ledh = LEDHFlow(
        ssm,
        num_lambda=cfg.edh_num_lambda,
        num_particles=cfg.edh_particles,
        ess_threshold=cfg.edh_ess_threshold,
    )

    runtime = {}
    reset_fn = (lambda: set_global_seed(seed)) if seed is not None else None
    ekf_res, runtime["ekf"] = timed_call(
        lambda: ekf.filter(y_obs, m0=m0, P0=P0),
        warmup=True,
        reset_fn=reset_fn,
    )
    ukf_res, runtime["ukf"] = timed_call(
        lambda: ukf.filter(y_obs, m0=m0, P0=P0),
        warmup=True,
        reset_fn=reset_fn,
    )
    y0 = y_obs[:, 0, :]
    r0 = y0[:, 0]
    theta0 = y0[:, 1]
    pos0 = tf.stack([r0 * tf.cos(theta0), r0 * tf.sin(theta0)], axis=-1)
    state_dim = tf.shape(m0)[-1]
    if state_dim > 2:
        tail_shape = tf.concat([tf.shape(pos0)[:-1], [state_dim - 2]], axis=0)
        tail = tf.broadcast_to(m0[2:], tail_shape)
        m0_init = tf.concat([pos0, tail], axis=-1)
    else:
        m0_init = pos0
    prior_scale = list(P0_scale)
    for i in range(2, len(prior_scale)):
        prior_scale[i] = prior_scale[i] * 10.0
    P0_init = tf.linalg.diag(tf.constant(prior_scale, dtype=tf.float32) ** 2)
    P0_init = tf.broadcast_to(
        P0_init,
        tf.concat([tf.shape(pos0)[:-1], [state_dim, state_dim]], axis=0),
    )
    L0_init = tf.linalg.cholesky(P0_init)

    def init_dist(shape):
        shape = tf.convert_to_tensor(shape, tf.int32)
        loc = tf.broadcast_to(
            m0_init[..., tf.newaxis, :],
            tf.concat([shape, [state_dim]], axis=0),
        )
        scale_tril = tf.broadcast_to(
            L0_init[..., tf.newaxis, :, :],
            tf.concat([shape, [state_dim, state_dim]], axis=0),
        )
        return tfp.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

    pf_out, runtime["pf"] = timed_call(
        lambda: pf.filter(y_obs, init_dist=init_dist, reweight=_reweight_mode(cfg.pf_reweight)),
        warmup=True,
        reset_fn=reset_fn,
    )
    pf_x, pf_w, pf_diag, pf_parent = pf_out
    edh_out, runtime["edh"] = timed_call(
        lambda: edh.filter(y_obs, init_dist=init_dist, reweight=_reweight_mode(cfg.edh_reweight)),
        warmup=True,
        reset_fn=reset_fn,
    )
    edh_x, edh_w, edh_diag, edh_parent = edh_out
    ledh_out, runtime["ledh"] = timed_call(
        lambda: ledh.filter(y_obs, init_dist=init_dist, reweight=_reweight_mode(cfg.edh_reweight)),
        warmup=True,
        reset_fn=reset_fn,
    )
    ledh_x, ledh_w, ledh_diag, ledh_parent = ledh_out
    print_particle_log_terms(
        "EDH",
        edh_w,
        edh_diag,
        enabled=bool(getattr(edh, "print_log_terms", False)),
    )
    print_particle_log_terms(
        "LEDH",
        ledh_w,
        ledh_diag,
        enabled=bool(getattr(ledh, "print_log_terms", False)),
    )

    pos_true = x_true[:, :, :2]
    pos_ekf = ekf_res["m_filt"][:, :, :2]
    pos_ukf = ukf_res["m_filt"][:, :, :2]
    pos_pf = weighted_mean(pf_x, pf_w, axis=-2)[:, :, :2]
    pos_edh = weighted_mean(edh_x, edh_w, axis=-2)[:, :, :2]
    pos_ledh = weighted_mean(ledh_x, ledh_w, axis=-2)[:, :, :2]
    mean_ekf = ekf_res["m_filt"]
    mean_ukf = ukf_res["m_filt"]
    mean_pf = weighted_mean(pf_x, pf_w, axis=-2)
    mean_edh = weighted_mean(edh_x, edh_w, axis=-2)
    mean_ledh = weighted_mean(ledh_x, ledh_w, axis=-2)

    metrics = {
        "rmse_pos_ekf": float(rmse_all(pos_true, pos_ekf).numpy()),
        "rmse_pos_ukf": float(rmse_all(pos_true, pos_ukf).numpy()),
        "rmse_pos_pf": float(rmse_all(pos_true, pos_pf).numpy()),
        "rmse_pos_edh": float(rmse_all(pos_true, pos_edh).numpy()),
        "rmse_pos_ledh": float(rmse_all(pos_true, pos_ledh).numpy()),
        "rmse_state_ekf": float(rmse_all(x_true, mean_ekf).numpy()),
        "rmse_state_ukf": float(rmse_all(x_true, mean_ukf).numpy()),
        "rmse_state_pf": float(rmse_all(x_true, mean_pf).numpy()),
        "rmse_state_edh": float(rmse_all(x_true, mean_edh).numpy()),
        "rmse_state_ledh": float(rmse_all(x_true, mean_ledh).numpy()),
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
        "mean_ess_edh": float(tf.reduce_mean(edh_diag["ess"]).numpy()),
        "mean_ess_ledh": float(tf.reduce_mean(ledh_diag["ess"]).numpy()),
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
        "edh": {
            "x": edh_x,
            "w": edh_w,
            "diagnostics": edh_diag,
            "parents": edh_parent,
            "ess_threshold": cfg.edh_ess_threshold,
            "num_particles": cfg.edh_particles,
            "num_lambda": cfg.edh_num_lambda,
        },
        "ledh": {
            "x": ledh_x,
            "w": ledh_w,
            "diagnostics": ledh_diag,
            "parents": ledh_parent,
            "ess_threshold": cfg.edh_ess_threshold,
            "num_particles": cfg.edh_particles,
            "num_lambda": cfg.edh_num_lambda,
        },
        "metrics": metrics,
        "runtime": runtime,
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
    edh_x = to_numpy(result["edh"]["x"])[0]
    edh_w = to_numpy(result["edh"]["w"])[0]
    pos_edh = to_numpy(weighted_mean(tf.convert_to_tensor(edh_x), tf.convert_to_tensor(edh_w), axis=-2))[:, :2]
    ledh_x = to_numpy(result["ledh"]["x"])[0]
    ledh_w = to_numpy(result["ledh"]["w"])[0]
    pos_ledh = to_numpy(weighted_mean(tf.convert_to_tensor(ledh_x), tf.convert_to_tensor(ledh_w), axis=-2))[:, :2]
    pf_parents = to_numpy(result["pf"]["parents"])[0]
    pf_ess = to_numpy(result["pf"]["diagnostics"]["ess"])[0, :]
    ess_threshold = result["pf"].get("ess_threshold")
    num_particles = result["pf"].get("num_particles")
    edh_ess = to_numpy(result["edh"]["diagnostics"]["ess"])[0, :]
    edh_ess_threshold = result["edh"].get("ess_threshold")
    edh_num_particles = result["edh"].get("num_particles")
    ledh_ess = to_numpy(result["ledh"]["diagnostics"]["ess"])[0, :]
    ledh_ess_threshold = result["ledh"].get("ess_threshold")
    ledh_num_particles = result["ledh"].get("num_particles")

    obs_x = y_obs[:, 0] * np.cos(y_obs[:, 1])
    obs_y = y_obs[:, 0] * np.sin(y_obs[:, 1])

    rmse_ekf = result["metrics"]["rmse_pos_ekf"]
    rmse_ukf = result["metrics"]["rmse_pos_ukf"]
    rmse_pf = result["metrics"]["rmse_pos_pf"]
    rmse_edh = result["metrics"]["rmse_pos_edh"]
    rmse_ledh = result["metrics"]["rmse_pos_ledh"]
    colors = {"ekf": "tab:blue", "ukf": "tab:orange", "pf": "tab:green", "edh": "tab:red", "ledh": "tab:purple"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
    axes[0].scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    axes[0].plot(pos_ekf[:, 0], pos_ekf[:, 1], linestyle="--", color=colors["ekf"],
                 label=f"EKF (RMSE={rmse_ekf:.3f})")
    axes[0].plot(pos_ukf[:, 0], pos_ukf[:, 1], linestyle="-.", color=colors["ukf"],
                 label=f"UKF (RMSE={rmse_ukf:.3f})")
    axes[0].plot(pos_pf[:, 0], pos_pf[:, 1], linestyle=":", color=colors["pf"],
                 label=f"PF (RMSE={rmse_pf:.3f})")
    axes[0].plot(pos_edh[:, 0], pos_edh[:, 1], linestyle="--", color=colors["edh"],
                 label=f"EDH (RMSE={rmse_edh:.3f})")
    axes[0].plot(pos_ledh[:, 0], pos_ledh[:, 1], linestyle="-", color=colors["ledh"],
                 label=f"LEDH (RMSE={rmse_ledh:.3f})")
    axes[0].set_title(f"Range-Bearing: EKF vs UKF {title_suffix}".strip())
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].axis("equal")
    axes[0].legend()

    err_ekf = np.linalg.norm(pos_true - pos_ekf, axis=1)
    err_ukf = np.linalg.norm(pos_true - pos_ukf, axis=1)
    err_pf = np.linalg.norm(pos_true - pos_pf, axis=1)
    err_edh = np.linalg.norm(pos_true - pos_edh, axis=1)
    err_ledh = np.linalg.norm(pos_true - pos_ledh, axis=1)
    axes[1].plot(err_ekf, linestyle="--", color=colors["ekf"], label="EKF position error")
    axes[1].plot(err_ukf, linestyle="-.", color=colors["ukf"], label="UKF position error")
    axes[1].plot(err_pf, linestyle=":", color=colors["pf"], label="PF position error")
    axes[1].plot(err_edh, linestyle="--", color=colors["edh"], label="EDH position error")
    axes[1].plot(err_ledh, linestyle="-", color=colors["ledh"], label="LEDH position error")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("pos error")
    axes[1].grid(True)
    axes[1].legend()

    if save_path is not None:
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
    line_edh, = ax.plot(pos_edh[:, 0], pos_edh[:, 1], t, linestyle="--", color=colors["edh"], label="EDH")
    line_ledh, = ax.plot(pos_ledh[:, 0], pos_ledh[:, 1], t, linestyle="-", color=colors["ledh"], label="LEDH")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(f"Range-Bearing: Trajectory over Time {title_suffix}".strip())
    ax.legend()
    if show:
        lines = [line_true, line_ekf, line_ukf, line_pf, line_edh, line_ledh]
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
        plt.savefig(str(traj_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(pf_ess, color=colors["pf"], label="PF ESS")
    plt.plot(edh_ess, color=colors["edh"], label="EDH ESS")
    plt.plot(ledh_ess, color=colors["ledh"], label="LEDH ESS")
    if ess_threshold is not None and num_particles is not None:
        plt.axhline(ess_threshold * num_particles, color="red", linestyle="--", linewidth=1.0,
                    label="ESS threshold")
    if edh_ess_threshold is not None and edh_num_particles is not None:
        plt.axhline(edh_ess_threshold * edh_num_particles, color=colors["edh"], linestyle="--", linewidth=1.0,
                    label="_nolegend_")
    if ledh_ess_threshold is not None and ledh_num_particles is not None:
        plt.axhline(ledh_ess_threshold * ledh_num_particles, color=colors["ledh"], linestyle="--", linewidth=1.0,
                    label="_nolegend_")
    plt.title(f"Range-Bearing: PF/EDH ESS {title_suffix}".strip())
    plt.xlabel("t")
    plt.ylabel("ESS")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        ess_path = save_path.parent / "pf_ess.png"
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

    fig, ax = plt.subplots(figsize=(7, 7))

    def _collect_points(x, w):
        pts = []
        for t in times:
            wt = w[t]
            idx = np.argsort(wt)[::-1][:max_pts]
            pts.append(x[t, idx, :2])
        return np.vstack(pts) if pts else np.zeros((0, 2), dtype=np.float32)

    pf_pts = _collect_points(pf_x, pf_w)
    edh_pts = _collect_points(edh_x, edh_w)
    ledh_pts = _collect_points(ledh_x, ledh_w)
    pf_cloud = ax.scatter(pf_pts[:, 0], pf_pts[:, 1], s=8, alpha=0.18, color=colors["pf"], label="PF cloud")
    edh_cloud = ax.scatter(edh_pts[:, 0], edh_pts[:, 1], s=8, alpha=0.18, color=colors["edh"], label="EDH cloud")
    ledh_cloud = ax.scatter(ledh_pts[:, 0], ledh_pts[:, 1], s=8, alpha=0.18, color=colors["ledh"], label="LEDH cloud")

    if len(times) > 1:
        x0 = pos_pf[times[:-1], 0]
        y0 = pos_pf[times[:-1], 1]
        u = pos_pf[times[1:], 0] - x0
        v = pos_pf[times[1:], 1] - y0
        ax.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1.0,
                  width=0.0025, color=colors["pf"], label="PF direction")

    ax.plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
    ax.plot(pos_pf[:, 0], pos_pf[:, 1], linestyle=":", color=colors["pf"], label="PF mean")
    ax.plot(pos_edh[:, 0], pos_edh[:, 1], linestyle="--", color=colors["edh"], label="EDH mean")
    ax.plot(pos_ledh[:, 0], pos_ledh[:, 1], linestyle="-", color=colors["ledh"], label="LEDH mean")
    ax.scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    ax.set_title(f"Range-Bearing: Particle Clouds (step={step}) {title_suffix}".strip())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    if show:
        cloud_labels = ["PF cloud", "EDH cloud", "LEDH cloud"]
        cloud_artists = [pf_cloud, edh_cloud, ledh_cloud]
        visibility = [artist.get_visible() for artist in cloud_artists]
        rax = fig.add_axes([0.02, 0.68, 0.18, 0.18])
        check = CheckButtons(rax, cloud_labels, visibility)

        def _toggle_cloud(label):
            idx = cloud_labels.index(label)
            artist = cloud_artists[idx]
            artist.set_visible(not artist.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(_toggle_cloud)

    if save_path is not None:
        path_plot = save_path.parent / "pf_paths.png"
        plt.savefig(str(path_plot), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def run(common: CommonConfig, cfg: RBConfig, seeds: List[Optional[int]]) -> Dict[str, Any]:
    out_root = Path(common.out_dir)
    ensure_dir(out_root)

    rb_runs = []
    for sd in seeds:
        r = run_rb_once(common, cfg, seed=sd)
        rb_runs.append(r)
        label = "None" if sd is None else str(sd)
        print_metrics(f"[RB seed={label}]", r["metrics"])
        print_runtime(f"[RB seed={label}]", r["runtime"])

        if common.save:
            run_dir = out_root / f"rb_seed{sd}"
            ensure_dir(run_dir)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"common": asdict(common), "rb": asdict(cfg), "metrics": r["metrics"], "runtime": r["runtime"]},
                    f,
                    indent=2,
                )
            save_runtime(run_dir / "runtime.csv", r["runtime"])
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
                edh_x=r["edh"]["x"], edh_w=r["edh"]["w"],
                edh_ess=r["edh"]["diagnostics"]["ess"], edh_logZ=r["edh"]["diagnostics"]["logZ"],
                edh_parents=r["edh"]["parents"],
                ledh_x=r["ledh"]["x"], ledh_w=r["ledh"]["w"],
                ledh_ess=r["ledh"]["diagnostics"]["ess"], ledh_logZ=r["ledh"]["diagnostics"]["logZ"],
                ledh_parents=r["ledh"]["parents"],
            )
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
        else:
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

    mean_rmse_ekf, std_rmse_ekf = aggregate_metrics(rb_runs, "rmse_pos_ekf")
    mean_rmse_ukf, std_rmse_ukf = aggregate_metrics(rb_runs, "rmse_pos_ukf")
    mean_rmse_pf, std_rmse_pf = aggregate_metrics(rb_runs, "rmse_pos_pf")
    mean_rmse_edh, std_rmse_edh = aggregate_metrics(rb_runs, "rmse_pos_edh")
    mean_rmse_ledh, std_rmse_ledh = aggregate_metrics(rb_runs, "rmse_pos_ledh")
    mean_rmse_state_ekf, std_rmse_state_ekf = aggregate_metrics(rb_runs, "rmse_state_ekf")
    mean_rmse_state_ukf, std_rmse_state_ukf = aggregate_metrics(rb_runs, "rmse_state_ukf")
    mean_rmse_state_pf, std_rmse_state_pf = aggregate_metrics(rb_runs, "rmse_state_pf")
    mean_rmse_state_edh, std_rmse_state_edh = aggregate_metrics(rb_runs, "rmse_state_edh")
    mean_rmse_state_ledh, std_rmse_state_ledh = aggregate_metrics(rb_runs, "rmse_state_ledh")
    summary = {
        "rmse_pos_ekf": {"mean": mean_rmse_ekf, "std": std_rmse_ekf},
        "rmse_pos_ukf": {"mean": mean_rmse_ukf, "std": std_rmse_ukf},
        "rmse_pos_pf": {"mean": mean_rmse_pf, "std": std_rmse_pf},
        "rmse_pos_edh": {"mean": mean_rmse_edh, "std": std_rmse_edh},
        "rmse_pos_ledh": {"mean": mean_rmse_ledh, "std": std_rmse_ledh},
        "rmse_state_ekf": {"mean": mean_rmse_state_ekf, "std": std_rmse_state_ekf},
        "rmse_state_ukf": {"mean": mean_rmse_state_ukf, "std": std_rmse_state_ukf},
        "rmse_state_pf": {"mean": mean_rmse_state_pf, "std": std_rmse_state_pf},
        "rmse_state_edh": {"mean": mean_rmse_state_edh, "std": std_rmse_state_edh},
        "rmse_state_ledh": {"mean": mean_rmse_state_ledh, "std": std_rmse_state_ledh},
    }
    print("\n==== Range-Bearing Summary ====")
    print(f"EKF Pos RMSE (mean±std): {mean_rmse_ekf:.6g} ± {std_rmse_ekf:.6g}")
    print(f"UKF Pos RMSE (mean±std): {mean_rmse_ukf:.6g} ± {std_rmse_ukf:.6g}")
    print(f"PF  Pos RMSE (mean±std): {mean_rmse_pf:.6g} ± {std_rmse_pf:.6g}")
    print(f"EDH Pos RMSE (mean±std): {mean_rmse_edh:.6g} ± {std_rmse_edh:.6g}")
    print(f"LEDH Pos RMSE (mean±std): {mean_rmse_ledh:.6g} ± {std_rmse_ledh:.6g}")
    print(f"EKF State RMSE (mean±std): {mean_rmse_state_ekf:.6g} ± {std_rmse_state_ekf:.6g}")
    print(f"UKF State RMSE (mean±std): {mean_rmse_state_ukf:.6g} ± {std_rmse_state_ukf:.6g}")
    print(f"PF  State RMSE (mean±std): {mean_rmse_state_pf:.6g} ± {std_rmse_state_pf:.6g}")
    print(f"EDH State RMSE (mean±std): {mean_rmse_state_edh:.6g} ± {std_rmse_state_edh:.6g}")
    print(f"LEDH State RMSE (mean±std): {mean_rmse_state_ledh:.6g} ± {std_rmse_state_ledh:.6g}")
    print("====")
    return summary


def main():
    common = CommonConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rb_motion", choices=["cv", "ctrv"], default=None)
    parser.add_argument("--q_scale_pos", type=float, default=None)
    parser.add_argument("--q_scale_v", type=float, default=None)
    parser.add_argument("--q_scale_psi", type=float, default=None)
    parser.add_argument("--q_scale_omega", type=float, default=None)
    parser.add_argument("--r_range", type=float, default=None)
    parser.add_argument("--r_bearing", type=float, default=None)
    parser.add_argument("--r_kappa", type=float, default=None)
    parser.add_argument("--x0_true", type=float, nargs="+", default=None)
    parser.add_argument("--m0_est", type=float, nargs="+", default=None)
    parser.add_argument("--P0_scale", type=float, default=None)
    parser.add_argument("--ukf_alpha", type=float, default=None)
    parser.add_argument("--ukf_beta", type=float, default=None)
    parser.add_argument("--ukf_kappa", type=float, default=None)
    parser.add_argument("--pf_particles", type=int, default=None)
    parser.add_argument("--pf_ess_threshold", type=float, default=None)
    parser.add_argument("--pf_reweight", type=str, choices=["never", "auto", "always"], default=None)
    parser.add_argument("--edh_particles", type=int, default=None)
    parser.add_argument("--edh_num_lambda", type=int, default=None)
    parser.add_argument("--edh_ess_threshold", type=float, default=None)
    parser.add_argument("--edh_reweight", type=str, choices=["never", "auto", "always"], default=None)
    parser.add_argument("--T", type=int, default=common.T)
    parser.add_argument("--batch", type=int, default=common.batch_size)
    parser.add_argument("--seeds", type=int, nargs="*", default=common.seed)
    parser.add_argument("--num_seeds", type=int, default=None)
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

    seeds = args.seeds
    if not args.no_seed and args.num_seeds is not None:
        base_seed = seeds[0] if isinstance(seeds, (list, tuple)) and len(seeds) > 0 else None
        seeds = make_seeds(args.num_seeds, base_seed=base_seed)
    common.seed = None if args.no_seed else seeds
    common.save = (not args.no_save)
    common.show = (not args.no_show)

    cfg = RBConfig()
    if args.rb_motion is not None:
        cfg.motion_model = args.rb_motion
    if args.q_scale_pos is not None:
        cfg.q_scale_pos = args.q_scale_pos
    if args.q_scale_v is not None:
        cfg.q_scale_v = args.q_scale_v
    if args.q_scale_psi is not None:
        cfg.q_scale_psi = args.q_scale_psi
    if args.q_scale_omega is not None:
        cfg.q_scale_omega = args.q_scale_omega
    if args.r_range is not None:
        cfg.r_range = args.r_range
    if args.r_bearing is not None:
        cfg.r_bearing = args.r_bearing
    elif args.r_kappa is not None:
        cfg.r_bearing = float(np.sqrt(1.0 / args.r_kappa))
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
    if args.pf_reweight is not None:
        cfg.pf_reweight = args.pf_reweight
    if args.edh_particles is not None:
        cfg.edh_particles = args.edh_particles
    if args.edh_num_lambda is not None:
        cfg.edh_num_lambda = args.edh_num_lambda
    if args.edh_ess_threshold is not None:
        cfg.edh_ess_threshold = args.edh_ess_threshold
    if args.edh_reweight is not None:
        cfg.edh_reweight = args.edh_reweight
    if args.x0_true is not None:
        cfg.x0_true = tuple(args.x0_true)
    if args.m0_est is not None:
        cfg.m0_est = tuple(args.m0_est)
    if args.P0_scale is not None:
        cfg.P0_scale = tuple(args.P0_scale)
    seeds = [None] if args.no_seed else seeds
    run(common, cfg, seeds)


if __name__ == "__main__":
    main()
