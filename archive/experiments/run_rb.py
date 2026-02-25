import os
import json
import argparse
from dataclasses import asdict
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
from config import CommonConfig, RBConfig
from experiment_helper import (
    set_global_seed,
    to_numpy,
    rmse_all,
    innovation_rmse,
    ensure_dir,
    save_npz,
    print_metrics,
    print_runtime,
    aggregate_metrics,
    save_runtime,
    make_seeds,
)
from experiments.filter_runner import run_filters

def run_rb_once(common: CommonConfig, cfg: RBConfig, seed: Optional[int]) -> Dict[str, Any]:
    if seed is not None:
        set_global_seed(seed)

    cov_eps_y = np.diag([cfg.r_range**2, cfg.r_bearing**2]).astype(np.float32)
    if cfg.motion_model == "cv":
        cov_eps_x = np.diag([0.0, 0.0, cfg.q_scale_v**2, cfg.q_scale_v**2]).astype(np.float32)
        motion_model = ConstantVelocityMotionModel(
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

    reset_fn = (lambda: set_global_seed(seed)) if seed is not None else None
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

    def tfp_init_dist(shape):
        shape = tf.convert_to_tensor(shape, tf.int32)
        dist = init_dist(tf.concat([shape, [1]], axis=0))
        loc = tf.squeeze(dist.loc, axis=-2)
        scale_tril = tf.squeeze(dist.scale_tril, axis=-3)
        return tfp.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

    filter_out = run_filters(
        ssm,
        y_obs,
        cfg,
        m0=m0,
        P0=P0,
        init_dist=init_dist,
        tfp_init_dist=tfp_init_dist,
        reset_fn=reset_fn,
    )
    runtime = filter_out["runtime"]
    ekf_res = filter_out["ekf"]
    ukf_res = filter_out["ukf"]
    pf_pack = filter_out["pf"]
    tfp_pf_pack = filter_out["tfp_pf"]
    edh_pack = filter_out["edh"]
    ledh_pack = filter_out["ledh"]
    kflow_pack = filter_out["kflow"]

    pos_true = x_true[:, :, :2]
    metrics = {}
    if ekf_res is not None:
        pos_ekf = ekf_res["m_filt"][:, :, :2]
        mean_ekf = ekf_res["m_filt"]
        metrics["rmse_pos_ekf"] = float(rmse_all(pos_true, pos_ekf).numpy())
        metrics["rmse_state_ekf"] = float(rmse_all(x_true, mean_ekf).numpy())
        metrics["rmse_innov_ekf"] = float(innovation_rmse(ssm, y_obs, ekf_res["m_pred"]).numpy())
        metrics["mean_condP_ekf"] = float(tf.reduce_mean(ekf_res["cond_P"]).numpy())
        metrics["mean_condS_ekf"] = float(tf.reduce_mean(ekf_res["cond_S"]).numpy())
        metrics["max_condP_ekf"] = float(tf.reduce_max(ekf_res["cond_P"]).numpy())
        metrics["max_condS_ekf"] = float(tf.reduce_max(ekf_res["cond_S"]).numpy())
    if ukf_res is not None:
        pos_ukf = ukf_res["m_filt"][:, :, :2]
        mean_ukf = ukf_res["m_filt"]
        metrics["rmse_pos_ukf"] = float(rmse_all(pos_true, pos_ukf).numpy())
        metrics["rmse_state_ukf"] = float(rmse_all(x_true, mean_ukf).numpy())
        metrics["rmse_innov_ukf"] = float(innovation_rmse(ssm, y_obs, ukf_res["m_pred"]).numpy())
        metrics["mean_condP_ukf"] = float(tf.reduce_mean(ukf_res["cond_P"]).numpy())
        metrics["mean_condS_ukf"] = float(tf.reduce_mean(ukf_res["cond_S"]).numpy())
        metrics["max_condP_ukf"] = float(tf.reduce_max(ukf_res["cond_P"]).numpy())
        metrics["max_condS_ukf"] = float(tf.reduce_max(ukf_res["cond_S"]).numpy())
    if pf_pack is not None:
        mean_pf = weighted_mean(pf_pack["x"], pf_pack["w"], axis=-2)
        pos_pf = mean_pf[:, :, :2]
        metrics["rmse_pos_pf"] = float(rmse_all(pos_true, pos_pf).numpy())
        metrics["rmse_state_pf"] = float(rmse_all(x_true, mean_pf).numpy())
        metrics["mean_ess_pf"] = float(tf.reduce_mean(pf_pack["diagnostics"]["ess"]).numpy())
    if tfp_pf_pack is not None:
        mean_tfp_pf = weighted_mean(tfp_pf_pack["x"], tfp_pf_pack["w"], axis=-2)
        pos_tfp_pf = mean_tfp_pf[:, :, :2]
        metrics["rmse_pos_tfp_pf"] = float(rmse_all(pos_true, pos_tfp_pf).numpy())
        metrics["rmse_state_tfp_pf"] = float(rmse_all(x_true, mean_tfp_pf).numpy())
        metrics["mean_ess_tfp_pf"] = float(tf.reduce_mean(tfp_pf_pack["diagnostics"]["ess"]).numpy())
    if edh_pack is not None:
        mean_edh = weighted_mean(edh_pack["x"], edh_pack["w"], axis=-2)
        pos_edh = mean_edh[:, :, :2]
        metrics["rmse_pos_edh"] = float(rmse_all(pos_true, pos_edh).numpy())
        metrics["rmse_state_edh"] = float(rmse_all(x_true, mean_edh).numpy())
        metrics["mean_ess_edh"] = float(tf.reduce_mean(edh_pack["diagnostics"]["ess"]).numpy())
    if ledh_pack is not None:
        mean_ledh = weighted_mean(ledh_pack["x"], ledh_pack["w"], axis=-2)
        pos_ledh = mean_ledh[:, :, :2]
        metrics["rmse_pos_ledh"] = float(rmse_all(pos_true, pos_ledh).numpy())
        metrics["rmse_state_ledh"] = float(rmse_all(x_true, mean_ledh).numpy())
        metrics["mean_ess_ledh"] = float(tf.reduce_mean(ledh_pack["diagnostics"]["ess"]).numpy())
    for kernel_type, pack in kflow_pack.items():
        mean_kflow = tf.reduce_mean(pack["x"], axis=-2)
        pos_kflow = mean_kflow[:, :, :2]
        metrics[f"rmse_pos_kflow_{kernel_type}"] = float(rmse_all(pos_true, pos_kflow).numpy())
        metrics[f"rmse_state_kflow_{kernel_type}"] = float(rmse_all(x_true, mean_kflow).numpy())
        metrics[f"mean_ess_kflow_{kernel_type}"] = float(tf.reduce_mean(pack["diagnostics"]["ess"]).numpy())

    return {
        "x_true": x_true,
        "y_obs": y_obs,
        "ekf": ekf_res,
        "ukf": ukf_res,
        "pf": pf_pack,
        "tfp_pf": tfp_pf_pack,
        "edh": edh_pack,
        "ledh": ledh_pack,
        "kflow": kflow_pack,
        "metrics": metrics,
        "runtime": runtime,
    }


def plot_rb(result: Dict[str, Any], title_suffix: str = "", save_path: Optional[Path] = None, show: bool = True) -> None:
    x_true = to_numpy(result["x_true"])[0]
    y_obs = to_numpy(result["y_obs"])[0]
    pos_true = x_true[:, :2]
    metrics = result.get("metrics", {})

    colors = {
        "ekf": "tab:blue",
        "ukf": "tab:orange",
        "pf": "tab:green",
        "tfp_pf": "tab:cyan",
        "edh": "tab:red",
        "ledh": "tab:purple",
        "kflow_scalar": "tab:brown",
        "kflow_diag": "tab:olive",
    }
    styles = {
        "ekf": "--",
        "ukf": "-.",
        "pf": ":",
        "tfp_pf": "--",
        "edh": "--",
        "ledh": "-",
        "kflow_scalar": "-.",
        "kflow_diag": "-",
    }

    series = []
    particle_series = {}
    ekf_res = result.get("ekf")
    if ekf_res is not None:
        ekf_m = to_numpy(ekf_res["m_filt"])[0]
        pos_ekf = ekf_m[:, :2]
        rmse = metrics.get("rmse_pos_ekf")
        label = "EKF" if rmse is None else f"EKF (RMSE={rmse:.3f})"
        series.append(("ekf", label, pos_ekf))
    ukf_res = result.get("ukf")
    if ukf_res is not None:
        ukf_m = to_numpy(ukf_res["m_filt"])[0]
        pos_ukf = ukf_m[:, :2]
        rmse = metrics.get("rmse_pos_ukf")
        label = "UKF" if rmse is None else f"UKF (RMSE={rmse:.3f})"
        series.append(("ukf", label, pos_ukf))
    pf_res = result.get("pf")
    if pf_res is not None:
        pf_x = to_numpy(pf_res["x"])[0]
        pf_w = to_numpy(pf_res["w"])[0]
        mean_pf = to_numpy(weighted_mean(tf.convert_to_tensor(pf_x), tf.convert_to_tensor(pf_w), axis=-2))
        pos_pf = mean_pf[:, :2]
        rmse = metrics.get("rmse_pos_pf")
        label = "PF" if rmse is None else f"PF (RMSE={rmse:.3f})"
        series.append(("pf", label, pos_pf))
        particle_series["pf"] = {"x": pf_x, "w": pf_w, "pos": pos_pf}
    tfp_pf_res = result.get("tfp_pf")
    if tfp_pf_res is not None:
        tfp_pf_x = to_numpy(tfp_pf_res["x"])[0]
        tfp_pf_w = to_numpy(tfp_pf_res["w"])[0]
        mean_tfp_pf = to_numpy(weighted_mean(tf.convert_to_tensor(tfp_pf_x), tf.convert_to_tensor(tfp_pf_w), axis=-2))
        pos_tfp_pf = mean_tfp_pf[:, :2]
        rmse = metrics.get("rmse_pos_tfp_pf")
        label = "TFP-PF" if rmse is None else f"TFP-PF (RMSE={rmse:.3f})"
        series.append(("tfp_pf", label, pos_tfp_pf))
        particle_series["tfp_pf"] = {"x": tfp_pf_x, "w": tfp_pf_w, "pos": pos_tfp_pf}
    edh_res = result.get("edh")
    if edh_res is not None:
        edh_x = to_numpy(edh_res["x"])[0]
        edh_w = to_numpy(edh_res["w"])[0]
        mean_edh = to_numpy(weighted_mean(tf.convert_to_tensor(edh_x), tf.convert_to_tensor(edh_w), axis=-2))
        pos_edh = mean_edh[:, :2]
        rmse = metrics.get("rmse_pos_edh")
        label = "EDH" if rmse is None else f"EDH (RMSE={rmse:.3f})"
        series.append(("edh", label, pos_edh))
        particle_series["edh"] = {"x": edh_x, "w": edh_w, "pos": pos_edh}
    ledh_res = result.get("ledh")
    if ledh_res is not None:
        ledh_x = to_numpy(ledh_res["x"])[0]
        ledh_w = to_numpy(ledh_res["w"])[0]
        mean_ledh = to_numpy(weighted_mean(tf.convert_to_tensor(ledh_x), tf.convert_to_tensor(ledh_w), axis=-2))
        pos_ledh = mean_ledh[:, :2]
        rmse = metrics.get("rmse_pos_ledh")
        label = "LEDH" if rmse is None else f"LEDH (RMSE={rmse:.3f})"
        series.append(("ledh", label, pos_ledh))
        particle_series["ledh"] = {"x": ledh_x, "w": ledh_w, "pos": pos_ledh}
    kflow = result.get("kflow", {})
    for kernel_type, pack in kflow.items():
        kflow_x = to_numpy(pack["x"])[0]
        mean_kflow = to_numpy(tf.reduce_mean(tf.convert_to_tensor(kflow_x), axis=-2))
        pos_kflow = mean_kflow[:, :2]
        rmse = metrics.get(f"rmse_pos_kflow_{kernel_type}")
        label = f"KFlow-{kernel_type}" if rmse is None else f"KFlow-{kernel_type} (RMSE={rmse:.3f})"
        series.append((f"kflow_{kernel_type}", label, pos_kflow))
        particle_series[f"kflow_{kernel_type}"] = {"x": kflow_x, "w": None, "pos": pos_kflow}

    obs_x = y_obs[:, 0] * np.cos(y_obs[:, 1])
    obs_y = y_obs[:, 0] * np.sin(y_obs[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
    axes[0].scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    for key, label, pos in series:
        axes[0].plot(pos[:, 0], pos[:, 1], linestyle=styles.get(key, "-"), color=colors.get(key, "tab:gray"),
                     label=label)
    axes[0].set_title(f"Range-Bearing: EKF vs UKF {title_suffix}".strip())
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].axis("equal")
    axes[0].legend()

    for key, label, pos in series:
        err = np.linalg.norm(pos_true - pos, axis=1)
        axes[1].plot(err, linestyle=styles.get(key, "-"), color=colors.get(key, "tab:gray"),
                     label=f"{label.split(' ')[0]} position error")
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
    lines = [line_true]
    labels = ["True"]
    for key, label, pos in series:
        line, = ax.plot(pos[:, 0], pos[:, 1], t, linestyle=styles.get(key, "-"),
                        color=colors.get(key, "tab:gray"), label=label.split(" ")[0])
        lines.append(line)
        labels.append(line.get_label())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(f"Range-Bearing: Trajectory over Time {title_suffix}".strip())
    ax.legend()
    if show:
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

    ess_series = []
    if pf_res is not None:
        ess_series.append(("pf", "PF ESS", to_numpy(pf_res["diagnostics"]["ess"])[0, :]))
    if tfp_pf_res is not None:
        ess_series.append(("tfp_pf", "TFP-PF ESS", to_numpy(tfp_pf_res["diagnostics"]["ess"])[0, :]))
    if edh_res is not None:
        ess_series.append(("edh", "EDH ESS", to_numpy(edh_res["diagnostics"]["ess"])[0, :]))
    if ledh_res is not None:
        ess_series.append(("ledh", "LEDH ESS", to_numpy(ledh_res["diagnostics"]["ess"])[0, :]))
    for kernel_type, pack in kflow.items():
        ess_series.append((f"kflow_{kernel_type}", f"KFlow-{kernel_type} ESS",
                           to_numpy(pack["diagnostics"]["ess"])[0, :]))
    if ess_series:
        plt.figure(figsize=(8, 3))
        for key, label, ess in ess_series:
            plt.plot(ess, color=colors.get(key, "tab:gray"), label=label)
        if pf_res is not None:
            ess_threshold = pf_res.get("ess_threshold")
            num_particles = pf_res.get("num_particles")
            if ess_threshold is not None and num_particles is not None:
                plt.axhline(ess_threshold * num_particles, color="red", linestyle="--", linewidth=1.0,
                            label="ESS threshold")
        if edh_res is not None:
            edh_ess_threshold = edh_res.get("ess_threshold")
            edh_num_particles = edh_res.get("num_particles")
            if edh_ess_threshold is not None and edh_num_particles is not None:
                plt.axhline(edh_ess_threshold * edh_num_particles, color=colors["edh"], linestyle="--", linewidth=1.0,
                            label="_nolegend_")
        if ledh_res is not None:
            ledh_ess_threshold = ledh_res.get("ess_threshold")
            ledh_num_particles = ledh_res.get("num_particles")
            if ledh_ess_threshold is not None and ledh_num_particles is not None:
                plt.axhline(ledh_ess_threshold * ledh_num_particles, color=colors["ledh"], linestyle="--", linewidth=1.0,
                            label="_nolegend_")
        plt.title(f"Range-Bearing: ESS {title_suffix}".strip())
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

    if particle_series:
        step = 5
        times = np.arange(0, T, step, dtype=int)
        fig, ax = plt.subplots(figsize=(7, 7))
        clouds = []
        cloud_labels = []

        def _collect_points(x, w, max_pts):
            pts = []
            for t in times:
                if w is None:
                    idx = np.arange(min(max_pts, x.shape[1]))
                else:
                    wt = w[t]
                    idx = np.argsort(wt)[::-1][:max_pts]
                pts.append(x[t, idx, :2])
            return np.vstack(pts) if pts else np.zeros((0, 2), dtype=np.float32)

        for key, pack in particle_series.items():
            x = pack["x"]
            w = pack["w"]
            max_pts = min(200, x.shape[1])
            pts = _collect_points(x, w, max_pts)
            if key.startswith("kflow_"):
                label = f"KFlow-{key.split('_', 1)[1]} cloud"
            else:
                label = f"{key.upper()} cloud"
            color = colors.get(key, "tab:gray")
            cloud = ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.18, color=color, label=label)
            clouds.append(cloud)
            cloud_labels.append(label)

        base_key = "pf" if "pf" in particle_series else next(iter(particle_series.keys()))
        base_pos = particle_series[base_key]["pos"]
        if len(times) > 1:
            x0 = base_pos[times[:-1], 0]
            y0 = base_pos[times[:-1], 1]
            u = base_pos[times[1:], 0] - x0
            v = base_pos[times[1:], 1] - y0
            ax.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1.0,
                      width=0.0025, color=colors.get(base_key, "tab:gray"), label="direction")

        ax.plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, color="black", label="True trajectory")
        for key, label, pos in series:
            ax.plot(pos[:, 0], pos[:, 1], linestyle=styles.get(key, "-"),
                    color=colors.get(key, "tab:gray"), label=label.split(" ")[0])
        ax.scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
        ax.set_title(f"Range-Bearing: Particle Clouds (step={step}) {title_suffix}".strip())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.axis("equal")
        ax.legend()

        if show and clouds:
            visibility = [artist.get_visible() for artist in clouds]
            rax = fig.add_axes([0.02, 0.68, 0.18, 0.18])
            check = CheckButtons(rax, cloud_labels, visibility)

            def _toggle_cloud(label):
                idx = cloud_labels.index(label)
                artist = clouds[idx]
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
            save_payload = {"x_true": r["x_true"], "y_obs": r["y_obs"]}
            if r.get("ekf") is not None:
                save_payload.update(
                    ekf_m=r["ekf"]["m_filt"],
                    ekf_P=r["ekf"]["P_filt"],
                    ekf_condP=r["ekf"]["cond_P"],
                    ekf_condS=r["ekf"]["cond_S"],
                )
            if r.get("ukf") is not None:
                save_payload.update(
                    ukf_m=r["ukf"]["m_filt"],
                    ukf_P=r["ukf"]["P_filt"],
                    ukf_condP=r["ukf"]["cond_P"],
                    ukf_condS=r["ukf"]["cond_S"],
                )
            if r.get("pf") is not None:
                save_payload.update(
                    pf_x=r["pf"]["x"],
                    pf_w=r["pf"]["w"],
                    pf_ess=r["pf"]["diagnostics"]["ess"],
                    pf_logZ=r["pf"]["diagnostics"]["logZ"],
                    pf_parents=r["pf"]["parents"],
                )
            if r.get("edh") is not None:
                save_payload.update(
                    edh_x=r["edh"]["x"],
                    edh_w=r["edh"]["w"],
                    edh_ess=r["edh"]["diagnostics"]["ess"],
                    edh_logZ=r["edh"]["diagnostics"]["logZ"],
                    edh_parents=r["edh"]["parents"],
                )
            if r.get("ledh") is not None:
                save_payload.update(
                    ledh_x=r["ledh"]["x"],
                    ledh_w=r["ledh"]["w"],
                    ledh_ess=r["ledh"]["diagnostics"]["ess"],
                    ledh_logZ=r["ledh"]["diagnostics"]["logZ"],
                    ledh_parents=r["ledh"]["parents"],
                )
            kflow = r.get("kflow", {})
            for kernel_type, pack in kflow.items():
                save_payload.update(
                    {
                        f"kflow_{kernel_type}_x": pack["x"],
                        f"kflow_{kernel_type}_ess": pack["diagnostics"]["ess"],
                        f"kflow_{kernel_type}_logZ": pack["diagnostics"]["logZ"],
                        f"kflow_{kernel_type}_parents": pack["parents"],
                    }
                )
            save_npz(run_dir / "results.npz", **save_payload)
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
        else:
            plot_rb(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

    metric_sets = [set(r["metrics"].keys()) for r in rb_runs]
    metric_keys = sorted(set.intersection(*metric_sets)) if metric_sets else []
    summary = {}
    print("\n==== Range-Bearing Summary ====")
    for key in metric_keys:
        mean_val, std_val = aggregate_metrics(rb_runs, key)
        summary[key] = {"mean": mean_val, "std": std_val}
        print(f"{key} (mean±std): {mean_val:.6g} ± {std_val:.6g}")
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
    parser.add_argument("--kflow_particles", type=int, default=None)
    parser.add_argument("--kflow_num_lambda", type=int, default=None)
    parser.add_argument("--kflow_ds_init", type=float, default=None)
    parser.add_argument("--kflow_alpha", type=float, default=None)
    parser.add_argument("--kflow_kernel_types", type=str, nargs="*", default=None)
    parser.add_argument("--kflow_adaptive_step", action="store_true")
    parser.add_argument("--kflow_adaptive_window", type=int, default=None)
    parser.add_argument("--kflow_adaptive_factor", type=float, default=None)
    parser.add_argument("--kflow_adaptive_min", type=float, default=None)
    parser.add_argument("--kflow_adaptive_max", type=float, default=None)
    parser.add_argument("--kflow_debug", action="store_true")
    parser.add_argument("--kflow_debug_every", type=int, default=None)
    parser.add_argument("--kflow_max_flow_norm", type=float, default=None)
    parser.add_argument("--filters", type=str, nargs="*", default=None)
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
    if args.kflow_particles is not None:
        cfg.kflow_particles = args.kflow_particles
    if args.kflow_num_lambda is not None:
        cfg.kflow_num_lambda = args.kflow_num_lambda
    if args.kflow_ds_init is not None:
        cfg.kflow_ds_init = args.kflow_ds_init
    if args.kflow_alpha is not None:
        cfg.kflow_alpha = args.kflow_alpha
    if args.kflow_kernel_types is not None:
        cfg.kflow_kernel_types = tuple(args.kflow_kernel_types)
    if args.kflow_adaptive_step:
        cfg.kflow_adaptive_step = True
    if args.kflow_adaptive_window is not None:
        cfg.kflow_adaptive_window = args.kflow_adaptive_window
    if args.kflow_adaptive_factor is not None:
        cfg.kflow_adaptive_factor = args.kflow_adaptive_factor
    if args.kflow_adaptive_min is not None:
        if args.kflow_adaptive_min <= 0.0:
            cfg.kflow_adaptive_min = None
        else:
            cfg.kflow_adaptive_min = args.kflow_adaptive_min
    if args.kflow_adaptive_max is not None:
        if args.kflow_adaptive_max <= 0.0:
            cfg.kflow_adaptive_max = None
        else:
            cfg.kflow_adaptive_max = args.kflow_adaptive_max
    if args.kflow_debug:
        cfg.kflow_debug = True
    if args.kflow_debug_every is not None:
        cfg.kflow_debug_every = args.kflow_debug_every
    if args.kflow_max_flow_norm is not None:
        if args.kflow_max_flow_norm <= 0.0:
            cfg.kflow_max_flow_norm = None
        else:
            cfg.kflow_max_flow_norm = args.kflow_max_flow_norm
    if args.filters is not None:
        cfg.filters = tuple(args.filters)
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
