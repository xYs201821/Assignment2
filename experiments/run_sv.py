import os
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(this_dir)

from src.ssm import StochasticVolatilitySSM
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
class SVConfig:
    alpha: float = 0.98
    sigma: float = 0.3
    beta: float = 1.0
    noise_scale_func: bool = True
    obs_mode: str = "logy2"
    obs_eps: float = 1e-16

    x0_true: float = 0.5
    m0_est: float = 0.0
    P0_scale: float = 2.0

    ukf_alpha: float = 3e-1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    pf_particles: int = 500
    pf_ess_threshold: float = 0.5


def run_sv_once(common: CommonConfig, cfg: SVConfig, seed: Optional[int]) -> Dict[str, Any]:
    if seed is not None:
        set_global_seed(seed)

    ssm = StochasticVolatilitySSM(
        alpha=cfg.alpha,
        sigma=cfg.sigma,
        beta=cfg.beta,
        seed=seed,
        noise_scale_func=cfg.noise_scale_func,
        obs_mode=cfg.obs_mode,
        obs_eps=cfg.obs_eps,
    )
    if seed is not None:
        ssm.set_seed(seed)

    x0_true = tf.constant([cfg.x0_true], dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant([cfg.m0_est], dtype=tf.float32)
    P0 = tf.eye(ssm.state_dim, dtype=tf.float32) * cfg.P0_scale

    ekf = ExtendedKalmanFilter(ssm)
    ukf = UnscentedKalmanFilter(ssm, alpha=cfg.ukf_alpha, beta=cfg.ukf_beta, kappa=cfg.ukf_kappa)
    pf = ParticleFilter(ssm, num_particles=cfg.pf_particles, ess_threshold=cfg.pf_ess_threshold)

    ekf_res = ekf.filter(y_obs, m0=m0, P0=P0)
    ukf_res = ukf.filter(y_obs, m0=m0, P0=P0)
    init_dist = make_init_dist(m0, P0)
    pf_x, pf_w, pf_diag, pf_parent = pf.filter(y_obs, init_dist=init_dist)

    x_true_x = x_true[:, :, :1]
    ekf_m_x = ekf_res["m_filt"][:, :, :1]
    ukf_m_x = ukf_res["m_filt"][:, :, :1]
    pf_m_x = weighted_mean(pf_x, pf_w, axis=-2)[:, :, :1]

    metrics = {
        "rmse_ekf_x": float(rmse_all(x_true_x, ekf_m_x).numpy()),
        "rmse_ukf_x": float(rmse_all(x_true_x, ukf_m_x).numpy()),
        "rmse_pf_x": float(rmse_all(x_true_x, pf_m_x).numpy()),
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
        "final_x_ekf": float(ekf_res["m_filt"][0, -1, 0].numpy()),
        "final_x_ukf": float(ukf_res["m_filt"][0, -1, 0].numpy()),
        "final_x_pf": float(pf_m_x[0, -1, 0].numpy()),
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


def plot_sv(result: Dict[str, Any], title_suffix: str = "", save_path: Optional[Path] = None, show: bool = True) -> None:
    x_true = to_numpy(result["x_true"])[0, :, 0]
    ekf_m = to_numpy(result["ekf"]["m_filt"])[0, :, 0]
    ukf_m = to_numpy(result["ukf"]["m_filt"])[0, :, 0]
    pf_x = to_numpy(result["pf"]["x"])[0]
    pf_w = to_numpy(result["pf"]["w"])[0]
    pf_m = to_numpy(weighted_mean(tf.convert_to_tensor(pf_x), tf.convert_to_tensor(pf_w), axis=-2))[:, 0]
    pf_ess = to_numpy(result["pf"]["diagnostics"]["ess"])[0, :]
    ess_threshold = result["pf"].get("ess_threshold")
    num_particles = result["pf"].get("num_particles")

    rmse_ekf = result["metrics"]["rmse_ekf_x"]
    rmse_ukf = result["metrics"]["rmse_ukf_x"]
    rmse_pf = result["metrics"]["rmse_pf_x"]
    colors = {"ekf": "tab:blue", "ukf": "tab:orange", "pf": "tab:green"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x_true, linewidth=2, color="black", label="True x_t")
    axes[0].plot(ekf_m, linestyle="--", color=colors["ekf"], label=f"EKF (RMSE={rmse_ekf:.3f})")
    axes[0].plot(ukf_m, linestyle="-.", color=colors["ukf"], label=f"UKF (RMSE={rmse_ukf:.3f})")
    axes[0].plot(pf_m, linestyle=":", color=colors["pf"], label=f"PF (RMSE={rmse_pf:.3f})")
    axes[0].set_title(f"SV: EKF vs UKF {title_suffix}".strip())
    axes[0].set_ylabel("x_t (log-vol)")
    axes[0].grid(True)
    axes[0].legend()

    err_ekf = np.abs(x_true - ekf_m)
    err_ukf = np.abs(x_true - ukf_m)
    err_pf = np.abs(x_true - pf_m)
    axes[1].plot(err_ekf, linestyle="--", color=colors["ekf"], label="|x_true - EKF|")
    axes[1].plot(err_ukf, linestyle="-.", color=colors["ukf"], label="|x_true - UKF|")
    axes[1].plot(err_pf, linestyle=":", color=colors["pf"], label="|x_true - PF|")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("abs error")
    axes[1].grid(True)
    axes[1].legend()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(pf_ess, color=colors["pf"], label="PF ESS")
    if ess_threshold is not None and num_particles is not None:
        plt.axhline(ess_threshold * num_particles, color="red", linestyle="--", linewidth=1.0,
                    label="ESS threshold")
    plt.title(f"SV: PF ESS {title_suffix}".strip())
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


def run(common: CommonConfig, cfg: SVConfig, seeds: List[Optional[int]]) -> None:
    out_root = Path(common.out_dir)
    ensure_dir(out_root)

    sv_runs = []
    for sd in seeds:
        r = run_sv_once(common, cfg, seed=sd)
        sv_runs.append(r)
        label = "None" if sd is None else str(sd)
        print_metrics(f"[SV seed={label}]", r["metrics"])

        if common.save:
            run_dir = out_root / f"sv_seed{sd}"
            ensure_dir(run_dir)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"common": asdict(common), "sv": asdict(cfg), "metrics": r["metrics"]}, f, indent=2)
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
            plot_sv(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
        else:
            plot_sv(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

    mean_rmse_ekf, std_rmse_ekf = aggregate_metrics(sv_runs, "rmse_ekf_x")
    mean_rmse_ukf, std_rmse_ukf = aggregate_metrics(sv_runs, "rmse_ukf_x")
    mean_rmse_pf, std_rmse_pf = aggregate_metrics(sv_runs, "rmse_pf_x")
    print("\n[SV Summary]")
    print(f"EKF RMSE_x (mean±std): {mean_rmse_ekf:.6g} ± {std_rmse_ekf:.6g}")
    print(f"UKF RMSE_x (mean±std): {mean_rmse_ukf:.6g} ± {std_rmse_ukf:.6g}")
    print(f"PF  RMSE_x (mean±std): {mean_rmse_pf:.6g} ± {std_rmse_pf:.6g}\n")


def main():
    common = CommonConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv_obs_mode", choices=["y", "logy2"], default=None)
    parser.add_argument("--sv_obs_eps", type=float, default=None)
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

    cfg = SVConfig()
    if args.sv_obs_mode is not None:
        cfg.obs_mode = args.sv_obs_mode
    if args.sv_obs_eps is not None:
        cfg.obs_eps = args.sv_obs_eps

    seeds = args.seeds if not args.no_seed else [None]
    run(common, cfg, seeds)


if __name__ == "__main__":
    main()
