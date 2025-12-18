import os
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ssm import StochasticVolatilitySSM, RangeBearingSSM
from src.motion_model import ConstantVelocityMotionModel
from src.filter import ExtendedKalmanFilter, UnscentedKalmanFilter


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def rmse_all(x_true: tf.Tensor, x_est: tf.Tensor) -> tf.Tensor:
    err2 = tf.square(x_true - x_est)
    return tf.sqrt(tf.reduce_mean(err2))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(str(path), **{k: to_numpy(v) for k, v in arrays.items()})


def print_metrics(prefix: str, metrics: Dict[str, Any]) -> None:
    keys = sorted(metrics.keys())
    items = ", ".join([f"{k}={metrics[k]:.6g}" if isinstance(metrics[k], (float, int)) else f"{k}={metrics[k]}"
                       for k in keys])
    print(f"{prefix} {items}")


@dataclass
class CommonConfig:
    T: int = 100
    batch_size: int = 1
    seed: int = 42
    out_dir: str = "runs"
    save: bool = True
    show: bool = True


@dataclass
class SVConfig:
    alpha: float = 0.95
    sigma: float = 0.15
    beta: float = 1.0

    x0_true: Tuple[float, float] = (1.0)
    m0_est: float = 0.75
    P0_scale: float = 2.0

    ukf_alpha: float = 3e-1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0


@dataclass
class RBConfig:
    dt: float = 0.1
    q_scale_pos: float = 0.1
    r_range: float = 0.5
    r_bearing: float = 0.05

    x0_true: Tuple[float, float, float, float] = (10.0, 10.0, 1.0, 0.5)
    m0_est: Tuple[float, float, float, float] = (8.0, 12.0, 0.0, 0.0)
    P0_scale: float = 5.0

    ukf_alpha: float = 3e-1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0


def run_sv_once(common: CommonConfig, cfg: SVConfig, seed: int) -> Dict[str, Any]:
    set_global_seed(seed)

    ssm = StochasticVolatilitySSM(alpha=cfg.alpha, sigma=cfg.sigma, beta=cfg.beta, seed=seed)
    ssm.set_seed(seed)

    x0_true = tf.constant([cfg.x0_true], dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant([cfg.m0_est], dtype=tf.float32)
    P0 = tf.eye(ssm.state_dim, dtype=tf.float32) * cfg.P0_scale

    ekf = ExtendedKalmanFilter(ssm)
    ukf = UnscentedKalmanFilter(ssm, alpha=cfg.ukf_alpha, beta=cfg.ukf_beta, kappa=cfg.ukf_kappa)

    ekf_res = ekf.filter(y_obs, m0=m0, P0=P0)
    ukf_res = ukf.filter(y_obs, m0=m0, P0=P0)

    x_true_x = x_true[:, :, :1]
    ekf_m_x = ekf_res["m_filt"][:, :, :1]
    ukf_m_x = ukf_res["m_filt"][:, :, :1]

    metrics = {
        "rmse_ekf_x": float(rmse_all(x_true_x, ekf_m_x).numpy()),
        "rmse_ukf_x": float(rmse_all(x_true_x, ukf_m_x).numpy()),
        "mean_condP_ekf": float(tf.reduce_mean(ekf_res["cond_P"]).numpy()),
        "mean_condS_ekf": float(tf.reduce_mean(ekf_res["cond_S"]).numpy()),
        "mean_condP_ukf": float(tf.reduce_mean(ukf_res["cond_P"]).numpy()),
        "mean_condS_ukf": float(tf.reduce_mean(ukf_res["cond_S"]).numpy()),
        "final_x_ekf": float(ekf_res["m_filt"][0, -1, 0].numpy()),
        "final_x_ukf": float(ukf_res["m_filt"][0, -1, 0].numpy()),
    }

    return {
        "x_true": x_true,
        "y_obs": y_obs,
        "ekf": ekf_res,
        "ukf": ukf_res,
        "metrics": metrics,
    }


def plot_sv(result: Dict[str, Any], title_suffix: str = "", save_path: Optional[Path] = None, show: bool = True) -> None:
    x_true = to_numpy(result["x_true"])[0, :, 0]
    ekf_m = to_numpy(result["ekf"]["m_filt"])[0, :, 0]
    ukf_m = to_numpy(result["ukf"]["m_filt"])[0, :, 0]

    rmse_ekf = result["metrics"]["rmse_ekf_x"]
    rmse_ukf = result["metrics"]["rmse_ukf_x"]

    plt.figure(figsize=(10, 5))
    plt.plot(x_true, linewidth=2, label="True x_t")
    plt.plot(ekf_m, linestyle="--", label=f"EKF (RMSE={rmse_ekf:.3f})")
    plt.plot(ukf_m, linestyle="--", label=f"UKF (RMSE={rmse_ukf:.3f})")
    plt.title(f"SV: EKF vs UKF {title_suffix}".strip())
    plt.xlabel("t")
    plt.ylabel("x_t (log-vol)")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def run_rb_once(common: CommonConfig, cfg: RBConfig, seed: int) -> Dict[str, Any]:
    set_global_seed(seed)

    cov_eps_x = np.diag([cfg.q_scale_pos**2, cfg.q_scale_pos**2]).astype(np.float32)
    cov_eps_y = np.diag([cfg.r_range**2, cfg.r_bearing**2]).astype(np.float32)

    cv_model = ConstantVelocityMotionModel(v=tf.zeros([2], dtype=tf.float32), dt=cfg.dt, cov_eps=cov_eps_x)
    ssm = RangeBearingSSM(motion_model=cv_model, cov_eps_y=cov_eps_y, seed=seed)
    ssm.set_seed(seed)

    x0_true = tf.constant(cfg.x0_true, dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant(cfg.m0_est, dtype=tf.float32)
    P0 = tf.eye(ssm.state_dim, dtype=tf.float32) * cfg.P0_scale

    ekf = ExtendedKalmanFilter(ssm)
    ukf = UnscentedKalmanFilter(ssm, alpha=cfg.ukf_alpha, beta=cfg.ukf_beta, kappa=cfg.ukf_kappa)

    ekf_res = ekf.filter(y_obs, m0=m0, P0=P0)
    ukf_res = ukf.filter(y_obs, m0=m0, P0=P0)

    pos_true = x_true[:, :, :2]
    pos_ekf = ekf_res["m_filt"][:, :, :2]
    pos_ukf = ukf_res["m_filt"][:, :, :2]

    metrics = {
        "rmse_pos_ekf": float(rmse_all(pos_true, pos_ekf).numpy()),
        "rmse_pos_ukf": float(rmse_all(pos_true, pos_ukf).numpy()),
        "mean_condP_ekf": float(tf.reduce_mean(ekf_res["cond_P"]).numpy()),
        "mean_condS_ekf": float(tf.reduce_mean(ekf_res["cond_S"]).numpy()),
        "mean_condP_ukf": float(tf.reduce_mean(ukf_res["cond_P"]).numpy()),
        "mean_condS_ukf": float(tf.reduce_mean(ukf_res["cond_S"]).numpy()),
    }

    return {
        "x_true": x_true,
        "y_obs": y_obs,
        "ekf": ekf_res,
        "ukf": ukf_res,
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

    obs_x = y_obs[:, 0] * np.cos(y_obs[:, 1])
    obs_y = y_obs[:, 0] * np.sin(y_obs[:, 1])

    rmse_ekf = result["metrics"]["rmse_pos_ekf"]
    rmse_ukf = result["metrics"]["rmse_pos_ukf"]

    plt.figure(figsize=(7, 7))
    plt.plot(pos_true[:, 0], pos_true[:, 1], linewidth=2, label="True trajectory")
    plt.scatter(obs_x, obs_y, s=10, alpha=0.25, label="Measurements")
    plt.plot(pos_ekf[:, 0], pos_ekf[:, 1], linestyle="--", label=f"EKF (RMSE={rmse_ekf:.3f})")
    plt.plot(pos_ukf[:, 0], pos_ukf[:, 1], linestyle="--", label=f"UKF (RMSE={rmse_ukf:.3f})")

    plt.title(f"Range-Bearing: EKF vs UKF {title_suffix}".strip())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def aggregate_metrics(runs: List[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = np.array([r["metrics"][key] for r in runs], dtype=np.float32)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["sv", "rb", "all"], default="all")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    common = CommonConfig(
        T=args.T,
        batch_size=args.batch,
        seed=args.seeds[0] if len(args.seeds) > 0 else 42,
        out_dir=args.out_dir,
        save=(not args.no_save),
        show=(not args.no_show),
    )

    out_root = Path(common.out_dir)
    ensure_dir(out_root)

    if args.exp in ["sv", "all"]:
        sv_cfg = SVConfig()
        sv_runs = []
        for sd in args.seeds:
            r = run_sv_once(common, sv_cfg, seed=sd)
            sv_runs.append(r)
            print_metrics(f"[SV seed={sd}]", r["metrics"])

            if common.save:
                run_dir = out_root / f"sv_seed{sd}"
                ensure_dir(run_dir)
                with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump({"common": asdict(common), "sv": asdict(sv_cfg), "metrics": r["metrics"]}, f, indent=2)
                save_npz(
                    run_dir / "results.npz",
                    x_true=r["x_true"], y_obs=r["y_obs"],
                    ekf_m=r["ekf"]["m_filt"], ukf_m=r["ukf"]["m_filt"],
                    ekf_P=r["ekf"]["P_filt"], ukf_P=r["ukf"]["P_filt"],
                    ekf_condP=r["ekf"]["cond_P"], ukf_condP=r["ukf"]["cond_P"],
                    ekf_condS=r["ekf"]["cond_S"], ukf_condS=r["ukf"]["cond_S"],
                )
                plot_sv(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
            else:
                plot_sv(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

        mean_rmse_ekf, std_rmse_ekf = aggregate_metrics(sv_runs, "rmse_ekf_x")
        mean_rmse_ukf, std_rmse_ukf = aggregate_metrics(sv_runs, "rmse_ukf_x")
        print("\n[SV Summary]")
        print(f"EKF RMSE_x (mean±std): {mean_rmse_ekf:.6g} ± {std_rmse_ekf:.6g}")
        print(f"UKF RMSE_x (mean±std): {mean_rmse_ukf:.6g} ± {std_rmse_ukf:.6g}")

    if args.exp in ["rb", "all"]:
        rb_cfg = RBConfig()
        rb_runs = []
        for sd in args.seeds:
            r = run_rb_once(common, rb_cfg, seed=sd)
            rb_runs.append(r)
            print_metrics(f"[RB seed={sd}]", r["metrics"])

            if common.save:
                run_dir = out_root / f"rb_seed{sd}"
                ensure_dir(run_dir)
                with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump({"common": asdict(common), "rb": asdict(rb_cfg), "metrics": r["metrics"]}, f, indent=2)
                save_npz(
                    run_dir / "results.npz",
                    x_true=r["x_true"], y_obs=r["y_obs"],
                    ekf_m=r["ekf"]["m_filt"], ukf_m=r["ukf"]["m_filt"],
                    ekf_P=r["ekf"]["P_filt"], ukf_P=r["ukf"]["P_filt"],
                    ekf_condP=r["ekf"]["cond_P"], ukf_condP=r["ukf"]["cond_P"],
                    ekf_condS=r["ekf"]["cond_S"], ukf_condS=r["ukf"]["cond_S"],
                )
                plot_rb(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
            else:
                plot_rb(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

        mean_rmse_ekf, std_rmse_ekf = aggregate_metrics(rb_runs, "rmse_pos_ekf")
        mean_rmse_ukf, std_rmse_ukf = aggregate_metrics(rb_runs, "rmse_pos_ukf")
        print("\n[Range-Bearing Summary]")
        print(f"EKF Pos RMSE (mean±std): {mean_rmse_ekf:.6g} ± {std_rmse_ekf:.6g}")
        print(f"UKF Pos RMSE (mean±std): {mean_rmse_ukf:.6g} ± {std_rmse_ukf:.6g}")


if __name__ == "__main__":
    main()
