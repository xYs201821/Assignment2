import os
import json
import argparse
from dataclasses import asdict
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
from config import CommonConfig, SVConfig
from experiment_helper import (
    set_global_seed,
    to_numpy,
    rmse_all,
    innovation_rmse,
    ensure_dir,
    save_npz,
    print_metrics,
    print_runtime,
    make_init_dist,
    aggregate_metrics,
    save_runtime,
    make_seeds,
)
from experiments.filter_runner import run_filters

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

    x0_true = tf.constant([cfg.x0_true], dtype=tf.float32)
    x_true, y_obs = ssm.simulate(common.T, shape=(common.batch_size, ), x0=x0_true)

    m0 = tf.constant([cfg.m0_est], dtype=tf.float32)
    P0 = tf.eye(ssm.state_dim, dtype=tf.float32) * cfg.P0_scale

    reset_fn = (lambda: set_global_seed(seed)) if seed is not None else None
    init_dist = make_init_dist(m0, P0)
    filter_out = run_filters(
        ssm,
        y_obs,
        cfg,
        m0=m0,
        P0=P0,
        init_dist=init_dist,
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

    x_true_x = x_true[:, :, :1]
    metrics = {}
    if ekf_res is not None:
        ekf_m_x = ekf_res["m_filt"][:, :, :1]
        metrics["rmse_ekf_x"] = float(rmse_all(x_true_x, ekf_m_x).numpy())
        metrics["rmse_innov_ekf"] = float(innovation_rmse(ssm, y_obs, ekf_res["m_pred"]).numpy())
        metrics["mean_condP_ekf"] = float(tf.reduce_mean(ekf_res["cond_P"]).numpy())
        metrics["mean_condS_ekf"] = float(tf.reduce_mean(ekf_res["cond_S"]).numpy())
        metrics["max_condP_ekf"] = float(tf.reduce_max(ekf_res["cond_P"]).numpy())
        metrics["max_condS_ekf"] = float(tf.reduce_max(ekf_res["cond_S"]).numpy())
        metrics["final_x_ekf"] = float(ekf_res["m_filt"][0, -1, 0].numpy())
    if ukf_res is not None:
        ukf_m_x = ukf_res["m_filt"][:, :, :1]
        metrics["rmse_ukf_x"] = float(rmse_all(x_true_x, ukf_m_x).numpy())
        metrics["rmse_innov_ukf"] = float(innovation_rmse(ssm, y_obs, ukf_res["m_pred"]).numpy())
        metrics["mean_condP_ukf"] = float(tf.reduce_mean(ukf_res["cond_P"]).numpy())
        metrics["mean_condS_ukf"] = float(tf.reduce_mean(ukf_res["cond_S"]).numpy())
        metrics["max_condP_ukf"] = float(tf.reduce_max(ukf_res["cond_P"]).numpy())
        metrics["max_condS_ukf"] = float(tf.reduce_max(ukf_res["cond_S"]).numpy())
        metrics["final_x_ukf"] = float(ukf_res["m_filt"][0, -1, 0].numpy())
    if pf_pack is not None:
        pf_m_x = weighted_mean(pf_pack["x"], pf_pack["w"], axis=-2)[:, :, :1]
        metrics["rmse_pf_x"] = float(rmse_all(x_true_x, pf_m_x).numpy())
        metrics["final_x_pf"] = float(pf_m_x[0, -1, 0].numpy())
        metrics["mean_ess_pf"] = float(tf.reduce_mean(pf_pack["diagnostics"]["ess"]).numpy())
    if tfp_pf_pack is not None:
        tfp_pf_m_x = weighted_mean(tfp_pf_pack["x"], tfp_pf_pack["w"], axis=-2)[:, :, :1]
        metrics["rmse_tfp_pf_x"] = float(rmse_all(x_true_x, tfp_pf_m_x).numpy())
        metrics["final_x_tfp_pf"] = float(tfp_pf_m_x[0, -1, 0].numpy())
        metrics["mean_ess_tfp_pf"] = float(tf.reduce_mean(tfp_pf_pack["diagnostics"]["ess"]).numpy())
    if edh_pack is not None:
        edh_m_x = weighted_mean(edh_pack["x"], edh_pack["w"], axis=-2)[:, :, :1]
        metrics["rmse_edh_x"] = float(rmse_all(x_true_x, edh_m_x).numpy())
        metrics["final_x_edh"] = float(edh_m_x[0, -1, 0].numpy())
        metrics["mean_ess_edh"] = float(tf.reduce_mean(edh_pack["diagnostics"]["ess"]).numpy())
    if ledh_pack is not None:
        ledh_m_x = weighted_mean(ledh_pack["x"], ledh_pack["w"], axis=-2)[:, :, :1]
        metrics["rmse_ledh_x"] = float(rmse_all(x_true_x, ledh_m_x).numpy())
        metrics["final_x_ledh"] = float(ledh_m_x[0, -1, 0].numpy())
        metrics["mean_ess_ledh"] = float(tf.reduce_mean(ledh_pack["diagnostics"]["ess"]).numpy())
    for kernel_type, pack in kflow_pack.items():
        kflow_m_x = tf.reduce_mean(pack["x"], axis=-2)[:, :, :1]
        metrics[f"rmse_kflow_{kernel_type}_x"] = float(rmse_all(x_true_x, kflow_m_x).numpy())
        metrics[f"final_x_kflow_{kernel_type}"] = float(kflow_m_x[0, -1, 0].numpy())
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


def plot_sv(result: Dict[str, Any], title_suffix: str = "", save_path: Optional[Path] = None, show: bool = True) -> None:
    x_true = to_numpy(result["x_true"])[0, :, 0]
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
    ekf_res = result.get("ekf")
    if ekf_res is not None:
        ekf_m = to_numpy(ekf_res["m_filt"])[0, :, 0]
        rmse = metrics.get("rmse_ekf_x")
        label = "EKF" if rmse is None else f"EKF (RMSE={rmse:.3f})"
        series.append(("ekf", label, ekf_m))
    ukf_res = result.get("ukf")
    if ukf_res is not None:
        ukf_m = to_numpy(ukf_res["m_filt"])[0, :, 0]
        rmse = metrics.get("rmse_ukf_x")
        label = "UKF" if rmse is None else f"UKF (RMSE={rmse:.3f})"
        series.append(("ukf", label, ukf_m))
    pf_res = result.get("pf")
    if pf_res is not None:
        pf_x = to_numpy(pf_res["x"])[0]
        pf_w = to_numpy(pf_res["w"])[0]
        pf_m = to_numpy(weighted_mean(tf.convert_to_tensor(pf_x), tf.convert_to_tensor(pf_w), axis=-2))[:, 0]
        rmse = metrics.get("rmse_pf_x")
        label = "PF" if rmse is None else f"PF (RMSE={rmse:.3f})"
        series.append(("pf", label, pf_m))
    tfp_pf_res = result.get("tfp_pf")
    if tfp_pf_res is not None:
        tfp_pf_x = to_numpy(tfp_pf_res["x"])[0]
        tfp_pf_w = to_numpy(tfp_pf_res["w"])[0]
        tfp_pf_m = to_numpy(weighted_mean(tf.convert_to_tensor(tfp_pf_x), tf.convert_to_tensor(tfp_pf_w), axis=-2))[:, 0]
        rmse = metrics.get("rmse_tfp_pf_x")
        label = "TFP-PF" if rmse is None else f"TFP-PF (RMSE={rmse:.3f})"
        series.append(("tfp_pf", label, tfp_pf_m))
    edh_res = result.get("edh")
    if edh_res is not None:
        edh_x = to_numpy(edh_res["x"])[0]
        edh_w = to_numpy(edh_res["w"])[0]
        edh_m = to_numpy(weighted_mean(tf.convert_to_tensor(edh_x), tf.convert_to_tensor(edh_w), axis=-2))[:, 0]
        rmse = metrics.get("rmse_edh_x")
        label = "EDH" if rmse is None else f"EDH (RMSE={rmse:.3f})"
        series.append(("edh", label, edh_m))
    ledh_res = result.get("ledh")
    if ledh_res is not None:
        ledh_x = to_numpy(ledh_res["x"])[0]
        ledh_w = to_numpy(ledh_res["w"])[0]
        ledh_m = to_numpy(weighted_mean(tf.convert_to_tensor(ledh_x), tf.convert_to_tensor(ledh_w), axis=-2))[:, 0]
        rmse = metrics.get("rmse_ledh_x")
        label = "LEDH" if rmse is None else f"LEDH (RMSE={rmse:.3f})"
        series.append(("ledh", label, ledh_m))
    kflow = result.get("kflow", {})
    for kernel_type, pack in kflow.items():
        kflow_x = to_numpy(pack["x"])[0]
        kflow_m = to_numpy(tf.reduce_mean(tf.convert_to_tensor(kflow_x), axis=-2))[:, 0]
        rmse = metrics.get(f"rmse_kflow_{kernel_type}_x")
        label = f"KFlow-{kernel_type}" if rmse is None else f"KFlow-{kernel_type} (RMSE={rmse:.3f})"
        series.append((f"kflow_{kernel_type}", label, kflow_m))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x_true, linewidth=2, color="black", label="True x_t")
    for key, label, mean in series:
        axes[0].plot(mean, linestyle=styles.get(key, "-"), color=colors.get(key, "tab:gray"), label=label)
    axes[0].set_title(f"SV: EKF vs UKF {title_suffix}".strip())
    axes[0].set_ylabel("x_t (log-vol)")
    axes[0].grid(True)
    axes[0].legend()

    for key, label, mean in series:
        err = np.abs(x_true - mean)
        axes[1].plot(err, linestyle=styles.get(key, "-"), color=colors.get(key, "tab:gray"),
                     label=f"|x_true - {label.split(' ')[0]}|")
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
                plt.axhline(edh_ess_threshold * edh_num_particles, color=colors["edh"], linestyle="--",
                            linewidth=1.0, label="_nolegend_")
        if ledh_res is not None:
            ledh_ess_threshold = ledh_res.get("ess_threshold")
            ledh_num_particles = ledh_res.get("num_particles")
            if ledh_ess_threshold is not None and ledh_num_particles is not None:
                plt.axhline(ledh_ess_threshold * ledh_num_particles, color=colors["ledh"], linestyle="--",
                            linewidth=1.0, label="_nolegend_")
        plt.title(f"SV: ESS {title_suffix}".strip())
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


def run(common: CommonConfig, cfg: SVConfig, seeds: List[Optional[int]]) -> Dict[str, Any]:
    out_root = Path(common.out_dir)
    ensure_dir(out_root)

    sv_runs = []
    for sd in seeds:
        r = run_sv_once(common, cfg, seed=sd)
        sv_runs.append(r)
        label = "None" if sd is None else str(sd)
        print_metrics(f"[SV seed={label}]", r["metrics"])
        print_runtime(f"[SV seed={label}]", r["runtime"])

        if common.save:
            run_dir = out_root / f"sv_seed{sd}"
            ensure_dir(run_dir)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"common": asdict(common), "sv": asdict(cfg), "metrics": r["metrics"], "runtime": r["runtime"]},
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
            plot_sv(r, title_suffix=f"(seed={sd})", save_path=run_dir / "plot.png", show=common.show)
        else:
            plot_sv(r, title_suffix=f"(seed={sd})", save_path=None, show=common.show)

    metric_sets = [set(r["metrics"].keys()) for r in sv_runs]
    metric_keys = sorted(set.intersection(*metric_sets)) if metric_sets else []
    summary = {}
    print("\n==== SV Summary ====")
    for key in metric_keys:
        mean_val, std_val = aggregate_metrics(sv_runs, key)
        summary[key] = {"mean": mean_val, "std": std_val}
        print(f"{key} (mean±std): {mean_val:.6g} ± {std_val:.6g}")
    print("====")
    return summary


def main():
    common = CommonConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv_obs_mode", choices=["y", "logy2"], default=None)
    parser.add_argument("--sv_obs_eps", type=float, default=None)
    parser.add_argument("--edh_particles", type=int, default=None)
    parser.add_argument("--edh_num_lambda", type=int, default=None)
    parser.add_argument("--edh_ess_threshold", type=float, default=None)
    parser.add_argument("--kflow_particles", type=int, default=None)
    parser.add_argument("--kflow_num_lambda", type=int, default=None)
    parser.add_argument("--kflow_alpha", type=float, default=None)
    parser.add_argument("--kflow_kernel_types", type=str, nargs="*", default=None)
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

    cfg = SVConfig()
    if args.sv_obs_mode is not None:
        cfg.obs_mode = args.sv_obs_mode
    if args.sv_obs_eps is not None:
        cfg.obs_eps = args.sv_obs_eps
    if args.edh_particles is not None:
        cfg.edh_particles = args.edh_particles
    if args.edh_num_lambda is not None:
        cfg.edh_num_lambda = args.edh_num_lambda
    if args.edh_ess_threshold is not None:
        cfg.edh_ess_threshold = args.edh_ess_threshold
    if args.kflow_particles is not None:
        cfg.kflow_particles = args.kflow_particles
    if args.kflow_num_lambda is not None:
        cfg.kflow_num_lambda = args.kflow_num_lambda
    if args.kflow_alpha is not None:
        cfg.kflow_alpha = args.kflow_alpha
    if args.kflow_kernel_types is not None:
        cfg.kflow_kernel_types = tuple(args.kflow_kernel_types)
    if args.filters is not None:
        cfg.filters = tuple(args.filters)

    seeds = [None] if args.no_seed else seeds
    run(common, cfg, seeds)


if __name__ == "__main__":
    main()
