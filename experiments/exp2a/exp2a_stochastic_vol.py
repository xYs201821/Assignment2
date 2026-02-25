from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import numpy as np
import tensorflow as tf

# Suppress retracing warnings - expected when running multiple filter types
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from experiments.common.exp_helper import (
    SUMMARY_KEYS,
    aggregate_metrics_by_method,
    print_method_summary_table,
    print_metrics_compare,
    print_separator,
    record_metrics,
)
from experiments.common.exp_utils import (
    as_list,
    build_init_dist,
    cfg_section,
    cfg_subsection,
    ensure_dir,
    ess_from_weights,
    expand_sweep_values,
    ess_threshold_for_method,
    first_non_null,
    get_summary_keys_and_prefixes,
    is_particle_like_method,
    load_config,
    parse_percentile_band,
    parse_positive_int_or_none,
    particle_pairs,
    resolve_filter_model_cfg,
    resolve_plot_controls,
    resolve_optional_float_list,
    save_npz,
    select_pre_resample_weights,
    set_seed,
    tag_from_cfg,
)
from experiments.common.filter_cfg import build_filter_cfg
from experiments.common.plot_utils import plot_stability_over_time, plot_ess_over_time
from experiments.common.runner import run_filter
from src.ssm import StochasticVolatilitySSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp2a_config.yaml")


def build_ssm(
    alpha: float,
    sigma: float,
    beta: float,
    obs_mode: str,
    mu: float,
    noise_scale_func: bool,
    obs_eps: float,
    seed: int,
) -> StochasticVolatilitySSM:
    return StochasticVolatilitySSM(
        alpha=alpha,
        sigma=sigma,
        beta=beta,
        mu=mu,
        noise_scale_func=noise_scale_func,
        obs_mode=obs_mode,
        obs_eps=obs_eps,
        seed=seed,
    )


def finite_diff_jacobian(h_fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    grad = np.zeros_like(x)
    for i in range(x.shape[-1]):
        dx = np.zeros_like(x)
        dx[..., i] = eps
        f1 = h_fn(x + dx)
        f0 = h_fn(x - dx)
        grad[..., i] = (f1 - f0) / (2.0 * eps)
    return grad


def _rmse_vol(x_true: tf.Tensor, mean: tf.Tensor) -> float:
    vol_true = tf.exp(0.5 * x_true)
    vol_est = tf.exp(0.5 * mean)
    diff = vol_true - vol_est
    return float(tf.sqrt(tf.reduce_mean(tf.square(diff))).numpy())


def _sv_extra_metrics(x_true: tf.Tensor, mean: tf.Tensor) -> Dict[str, float]:
    return {"rmse_vol": _rmse_vol(x_true, mean)}


def _plot_state_trajectory(
    path: Path,
    x_true: tf.Tensor,
    outputs: Dict[str, Dict[str, Any]],
    method_order: List[str],
    title: Optional[str] = None,
    show: bool = False,
    interactive: bool = False,
    time_gap: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    x_true_np = np.asarray(x_true)[0, :, 0]
    t_axis = np.arange(len(x_true_np))
    markevery = max(1, int(len(x_true_np) / 12))
    fig, ax = plt.subplots(figsize=(8, 3))
    line_true, = ax.plot(x_true_np, color="k", label="true", linestyle="-")
    style_cycle = cycle(["-", "--", "-.", ":", (0, (3, 1, 1, 1))])
    marker_cycle = cycle(["o", "s", "^", "v", "D", "x", "P", "*"])
    for method in method_order:
        mean = outputs.get(method, {}).get("mean")
        if mean is None:
            continue
        line, = ax.plot(
            np.asarray(mean)[0, :, 0],
            label=method,
            linestyle=next(style_cycle),
            marker=next(marker_cycle),
            markevery=markevery,
            markersize=4,
        )
    ax.set_xlabel("t")
    ax.set_ylabel("x_t")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

# Common config helpers are imported from exp_utils.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2a (stochastic volatility) runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: key=value (dot-separated keys).",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable memory tracking (faster; no memory.* in output).",
    )
    parser.add_argument(
        "--track-profile",
        action="store_true",
        help="Enable TF profiler (adds overhead; writes to results/tf_profiler).",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Run TensorFlow functions eagerly (debug; slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, args.overrides)

    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    filters_cfg = cfg_section(cfg, "filters")
    model_filter_cfg = resolve_filter_model_cfg(cfg, filters_cfg)
    pf_cfg = cfg_subsection(filters_cfg, "pf", "filters")
    flow_cfg = cfg_subsection(filters_cfg, "flow", "filters")
    spf_cfg = cfg_subsection(filters_cfg, "stochastic_pf", "filters")
    kflow_cfg = cfg_subsection(filters_cfg, "kflow", "filters")
    ukf_cfg = cfg_subsection(filters_cfg, "ukf", "filters")
    init_cfg = cfg_section(cfg, "init")

    out_root = Path(exp_cfg.get("output_root", "results/exp2a_stochastic_vol"))
    ensure_dir(out_root)

    # CLI: --no-tracking disables memory; --track-profile enables TF profiler (no other profiler).
    track_memory = not args.no_tracking and exp_cfg.get("track_memory", True)
    track_profile = bool(args.track_profile or exp_cfg.get("track_profile", False))
    eager = bool(args.eager or exp_cfg.get("eager", False))
    if eager:
        tf.config.run_functions_eagerly(True)

    T = int(exp_cfg.get("T", 80))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    calc_ekf_jacobian = bool(exp_cfg.get("calc_ekf_jacobian_error", True))
    plot_controls = resolve_plot_controls(
        exp_cfg,
        [
            ("plot_state", "plot_seed0_only", None),
            ("plot_pf_ess", "plot_pf_ess_seed0_only", "plot_pf_ess_show"),
            ("plot_stability", "plot_stability_seed0_only", "plot_stability_show"),
        ],
    )
    plot_state = plot_controls["plot_state"]
    plot_seed0_only = plot_controls["plot_seed0_only"]
    show_plots = plot_controls["show_plots"]
    plot_interactive = plot_controls["plot_interactive"]
    plot_pf_ess = plot_controls["plot_pf_ess"]
    plot_pf_ess_seed0_only = plot_controls["plot_pf_ess_seed0_only"]
    plot_pf_ess_show = plot_controls["plot_pf_ess_show"]
    plot_stability = plot_controls["plot_stability"]
    plot_stability_seed0_only = plot_controls["plot_stability_seed0_only"]
    plot_stability_show = plot_controls["plot_stability_show"]
    plot_time_gap = parse_positive_int_or_none(exp_cfg.get("plot_time_gap"))
    plot_pf_ess_percentiles = parse_percentile_band(
        exp_cfg.get("plot_pf_ess_percentiles"),
        (10.0, 90.0),
    )
    plot_stability_percentiles = parse_percentile_band(
        exp_cfg.get("plot_stability_percentiles"),
        (25.0, 75.0),
    )
    metrics_cfg_override = cfg_section(cfg, "metrics")
    summary_keys, exclude_prefixes = get_summary_keys_and_prefixes(exp_cfg, SUMMARY_KEYS)

    alpha = float(model_cfg.get("alpha", 0.99))
    mu = float(model_cfg.get("mu", 0.01))
    noise_scale_func = bool(model_cfg.get("noise_scale_func", False))
    obs_eps = float(model_cfg.get("obs_eps", 1e-16))
    betas = [float(b) for b in as_list(model_cfg.get("betas", [2.0]))]
    sigmas = [float(s) for s in as_list(model_cfg.get("sigmas", [0.6]))]
    sigmas_filter = expand_sweep_values(
        resolve_optional_float_list(
            first_non_null(
                model_filter_cfg.get("sigmas"),
                model_filter_cfg.get("sigma"),
                model_cfg.get("sigmas_filter"),
                model_cfg.get("sigma_filter"),
            )
        ),
        sigmas,
        "filters.model.sigmas",
    )
    obs_eps_filter_raw = first_non_null(
        model_filter_cfg.get("obs_eps"),
        model_cfg.get("obs_eps_filter"),
    )
    obs_eps_filter = float(obs_eps_filter_raw) if obs_eps_filter_raw is not None else obs_eps
    obs_modes = [str(m) for m in as_list(model_cfg.get("obs_modes", ["y", "logy2"]))]

    base_methods = [
        str(m).lower() for m in as_list(filters_cfg.get("methods", ["ekf", "ukf", "pf"]))
    ]
    pf_particles = [int(n) for n in as_list(pf_cfg.get("num_particles", [200]))]
    flow_particles = [int(n) for n in as_list(flow_cfg.get("num_particles", [200]))]

    num_lambda_flow = int(flow_cfg.get("num_lambda", 20))
    pf_ess_threshold = float(pf_cfg.get("ess_threshold", 0.5))
    flow_ess_threshold = float(flow_cfg.get("ess_threshold", 0.5))
    pf_reweight = str(pf_cfg.get("reweight", "auto"))
    flow_reweight = str(flow_cfg.get("reweight", "auto"))
    ukf_alpha = ukf_cfg.get("alpha")
    ukf_beta = ukf_cfg.get("beta")
    ukf_kappa = ukf_cfg.get("kappa")
    ukf_jitter = ukf_cfg.get("jitter")

    m0 = init_cfg.get("m0")
    P0 = init_cfg.get("P0")
    m0_arr = np.array(m0, dtype=np.float32) if m0 is not None else None
    P0_arr = np.array(P0, dtype=np.float32) if P0 is not None else None
    init_dist = None
    if m0_arr is not None and P0_arr is not None:
        init_dist = build_init_dist(m0_arr, P0_arr)

    pairs_list = particle_pairs(pf_particles, flow_particles, pair_particles)

    for obs_mode in obs_modes:
        metrics_cfg: Dict[str, Any] = {
            "rmse_obs": False,
            "rmse_unobs": False,
            "rmse_y": obs_mode != "y",
            "nees": False,
            "nis": False,
            "rank_hist": False,
        }
        if isinstance(metrics_cfg_override, dict):
            metrics_cfg.update(metrics_cfg_override)
        for beta in betas:
            for sigma_idx, sigma in enumerate(sigmas):
                sigma_filter = sigmas_filter[sigma_idx]
                for N_pf, N_flow in pairs_list:
                    cfg_tag = tag_from_cfg(
                        {
                            "obs": obs_mode,
                            "beta": beta,
                            "sigma": sigma,
                            "Npf": N_pf,
                            "Nflow": N_flow,
                            "lambda": num_lambda_flow,
                            "B": batch_size,
                        }
                    )
                    methods, filter_cfg = build_filter_cfg(
                        num_particles_pf=N_pf,
                        num_particles_flow=N_flow,
                        num_lambda_flow=num_lambda_flow,
                        ukf_alpha=ukf_alpha,
                        ukf_beta=ukf_beta,
                        ukf_kappa=ukf_kappa,
                        ukf_jitter=ukf_jitter,
                        ess_threshold_pf=pf_ess_threshold,
                        ess_threshold_flow=flow_ess_threshold,
                        reweight_pf=pf_reweight,
                        reweight_flow=flow_reweight,
                        methods=base_methods,
                        flow_cfg=flow_cfg,
                        kflow_cfg=kflow_cfg,
                        stochastic_pf_cfg=spf_cfg,
                    )
                    metrics_across_seeds: Dict[str, List[Dict[str, Any]]] = {
                        method: [] for method in methods
                    }
                    for seed in seeds:
                        set_seed(seed)
                        sim_ssm = build_ssm(
                            alpha=alpha,
                            sigma=sigma,
                            beta=beta,
                            obs_mode=obs_mode,
                            mu=mu,
                            noise_scale_func=noise_scale_func,
                            obs_eps=obs_eps,
                            seed=seed,
                        )
                        filter_ssm = build_ssm(
                            alpha=alpha,
                            sigma=sigma_filter,
                            beta=beta,
                            obs_mode=obs_mode,
                            mu=mu,
                            noise_scale_func=noise_scale_func,
                            obs_eps=obs_eps_filter,
                            seed=seed,
                        )
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,))
                        rng_state = filter_ssm.rng.state.read_value()

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            filter_ssm.rng.state.assign(rng_state)
                            method_ssm = filter_ssm
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "kalman", "ekf", "ukf"):
                                if m0_arr is not None:
                                    method_cfg["m0"] = m0_arr
                                if P0_arr is not None:
                                    method_cfg["P0"] = P0_arr
                            elif init_dist is not None:
                                method_cfg["init_dist"] = init_dist
                            method_cfg["init_seed"] = seed
                            method_cfg["track_memory"] = track_memory
                            method_cfg["track_profile"] = track_profile
                            out = run_filter(
                                method_ssm,
                                y_obs,
                                method,
                                **method_cfg,
                            )
                            outputs[method] = out

                        metrics_by_method: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            out = outputs[method]
                            method_dir = per_seed_dir / method
                            extra_metrics = _sv_extra_metrics(x_true, out["mean"])
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                metrics_cfg=metrics_cfg,
                                extra_metrics=extra_metrics,
                                prefix=f"exp2a_stochastic_vol {cfg_tag} seed{seed} {method}",
                                print_full=False,
                            )
                            metrics_by_method[method] = metrics
                            metrics_across_seeds[method].append(metrics)

                            if plot_pf_ess and (not plot_pf_ess_seed0_only or seed == seeds[0]):
                                if is_particle_like_method(method) and not out.get("is_gaussian", False):
                                    w_pre = select_pre_resample_weights(out)
                                    if w_pre is not None:
                                        ess_threshold = ess_threshold_for_method(
                                            method,
                                            pf_ess_threshold,
                                            flow_ess_threshold,
                                        )
                                        plot_path = method_dir / "pf_ess_over_time.png"
                                        ess_arr = ess_from_weights(w_pre)
                                        if ess_arr is not None:
                                            plot_ess_over_time(
                                                plot_path,
                                                {method: ess_arr},
                                                ess_threshold=ess_threshold,
                                                band_percentiles=plot_pf_ess_percentiles,
                                                show=plot_pf_ess_show,
                                            )

                            diag = {
                                k: v
                                for k, v in out.get("diagnostics", {}).items()
                                if v is not None and not isinstance(v, dict)
                            }
                            diag["mean"] = out["mean"]
                            diag["cov"] = out["cov"]
                            diff = x_true - out["mean"]
                            diag["rmse_t"] = tf.norm(diff, axis=-1)
                            save_npz(method_dir / "diagnostics.npz", **diag)
                            if plot_stability and (
                                not plot_stability_seed0_only or seed == seeds[0]
                            ):
                                diag_src = out.get("diagnostics", {})
                                if isinstance(diag_src, dict):
                                    plot_stability_over_time(
                                        method_dir,
                                        diag_src,
                                        band_percentiles=plot_stability_percentiles,
                                        show=plot_stability_show,
                                    )

                        print_separator(f"exp2a_stochastic_vol {cfg_tag} seed{seed} summary")
                        print_method_summary_table(
                            metrics_by_method,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2a_stochastic_vol {cfg_tag} seed{seed} compare")
                        print_metrics_compare(
                            metrics_by_method,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )

                        if plot_state and (not plot_seed0_only or seed == seeds[0]):
                            plot_path = per_seed_dir / "state_trajectory.png"
                            plot_title = f"exp2a_stochastic_vol {cfg_tag} seed{seed}"
                            _plot_state_trajectory(
                                plot_path,
                                x_true,
                                outputs,
                                methods,
                                title=plot_title,
                                show=show_plots,
                                interactive=plot_interactive,
                                time_gap=plot_time_gap,
                            )

                        if calc_ekf_jacobian and "ekf" in outputs:
                            ekf = outputs["ekf"]
                            m_pred = ekf.get("m_pred", ekf["mean"])
                            x_series = tf.convert_to_tensor(m_pred[0], tf.float32)
                            with tf.GradientTape() as tape:
                                tape.watch(x_series)
                                y_series = sim_ssm.h(x_series)
                            jac_series = tape.batch_jacobian(y_series, x_series).numpy()
                            x_series_np = x_series.numpy()
                            jac_err = []
                            for t in range(T):
                                x_t = x_series_np[t]
                                fd = finite_diff_jacobian(
                                    lambda z: sim_ssm.h(z[None, :])[0].numpy(), x_t
                                )
                                jac_err.append(np.linalg.norm(fd - jac_series[t]))
                            save_npz(
                                per_seed_dir / "ekf_jacobian_error.npz",
                                jac_error=np.array(jac_err),
                            )

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp2a_stochastic_vol {cfg_tag} avg summary")
                        print_method_summary_table(
                            mean_metrics,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2a_stochastic_vol {cfg_tag} avg compare")
                        print_metrics_compare(
                            mean_metrics,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )


if __name__ == "__main__":
    main()
