from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

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
    ess_threshold_for_method,
    ensure_dir,
    ess_from_weights,
    expand_sweep_values,
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
from src.metrics import rmse
from src.motion_model import ConstantVelocityMotionModel
from src.ssm import RangeBearingSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp2b_config.yaml")


def build_ssm(
    sigma_r: float,
    sigma_theta: float,
    dt: float,
    cov_eps_x: np.ndarray,
    jitter: float,
    seed: int,
    jacobian_r_min: Optional[float] = None,
) -> RangeBearingSSM:
    motion_model = ConstantVelocityMotionModel(
        dt=dt,
        cov_eps=cov_eps_x,
        seed=seed,
        jitter=jitter,
    )
    cov_eps_y = np.diag([sigma_r**2, sigma_theta ** 2]).astype(np.float32)
    return RangeBearingSSM(
        motion_model=motion_model,
        cov_eps_y=cov_eps_y,
        jitter=jitter,
        seed=seed,
        jacobian_r_min=jacobian_r_min,
    )


def build_initial_state(dist: float, init_cfg: Dict[str, Any]) -> np.ndarray:
    x0_cfg = init_cfg.get("x0")
    if x0_cfg is not None:
        return np.array(x0_cfg, dtype=np.float32)
    speed = float(init_cfg.get("x0_speed", 0.5))
    return np.array([dist, 0.0, speed, 0.0], dtype=np.float32)


def build_initial_mean(x0: np.ndarray, init_cfg: Dict[str, Any]) -> np.ndarray:
    offset = init_cfg.get("m0_offset")
    if offset is None:
        offset = [0.1, 0.1, 0.0, 0.0]
    return x0 + np.array(offset, dtype=np.float32)


def build_initial_cov(init_cfg: Dict[str, Any]) -> np.ndarray:
    P0 = init_cfg.get("P0")
    if P0 is not None:
        return np.array(P0, dtype=np.float32)
    diag = init_cfg.get("P0_diag")
    if diag is None:
        diag = [1.0, 1.0, 0.5, 0.5]
    diag = np.array(diag, dtype=np.float32)
    return np.diag(diag)


def build_initial_cov_filter(init_cfg: Dict[str, Any]) -> np.ndarray:
    """Optional filter-specific initial covariance."""
    P0 = init_cfg.get("P0_filter")
    if P0 is not None:
        return np.array(P0, dtype=np.float32)
    diag = init_cfg.get("P0_diag_filter")
    if diag is not None:
        diag = np.array(diag, dtype=np.float32)
        return np.diag(diag)
    return build_initial_cov(init_cfg)


def build_init_particles(
    m0: np.ndarray,
    P0: np.ndarray,
    num_particles: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dx = int(m0.shape[0])
    z = rng.standard_normal((batch_size, num_particles, dx)).astype(np.float32)
    z = z - z.mean(axis=1, keepdims=True)
    L = np.linalg.cholesky(P0).astype(np.float32)
    return m0.astype(np.float32) + np.matmul(z, L.T)


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

    x_true_np = np.asarray(x_true)[0]
    markevery = max(1, int(len(x_true_np) / 12))
    fig, ax = plt.subplots(figsize=(5, 5))
    line_true, = ax.plot(x_true_np[:, 0], x_true_np[:, 1], color="k", label="true", linestyle="-")
    style_cycle = cycle(["-", "--", "-.", ":", (0, (3, 1, 1, 1))])
    marker_cycle = cycle(["o", "s", "^", "v", "D", "x", "P", "*"])
    for method in method_order:
        mean = outputs.get(method, {}).get("mean")
        if mean is None:
            continue
        mean_np = np.asarray(mean)[0]
        line, = ax.plot(
            mean_np[:, 0],
            mean_np[:, 1],
            label=method,
            linestyle=next(style_cycle),
            marker=next(marker_cycle),
            markevery=markevery,
            markersize=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ess_from_weights moved to exp_utils.ess_from_weights


def _impoverishment_from_parents(parents: np.ndarray) -> Optional[np.ndarray]:
    parents_np = np.asarray(parents)
    if parents_np.ndim == 2:
        parents_np = parents_np[np.newaxis, ...]
    if parents_np.ndim != 3:
        return None
    batch, T, N = parents_np.shape
    if N == 0 or T == 0:
        return None
    unique_frac = np.zeros((batch, T), dtype=np.float32)
    for b in range(batch):
        for t in range(T):
            unique_frac[b, t] = np.unique(parents_np[b, t]).size / float(N)
    return 1.0 - unique_frac


def _plot_particle_cloud(
    path: Path,
    x_particles: np.ndarray,
    w: np.ndarray,
    x_true: np.ndarray,
    time_indices: List[int],
    show: bool = False,
    title: Optional[str] = None,
    dims: Tuple[int, int] = (0, 1),
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    x_np = np.asarray(x_particles)
    w_np = np.asarray(w)
    x_true_np = np.asarray(x_true)
    if x_np.ndim == 3:
        x_np = x_np[np.newaxis, ...]
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if x_np.ndim != 4 or w_np.ndim != 3 or x_true_np.ndim < 3:
        return
    T = min(x_np.shape[1], w_np.shape[1], x_true_np.shape[1])
    times = [t for t in time_indices if 0 <= t < T]
    if not times:
        return
    b = 0
    if x_np.shape[0] == 0:
        return

    idx0, idx1 = dims
    if x_np.shape[-1] <= max(idx0, idx1):
        return

    weights: List[np.ndarray] = []
    w_max = 0.0
    for t in times:
        w_t = w_np[b, t]
        w_sum = np.sum(w_t)
        if w_sum > 0:
            w_t = w_t / w_sum
        weights.append(w_t)
        if w_t.size > 0:
            w_max = max(w_max, float(np.max(w_t)))
    if w_max <= 0:
        w_max = 1.0

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=w_max)
    markers = ["o", "^", "s"]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(
        x_true_np[b, :, idx0],
        x_true_np[b, :, idx1],
        color="k",
        linewidth=1.2,
        label="true",
    )
    for i, t in enumerate(times):
        post = x_np[b, t][:, [idx0, idx1]]
        ax.scatter(
            post[:, 0],
            post[:, 1],
            s=8.0,
            c=weights[i],
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            edgecolors="none",
            marker=markers[i % len(markers)],
            label=f"t={t}",
        )
    ax.set_xlabel(f"x{idx0}")
    ax.set_ylabel(f"x{idx1}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="weight")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)




def _select_pre_resample_particles(out: Dict[str, Any]) -> Optional[np.ndarray]:
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    x_pre = diagnostics.get("x_pre")
    if x_pre is None:
        x_pre = diagnostics.get("x_pred")
    if x_pre is None:
        x_pre = out.get("x_particles") if isinstance(out, dict) else None
    return x_pre




# Common config helpers are imported from exp_utils.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2b (range/bearing) runner.")
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
    return parser.parse_args()


def _resolve_model_cov(model_cfg: Dict[str, Any]) -> np.ndarray:
    cov_eps_cfg = model_cfg.get("cov_eps")
    if cov_eps_cfg is not None:
        cov_eps_x = np.array(cov_eps_cfg, dtype=np.float32)
        if cov_eps_x.shape != (4, 4):
            raise ValueError("model.cov_eps must be a 4x4 matrix for range-bearing CV model")
        return cov_eps_x
    if "q_scale_v" not in model_cfg:
        raise ValueError("Set model.cov_eps (4x4) or model.q_scale_v for range-bearing CV model")
    q_scale_v = float(model_cfg.get("q_scale_v", 0.2))
    return np.diag([0.0, 0.0, q_scale_v**2, q_scale_v**2]).astype(np.float32)


def _resolve_filter_cov(
    model_cfg: Dict[str, Any],
    model_filter_cfg: Dict[str, Any],
    cov_eps_x: np.ndarray,
) -> np.ndarray:
    cov_eps_filter_cfg = first_non_null(
        model_filter_cfg.get("cov_eps"),
        model_filter_cfg.get("cov_eps_x"),
        model_cfg.get("cov_eps_filter"),
        model_cfg.get("cov_eps_x_filter"),
    )
    if cov_eps_filter_cfg is None:
        return cov_eps_x
    cov_eps_x_filter = np.array(cov_eps_filter_cfg, dtype=np.float32)
    if cov_eps_x_filter.shape != (4, 4):
        raise ValueError("filters.model.cov_eps must be a 4x4 matrix for range-bearing CV model")
    return cov_eps_x_filter


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

    out_root = Path(exp_cfg.get("output_root", "results/exp2b_range_bearing"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 80))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    save_particles_seed0 = bool(exp_cfg.get("save_particles_seed0", True))
    plot_controls = resolve_plot_controls(
        exp_cfg,
        [
            ("plot_state", "plot_seed0_only", None),
            ("plot_pf_ess", "plot_pf_ess_seed0_only", "plot_pf_ess_show"),
            ("plot_pf_degeneracy", "plot_pf_degeneracy_seed0_only", "plot_pf_degeneracy_show"),
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
    plot_pf_degeneracy = plot_controls["plot_pf_degeneracy"]
    plot_pf_degeneracy_seed0_only = plot_controls["plot_pf_degeneracy_seed0_only"]
    plot_pf_degeneracy_show = plot_controls["plot_pf_degeneracy_show"]
    plot_stability = plot_controls["plot_stability"]
    plot_stability_seed0_only = plot_controls["plot_stability_seed0_only"]
    plot_stability_show = plot_controls["plot_stability_show"]
    plot_time_gap = parse_positive_int_or_none(exp_cfg.get("plot_time_gap"))
    plot_pf_degeneracy_times = [0, 9, 29]
    plot_pf_degeneracy_times = [t for t in plot_pf_degeneracy_times if 0 <= t < T]
    plot_stability_percentiles = parse_percentile_band(
        exp_cfg.get("plot_stability_percentiles"),
        (25.0, 75.0),
    )

    summary_keys, exclude_prefixes = get_summary_keys_and_prefixes(exp_cfg, SUMMARY_KEYS)

    distances = [float(d) for d in as_list(model_cfg.get("distances", [0.5, 2.0, 10.0]))]
    sigma_thetas = [float(s) for s in as_list(model_cfg.get("sigma_thetas", [1.0, 5.0, 15.0]))]
    sigma_rs = [float(s) for s in as_list(model_cfg.get("sigma_rs", [0.05, 0.2, 1.0]))]
    sigma_thetas_filter = resolve_optional_float_list(
        first_non_null(
            model_filter_cfg.get("sigma_thetas"),
            model_filter_cfg.get("sigma_theta"),
            model_cfg.get("sigma_thetas_filter"),
            model_cfg.get("sigma_theta_filter"),
        )
    )
    sigma_rs_filter = resolve_optional_float_list(
        first_non_null(
            model_filter_cfg.get("sigma_rs"),
            model_filter_cfg.get("sigma_r"),
            model_cfg.get("sigma_rs_filter"),
            model_cfg.get("sigma_r_filter"),
        )
    )
    sigma_thetas_filter = expand_sweep_values(
        sigma_thetas_filter,
        sigma_thetas,
        "filters.model.sigma_thetas",
    )
    sigma_rs_filter = expand_sweep_values(
        sigma_rs_filter,
        sigma_rs,
        "filters.model.sigma_rs",
    )
    dt = float(model_cfg.get("dt", 1.0))
    jitter = float(model_cfg.get("jitter", 1e-12))
    jacobian_r_min = model_cfg.get("jacobian_r_min")
    if jacobian_r_min is not None:
        jacobian_r_min = float(jacobian_r_min)
    cov_eps_x = _resolve_model_cov(model_cfg)
    cov_eps_x_filter = _resolve_filter_cov(model_cfg, model_filter_cfg, cov_eps_x)
    jacobian_r_min_filter = model_filter_cfg.get("jacobian_r_min", jacobian_r_min)
    if jacobian_r_min_filter is not None:
        jacobian_r_min_filter = float(jacobian_r_min_filter)

    methods_cfg = filters_cfg.get("methods")
    base_methods = [
        str(m).lower()
        for m in as_list(
            methods_cfg
            if methods_cfg is not None
            else [
                "ekf",
                "ukf",
                "pf",
                "edh",
                "edh(pfpf)",
                "ledh",
                "ledh(pfpf)",
                "kflow_scalar",
                "kflow_diag",
            ]
        )
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

    pairs_list = particle_pairs(pf_particles, flow_particles, pair_particles)

    for dist in distances:
        x0 = build_initial_state(dist, init_cfg)
        m0 = build_initial_mean(x0, init_cfg)
        P0 = build_initial_cov(init_cfg)
        P0_filter = build_initial_cov_filter(init_cfg)
        init_dist = build_init_dist(m0, P0_filter)
        for sigma_theta_idx, sigma_theta in enumerate(sigma_thetas):
            for sigma_r_idx, sigma_r in enumerate(sigma_rs):
                sigma_theta_filter = sigma_thetas_filter[sigma_theta_idx]
                sigma_r_filter = sigma_rs_filter[sigma_r_idx]
                for N_pf, N_flow in pairs_list:
                    cfg_tag = tag_from_cfg(
                        {
                            "dist": dist,
                            "sigma_theta": sigma_theta,
                            "sigma_r": sigma_r,
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
                        particle_counts = {
                            int(filter_cfg.get(method, {}).get("num_particles"))
                            for method in methods
                            if method not in ("kf", "kalman", "ekf", "ukf")
                            and filter_cfg.get(method, {}).get("num_particles") is not None
                        }
                        init_particles_by_n = {
                            n: build_init_particles(m0, P0_filter, n, batch_size, seed)
                            for n in particle_counts
                        }
                        sim_ssm = build_ssm(
                            sigma_r=sigma_r,
                            sigma_theta=np.deg2rad(sigma_theta),
                            dt=dt,
                            cov_eps_x=cov_eps_x,
                            jitter=jitter,
                            seed=seed,
                            jacobian_r_min=jacobian_r_min,
                        )
                        filter_ssm = build_ssm(
                            sigma_r=sigma_r_filter,
                            sigma_theta=np.deg2rad(sigma_theta_filter),
                            dt=dt,
                            cov_eps_x=cov_eps_x_filter,
                            jitter=jitter,
                            seed=seed,
                            jacobian_r_min=jacobian_r_min_filter,
                        )
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,), x0=x0)
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
                                method_cfg["m0"] = m0
                                method_cfg["P0"] = P0_filter
                            else:
                                method_cfg["init_dist"] = init_dist
                                num_particles = method_cfg.get("num_particles")
                                if num_particles is not None:
                                    init_particles = init_particles_by_n.get(int(num_particles))
                                    if init_particles is not None:
                                        method_cfg["init_particles"] = init_particles
                            method_cfg["init_seed"] = seed
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
                            extra_metrics = {
                                "rmse_pos": float(rmse(x_true, out["mean"], dims=[0, 1]).numpy()),
                                "rmse_vel": float(rmse(x_true, out["mean"], dims=[2, 3]).numpy()),
                            }
                            metrics = record_metrics(
                                sim_ssm,
                                x_true,
                                y_obs,
                                out,
                                method_dir,
                                metrics_cfg={"rmse_obs": False},
                                extra_metrics=extra_metrics,
                                prefix=f"exp2b_range_bearing {cfg_tag} seed{seed} {method}",
                                print_full=False,
                            )
                            metrics_by_method[method] = metrics
                            metrics_across_seeds[method].append(metrics)

                            if is_particle_like_method(method):
                                if plot_pf_ess and (
                                    not plot_pf_ess_seed0_only or seed == seeds[0]
                                ):
                                    w_pre = select_pre_resample_weights(out)
                                    if w_pre is not None:
                                        ess_threshold = ess_threshold_for_method(
                                            method,
                                            pf_ess_threshold,
                                            flow_ess_threshold,
                                        )
                                        plot_path = method_dir / "pf_ess_over_time.png"
                                        plot_ess_over_time(
                                            plot_path,
                                            w_pre,
                                            ess_threshold=ess_threshold,
                                            show=plot_pf_ess_show,
                                        )
                                if plot_pf_degeneracy and (
                                    not plot_pf_degeneracy_seed0_only or seed == seeds[0]
                                ):
                                    x_pre = _select_pre_resample_particles(out)
                                    w_pre = select_pre_resample_weights(out)
                                    if x_pre is not None and w_pre is not None:
                                        plot_path = method_dir / "pf_particles_t0_9_29.png"
                                        _plot_particle_cloud(
                                            plot_path,
                                            x_particles=x_pre,
                                            w=w_pre,
                                            x_true=x_true,
                                            time_indices=plot_pf_degeneracy_times,
                                            show=plot_pf_degeneracy_show,
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
                            w_pre = select_pre_resample_weights(out)
                            if w_pre is None:
                                w_pre = out.get("w")
                            if w_pre is not None and not out.get("is_gaussian", False):
                                ess_t = ess_from_weights(np.asarray(w_pre))
                                if ess_t is not None:
                                    diag["ess_t"] = ess_t
                                parents = out.get("parents")
                                if parents is not None:
                                    impoverishment_t = _impoverishment_from_parents(parents)
                                    if impoverishment_t is not None:
                                        diag["impoverishment_t"] = impoverishment_t
                            if save_particles_seed0 and seed == seeds[0]:
                                diag["x_particles"] = out["x_particles"]
                                diag["w"] = out["w"]
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

                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} summary")
                        print_method_summary_table(
                            metrics_by_method,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2b_range_bearing {cfg_tag} seed{seed} compare")
                        print_metrics_compare(
                            metrics_by_method,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )

                        if plot_state and (not plot_seed0_only or seed == seeds[0]):
                            plot_path = per_seed_dir / "state_trajectory.png"
                            plot_title = f"exp2b_range_bearing {cfg_tag} seed{seed}"
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

                    if len(seeds) > 1:
                        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
                        print_separator(f"exp2b_range_bearing {cfg_tag} avg summary")
                        print_method_summary_table(
                            mean_metrics,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2b_range_bearing {cfg_tag} avg compare")
                        print_metrics_compare(
                            mean_metrics,
                            method_order=tuple(methods),
                            exclude_prefixes=exclude_prefixes,
                        )


if __name__ == "__main__":
    main()
