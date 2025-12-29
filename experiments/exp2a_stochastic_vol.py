from __future__ import annotations

import argparse
import sys
from itertools import cycle, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import yaml

from experiments.exp_helper import (
    aggregate_metrics_by_method,
    print_method_summary_table,
    print_metrics_compare,
    print_separator,
    record_metrics,
)
from experiments.exp_utils import build_init_dist, ensure_dir, save_npz, set_seed, tag_from_cfg
from experiments.filter_cfg import build_filter_cfg
from experiments.runner import run_filter
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
    from matplotlib.widgets import CheckButtons

    x_true_np = np.asarray(x_true)[0, :, 0]
    t_axis = np.arange(len(x_true_np))
    markevery = max(1, int(len(x_true_np) / 12))
    fig, ax = plt.subplots(figsize=(8, 3))
    lines = []
    labels = []
    annotations: Dict[str, List[Any]] = {}

    def _annotate_series(label: str, xs: np.ndarray, ys: np.ndarray, color: Any) -> None:
        if time_gap is None or time_gap <= 0:
            return
        texts: List[Any] = []
        for t in range(0, len(xs), time_gap):
            texts.append(
                ax.annotate(
                    str(t),
                    (xs[t], ys[t]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                    color=color,
                    alpha=0.8,
                )
            )
        annotations[label] = texts
    line_true, = ax.plot(x_true_np, color="k", label="true", linestyle="-")
    lines.append(line_true)
    labels.append("true")
    _annotate_series("true", t_axis, x_true_np, line_true.get_color())
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
        lines.append(line)
        labels.append(method)
        mean_np = np.asarray(mean)[0, :, 0]
        _annotate_series(method, t_axis, mean_np, line.get_color())
    if title:
        ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x_t")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    if interactive and show:
        fig.tight_layout(rect=(0.0, 0.0, 0.75, 1.0))
        selector_ax = fig.add_axes([0.78, 0.2, 0.2, 0.6])
        selector_ax.set_title("Trajectories", fontsize=9)
        visibility = [line.get_visible() for line in lines]
        check = CheckButtons(selector_ax, labels, visibility)

        def _toggle(label):
            idx = labels.index(label)
            line = lines[idx]
            new_vis = not line.get_visible()
            line.set_visible(new_vis)
            for text in annotations.get(label, []):
                text.set_visible(new_vis)
            fig.canvas.draw_idle()

        check.on_clicked(_toggle)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)




def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _particle_pairs(
    pf_particles: List[int],
    flow_particles: List[int],
    pair_particles: bool,
) -> List[Tuple[int, int]]:
    if not pf_particles and not flow_particles:
        raise ValueError("num_particles must be set for pf/flow.")
    if not pf_particles:
        pf_particles = list(flow_particles)
    if not flow_particles:
        flow_particles = list(pf_particles)
    if pair_particles:
        if len(pf_particles) == 1 and len(flow_particles) > 1:
            pf_particles = pf_particles * len(flow_particles)
        if len(flow_particles) == 1 and len(pf_particles) > 1:
            flow_particles = flow_particles * len(pf_particles)
        if len(pf_particles) != len(flow_particles):
            raise ValueError("pair_particles requires pf/flow lists of equal length.")
        return list(zip(pf_particles, flow_particles))
    return list(product(pf_particles, flow_particles))


def _deep_set(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("override key cannot be empty")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _deep_set(cfg, key.strip(), value)
    return cfg


def _load_config(path: Path, overrides: List[str]) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return _apply_overrides(cfg, overrides)


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config, args.overrides)

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    filters_cfg = cfg.get("filters", {})
    pf_cfg = filters_cfg.get("pf", {})
    flow_cfg = filters_cfg.get("flow", {})
    kflow_cfg = filters_cfg.get("kflow", {})
    ukf_cfg = filters_cfg.get("ukf", {})
    init_cfg = cfg.get("init", {})

    out_root = Path(exp_cfg.get("output_root", "results/exp2a_stochastic_vol"))
    ensure_dir(out_root)

    T = int(exp_cfg.get("T", 80))
    batch_size = int(exp_cfg.get("batch_size", 1))
    seeds = [int(s) for s in _as_list(exp_cfg.get("seeds", [0]))]
    pair_particles = bool(exp_cfg.get("pair_particles", True))
    calc_ekf_jacobian = bool(exp_cfg.get("calc_ekf_jacobian_error", True))
    plot_state = bool(exp_cfg.get("plot_state", False))
    plot_seed0_only = bool(exp_cfg.get("plot_seed0_only", True))
    show_plots = bool(exp_cfg.get("show_plots", False))
    plot_interactive = bool(exp_cfg.get("plot_interactive", False))
    plot_time_gap = exp_cfg.get("plot_time_gap")
    if plot_time_gap is not None:
        plot_time_gap = int(plot_time_gap)
        if plot_time_gap <= 0:
            plot_time_gap = None
    metrics_cfg_override = cfg.get("metrics", {})

    alpha = float(model_cfg.get("alpha", 0.99))
    mu = float(model_cfg.get("mu", 0.01))
    noise_scale_func = bool(model_cfg.get("noise_scale_func", False))
    obs_eps = float(model_cfg.get("obs_eps", 1e-16))
    betas = [float(b) for b in _as_list(model_cfg.get("betas", [2.0]))]
    sigmas = [float(s) for s in _as_list(model_cfg.get("sigmas", [0.6]))]
    obs_modes = [str(m) for m in _as_list(model_cfg.get("obs_modes", ["y", "logy2"]))]

    base_methods = [
        str(m).lower() for m in _as_list(filters_cfg.get("methods", ["ekf", "ukf", "pf"]))
    ]
    pf_particles = [int(n) for n in _as_list(pf_cfg.get("num_particles", [200]))]
    flow_particles = [int(n) for n in _as_list(flow_cfg.get("num_particles", [200]))]

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

    particle_pairs = _particle_pairs(pf_particles, flow_particles, pair_particles)

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
        if metrics_cfg.get("rmse_y"):
            summary_keys = (
                "rmse_state",
                "rmse_vol",
                "rmse_y",
                "nll",
                "ess_mean",
                "runtime.total_s",
                "memory.peak_rss_mb",
            )
        else:
            summary_keys = (
                "rmse_state",
                "rmse_vol",
                "nll",
                "ess_mean",
                "runtime.total_s",
                "memory.peak_rss_mb",
            )
        for beta in betas:
            for sigma in sigmas:
                for N_pf, N_flow in particle_pairs:
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
                        kflow_cfg=kflow_cfg,
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
                        x_true, y_obs = sim_ssm.simulate(T, shape=(batch_size,))

                        per_seed_dir = out_root / cfg_tag / f"seed{seed}"
                        ensure_dir(per_seed_dir)
                        save_npz(per_seed_dir / "data.npz", x_true=x_true, y_obs=y_obs)

                        outputs: Dict[str, Dict[str, Any]] = {}
                        for method in methods:
                            method_ssm = build_ssm(
                                alpha=alpha,
                                sigma=sigma,
                                beta=beta,
                                obs_mode=obs_mode,
                                mu=mu,
                                noise_scale_func=noise_scale_func,
                                obs_eps=obs_eps,
                                seed=seed,
                            )
                            method_cfg = dict(filter_cfg.get(method, {}))
                            if method in ("kf", "kalman", "ekf", "ukf"):
                                if m0_arr is not None:
                                    method_cfg["m0"] = m0_arr
                                if P0_arr is not None:
                                    method_cfg["P0"] = P0_arr
                            elif init_dist is not None:
                                method_cfg["init_dist"] = init_dist
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

                        print_separator(f"exp2a_stochastic_vol {cfg_tag} seed{seed} summary")
                        print_method_summary_table(
                            metrics_by_method,
                            method_order=tuple(methods),
                            keys=summary_keys,
                        )
                        print_separator(f"exp2a_stochastic_vol {cfg_tag} seed{seed} compare")
                        print_metrics_compare(metrics_by_method, method_order=tuple(methods))

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
                        print_metrics_compare(mean_metrics, method_order=tuple(methods))


if __name__ == "__main__":
    main()
