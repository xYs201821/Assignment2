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

def _ess_from_weights(w: np.ndarray) -> Optional[np.ndarray]:
    w_np = np.asarray(w, dtype=np.float64)
    if w_np.ndim == 2:
        w_np = w_np[np.newaxis, ...]
    if w_np.ndim != 3:
        return None
    w_sum = np.sum(w_np, axis=-1, keepdims=True)
    w_norm = np.divide(w_np, w_sum, out=np.zeros_like(w_np), where=w_sum > 0)
    ess_t = 1.0 / np.sum(np.square(w_norm), axis=-1)
    return ess_t


def _select_pre_resample_weights(out: Dict[str, Any]) -> Optional[np.ndarray]:
    diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    w_pre = diagnostics.get("w_pre")
    if w_pre is None:
        w_pre = out.get("w") if isinstance(out, dict) else None
    return w_pre


def _is_pf_method(method: str) -> bool:
    method = str(method).lower()
    return method in ("pf", "bootstrap") or method.startswith("pf")


def _is_stochastic_pf_method(method: str) -> bool:
    method = str(method).lower()
    return method.startswith("stochastic_pf") or method.startswith("stochastic-pf") or method == "spf"


def _is_pfpf_flow_method(method: str) -> bool:
    method = str(method).lower()
    return "pfpf" in method and (method.startswith("edh") or method.startswith("ledh"))


def _ess_threshold_for_method(
    method: str,
    pf_threshold: float,
    flow_threshold: float,
) -> Optional[float]:
    if _is_pf_method(method):
        return pf_threshold
    if _is_pfpf_flow_method(method):
        return flow_threshold
    if _is_stochastic_pf_method(method):
        return flow_threshold
    return None


def _plot_ess_over_time(
    path: Path,
    w: np.ndarray,
    ess_threshold: Optional[float] = None,
    band_percentiles: Optional[Tuple[float, float]] = (10.0, 90.0),
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    ess_t = _ess_from_weights(w)
    if ess_t is None:
        return
    T = ess_t.shape[1]
    t = np.arange(T)
    ess_mean = np.mean(ess_t, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(t, ess_mean, color="C0", linewidth=1.6, label="ESS mean")
    if ess_t.shape[0] > 1 and band_percentiles is not None:
        p_lo, p_hi = band_percentiles
        ess_lo = np.percentile(ess_t, p_lo, axis=0)
        ess_hi = np.percentile(ess_t, p_hi, axis=0)
        ax.fill_between(
            t,
            ess_lo,
            ess_hi,
            color="C0",
            alpha=0.2,
            label=f"ESS p{int(p_lo)}-p{int(p_hi)}",
        )
    if ess_threshold is not None:
        N = np.asarray(w).shape[-1]
        ax.axhline(
            ess_threshold * float(N),
            color="C3",
            linestyle="--",
            linewidth=1.0,
            label="ESS threshold",
        )
    ax.set_xlabel("time")
    ax.set_ylabel("ESS")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="best")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_stability_series(
    path: Path,
    values: np.ndarray,
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    arr = np.asarray(values)
    if arr.ndim == 1:
        mean = arr
        lo = hi = None
    else:
        flat = arr.reshape(-1, arr.shape[-1])
        mean = np.mean(flat, axis=0)
        if band_percentiles is None:
            lo = hi = None
        else:
            p_lo, p_hi = band_percentiles
            lo = np.percentile(flat, p_lo, axis=0)
            hi = np.percentile(flat, p_hi, axis=0)

    t = np.arange(mean.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(t, mean, color="C0", linewidth=1.6)
    if lo is not None and hi is not None:
        ax.fill_between(t, lo, hi, color="C0", alpha=0.25, linewidth=0)
    ax.set_xlabel("time")
    ax.grid(True, linestyle=":")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_stability_over_time(
    output_dir: Path,
    diagnostics: Dict[str, Any],
    band_percentiles: Optional[Tuple[float, float]] = (25.0, 75.0),
    show: bool = False,
) -> None:
    key_specs = [
        ("logdet_cov", "logdet_cov"),
        ("condH_log10_max", "condH_log10"),
        ("condJ_log10_max", "condJ_log10"),
        ("condK_log10_max", "condK_log10"),
        ("flow_norm_mean_max", "flow_norm_mean"),
    ]
    for key, label in key_specs:
        val = diagnostics.get(key)
        if val is None:
            continue
        title = f"stability_{label}"
        path = output_dir / f"{title}.png"
        _plot_stability_series(
            path,
            np.asarray(val),
            band_percentiles=band_percentiles,
            show=show,
            title=title,
        )


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
    spf_cfg = filters_cfg.get("stochastic_pf", {})
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
    plot_show = exp_cfg.get("plot_show")
    if plot_show is None:
        plot_state = bool(exp_cfg.get("plot_state", False))
        plot_seed0_only = bool(exp_cfg.get("plot_seed0_only", True))
        show_plots = bool(exp_cfg.get("show_plots", False))
        plot_interactive = bool(exp_cfg.get("plot_interactive", False))
        plot_pf_ess = bool(exp_cfg.get("plot_pf_ess", False))
        plot_pf_ess_seed0_only = bool(exp_cfg.get("plot_pf_ess_seed0_only", True))
        plot_pf_ess_show = bool(exp_cfg.get("plot_pf_ess_show", False))
        plot_stability = bool(exp_cfg.get("plot_stability", False))
        plot_stability_seed0_only = bool(exp_cfg.get("plot_stability_seed0_only", True))
        plot_stability_show = bool(exp_cfg.get("plot_stability_show", False))
    else:
        plot_show = bool(plot_show)
        plot_state = plot_show
        plot_pf_ess = plot_show
        plot_stability = plot_show
        plot_seed0_only = bool(exp_cfg.get("plot_seed0_only", True))
        plot_pf_ess_seed0_only = bool(exp_cfg.get("plot_pf_ess_seed0_only", True))
        plot_stability_seed0_only = bool(exp_cfg.get("plot_stability_seed0_only", True))
        show_plots = plot_show and bool(exp_cfg.get("show_plots", False))
        plot_interactive = plot_show and bool(exp_cfg.get("plot_interactive", False))
        plot_pf_ess_show = show_plots
        plot_stability_show = show_plots
    plot_time_gap = exp_cfg.get("plot_time_gap")
    if plot_time_gap is not None:
        plot_time_gap = int(plot_time_gap)
        if plot_time_gap <= 0:
            plot_time_gap = None
    plot_pf_ess_percentiles = exp_cfg.get("plot_pf_ess_percentiles")
    if plot_pf_ess_percentiles is None:
        plot_pf_ess_percentiles = (10.0, 90.0)
    else:
        vals = list(plot_pf_ess_percentiles) if isinstance(plot_pf_ess_percentiles, (list, tuple)) else []
        if len(vals) >= 2:
            plot_pf_ess_percentiles = (float(vals[0]), float(vals[1]))
        else:
            plot_pf_ess_percentiles = (10.0, 90.0)
    plot_stability_percentiles = exp_cfg.get("plot_stability_percentiles")
    if plot_stability_percentiles is None:
        plot_stability_percentiles = (25.0, 75.0)
    else:
        vals = list(plot_stability_percentiles) if isinstance(plot_stability_percentiles, (list, tuple)) else []
        if len(vals) >= 2:
            plot_stability_percentiles = (float(vals[0]), float(vals[1]))
        else:
            plot_stability_percentiles = (25.0, 75.0)
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

                            if plot_pf_ess and (not plot_pf_ess_seed0_only or seed == seeds[0]):
                                if (
                                    _is_pf_method(method)
                                    or _is_pfpf_flow_method(method)
                                    or _is_stochastic_pf_method(method)
                                ) and not out.get("is_gaussian", False):
                                    w_pre = _select_pre_resample_weights(out)
                                    if w_pre is not None:
                                        ess_threshold = _ess_threshold_for_method(
                                            method,
                                            pf_ess_threshold,
                                            flow_ess_threshold,
                                        )
                                        plot_path = method_dir / "pf_ess_over_time.png"
                                        _plot_ess_over_time(
                                            plot_path,
                                            w_pre,
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
                                    _plot_stability_over_time(
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
