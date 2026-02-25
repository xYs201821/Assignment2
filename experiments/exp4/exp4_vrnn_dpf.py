from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.common.exp_helper import (
    aggregate_metrics_by_method,
    ess_from_log_weights as _ess_from_log_weights,
    particle_mean as _particle_mean,
    print_method_summary_table,
    print_separator,
)
from experiments.common.exp_utils import (
    as_list,
    cfg_section,
    ensure_dir,
    load_config,
    save_json,
    save_npz,
)
from experiments.exp4.exp4_model import build_exp4_vrnn_ssm
from experiments.exp4.polyphonic_data import load_polyphonic_split
from src.filters import (
    DiffusionResamplingDPF,
    OTResamplingDPF,
    ParticleTransformerDPF,
    SoftResamplingDPF,
    StandardResamplingDPF,
)
from src.ssm import VRNNBinarySSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp4_config.yaml")
SUMMARY_KEYS = ("elbo", "nll", "bce", "rmse_state", "ess_mean", "train_sec")


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args(default_config: Path = DEFAULT_CONFIG_PATH) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: VRNN + DPF training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to YAML config (default: {default_config})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run, e.g. --methods pf soft ot diffusion transformer",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        default=None,
        help="Seed to run; repeat flag for multiple seeds.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override experiment.output_root.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name under output root.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override training.steps.",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=None,
        help="Override dpf.common.num_particles.",
    )
    return parser.parse_args()


def _resolve_methods(dpf_cfg: Dict[str, Any]) -> List[str]:
    methods: List[str] = []
    for raw in as_list(dpf_cfg.get("methods", ["pf"])):
        text = str(raw).strip().lower()
        if not text:
            continue
        if text in {"pt", "particle_transformer"} or "transform" in text:
            text = "transformer"
        if text not in methods:
            methods.append(text)
    if not methods:
        return ["pf"]
    return methods


def _method_cfg(dpf_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    common_cfg = dpf_cfg.get("common", {})
    if not isinstance(common_cfg, dict):
        common_cfg = {}
    specific_cfg = dpf_cfg.get(method, {})
    if not isinstance(specific_cfg, dict):
        specific_cfg = {}
    merged = dict(common_cfg)
    merged.update(specific_cfg)
    return merged


def _subset_sequences(
    y_all: np.ndarray,
    mask_all: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    max_sequences: int | None,
    shuffle: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    n_all = int(y_all.shape[0])
    if n_all <= 0:
        raise ValueError("Dataset split contains no sequences.")

    if max_sequences is not None and max_sequences > 0:
        n_target = min(n_all, int(max_sequences))
    else:
        n_target = n_all
    if batch_size > 0:
        n_target = min(n_target, int(batch_size))

    if bool(shuffle):
        rng = np.random.default_rng(int(seed))
        indices = rng.permutation(n_all)[:n_target]
    else:
        indices = np.arange(n_target, dtype=np.int32)

    y = tf.convert_to_tensor(y_all[indices], dtype=tf.float32)
    mask = tf.convert_to_tensor(mask_all[indices], dtype=tf.float32)
    return y, mask


def _build_data_batches(
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    *,
    seed: int,
) -> list[Dict[str, Any]]:
    data_cfg = exp_cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    source = str(data_cfg.get("source", "synthetic")).strip().lower()
    batch_size = int(exp_cfg.get("batch_size", 32))
    seq_len = int(exp_cfg.get("T", 150))

    if source == "synthetic":
        sim_seed_offset = int(exp_cfg.get("sim_seed_offset", 100_000))
        sim_ssm = build_exp4_vrnn_ssm(
            model_cfg=model_cfg,
            seed=seed + sim_seed_offset,
            trainable=False,
        )
        x_true, y_obs = sim_ssm.simulate(T=seq_len, shape=[batch_size])
        mask = tf.ones(tf.shape(y_obs)[:-1], dtype=tf.float32)
        return [
            {
                "dataset": "synthetic",
                "y_obs": tf.cast(y_obs, tf.float32),
                "mask": mask,
                "x_true": tf.cast(x_true, tf.float32),
                "meta": {"source": "synthetic"},
            }
        ]

    if source != "polyphonic":
        raise ValueError("experiment.data.source must be 'synthetic' or 'polyphonic'.")

    datasets = [str(s) for s in as_list(data_cfg.get("datasets", ["jsb_chorales", "musedata"]))]
    split = str(data_cfg.get("split", "train"))
    data_root = Path(str(data_cfg.get("root", "data/polyphonic")))
    download = bool(data_cfg.get("download", True))
    shuffle = bool(data_cfg.get("shuffle", True))
    max_sequences_raw = data_cfg.get("max_sequences")
    max_sequences = int(max_sequences_raw) if max_sequences_raw is not None else None
    obs_dim = int(model_cfg.get("obs_dim", 88))

    batches: list[Dict[str, Any]] = []
    for ds_name in datasets:
        y_all, mask_all, meta = load_polyphonic_split(
            dataset=ds_name,
            root=data_root,
            split=split,
            seq_len=seq_len,
            obs_dim=obs_dim,
            download=download,
        )
        y_obs, mask = _subset_sequences(
            y_all,
            mask_all,
            seed=seed,
            batch_size=batch_size,
            max_sequences=max_sequences,
            shuffle=shuffle,
        )
        batches.append(
            {
                "dataset": str(meta.get("dataset", ds_name)),
                "y_obs": y_obs,
                "mask": mask,
                "x_true": None,
                "meta": meta,
            }
        )
    return batches


def _build_dpf(method: str, ssm: VRNNBinarySSM, dpf_cfg: Dict[str, Any]):
    cfg = _method_cfg(dpf_cfg, method)
    num_particles = int(cfg.get("num_particles", 64))
    ess_threshold = float(cfg.get("ess_threshold", 0.5))
    resample = cfg.get("resample", "auto")
    stop_grad = bool(cfg.get("stop_grad_through_time", False))

    if method in ("pf", "standard"):
        dpf = StandardResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
        )
        dpf.stop_grad_through_time = stop_grad
        return dpf
    if method == "soft":
        lam = float(cfg.get("lam", 0.8))
        dpf = SoftResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            lam=lam,
            resample=resample,
        )
        dpf.stop_grad_through_time = stop_grad
        return dpf
    if method == "ot":
        return OTResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            ot_epsilon=float(cfg.get("ot_epsilon", 0.1)),
            ot_num_iters=int(cfg.get("ot_num_iters", 50)),
            ot_jitter=float(cfg.get("ot_jitter", 1e-6)),
            stop_grad_through_time=stop_grad,
            resample=resample,
        )
    if method == "diffusion":
        return DiffusionResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            diff_a=float(cfg.get("diff_a", -0.2)),
            diff_T=float(cfg.get("diff_T", 1.0)),
            diff_steps=int(cfg.get("diff_steps", 8)),
            diff_ode=bool(cfg.get("diff_ode", True)),
            diff_eps=float(cfg.get("diff_eps", 1e-6)),
            stop_grad_through_time=stop_grad,
            resample=resample,
        )
    if method == "transformer":
        d_model = int(cfg.get("pt_d_model", cfg.get("d_model", 128)))
        hidden = int(cfg.get("pt_hidden", cfg.get("hidden", 128)))
        num_heads = int(cfg.get("pt_num_heads", cfg.get("num_heads", 4)))
        num_encoder_layers = int(cfg.get("pt_num_encoder_layers", cfg.get("num_encoder_layers", 2)))
        num_decoder_layers = int(cfg.get("pt_num_decoder_layers", cfg.get("num_decoder_layers", 1)))
        dropout_rate = float(cfg.get("pt_dropout_rate", cfg.get("dropout_rate", 0.0)))
        stop_grad_t = bool(cfg.get("stop_grad_through_time", cfg.get("truncate_time_grad", True)))
        return ParticleTransformerDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            d_model=d_model,
            hidden=hidden,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            stop_grad_through_time=stop_grad_t,
            resample=resample,
        )
    raise ValueError(f"Unsupported method '{method}'.")


def _warmup_transformer_resampler(dpf: ParticleTransformerDPF) -> None:
    n = int(dpf.num_particles)
    dx = int(dpf.ssm.state_dim)
    log_uniform = -np.log(float(n))
    x0 = tf.zeros([1, n, dx], dtype=tf.float32)
    lw0 = tf.fill([1, n], tf.constant(log_uniform, dtype=tf.float32))
    _ = dpf.resampler_net(x0, lw0, training=False)


def _bce(y_true: tf.Tensor, y_prob: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.cast(y_prob, tf.float32)
    p = tf.clip_by_value(y_prob, 1e-6, 1.0 - 1e-6)
    loss_t = -(y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))
    if mask is None:
        return tf.reduce_mean(loss_t)
    mask = tf.cast(mask, tf.float32)
    masked = loss_t * mask[..., tf.newaxis]
    denom = tf.reduce_sum(mask) * tf.cast(tf.shape(y_true)[-1], tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(masked), denom)


def _evaluate_method(
    dpf,
    ssm: VRNNBinarySSM,
    y_obs: tf.Tensor,
    x_true: tf.Tensor | None,
    *,
    resample: str | int | bool,
    mask: tf.Tensor | None = None,
) -> tuple[Dict[str, float], Dict[str, tf.Tensor]]:
    x_seq, w_seq, diagnostics, _ = dpf.filter(y_obs, resample=resample)
    x_mean = _particle_mean(x_seq, w_seq)
    p_particles = ssm.h(x_seq)
    p_mean = _particle_mean(p_particles, w_seq)

    logz_t = tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32)
    if mask is None:
        mask_t = tf.ones_like(logz_t, dtype=tf.float32)
    else:
        mask_t = tf.cast(mask, tf.float32)
        tf.debugging.assert_equal(tf.shape(mask_t), tf.shape(logz_t), message="mask must have shape [B, T].")

    weighted_logz = tf.reduce_sum(logz_t * mask_t)
    valid_steps = tf.reduce_sum(mask_t)
    # Report ELBO on a per-time-step basis to keep scale comparable across sequence lengths.
    elbo = tf.math.divide_no_nan(weighted_logz, valid_steps)
    nll = -elbo
    if x_true is None:
        rmse_state = tf.constant(float("nan"), dtype=tf.float32)
    else:
        rmse_state = tf.sqrt(tf.reduce_mean(tf.square(x_mean - x_true)))
    bce = _bce(y_obs, p_mean, mask=mask_t)

    log_w_pre = tf.convert_to_tensor(diagnostics["log_w_pre"], dtype=tf.float32)
    ess = _ess_from_log_weights(log_w_pre)
    ess_mean = tf.reduce_mean(ess)

    metrics = {
        "elbo": float(elbo.numpy()),
        "nll": float(nll.numpy()),
        "bce": float(bce.numpy()),
        "rmse_state": float(rmse_state.numpy()),
        "ess_mean": float(ess_mean.numpy()),
    }
    traces = {
        "x_seq": x_seq,
        "w_seq": w_seq,
        "x_mean": x_mean,
        "p_mean": p_mean,
        "log_z_t": logz_t,
    }
    return metrics, traces


def _train_one_method(
    method: str,
    fit_ssm: VRNNBinarySSM,
    y_obs: tf.Tensor,
    x_true: tf.Tensor | None,
    *,
    train_cfg: Dict[str, Any],
    dpf_cfg: Dict[str, Any],
    mask: tf.Tensor | None = None,
) -> Dict[str, Any]:
    cfg = _method_cfg(dpf_cfg, method)
    dpf = _build_dpf(method, fit_ssm, dpf_cfg)
    if method == "transformer":
        freeze_resampler = bool(cfg.get("freeze_resampler", False))
        dpf.resampler_net.trainable = not freeze_resampler
        _warmup_transformer_resampler(dpf)
    resample = cfg.get("resample", "auto")
    steps = int(train_cfg.get("steps", 300))
    lr = float(train_cfg.get("lr", 1e-3))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    mc_samples = max(1, int(train_cfg.get("mc_samples", 1)))
    log_every = max(1, int(train_cfg.get("log_every", 20)))

    init_metrics, init_traces = _evaluate_method(
        dpf,
        fit_ssm,
        y_obs,
        x_true,
        resample=resample,
        mask=mask,
    )

    train_vars = list(fit_ssm.trainable_variables)
    if method == "transformer":
        train_vars.extend(
            v
            for v in dpf.resampler_net.trainable_variables
            if getattr(v, "trainable", False)
        )
    # Preserve insertion order while deduplicating by object identity.
    dedup_vars: List[tf.Variable] = []
    seen_var_ids: set[int] = set()
    for var in train_vars:
        var_id = id(var)
        if var_id in seen_var_ids:
            continue
        seen_var_ids.add(var_id)
        dedup_vars.append(var)
    train_vars = dedup_vars
    if not train_vars:
        raise RuntimeError("No trainable variables found for current method/model.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    grad_raw_hist: List[float] = []
    grad_hist: List[float] = []
    elbo_hist: List[float] = []
    bce_hist: List[float] = []

    train_t0 = time.perf_counter()
    for step in range(steps):
        with tf.GradientTape() as tape:
            elbo_terms = []
            bce_terms = []
            ess_terms = []
            nll_terms = []
            for _ in range(mc_samples):
                x_seq, w_seq, diagnostics, _ = dpf.filter(y_obs, resample=resample)
                logz_t = tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32)
                if mask is None:
                    mask_t = tf.ones_like(logz_t, dtype=tf.float32)
                else:
                    mask_t = tf.cast(mask, tf.float32)
                weighted_logz = tf.reduce_sum(logz_t * mask_t)
                valid_steps = tf.reduce_sum(mask_t)
                elbo_step = tf.math.divide_no_nan(weighted_logz, valid_steps)
                elbo_terms.append(elbo_step)
                nll_terms.append(-elbo_step)
                log_w_pre = tf.convert_to_tensor(diagnostics["log_w_pre"], dtype=tf.float32)
                ess_terms.append(tf.reduce_mean(_ess_from_log_weights(log_w_pre)))
                p_mean = _particle_mean(fit_ssm.h(x_seq), w_seq)
                bce_terms.append(_bce(y_obs, p_mean, mask=mask_t))
            elbo = tf.add_n(elbo_terms) / float(mc_samples)
            loss = -elbo
            nll_step = tf.add_n(nll_terms) / float(mc_samples)
            bce_step = tf.add_n(bce_terms) / float(mc_samples)
            ess_step = tf.add_n(ess_terms) / float(mc_samples)

        grads = tape.gradient(loss, train_vars)
        valid = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if valid:
            grad_tensors, var_tensors = zip(*valid)
            raw_grad_norm = tf.linalg.global_norm(grad_tensors)
            if grad_clip_norm > 0.0:
                clipped, _ = tf.clip_by_global_norm(grad_tensors, grad_clip_norm)
                grad_norm = tf.linalg.global_norm(clipped)
                optimizer.apply_gradients(zip(clipped, var_tensors))
            else:
                grad_norm = raw_grad_norm
                optimizer.apply_gradients(zip(grad_tensors, var_tensors))
        else:
            raw_grad_norm = tf.constant(0.0, dtype=tf.float32)
            grad_norm = tf.constant(0.0, dtype=tf.float32)

        elbo_hist.append(float(elbo.numpy()))
        bce_hist.append(float(bce_step.numpy()))
        grad_raw_hist.append(float(raw_grad_norm.numpy()))
        grad_hist.append(float(grad_norm.numpy()))

        if ((step + 1) % log_every == 0) or (step == 0) or (step + 1 == steps):
            _log(
                f"[{method}] step {step + 1}/{steps} "
                f"elbo={float(elbo.numpy()):.4f} "
                f"nll={float(nll_step.numpy()):.4f} "
                f"bce={float(bce_step.numpy()):.4f} "
                f"ess={float(ess_step.numpy()):.2f} "
                f"g={float(grad_norm.numpy()):.4f} "
                f"g_raw={float(raw_grad_norm.numpy()):.4f}"
            )

    runtime_train_sec = float(time.perf_counter() - train_t0)
    final_metrics, final_traces = _evaluate_method(
        dpf,
        fit_ssm,
        y_obs,
        x_true,
        resample=resample,
        mask=mask,
    )
    final_metrics["train_sec"] = runtime_train_sec

    return {
        "method": method,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "runtime_train_sec": runtime_train_sec,
        "elbo_hist": np.asarray(elbo_hist, dtype=np.float32),
        "bce_hist": np.asarray(bce_hist, dtype=np.float32),
        "grad_raw_hist": np.asarray(grad_raw_hist, dtype=np.float32),
        "grad_hist": np.asarray(grad_hist, dtype=np.float32),
        "init_traces": init_traces,
        "final_traces": final_traces,
    }


def run_exp4_vrnn_dpf(
    cfg: Dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    exp_name: str | None = None,
) -> None:
    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    train_cfg = cfg_section(cfg, "training")
    dpf_cfg = cfg_section(cfg, "dpf")

    out_root_cfg = exp_cfg.get("output_root", f"results/{config_path.stem}")
    base_out_root = Path(out_root_cfg)
    ensure_dir(base_out_root)
    experiment_name = str(exp_name or exp_cfg.get("experiment_name", "exp4")).strip() or "exp4"
    out_root = base_out_root / experiment_name
    ensure_dir(out_root)

    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    methods = _resolve_methods(dpf_cfg)
    T = int(exp_cfg.get("T", 150))
    batch_size = int(exp_cfg.get("batch_size", 32))
    save_traces = bool(exp_cfg.get("save_traces", True))
    fit_seed_offset = int(exp_cfg.get("fit_seed_offset", 200_000))

    _log(f"[exp4] output={out_root}")
    _log(f"[exp4] seeds={seeds} methods={methods} T={T} batch_size={batch_size}")

    per_seed_metrics: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    metrics_across_seeds: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for seed in seeds:
        data_batches = _build_data_batches(exp_cfg, model_cfg, seed=seed)
        per_seed_metrics[seed] = {}

        for batch in data_batches:
            dataset_name = str(batch["dataset"])
            y_obs = tf.cast(batch["y_obs"], tf.float32)
            x_true = batch["x_true"]
            if x_true is not None:
                x_true = tf.cast(x_true, tf.float32)
            mask = tf.cast(batch["mask"], tf.float32)

            _log(
                f"[seed={seed}][{dataset_name}] "
                f"data_shape={tuple(y_obs.shape)}"
            )
            meta = batch.get("meta", {})
            if meta:
                file_path = meta.get("file")
                if file_path:
                    _log(f"[seed={seed}][{dataset_name}] file={file_path}")

            per_seed_metrics[seed][dataset_name] = {}
            if dataset_name not in metrics_across_seeds:
                metrics_across_seeds[dataset_name] = {m: [] for m in methods}

            for method in methods:
                _log(f"[seed={seed}][{dataset_name}][{method}] training start")
                fit_ssm = build_exp4_vrnn_ssm(
                    model_cfg=model_cfg,
                    seed=seed + fit_seed_offset,
                    trainable=True,
                )
                result = _train_one_method(
                    method=method,
                    fit_ssm=fit_ssm,
                    y_obs=y_obs,
                    x_true=x_true,
                    train_cfg=train_cfg,
                    dpf_cfg=dpf_cfg,
                    mask=mask,
                )
                final_metrics = result["final_metrics"]
                per_seed_metrics[seed][dataset_name][method] = final_metrics
                metrics_across_seeds[dataset_name][method].append(final_metrics)
                _log(
                    f"[seed={seed}][{dataset_name}][{method}] final "
                    f"elbo={final_metrics['elbo']:.4f} "
                    f"nll={final_metrics['nll']:.4f} "
                    f"bce={final_metrics['bce']:.4f} "
                    f"rmse_state={final_metrics['rmse_state']:.4f} "
                    f"ess={final_metrics['ess_mean']:.2f} "
                    f"train_sec={final_metrics['train_sec']:.2f}"
                )

                if save_traces:
                    seed_dir = out_root / f"seed{seed}" / dataset_name
                    ensure_dir(seed_dir)
                    arrays = {
                        "y_obs": y_obs,
                        "mask": mask,
                        "x_init": result["init_traces"]["x_seq"],
                        "w_init": result["init_traces"]["w_seq"],
                        "p_init": result["init_traces"]["p_mean"],
                        "x_final": result["final_traces"]["x_seq"],
                        "w_final": result["final_traces"]["w_seq"],
                        "p_final": result["final_traces"]["p_mean"],
                        "elbo_hist": result["elbo_hist"],
                        "bce_hist": result["bce_hist"],
                        "grad_raw_hist": result["grad_raw_hist"],
                        "grad_hist": result["grad_hist"],
                    }
                    if x_true is not None:
                        arrays["x_true"] = x_true
                    save_npz(seed_dir / f"{method}_trace.npz", **arrays)

            print_separator(f"exp4 {dataset_name} seed{seed} summary")
            print_method_summary_table(
                per_seed_metrics[seed][dataset_name],
                method_order=tuple(methods),
                keys=SUMMARY_KEYS,
            )

    mean_metrics_by_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name in sorted(metrics_across_seeds):
        mean_metrics = aggregate_metrics_by_method(metrics_across_seeds[dataset_name])
        mean_metrics_by_dataset[dataset_name] = mean_metrics
        print_separator(f"exp4 {dataset_name} avg summary")
        print_method_summary_table(
            mean_metrics,
            method_order=tuple(methods),
            keys=SUMMARY_KEYS,
        )

    summary = {
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "seeds": seeds,
        "methods": methods,
        "per_seed_final": per_seed_metrics,
        "mean_final": mean_metrics_by_dataset,
    }
    summary_path = out_root / "summary.json"
    save_json(summary_path, summary)
    _log(f"[exp4] summary saved: {summary_path}")


def main() -> None:
    args = _parse_args(DEFAULT_CONFIG_PATH)
    cfg = load_config(args.config, [])
    exp_cfg = cfg.setdefault("experiment", {})
    if args.methods:
        cfg.setdefault("dpf", {})["methods"] = args.methods
    if args.seeds:
        exp_cfg["seeds"] = args.seeds
    if args.output_root is not None:
        exp_cfg["output_root"] = args.output_root
    if args.steps is not None:
        cfg.setdefault("training", {})["steps"] = int(args.steps)
    if args.num_particles is not None:
        dpf_cfg = cfg.setdefault("dpf", {})
        common_cfg = dpf_cfg.setdefault("common", {})
        if isinstance(common_cfg, dict):
            common_cfg["num_particles"] = int(args.num_particles)
        else:
            dpf_cfg["num_particles"] = int(args.num_particles)

    run_exp4_vrnn_dpf(
        cfg=cfg,
        config_path=args.config,
        exp_name=args.name if args.name is not None else None,
    )


if __name__ == "__main__":
    main()
