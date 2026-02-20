from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import tqdm
from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

from experiments.common.exp_utils import (
    as_list,
    cfg_section,
    ensure_dir,
    load_config,
    save_json,
    save_npz,
    set_seed,
)
from experiments.common.exp_helper import (
    aggregate_metrics_by_method,
    ess_from_log_weights as _ess_from_log_weights,
    ess_stats_from_tensor as _ess_stats_from_tensor,
    grad_metrics_from_history as _grad_metrics_from_history,
    particle_mean as _particle_mean,
    print_method_summary_table,
    print_separator,
    resampled_from_parent_index as _resampled_from_parent_index,
    result_stage_metrics as _result_stage_metrics,
    safe_nanmean as _safe_nanmean,
)
from experiments.exp3.exp3_model import LinearGaussianProposalPhi, build_exp3_linear_ssm_pair
from src.filters import (
    DPFBase,
    DiffusionResamplingDPF,
    OTResamplingDPF,
    ParticleTransformerDPF,
    SoftResamplingDPF,
    StandardResamplingDPF,
)
from src.filters.kalman import KalmanFilter
from src.ssm import LinearGaussianSSM

tfd = tfp.distributions

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp3_config.yaml")
SUMMARY_KEYS = (
    "loss",
    "nll",
    "rmse",
    "rmse_phi",
    "ess_mean",
    "grad_snr",
    "grad_var",
    "train_sec",
)


def _log(message: str) -> None:
    tqdm.write(message)


def _parse_args(default_config: Path = DEFAULT_CONFIG_PATH) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGSSM DPF backprop experiment runner.")
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
        help=(
            "Methods to run, e.g. "
            "--methods [kalman, baseline, pf, soft, ot, diffusion, transformer]"
        ),
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
        help=(
            "Experiment name. Results are saved under "
            "<output-root>/<name>/. If omitted, uses experiment.experiment_name "
            "or 'exp3'."
        ),
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
        help="Override dpf.num_particles.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class _SeedContext:
    """Per-seed immutable context shared by method runners."""
    rng: np.random.Generator
    fit_ssm: LinearGaussianSSM
    x_true: Any
    y_obs: Any
    state_dim: int
    obs_dim: int
    rng_base: int


def _build_seed_context(
    seed: int,
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> _SeedContext:
    """Build one seed's simulation data + fit model once and reuse downstream."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    T = int(exp_cfg.get("T", 50))
    batch_size = int(exp_cfg.get("batch_size", 1))
    state_dim = int(model_cfg.get("state_dim", 8))
    obs_dim = int(model_cfg.get("obs_dim", max(1, state_dim // 2)))

    sim_ssm, fit_ssm = build_exp3_linear_ssm_pair(
        model_cfg=model_cfg,
        seed=int(seed),
        fit_trainable=False,
    )
    x_true, y_obs = sim_ssm.simulate(T=T, shape=[batch_size])

    return _SeedContext(
        rng=rng,
        fit_ssm=fit_ssm,
        x_true=x_true,
        y_obs=y_obs,
        state_dim=state_dim,
        obs_dim=obs_dim,
        rng_base=int(seed) * 1_000_000,
    )


def _evaluate_filter(
    dpf: Any,
    y_obs: tf.Tensor,
    x_true: tf.Tensor,
    resample: str | int | bool,
) -> Dict[str, Any]:
    resample_mode = dpf._normalize_reweight(resample)
    x_seq, w_seq, diagnostics, _ = dpf.filter(y_obs, resample=resample_mode)
    mean_seq = _particle_mean(x_seq, w_seq)
    logz_t = tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32)
    logz_total = tf.reduce_sum(logz_t, axis=1)
    loss = -tf.reduce_mean(logz_total)
    nll = -tf.reduce_mean(logz_t)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(mean_seq - x_true)))
    log_w_pre = tf.convert_to_tensor(diagnostics["log_w_pre"], dtype=tf.float32)
    ess_stats = _ess_stats_from_tensor(_ess_from_log_weights(log_w_pre))
    parent_index = tf.convert_to_tensor(diagnostics["parent_index"], dtype=tf.int32)
    resampled = _resampled_from_parent_index(parent_index)
    resample_rate = float(tf.reduce_mean(tf.cast(resampled, tf.float32)).numpy())
    return {
        "loss": float(loss.numpy()),
        "nll": float(nll.numpy()),
        "rmse": float(rmse.numpy()),
        "ess_mean": float(ess_stats["ess_mean"]),
        "resample_rate": resample_rate,
        "ess_over_time": np.asarray(ess_stats["ess_over_time"], dtype=np.float32),
        "logZ_t": np.asarray(logz_t, dtype=np.float32),
        "resampled_t": np.asarray(resampled, dtype=np.bool_),
        "mean": mean_seq,
        "logZ_total": logz_total,
        "x_particles": tf.convert_to_tensor(diagnostics["x"], dtype=tf.float32),
        "weights": tf.exp(tf.convert_to_tensor(diagnostics["log_w"], dtype=tf.float32)),
        "diagnostics": diagnostics,
    }


def _evaluate_kalman(
    ssm: LinearGaussianSSM,
    y_obs: tf.Tensor,
    x_true: tf.Tensor,
) -> Dict[str, Any]:
    kf = KalmanFilter(ssm)
    out = kf.filter(y_obs, m0=ssm.m0, P0=ssm.P0)
    mean_seq = tf.convert_to_tensor(out["m_filt"], dtype=tf.float32)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(mean_seq - x_true)))

    y_obs_t = tf.convert_to_tensor(y_obs, dtype=tf.float32)
    if y_obs_t.shape.rank == 2:
        y_obs_t = y_obs_t[tf.newaxis, ...]
    m_pred = tf.convert_to_tensor(out["m_pred"], dtype=tf.float32)
    P_pred = tf.convert_to_tensor(out["P_pred"], dtype=tf.float32)
    C = tf.convert_to_tensor(ssm.C, dtype=tf.float32)
    R = tf.convert_to_tensor(ssm.cov_eps_y, dtype=tf.float32)

    C_bt = C[tf.newaxis, tf.newaxis, :, :]  # [1,1,dy,dx]
    Ct_bt = tf.transpose(C_bt, perm=[0, 1, 3, 2])  # [1,1,dx,dy]
    y_pred = tf.einsum("ij,btj->bti", C, m_pred)  # [B,T,dy]
    S = tf.matmul(C_bt, tf.matmul(P_pred, Ct_bt)) + R[tf.newaxis, tf.newaxis, :, :]
    S = 0.5 * (S + tf.transpose(S, perm=[0, 1, 3, 2]))
    L = tf.linalg.cholesky(S)
    logz_t = tfd.MultivariateNormalTriL(loc=y_pred, scale_tril=L).log_prob(y_obs_t)
    logz_total = tf.reduce_sum(logz_t, axis=1)  # [B]
    loss = -tf.reduce_mean(logz_total)
    nll = -tf.reduce_mean(logz_t)

    diagnostics = {
        "m_pred": m_pred,
        "P_pred": P_pred,
        "logZ_t": logz_t,
        "logZ_total": logz_total,
    }
    return {
        "loss": float(loss.numpy()),
        "nll": float(nll.numpy()),
        "rmse": float(rmse.numpy()),
        "ess_mean": float("nan"),
        "resample_rate": float("nan"),
        "ess_over_time": np.asarray([], dtype=np.float32),
        "logZ_t": np.asarray(logz_t),
        "resampled_t": np.asarray([], dtype=np.bool_),
        "mean": mean_seq,
        "logZ_total": logz_total,
        "diagnostics": diagnostics,
    }


def _select_train_var_groups(
    proposal: LinearGaussianProposalPhi,
    method: str,
    dpf: DPFBase | None = None,
    method_cfg: Dict[str, Any] | None = None,
) -> tuple[List[tf.Variable], List[tf.Variable], List[tf.Variable]]:
    proposal_vars = [v for v in proposal.trainable_variables if getattr(v, "trainable", False)]
    resampler_vars: List[tf.Variable] = []
    if method == "transformer" and dpf is not None:
        cfg = {} if method_cfg is None else method_cfg
        freeze_resampler = bool(cfg.get("freeze_resampler", True))
        if not freeze_resampler:
            resampler_vars.extend(
                v
                for v in dpf.resampler_net.trainable_variables
                if getattr(v, "trainable", False)
            )
    train_vars = proposal_vars + resampler_vars
    return proposal_vars, resampler_vars, train_vars


def _warmup_transformer_resampler(dpf: ParticleTransformerDPF) -> None:
    n = int(dpf.num_particles)
    dx = int(dpf.ssm.state_dim)
    log_uniform = -np.log(float(n))
    x0 = tf.zeros([1, n, dx], dtype=tf.float32)
    lw0 = tf.fill([1, n], tf.constant(log_uniform, dtype=tf.float32))
    _ = dpf.resampler_net(x0, lw0, training=False)


def _resolve_seeded_path(path_value: Any, seed: int) -> Path:
    raw = str(path_value).strip()
    if not raw:
        raise ValueError("Empty pretrained_weights path.")
    resolved = raw.format(seed=int(seed))
    return Path(resolved).expanduser()


def _is_skip_pretrain_token(value: Any) -> bool:
    """Return True when pretrained loading/pretraining should be skipped."""
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, np.integer)):
        return int(value) == -1
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"-1", "false"}
    return False


def _trace_filename(experiment_name: str, seed: int, method: str) -> str:
    exp_tag = str(experiment_name).strip() or "exp3"
    return f"{exp_tag}_seed{int(seed)}_{method}_trace.npz"


def _resolve_pretrain_output_root(
    method_cfg: Dict[str, Any],
    experiment_name: str,
    experiment_dir: str | Path | None,
    seed: int,
) -> str:
    exp_name = str(experiment_name).strip() or "exp3"
    if experiment_dir is not None:
        default_root = str(Path(experiment_dir) / "transformer_pretrain")
    else:
        default_root = str(Path("results") / f"{exp_name}_transformer_pretrain")
    value = method_cfg.get("pretrain_output_root")
    if value is None:
        return default_root
    raw = str(value).strip()
    if not raw:
        return default_root
    exp_dir_str = str(Path(experiment_dir)) if experiment_dir is not None else default_root
    try:
        return raw.format(
            seed=int(seed),
            experiment_name=exp_name,
            run_name=exp_name,
            experiment_dir=exp_dir_str,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder in transformer.pretrain_output_root: {raw!r}; "
            "supported placeholders: {experiment_name}, {run_name}, {seed}, {experiment_dir}."
        ) from exc


def _auto_pretrain_transformer_weights(
    dpf: ParticleTransformerDPF,
    method_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    seed: int,
    experiment_name: str | None = None,
    experiment_dir: str | Path | None = None,
) -> Path:
    from experiments.exp3.exp3_transformer_pretrain import (
        pretrain_transformer_resampler_for_seed,
    )

    exp_name = str(experiment_name).strip() or "exp3"
    pretrain_output_root = _resolve_pretrain_output_root(
        method_cfg=method_cfg,
        experiment_name=exp_name,
        experiment_dir=experiment_dir,
        seed=seed,
    )
    ess_default = float(tf.convert_to_tensor(dpf.ess_threshold, dtype=tf.float32).numpy())
    result = pretrain_transformer_resampler_for_seed(
        seed=int(seed),
        model_cfg=model_cfg,
        proposal_cfg=proposal_cfg,
        num_particles=int(method_cfg.get("pretrain_num_particles", dpf.num_particles)),
        ess_threshold=float(method_cfg.get("pretrain_ess_threshold", ess_default)),
        d_model=int(method_cfg.get("pretrain_d_model", dpf.d_model)),
        hidden=int(method_cfg.get("pretrain_hidden", dpf.hidden)),
        num_heads=int(method_cfg.get("pretrain_num_heads", dpf.num_heads)),
        num_encoder_layers=int(
            method_cfg.get("pretrain_num_encoder_layers", dpf.num_encoder_layers)
        ),
        num_decoder_layers=int(
            method_cfg.get("pretrain_num_decoder_layers", dpf.num_decoder_layers)
        ),
        dropout_rate=float(method_cfg.get("pretrain_dropout_rate", dpf.dropout_rate)),
        steps=int(method_cfg.get("pretrain_steps", 300)),
        batch_size=int(method_cfg.get("pretrain_batch_size", 256)),
        lr=float(method_cfg.get("pretrain_lr", 1e-3)),
        log_every=int(method_cfg.get("pretrain_log_every", 50)),
        loss_name=str(method_cfg.get("pretrain_loss", "energy")),
        gmm_sigma=float(method_cfg.get("pretrain_gmm_sigma", 0.25)),
        gmm_symmetric=bool(method_cfg.get("pretrain_gmm_symmetric", True)),
        loss_mix_alpha=float(method_cfg.get("pretrain_loss_mix_alpha", 0.5)),
        sim_T=int(method_cfg.get("pretrain_sim_T", exp_cfg.get("T", 80))),
        sim_batch_size=int(
            method_cfg.get("pretrain_sim_batch_size", exp_cfg.get("batch_size", 100))
        ),
        output_root=pretrain_output_root,
    )
    return Path(result["weights_path"]).expanduser()


def _maybe_init_transformer_from_pretrain(
    dpf: ParticleTransformerDPF,
    method_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    seed: int,
    experiment_name: str | None = None,
    experiment_dir: str | Path | None = None,
) -> str | None:
    path_value = method_cfg.get(
        "pretrained_weights",
        method_cfg.get("resampler_pretrained_weights"),
    )
    if _is_skip_pretrain_token(path_value):
        _log(
            f"[seed={seed}][transformer] pretrained_weights={path_value!r}, "
            "skip pretrain and skip loading."
        )
        return None
    if path_value is None:
        _log(f"[seed={seed}][transformer] pretrained_weights is None, running auto pretrain...")
        weights_path = _auto_pretrain_transformer_weights(
            dpf=dpf,
            method_cfg=method_cfg,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            proposal_cfg=proposal_cfg,
            seed=seed,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
        )
    else:
        weights_path = _resolve_seeded_path(path_value, seed)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Transformer pretrained weights not found: {weights_path}"
            )

    _warmup_transformer_resampler(dpf)
    if weights_path.suffix.lower() == ".npz":
        with np.load(weights_path) as arrays:
            names = sorted(arrays.files)
            dpf.resampler_net.set_weights([np.asarray(arrays[k]) for k in names])
    else:
        dpf.resampler_net.load_weights(str(weights_path))
    return str(weights_path)


def _reset_filter_rng(ssm: LinearGaussianSSM, seed_value: int) -> None:
    """Reset RNGs before each filter call for common-random-number comparison."""
    seed_i = int(seed_value)
    if seed_i < 0:
        seed_i = -seed_i
    if getattr(ssm, "rng", None) is None:
        ssm.rng = tf.random.Generator.from_seed(seed_i)
    else:
        ssm.rng.reset_from_seed(seed_i)
    tf.random.set_seed(seed_i)


def reset_filter_rng(ssm: LinearGaussianSSM, seed_value: int) -> None:
    """Public helper for experiments sharing exp3 LGSSM RNG behavior."""
    _reset_filter_rng(ssm, seed_value)


def _canonical_dpf_method(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError("dpf method must not be empty.")
    if "kalman" in text or text == "kf":
        return "kalman"
    if "baseline" in text:
        return "baseline"
    if "transform" in text or text == "pt":
        return "transformer"
    if "diff" in text:
        return "diffusion"
    if "soft" in text:
        return "soft"
    if "bootstrap" in text or "pf" in text:
        return "pf"
    if "ot" in text:
        return "ot"
    raise ValueError(
        f"Unknown dpf method '{value}'. Expected one of "
        "'kalman', 'baseline', 'pf', 'soft', 'ot', 'diffusion' or 'transformer'."
    )


def _resolve_dpf_methods(
    dpf_cfg: Dict[str, Any],
) -> List[str]:
    methods_raw = dpf_cfg.get("methods", ["kalman", "soft", "ot"])
    methods: List[str] = []
    for method_raw in as_list(methods_raw):
        method = _canonical_dpf_method(method_raw)
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("dpf.methods is empty; please provide at least one method.")
    return methods


def _method_dpf_cfg(dpf_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Resolve method configuration with `common` defaults and method overrides."""
    out: Dict[str, Any] = {}

    common_cfg = dpf_cfg.get("common")
    if isinstance(common_cfg, dict):
        out.update(common_cfg)

    reserved = {
        "methods",
        "common",
        "kalman",
        "baseline",
        "pf",
        "soft",
        "ot",
        "diffusion",
        "transformer",
    }
    for key, value in dpf_cfg.items():
        if key in reserved:
            continue
        if isinstance(value, dict):
            continue
        out.setdefault(key, value)

    method_cfg = dpf_cfg.get(method)
    if isinstance(method_cfg, dict):
        out.update(method_cfg)

    if method == "baseline":
        has_explicit_baseline_cfg = isinstance(dpf_cfg.get("baseline"), dict)
        if (not has_explicit_baseline_cfg) and isinstance(dpf_cfg.get("pf"), dict):
            out.update(dpf_cfg["pf"])

    return out


def _build_dpf(
    ssm: LinearGaussianSSM,
    dpf_cfg: Dict[str, Any],
    method: str,
    proposal=None,
) -> Any:
    cfg = _method_dpf_cfg(dpf_cfg, method)
    num_particles = int(cfg.get("num_particles", 256))
    ess_threshold = float(cfg.get("ess_threshold", 0.5))
    resample = cfg.get("resample", "auto")
    if method == "baseline":
        # Fixed-phi PF baseline with systematic resampling.
        return StandardResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )
    if method == "pf":
        # Standard PF with proposal q_phi and systematic resampling.
        return StandardResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )
    if method == "soft":
        lam = float(cfg.get("lam", 0.95))
        return SoftResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            lam=lam,
            resample=resample,
            proposal=proposal,
        )
    if method == "ot":
        ot_epsilon = float(cfg.get("ot_epsilon", cfg.get("epsilon", 0.1)))
        ot_num_iters = int(cfg.get("ot_num_iters", cfg.get("num_iters", 50)))
        ot_jitter = float(cfg.get("ot_jitter", 1e-6))
        return OTResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            ot_jitter=ot_jitter,
            resample=resample,
            proposal=proposal,
        )
    if method == "diffusion":
        diff_a = float(cfg.get("diff_a", -1.0))
        diff_T = float(cfg.get("diff_T", 1.0))
        diff_steps = int(cfg.get("diff_steps", 8))
        diff_ode = bool(cfg.get("diff_ode", True))
        diff_eps = float(cfg.get("diff_eps", 1e-6))
        return DiffusionResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            diff_a=diff_a,
            diff_T=diff_T,
            diff_steps=diff_steps,
            diff_ode=diff_ode,
            diff_eps=diff_eps,
            resample=resample,
            proposal=proposal,
        )
    if method == "transformer":
        d_model = int(cfg.get("pt_d_model", cfg.get("d_model", 128)))
        hidden = int(cfg.get("pt_hidden", cfg.get("hidden", 128)))
        num_heads = int(cfg.get("pt_num_heads", cfg.get("num_heads", 4)))
        num_encoder_layers = int(
            cfg.get("pt_num_encoder_layers", cfg.get("num_encoder_layers", 2))
        )
        num_decoder_layers = int(
            cfg.get("pt_num_decoder_layers", cfg.get("num_decoder_layers", 1))
        )
        dropout_rate = float(cfg.get("pt_dropout_rate", cfg.get("dropout_rate", 0.0)))
        stop_grad_through_time = bool(
            cfg.get("stop_grad_through_time", cfg.get("truncate_time_grad", True))
        )
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
            stop_grad_through_time=stop_grad_through_time,
            resample=resample,
            proposal=proposal,
        )
    raise ValueError(f"Unsupported DPF method '{method}'.")


def _build_optimizer(
    train_cfg: Dict[str, Any],
    *,
    lr_override: float | None = None,
) -> tf.keras.optimizers.Optimizer:
    name = str(train_cfg.get("optimizer", "adam")).strip().lower()
    lr = float(train_cfg.get("lr", 1e-3)) if lr_override is None else float(lr_override)
    beta1 = float(train_cfg.get("beta1", 0.9))
    beta2 = float(train_cfg.get("beta2", 0.999))
    if name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
            amsgrad=bool(train_cfg.get("amsgrad", False)),
        )
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            beta_1=beta1,
            beta_2=beta2,
            amsgrad=bool(train_cfg.get("amsgrad", False)),
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=float(train_cfg.get("rho", 0.9)),
            momentum=float(train_cfg.get("momentum", 0.0)),
        )
    raise ValueError("training.optimizer must be one of: adam, adamw, rmsprop")


def _resolve_method_train_cfg(
    train_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve per-method training overrides from dpf.<method>.train."""
    out = dict(train_cfg)

    train_override = method_cfg.get("train")
    if train_override is None:
        return out
    if not isinstance(train_override, dict):
        raise ValueError("dpf.<method>.train must be a mapping.")
    out.update(train_override)
    return out


def _run_kalman_method(seed_ctx: _SeedContext) -> Dict[str, Any]:
    fit_ssm = seed_ctx.fit_ssm
    x_true = seed_ctx.x_true
    y_obs = seed_ctx.y_obs

    t_eval_init = time.perf_counter()
    init_eval = _evaluate_kalman(fit_ssm, y_obs, x_true)
    runtime_eval_init_sec = float(time.perf_counter() - t_eval_init)
    runtime_eval_final_sec = runtime_eval_init_sec

    return {
        "init_eval": init_eval,
        "final_eval": init_eval,
        "rmse_phi_init": float("nan"),
        "rmse_phi_final": float("nan"),
        "phi_init": np.asarray([], dtype=np.float32),
        "phi_final": np.asarray([], dtype=np.float32),
        "pretrained_weights_used": None,
        "loss_hist": np.asarray([], dtype=np.float32),
        "rmse_hist": np.asarray([], dtype=np.float32),
        "grad_hist": np.asarray([], dtype=np.float32),
        "grad_raw_hist": np.asarray([], dtype=np.float32),
        "phi_rmse_hist": np.asarray([], dtype=np.float32),
        "grad_snr_init": float("nan"),
        "grad_snr_final": float("nan"),
        "grad_var_init": float("nan"),
        "grad_var_final": float("nan"),
        "runtime_train_sec": 0.0,
        "runtime_train_per_step_sec": 0.0,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
    }


def _run_baseline_method(
    *,
    seed: int,
    seed_ctx: _SeedContext,
    dpf_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fit_ssm = seed_ctx.fit_ssm
    x_true = seed_ctx.x_true
    y_obs = seed_ctx.y_obs
    state_dim = int(seed_ctx.state_dim)
    obs_dim = int(seed_ctx.obs_dim)
    rng_base = int(seed_ctx.rng_base)

    method_cfg = _method_dpf_cfg(dpf_cfg, "baseline")
    resample = method_cfg.get("resample", "auto")
    proposal = LinearGaussianProposalPhi(
        A=fit_ssm.A,
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_delta=1.0, # optimal proposal
        init_gamma=1.0, # optimal proposal
        init_noise_std=0.0,
        rng=None,
    )
    dpf: DPFBase = _build_dpf(
        fit_ssm,
        dpf_cfg,
        method="baseline",
        proposal=proposal,
    )
    _log(
        f"[seed={seed}][baseline] fixed-phi baseline with phi=1 "
        "(no training)."
    )
    _reset_filter_rng(fit_ssm, rng_base + 11)
    t_eval_init = time.perf_counter()
    init_eval = _evaluate_filter(dpf, y_obs, x_true, resample=resample)
    runtime_eval_init_sec = float(time.perf_counter() - t_eval_init)

    rmse_phi_init = float(proposal.rmse_phi_to_one().numpy())
    phi_init = np.asarray(proposal.phi_vector().numpy(), dtype=np.float32)

    return {
        "init_eval": init_eval,
        "final_eval": init_eval,
        "rmse_phi_init": rmse_phi_init,
        "rmse_phi_final": rmse_phi_init,
        "phi_init": phi_init,
        "phi_final": np.asarray(phi_init, dtype=np.float32),
        "pretrained_weights_used": None,
        "loss_hist": np.asarray([], dtype=np.float32),
        "rmse_hist": np.asarray([], dtype=np.float32),
        "grad_hist": np.asarray([], dtype=np.float32),
        "grad_raw_hist": np.asarray([], dtype=np.float32),
        "phi_rmse_hist": np.asarray([], dtype=np.float32),
        "grad_snr_init": float("nan"),
        "grad_snr_final": float("nan"),
        "grad_var_init": float("nan"),
        "grad_var_final": float("nan"),
        "runtime_train_sec": 0.0,
        "runtime_train_per_step_sec": 0.0,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_init_sec,
    }


def _train_dpf(
    *,
    seed: int,
    method: str,
    fit_ssm: LinearGaussianSSM,
    dpf: DPFBase,
    proposal: LinearGaussianProposalPhi,
    y_obs: Any,
    x_true: Any,
    resample: str | int | bool,
    train_vars: List[tf.Variable],
    proposal_vars: List[tf.Variable],
    resampler_vars: List[tf.Variable],
    proposal_optimizer: tf.keras.optimizers.Optimizer,
    resampler_optimizer: tf.keras.optimizers.Optimizer | None,
    steps: int,
    mc_samples: int,
    grad_clip_norm: float,
    log_every: int,
    rng_base: int,
) -> Dict[str, Any]:
    proposal_var_ids = {id(v) for v in proposal_vars}
    resampler_var_ids = {id(v) for v in resampler_vars}

    loss_hist_ta = tf.TensorArray(tf.float32, size=steps, dynamic_size=False, clear_after_read=False)
    rmse_hist_ta = tf.TensorArray(tf.float32, size=steps, dynamic_size=False, clear_after_read=False)
    grad_hist_ta = tf.TensorArray(tf.float32, size=steps, dynamic_size=False, clear_after_read=False)
    grad_raw_hist_ta = tf.TensorArray(
        tf.float32, size=steps, dynamic_size=False, clear_after_read=False
    )
    phi_rmse_hist_ta = tf.TensorArray(tf.float32, size=steps, dynamic_size=False, clear_after_read=False)
    x_true_t = tf.convert_to_tensor(x_true, dtype=tf.float32)

    train_t0 = time.perf_counter()
    step_iter = tqdm(
        range(steps),
        total=steps,
        desc=f"seed={seed}:{method}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    for step in step_iter:
        loss_terms = []
        rmse_terms = []
        ess_terms = []
        mc_stride = 100
        with tf.GradientTape() as tape:
            for mc_idx in range(mc_samples):
                _reset_filter_rng(
                    fit_ssm,
                    rng_base + 1_000 + step * mc_stride + mc_idx,
                )
                x_seq, w_seq, diagnostics, _ = dpf.filter(y_obs, resample=resample)
                logz_total = tf.reduce_sum(
                    tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32),
                    axis=-1,
                )
                loss_terms.append(-tf.reduce_mean(logz_total))
                # Metrics are for logging only; keep them out of autodiff recording.
                with tape.stop_recording():
                    mean_seq = _particle_mean(x_seq, w_seq)
                    rmse_terms.append(
                        tf.sqrt(tf.reduce_mean(tf.square(mean_seq - x_true_t)))
                    )
                    ess_terms.append(
                        tf.reduce_mean(
                            _ess_from_log_weights(
                                tf.convert_to_tensor(diagnostics["log_w_pre"], dtype=tf.float32)
                            )
                        )
                    )
            loss = tf.add_n(loss_terms) / float(mc_samples)
        rmse_step_tensor = tf.add_n(rmse_terms) / float(mc_samples)
        ess_step_tensor = tf.add_n(ess_terms) / float(mc_samples)

        grads = tape.gradient(loss, train_vars)
        valid = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if valid:
            grad_values = [g for g, _ in valid]
            raw_grad_norm = tf.linalg.global_norm(grad_values)
            if grad_clip_norm > 0.0:
                grad_values, _ = tf.clip_by_global_norm(grad_values, grad_clip_norm)
            grad_norm = tf.linalg.global_norm(grad_values)  # post-clip norm
            valid = list(zip(grad_values, [v for _, v in valid]))
            if resampler_optimizer is None:
                proposal_optimizer.apply_gradients(valid)
            else:
                valid_prop = [(g, v) for g, v in valid if id(v) in proposal_var_ids]
                valid_res = [(g, v) for g, v in valid if id(v) in resampler_var_ids]
                if valid_prop:
                    proposal_optimizer.apply_gradients(valid_prop)
                if valid_res:
                    resampler_optimizer.apply_gradients(valid_res)
        else:
            raw_grad_norm = tf.constant(0.0, dtype=tf.float32)
            grad_norm = tf.constant(0.0, dtype=tf.float32)

        phi_rmse_step = tf.cast(proposal.rmse_phi_to_one(), tf.float32)
        loss_hist_ta = loss_hist_ta.write(step, tf.cast(loss, tf.float32))
        rmse_hist_ta = rmse_hist_ta.write(step, tf.cast(rmse_step_tensor, tf.float32))
        grad_hist_ta = grad_hist_ta.write(step, tf.cast(grad_norm, tf.float32))
        grad_raw_hist_ta = grad_raw_hist_ta.write(step, tf.cast(raw_grad_norm, tf.float32))
        phi_rmse_hist_ta = phi_rmse_hist_ta.write(step, phi_rmse_step)
        if log_every > 0 and (
            step == 0 or (step + 1) % log_every == 0 or (step + 1) == steps
        ):
            loss_step = float(loss.numpy())
            rmse_step = float(rmse_step_tensor.numpy())
            ess_step = float(ess_step_tensor.numpy())
            phi_rmse_step_val = float(phi_rmse_step.numpy())
            grad_raw_step = float(raw_grad_norm.numpy())
            grad_step = float(grad_norm.numpy())
            step_iter.set_postfix_str(
                "loss="
                f"{loss_step:.5f} "
                "rmse="
                f"{rmse_step:.5f} "
                "phi="
                f"{phi_rmse_step_val:.5f} "
                "ess="
                f"{ess_step:.2f} "
                "g="
                f"{grad_step:.4f} "
                "g_raw="
                f"{grad_raw_step:.4f} "
                "mc="
                f"{mc_samples}",
                refresh=False,
            )
    runtime_train_sec = float(time.perf_counter() - train_t0)
    runtime_train_per_step_sec = float(runtime_train_sec / max(steps, 1))
    loss_hist = np.asarray(loss_hist_ta.stack().numpy(), dtype=np.float32)
    rmse_hist = np.asarray(rmse_hist_ta.stack().numpy(), dtype=np.float32)
    grad_hist = np.asarray(grad_hist_ta.stack().numpy(), dtype=np.float32)
    grad_raw_hist = np.asarray(grad_raw_hist_ta.stack().numpy(), dtype=np.float32)
    phi_rmse_hist = np.asarray(phi_rmse_hist_ta.stack().numpy(), dtype=np.float32)
    grad_stats = _grad_metrics_from_history(grad_raw_hist, window=10)

    return {
        "loss_hist": loss_hist,
        "rmse_hist": rmse_hist,
        "grad_hist": grad_hist,
        "grad_raw_hist": grad_raw_hist,
        "phi_rmse_hist": phi_rmse_hist,
        "grad_snr_init": float(grad_stats["grad_snr_init"]),
        "grad_snr_final": float(grad_stats["grad_snr_final"]),
        "grad_var_init": float(grad_stats["grad_var_init"]),
        "grad_var_final": float(grad_stats["grad_var_final"]),
        "runtime_train_sec": runtime_train_sec,
        "runtime_train_per_step_sec": runtime_train_per_step_sec,
    }


def _run_trainable_dpf_method(
    *,
    seed: int,
    method: str,
    seed_ctx: _SeedContext,
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dpf_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    experiment_name: str | None,
    experiment_dir: str | Path | None,
) -> Dict[str, Any]:
    rng = seed_ctx.rng
    fit_ssm = seed_ctx.fit_ssm
    x_true = seed_ctx.x_true
    y_obs = seed_ctx.y_obs
    state_dim = int(seed_ctx.state_dim)
    obs_dim = int(seed_ctx.obs_dim)
    rng_base = int(seed_ctx.rng_base)

    method_cfg = _method_dpf_cfg(dpf_cfg, method)
    train_cfg_eff = _resolve_method_train_cfg(train_cfg, method_cfg)
    resample = method_cfg.get("resample", "auto")
    proposal = LinearGaussianProposalPhi(
        A=fit_ssm.A,
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_delta=proposal_cfg.get("init_delta", 1.0),
        init_gamma=proposal_cfg.get("init_gamma", 1.0),
        init_noise_std=float(proposal_cfg.get("init_noise_std", 0.0)),
        rng=rng,
    )
    dpf = _build_dpf(fit_ssm, dpf_cfg, method=method, proposal=proposal)
    pretrained_weights_used = None
    if method == "transformer":
        pretrained_weights_used = _maybe_init_transformer_from_pretrain(
            dpf=dpf,
            method_cfg=method_cfg,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            proposal_cfg=proposal_cfg,
            seed=seed,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
        )
        freeze_resampler = bool(method_cfg.get("freeze_resampler", True))
        dpf.resampler_net.trainable = not freeze_resampler
        if pretrained_weights_used is not None:
            _log(
                f"[seed={seed}][{method}] loaded pretrained resampler "
                f"weights: {pretrained_weights_used}"
            )

    proposal_vars, resampler_vars, train_vars = _select_train_var_groups(
        proposal,
        method=method,
        dpf=dpf,
        method_cfg=method_cfg,
    )
    if not train_vars:
        raise ValueError(f"No trainable variables selected for method '{method}'.")

    steps = int(train_cfg_eff.get("steps", 100))
    mc_samples = int(train_cfg_eff.get("mc_samples", 1))
    if mc_samples <= 0:
        raise ValueError("training.mc_samples must be a positive integer.")
    grad_clip_norm = float(train_cfg_eff.get("grad_clip_norm", 50.0))
    log_every = int(train_cfg_eff.get("log_every", 10))
    proposal_lr = float(train_cfg_eff.get("proposal_lr", train_cfg_eff.get("lr", 1e-3)))
    if "resampler_lr" in train_cfg_eff:
        resampler_lr = float(train_cfg_eff["resampler_lr"])
    elif "resampler_lr_scale" in train_cfg_eff:
        resampler_lr = proposal_lr * float(train_cfg_eff.get("resampler_lr_scale", 1.0))
    else:
        resampler_lr = proposal_lr

    proposal_optimizer = _build_optimizer(train_cfg_eff, lr_override=proposal_lr)
    resampler_optimizer = None
    if resampler_vars:
        resampler_optimizer = _build_optimizer(train_cfg_eff, lr_override=resampler_lr)
        _log(
            f"[seed={seed}][{method}] proposal_lr={proposal_lr:.3e} "
            f"resampler_lr={resampler_lr:.3e}"
        )
    else:
        _log(f"[seed={seed}][{method}] proposal_lr={proposal_lr:.3e}")

    _reset_filter_rng(fit_ssm, rng_base + 11)
    t_eval_init = time.perf_counter()
    init_eval = _evaluate_filter(dpf, y_obs, x_true, resample=resample)
    runtime_eval_init_sec = float(time.perf_counter() - t_eval_init)
    rmse_phi_init = float(proposal.rmse_phi_to_one().numpy())
    phi_init = np.asarray(proposal.phi_vector().numpy(), dtype=np.float32)
    train_out = _train_dpf(
        seed=seed,
        method=method,
        fit_ssm=fit_ssm,
        dpf=dpf,
        proposal=proposal,
        y_obs=y_obs,
        x_true=x_true,
        resample=resample,
        train_vars=train_vars,
        proposal_vars=proposal_vars,
        resampler_vars=resampler_vars,
        proposal_optimizer=proposal_optimizer,
        resampler_optimizer=resampler_optimizer,
        steps=steps,
        mc_samples=mc_samples,
        grad_clip_norm=grad_clip_norm,
        log_every=log_every,
        rng_base=rng_base,
    )

    _reset_filter_rng(fit_ssm, rng_base + 900_000)
    t_eval_final = time.perf_counter()
    final_eval = _evaluate_filter(dpf, y_obs, x_true, resample=resample)
    runtime_eval_final_sec = float(time.perf_counter() - t_eval_final)
    rmse_phi_final = float(proposal.rmse_phi_to_one().numpy())
    phi_final = np.asarray(proposal.phi_vector().numpy(), dtype=np.float32)

    return {
        "init_eval": init_eval,
        "final_eval": final_eval,
        "rmse_phi_init": rmse_phi_init,
        "rmse_phi_final": rmse_phi_final,
        "phi_init": phi_init,
        "phi_final": phi_final,
        "pretrained_weights_used": pretrained_weights_used,
        "loss_hist": np.asarray(train_out["loss_hist"], dtype=np.float32),
        "rmse_hist": np.asarray(train_out["rmse_hist"], dtype=np.float32),
        "grad_hist": np.asarray(train_out["grad_hist"], dtype=np.float32),
        "grad_raw_hist": np.asarray(train_out["grad_raw_hist"], dtype=np.float32),
        "phi_rmse_hist": np.asarray(train_out["phi_rmse_hist"], dtype=np.float32),
        "grad_snr_init": float(train_out["grad_snr_init"]),
        "grad_snr_final": float(train_out["grad_snr_final"]),
        "grad_var_init": float(train_out["grad_var_init"]),
        "grad_var_final": float(train_out["grad_var_final"]),
        "runtime_train_sec": float(train_out["runtime_train_sec"]),
        "runtime_train_per_step_sec": float(train_out["runtime_train_per_step_sec"]),
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
    }


def _run_method_with_registry(
    *,
    method: str,
    seed: int,
    seed_ctx: _SeedContext,
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dpf_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    experiment_name: str | None,
    experiment_dir: str | Path | None,
) -> Dict[str, Any]:
    registry = {
        "kalman": lambda: _run_kalman_method(seed_ctx),
        "baseline": lambda: _run_baseline_method(
            seed=seed,
            seed_ctx=seed_ctx,
            dpf_cfg=dpf_cfg,
        ),
    }
    runner = registry.get(method)
    if runner is not None:
        return runner()

    trainable_methods = {"pf", "soft", "ot", "diffusion", "transformer"}
    if method in trainable_methods:
        return _run_trainable_dpf_method(
            seed=seed,
            method=method,
            seed_ctx=seed_ctx,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dpf_cfg=dpf_cfg,
            proposal_cfg=proposal_cfg,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
        )
    raise ValueError(f"Unsupported dpf method '{method}'.")


def _run_single_seed(
    seed: int,
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dpf_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    method: str,
    experiment_name: str | None = None,
    experiment_dir: str | Path | None = None,
) -> Dict[str, Any]:
    seed_ctx = _build_seed_context(seed, exp_cfg, model_cfg)
    x_true = seed_ctx.x_true
    y_obs = seed_ctx.y_obs
    method_out = _run_method_with_registry(
        method=method,
        seed=seed,
        seed_ctx=seed_ctx,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dpf_cfg=dpf_cfg,
        proposal_cfg=proposal_cfg,
        experiment_name=experiment_name,
        experiment_dir=experiment_dir,
    )

    init_eval = method_out["init_eval"]
    final_eval = method_out["final_eval"]
    rmse_phi_init = float(method_out["rmse_phi_init"])
    rmse_phi_final = float(method_out["rmse_phi_final"])
    phi_init = np.asarray(method_out["phi_init"], dtype=np.float32)
    phi_final = np.asarray(method_out["phi_final"], dtype=np.float32)
    pretrained_weights_used = method_out["pretrained_weights_used"]
    loss_hist = np.asarray(method_out["loss_hist"], dtype=np.float32)
    rmse_hist = np.asarray(method_out["rmse_hist"], dtype=np.float32)
    grad_hist = np.asarray(method_out["grad_hist"], dtype=np.float32)
    grad_raw_hist = np.asarray(method_out["grad_raw_hist"], dtype=np.float32)
    phi_rmse_hist = np.asarray(method_out["phi_rmse_hist"], dtype=np.float32)
    grad_snr_init = float(method_out["grad_snr_init"])
    grad_snr_final = float(method_out["grad_snr_final"])
    grad_var_init = float(method_out["grad_var_init"])
    grad_var_final = float(method_out["grad_var_final"])
    runtime_train_sec = float(method_out["runtime_train_sec"])
    runtime_train_per_step_sec = float(method_out["runtime_train_per_step_sec"])
    runtime_eval_init_sec = float(method_out["runtime_eval_init_sec"])
    runtime_eval_final_sec = float(method_out["runtime_eval_final_sec"])

    result = {
        "seed": seed,
        "method": method,
        "loss_init": init_eval["loss"],
        "loss_final": final_eval["loss"],
        "nll_init": init_eval["nll"],
        "nll_final": final_eval["nll"],
        "rmse_init": init_eval["rmse"],
        "rmse_final": final_eval["rmse"],
        "rmse_phi_init": rmse_phi_init,
        "rmse_phi_final": rmse_phi_final,
        "ess_mean_init": float(init_eval["ess_mean"]),
        "ess_mean_final": float(final_eval["ess_mean"]),
        "resample_rate_init": float(init_eval["resample_rate"]),
        "resample_rate_final": float(final_eval["resample_rate"]),
        "grad_snr_init": grad_snr_init,
        "grad_snr_final": grad_snr_final,
        "grad_var_init": grad_var_init,
        "grad_var_final": grad_var_final,
        "runtime_train_sec": runtime_train_sec,
        "runtime_train_per_step_sec": runtime_train_per_step_sec,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
        "loss_history": loss_hist,
        "rmse_history": rmse_hist,
        "grad_raw_norm_history": grad_raw_hist,
        "grad_norm_history": grad_hist,
        "phi_rmse_history": phi_rmse_hist,
        "ess_over_time": np.asarray(final_eval["ess_over_time"], dtype=np.float32),
        "logz_t": np.asarray(final_eval["logZ_t"], dtype=np.float32),
        "resampled_t": np.asarray(final_eval["resampled_t"], dtype=np.bool_),
        "x_true": np.asarray(x_true),
        "y_obs": np.asarray(y_obs),
        "mean_final": np.asarray(final_eval["mean"]),
        "phi_init": phi_init,
        "phi_final": phi_final,
        "pretrained_weights_used": pretrained_weights_used,
    }
    return result


def _persist_trace(
    *,
    out_root: Path,
    experiment_name: str,
    seed: int,
    method: str,
    result: Dict[str, Any],
) -> None:
    method_dir = out_root / method
    ensure_dir(method_dir)
    save_npz(
        method_dir / _trace_filename(experiment_name, seed, method),
        loss_history=result["loss_history"],
        rmse_history=result["rmse_history"],
        grad_raw_norm_history=result["grad_raw_norm_history"],
        grad_norm_history=result["grad_norm_history"],
        phi_rmse_history=result["phi_rmse_history"],
        ess_over_time=result["ess_over_time"],
        logz_t=result["logz_t"],
        resampled_t=result["resampled_t"].astype(np.int8),
        x_true=result["x_true"],
        y_obs=result["y_obs"],
        mean_final=result["mean_final"],
        phi_init=result["phi_init"],
        phi_final=result["phi_final"],
    )


def _log_method_result(
    *,
    tag: str,
    seed: int,
    method: str,
    result: Dict[str, Any],
) -> None:
    _log(
        f"[{tag}] seed={seed} method={method} "
        f"loss {result['loss_init']:.6f}->{result['loss_final']:.6f} | "
        f"nll {result['nll_init']:.6f}->{result['nll_final']:.6f} | "
        f"rmse {result['rmse_init']:.6f}->{result['rmse_final']:.6f} | "
        f"phi_rmse {result['rmse_phi_init']:.6f}->{result['rmse_phi_final']:.6f} | "
        f"ess_mean {result['ess_mean_init']:.2f}->{result['ess_mean_final']:.2f} | "
        f"grad_snr {result['grad_snr_init']:.3f}->{result['grad_snr_final']:.3f} | "
        f"grad_var {result['grad_var_init']:.3e}->{result['grad_var_final']:.3e} | "
        f"train_sec {result['runtime_train_sec']:.3f}"
    )


def _build_method_summary(
    *,
    per_seed: Dict[str, List[Dict[str, Any]]],
    methods: List[str],
) -> Dict[str, Dict[str, float]]:
    return {
        method: {
            "loss_init_mean": float(np.mean([r["loss_init"] for r in per_seed[method]])),
            "loss_final_mean": float(np.mean([r["loss_final"] for r in per_seed[method]])),
            "nll_init_mean": float(np.mean([r["nll_init"] for r in per_seed[method]])),
            "nll_final_mean": float(np.mean([r["nll_final"] for r in per_seed[method]])),
            "rmse_init_mean": float(np.mean([r["rmse_init"] for r in per_seed[method]])),
            "rmse_final_mean": float(np.mean([r["rmse_final"] for r in per_seed[method]])),
            "rmse_phi_init_mean": _safe_nanmean([r["rmse_phi_init"] for r in per_seed[method]]),
            "rmse_phi_final_mean": _safe_nanmean([r["rmse_phi_final"] for r in per_seed[method]]),
            "ess_mean_init_mean": _safe_nanmean([r["ess_mean_init"] for r in per_seed[method]]),
            "ess_mean_final_mean": _safe_nanmean([r["ess_mean_final"] for r in per_seed[method]]),
            "resample_rate_init_mean": _safe_nanmean(
                [r["resample_rate_init"] for r in per_seed[method]]
            ),
            "resample_rate_final_mean": _safe_nanmean(
                [r["resample_rate_final"] for r in per_seed[method]]
            ),
            "grad_snr_init_mean": _safe_nanmean([r["grad_snr_init"] for r in per_seed[method]]),
            "grad_snr_final_mean": _safe_nanmean([r["grad_snr_final"] for r in per_seed[method]]),
            "grad_var_init_mean": _safe_nanmean([r["grad_var_init"] for r in per_seed[method]]),
            "grad_var_final_mean": _safe_nanmean([r["grad_var_final"] for r in per_seed[method]]),
            "runtime_train_sec_mean": _safe_nanmean(
                [r["runtime_train_sec"] for r in per_seed[method]]
            ),
            "runtime_train_per_step_sec_mean": _safe_nanmean(
                [r["runtime_train_per_step_sec"] for r in per_seed[method]]
            ),
            "runtime_eval_init_sec_mean": _safe_nanmean(
                [r["runtime_eval_init_sec"] for r in per_seed[method]]
            ),
            "runtime_eval_final_sec_mean": _safe_nanmean(
                [r["runtime_eval_final_sec"] for r in per_seed[method]]
            ),
        }
        for method in methods
    }


def run_lgssm_dpf_backprop(
    cfg: Dict[str, Any],
    config_path: Path,
    tag: str,
    *,
    exp_name: str | None = None,
) -> None:
    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    train_cfg = cfg_section(cfg, "training")
    proposal_cfg = cfg_section(cfg, "proposal")
    dpf_cfg = cfg_section(cfg, "dpf")

    base_out_root = Path(exp_cfg.get("output_root", f"results/{tag}"))
    experiment_name = str(
        exp_name if exp_name is not None else exp_cfg.get("experiment_name", "exp3")
    ).strip() or "exp3"
    out_root = base_out_root / experiment_name
    ensure_dir(out_root)
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    save_traces = bool(exp_cfg.get("save_traces", True))
    methods = _resolve_dpf_methods(dpf_cfg)

    _log(f"[{tag}] config={config_path}")
    _log(f"[{tag}] exp_name={experiment_name}")
    _log(f"[{tag}] output_root={out_root}")
    _log(f"[{tag}] seeds={seeds}")
    _log(f"[{tag}] methods={methods}")

    per_seed: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}
    metrics_across_seeds: Dict[str, List[Dict[str, Any]]] = {
        method: [] for method in methods
    }
    for seed in seeds:
        per_seed_metrics: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            result = _run_single_seed(
                seed,
                exp_cfg,
                model_cfg,
                train_cfg,
                dpf_cfg,
                proposal_cfg,
                method=method,
                experiment_name=experiment_name,
                experiment_dir=out_root,
            )
            per_seed[method].append(result)
            final_metrics = _result_stage_metrics(result, "final")
            per_seed_metrics[method] = final_metrics
            metrics_across_seeds[method].append(final_metrics)

            _log_method_result(
                tag=tag,
                seed=seed,
                method=method,
                result=result,
            )

            if save_traces:
                _persist_trace(
                    out_root=out_root,
                    experiment_name=experiment_name,
                    seed=seed,
                    method=method,
                    result=result,
                )

        print_separator(f"{tag} seed{seed} summary")
        print_method_summary_table(
            per_seed_metrics,
            method_order=tuple(methods),
            keys=SUMMARY_KEYS,
        )

    method_summary_new = _build_method_summary(
        per_seed=per_seed,
        methods=methods,
    )
    summary_path = out_root / "summary.json"
    summary = {
        "experiment_name": experiment_name,
        "num_seeds": int(len(seeds)),
        "methods": method_summary_new,
    }

    save_json(summary_path, summary)
    _log(f"[{tag}] summary saved: {summary_path}")
    for method in methods:
        method_summary = summary["methods"][method]
        _log(
            f"[{tag}] {method} mean loss "
            f"{method_summary['loss_init_mean']:.6f}->{method_summary['loss_final_mean']:.6f}, "
            f"mean nll {method_summary['nll_init_mean']:.6f}->{method_summary['nll_final_mean']:.6f}, "
            f"mean rmse {method_summary['rmse_init_mean']:.6f}->{method_summary['rmse_final_mean']:.6f}, "
            f"mean phi_rmse {method_summary['rmse_phi_init_mean']:.6f}->{method_summary['rmse_phi_final_mean']:.6f}, "
            f"mean ess {method_summary['ess_mean_init_mean']:.2f}->{method_summary['ess_mean_final_mean']:.2f}, "
            f"mean grad_snr {method_summary['grad_snr_init_mean']:.3f}->{method_summary['grad_snr_final_mean']:.3f}, "
            f"mean grad_var {method_summary['grad_var_init_mean']:.3e}->{method_summary['grad_var_final_mean']:.3e}, "
            f"mean train_sec {method_summary['runtime_train_sec_mean']:.3f}"
        )

    mean_metrics = aggregate_metrics_by_method(metrics_across_seeds)
    avg_metrics = dict(mean_metrics)
    avg_method_order = tuple(methods)
    print_separator(f"{tag} avg summary")
    print_method_summary_table(
        avg_metrics,
        method_order=avg_method_order,
        keys=SUMMARY_KEYS,
    )


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
        if isinstance(dpf_cfg.get("common"), dict):
            dpf_cfg["common"]["num_particles"] = int(args.num_particles)
        else:
            dpf_cfg["num_particles"] = int(args.num_particles)
    exp_name = args.name if args.name is not None else None

    run_lgssm_dpf_backprop(
        cfg=cfg,
        config_path=args.config,
        tag="exp3_lgssm_dpf_backprop",
        exp_name=exp_name,
    )


if __name__ == "__main__":
    main()
