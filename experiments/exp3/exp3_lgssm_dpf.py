from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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
    print_metrics_compare,
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
    "grad_raw",
    "gpu_peak_mb",
    "train_sec",
)
# Metrics reported for the held-out test split (no grad/phi columns).
TEST_SUMMARY_KEYS = ("loss", "nll", "rmse", "ess_mean")


_log_fh: "IO[str] | None" = None


def _init_log_file(path: Path) -> None:
    global _log_fh
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(path, "a", buffering=1, encoding="utf-8")


def _log(message: str) -> None:
    print(message, flush=True)
    if _log_fh is not None:
        try:
            print(message, file=_log_fh, flush=True)
        except Exception:
            pass


def _seed_tag(seed: int, method: str | None = None) -> str:
    """Return a fixed-width seed prefix: [seed= 1024] or [seed= 1024][soft]."""
    base = f"[seed={seed:>5}]"
    return base if method is None else f"{base}[{method}]"


def _current_gpu_memory_mb() -> tuple[float | None, float | None]:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return None, None
        info = tf.config.experimental.get_memory_info("GPU:0")
        current = info.get("current")
        peak = info.get("peak")
        current_mb = float(current) / (1024.0 * 1024.0) if current is not None else None
        peak_mb = float(peak) / (1024.0 * 1024.0) if peak is not None else None
        return current_mb, peak_mb
    except Exception:
        return None, None


def _reset_gpu_peak() -> None:
    """Reset GPU peak memory counter so each method reports its own peak."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def _format_memory_mb(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.1f}MB"


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of seeds to run in parallel (one subprocess per seed). "
            "Each subprocess has its own TF global state, avoiding RNG races. "
            "Default: 1 (serial)."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class _SeedContext:
    """Per-seed immutable context shared by method runners."""
    rng: np.random.Generator
    fit_ssm: LinearGaussianSSM
    x_true: Any   # train split
    y_obs: Any    # train split
    x_test: Any   # test split
    y_test: Any   # test split
    state_dim: int
    obs_dim: int
    rng_base: int


def _build_seed_context(
    seed: int,
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> _SeedContext:
    """Build one seed's simulation data + fit model once and reuse downstream.

    The simulated batch is split 50/50 into train and test halves.  If
    ``batch_size`` is odd the last sequence is dropped so both halves have
    equal size.  A minimum of 2 sequences (1 train + 1 test) is required.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    T = int(exp_cfg.get("T", 50))
    batch_size = int(exp_cfg.get("batch_size", 1))
    state_dim = int(model_cfg.get("state_dim", 8))
    obs_dim = int(model_cfg.get("obs_dim", max(1, state_dim // 2)))

    if batch_size < 2:
        raise ValueError(
            f"experiment.batch_size must be >= 2 for train-test split, got {batch_size}. "
            "Set experiment.batch_size to an even integer >= 2 in the config."
        )
    B_half = batch_size // 2  # each split gets this many sequences
    if batch_size % 2 != 0:
        _log(
            f"[seed={seed}] WARNING: batch_size={batch_size} is odd; "
            f"using {B_half * 2} sequences ({B_half} train + {B_half} test)."
        )

    sim_ssm, fit_ssm = build_exp3_linear_ssm_pair(
        model_cfg=model_cfg,
        seed=int(seed),
        fit_trainable=False,
    )
    x_all, y_all = sim_ssm.simulate(T=T, shape=[B_half * 2])

    x_train = x_all[:B_half]
    y_train = y_all[:B_half]
    x_test  = x_all[B_half:]
    y_test  = y_all[B_half:]

    return _SeedContext(
        rng=rng,
        fit_ssm=fit_ssm,
        x_true=x_train,
        y_obs=y_train,
        x_test=x_test,
        y_test=y_test,
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
    loss = -tf.reduce_mean(logz_t)
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
    loss = -tf.reduce_mean(logz_t)
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
            f"{_seed_tag(seed, 'transformer')} pretrained_weights={path_value!r}, "
            "skip pretrain and skip loading."
        )
        return None
    if path_value is None:
        _log(f"{_seed_tag(seed, 'transformer')} pretrained_weights is None, running auto pretrain...")
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
        stop_grad_through_time = bool(cfg.get("stop_grad_through_time", False))
        return OTResamplingDPF(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            ot_jitter=ot_jitter,
            resample=resample,
            stop_grad_through_time=stop_grad_through_time,
            proposal=proposal,
        )
    if method == "diffusion":
        diff_a = float(cfg.get("diff_a", -1.0))
        diff_T = float(cfg.get("diff_T", 1.0))
        diff_steps = int(cfg.get("diff_steps", 8))
        diff_ode = bool(cfg.get("diff_ode", True))
        diff_eps = float(cfg.get("diff_eps", 1e-6))
        stop_grad_through_time = bool(cfg.get("stop_grad_through_time", False))
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
            stop_grad_through_time=stop_grad_through_time,
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
    x_true = seed_ctx.x_true  # train split
    y_obs = seed_ctx.y_obs    # train split

    t_eval_init = time.perf_counter()
    init_eval = _evaluate_kalman(fit_ssm, y_obs, x_true)
    runtime_eval_init_sec = float(time.perf_counter() - t_eval_init)
    runtime_eval_final_sec = runtime_eval_init_sec

    t_eval_test = time.perf_counter()
    test_eval = _evaluate_kalman(fit_ssm, seed_ctx.y_test, seed_ctx.x_test)
    runtime_eval_test_sec = float(time.perf_counter() - t_eval_test)

    return {
        "init_eval": init_eval,
        "final_eval": init_eval,
        "test_eval": test_eval,
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
        "grad_raw_init": float("nan"),
        "grad_raw_final": float("nan"),
        "gpu_peak_train_mb": float("nan"),
        "runtime_pretrain_sec": 0.0,
        "runtime_train_sec": 0.0,
        "runtime_train_per_step_sec": 0.0,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
        "runtime_eval_test_sec": runtime_eval_test_sec,
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
        f"{_seed_tag(seed, 'baseline')} fixed-phi baseline with phi=1 (no training)."
    )
    _reset_filter_rng(fit_ssm, rng_base + 11)
    t_eval_init = time.perf_counter()
    init_eval = _evaluate_filter(dpf, y_obs, x_true, resample=resample)
    runtime_eval_init_sec = float(time.perf_counter() - t_eval_init)

    t_eval_test = time.perf_counter()
    test_eval = _evaluate_filter(dpf, seed_ctx.y_test, seed_ctx.x_test, resample=resample)
    runtime_eval_test_sec = float(time.perf_counter() - t_eval_test)

    rmse_phi_init = float(proposal.rmse_phi_to_one().numpy())
    phi_init = np.asarray(proposal.phi_vector().numpy(), dtype=np.float32)

    return {
        "init_eval": init_eval,
        "final_eval": init_eval,
        "test_eval": test_eval,
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
        "grad_raw_init": float("nan"),
        "grad_raw_final": float("nan"),
        "gpu_peak_train_mb": float("nan"),
        "runtime_pretrain_sec": 0.0,
        "runtime_train_sec": 0.0,
        "runtime_train_per_step_sec": 0.0,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_init_sec,
        "runtime_eval_test_sec": runtime_eval_test_sec,
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
    step_time_hist: deque[float] = deque(maxlen=max(1, int(log_every)))
    mc_var_window: deque[float] = deque(maxlen=max(1, int(log_every)))
    for step in range(steps):
        step_t0 = time.perf_counter()
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
                x_seq, w_seq, diagnostics, _ = dpf.filter(
                    y_obs, resample=resample, training=True)
                logz_per_step = tf.reduce_mean(
                    tf.convert_to_tensor(diagnostics["log_z"], dtype=tf.float32),
                    axis=-1,
                )
                loss_terms.append(-tf.reduce_mean(logz_per_step))
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

        # MC loss variance — monitoring only, stop_gradient so it never enters backprop
        if mc_samples > 1:
            mc_losses = tf.stack(
                [tf.stop_gradient(t) for t in loss_terms], axis=0
            )
            mc_var_step = float(tf.math.reduce_variance(mc_losses).numpy())
        else:
            mc_var_step = 0.0
        mc_var_window.append(mc_var_step)

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

        step_sec = float(time.perf_counter() - step_t0)
        step_time_hist.append(step_sec)
        if log_every > 0 and (
            step == 0 or (step + 1) % log_every == 0 or (step + 1) == steps
        ):
            loss_step = float(loss.numpy())
            rmse_step = float(rmse_step_tensor.numpy())
            ess_step = float(ess_step_tensor.numpy())
            phi_rmse_step_val = float(phi_rmse_step.numpy())
            grad_raw_step = float(raw_grad_norm.numpy())
            grad_step = float(grad_norm.numpy())
            avg_step_sec = float(np.mean(step_time_hist))
            mc_var_avg = float(np.mean(mc_var_window)) if mc_var_window else float("nan")
            gpu_mem_cur_mb, gpu_mem_peak_mb = _current_gpu_memory_mb()
            _log(
                f"{_seed_tag(seed, method)} "
                f"step {step + 1}/{steps} "
                f"loss={loss_step:.5f} "
                f"rmse={rmse_step:.5f} "
                f"phi={phi_rmse_step_val:.5f} "
                f"ess={ess_step:.2f} "
                f"g={grad_step:.4f} "
                f"g_raw={grad_raw_step:.4f} "
                f"mc_var={mc_var_avg:.4e} "
                f"mc={mc_samples} "
                f"gpu_mem={_format_memory_mb(gpu_mem_cur_mb)} "
                f"gpu_peak={_format_memory_mb(gpu_mem_peak_mb)} "
                f"step_avg={avg_step_sec:.3f}s"
            )
    runtime_train_sec = float(time.perf_counter() - train_t0)
    runtime_train_per_step_sec = float(runtime_train_sec / max(steps, 1))
    _, gpu_peak_train_mb = _current_gpu_memory_mb()
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
        "grad_raw_init": float(grad_stats["grad_raw_init"]),
        "grad_raw_final": float(grad_stats["grad_raw_final"]),
        "runtime_train_sec": runtime_train_sec,
        "runtime_train_per_step_sec": runtime_train_per_step_sec,
        "gpu_peak_train_mb": float(gpu_peak_train_mb) if gpu_peak_train_mb is not None else float("nan"),
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
    runtime_pretrain_sec = 0.0
    if method == "transformer":
        t_pretrain_start = time.perf_counter()
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
        runtime_pretrain_sec = float(time.perf_counter() - t_pretrain_start)
        freeze_resampler = bool(method_cfg.get("freeze_resampler", True))
        dpf.resampler_net.trainable = not freeze_resampler
        if pretrained_weights_used is not None:
            _log(
                f"{_seed_tag(seed, method)} "
                f"loaded pretrained resampler weights: {pretrained_weights_used} "
                f"(pretrain took {runtime_pretrain_sec:.1f}s)"
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
            f"{_seed_tag(seed, method)} "
            f"proposal_lr={proposal_lr:.3e} resampler_lr={resampler_lr:.3e}"
        )
    else:
        _log(
            f"{_seed_tag(seed, method)} "
            f"proposal_lr={proposal_lr:.3e}"
        )

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

    # Test-split evaluation (no gradient computation, held-out data).
    t_eval_test = time.perf_counter()
    test_eval = _evaluate_filter(dpf, seed_ctx.y_test, seed_ctx.x_test, resample=resample)
    runtime_eval_test_sec = float(time.perf_counter() - t_eval_test)

    return {
        "init_eval": init_eval,
        "final_eval": final_eval,
        "test_eval": test_eval,
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
        "grad_raw_init": float(train_out["grad_raw_init"]),
        "grad_raw_final": float(train_out["grad_raw_final"]),
        "gpu_peak_train_mb": float(train_out["gpu_peak_train_mb"]),
        "runtime_pretrain_sec": runtime_pretrain_sec,
        "runtime_train_sec": float(train_out["runtime_train_sec"]),
        "runtime_train_per_step_sec": float(train_out["runtime_train_per_step_sec"]),
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
        "runtime_eval_test_sec": runtime_eval_test_sec,
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


def _result_test_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract held-out test-split metrics from a seed result for table display.

    Gradient / phi columns are identical between splits (they come from the
    training run) so they are copied from the existing ``_final`` keys.
    """
    return {
        "loss": float(result["loss_test"]),
        "nll": float(result["nll_test"]),
        "rmse": float(result["rmse_test"]),
        "rmse_phi": float(result["rmse_phi_final"]),
        "ess_mean": float(result["ess_mean_test"]),
        "grad_snr": float(result["grad_snr_final"]),
        "grad_var": float(result["grad_var_final"]),
        "grad_raw": float(result["grad_raw_final"]),
        "train_sec": float(result["runtime_train_sec"]),
    }


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
    x_true = seed_ctx.x_true  # train split
    y_obs = seed_ctx.y_obs    # train split
    _reset_gpu_peak()
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
    test_eval = method_out["test_eval"]
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
    grad_raw_init = float(method_out["grad_raw_init"])
    grad_raw_final = float(method_out["grad_raw_final"])
    gpu_peak_train_mb = float(method_out["gpu_peak_train_mb"])
    runtime_pretrain_sec = float(method_out["runtime_pretrain_sec"])
    runtime_train_sec = float(method_out["runtime_train_sec"])
    runtime_train_per_step_sec = float(method_out["runtime_train_per_step_sec"])
    runtime_eval_init_sec = float(method_out["runtime_eval_init_sec"])
    runtime_eval_final_sec = float(method_out["runtime_eval_final_sec"])
    runtime_eval_test_sec = float(method_out.get("runtime_eval_test_sec", float("nan")))

    result = {
        "seed": seed,
        "method": method,
        # --- train split metrics ---
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
        # --- test split metrics ---
        "loss_test": float(test_eval["loss"]),
        "nll_test": float(test_eval["nll"]),
        "rmse_test": float(test_eval["rmse"]),
        "ess_mean_test": float(test_eval["ess_mean"]),
        "resample_rate_test": float(test_eval["resample_rate"]),
        # --- gradient / training stats (shared, not split-specific) ---
        "grad_snr_init": grad_snr_init,
        "grad_snr_final": grad_snr_final,
        "grad_var_init": grad_var_init,
        "grad_var_final": grad_var_final,
        "grad_raw_init": grad_raw_init,
        "grad_raw_final": grad_raw_final,
        "gpu_peak_train_mb": gpu_peak_train_mb,
        "runtime_pretrain_sec": runtime_pretrain_sec,
        "runtime_train_sec": runtime_train_sec,
        "runtime_train_per_step_sec": runtime_train_per_step_sec,
        "runtime_eval_init_sec": runtime_eval_init_sec,
        "runtime_eval_final_sec": runtime_eval_final_sec,
        "runtime_eval_test_sec": runtime_eval_test_sec,
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
        "x_test": np.asarray(seed_ctx.x_test),
        "y_test": np.asarray(seed_ctx.y_test),
        "mean_final": np.asarray(final_eval["mean"]),
        "mean_test": np.asarray(test_eval["mean"]),
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
        x_test=result["x_test"],
        y_test=result["y_test"],
        mean_final=result["mean_final"],
        mean_test=result["mean_test"],
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
    pretrain_sec = float(result.get("runtime_pretrain_sec", 0.0))
    pretrain_part = f" | pretrain_sec {pretrain_sec:.3f}" if pretrain_sec > 0.0 else ""
    _log(
        f"[{tag}] seed={seed} method={method} "
        f"loss {result['loss_init']:.6f}->{result['loss_final']:.6f} (test {result['loss_test']:.6f}) | "
        f"nll {result['nll_init']:.6f}->{result['nll_final']:.6f} (test {result['nll_test']:.6f}) | "
        f"rmse {result['rmse_init']:.6f}->{result['rmse_final']:.6f} (test {result['rmse_test']:.6f}) | "
        f"phi_rmse {result['rmse_phi_init']:.6f}->{result['rmse_phi_final']:.6f} | "
        f"ess_mean {result['ess_mean_init']:.2f}->{result['ess_mean_final']:.2f} (test {result['ess_mean_test']:.2f}) | "
        f"grad_snr {result['grad_snr_init']:.3f}->{result['grad_snr_final']:.3f} | "
        f"grad_var {result['grad_var_init']:.3e}->{result['grad_var_final']:.3e} | "
        f"train_sec {result['runtime_train_sec']:.3f}"
        f"{pretrain_part}"
    )


def _build_method_summary(
    *,
    per_seed: Dict[str, List[Dict[str, Any]]],
    methods: List[str],
) -> Dict[str, Dict[str, float]]:
    return {
        method: {
            # train-split aggregates
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
            # test-split aggregates
            "loss_test_mean": _safe_nanmean([r["loss_test"] for r in per_seed[method]]),
            "nll_test_mean": _safe_nanmean([r["nll_test"] for r in per_seed[method]]),
            "rmse_test_mean": _safe_nanmean([r["rmse_test"] for r in per_seed[method]]),
            "ess_mean_test_mean": _safe_nanmean([r["ess_mean_test"] for r in per_seed[method]]),
            "resample_rate_test_mean": _safe_nanmean(
                [r["resample_rate_test"] for r in per_seed[method]]
            ),
            # gradient / training stats
            "grad_snr_init_mean": _safe_nanmean([r["grad_snr_init"] for r in per_seed[method]]),
            "grad_snr_final_mean": _safe_nanmean([r["grad_snr_final"] for r in per_seed[method]]),
            "grad_var_init_mean": _safe_nanmean([r["grad_var_init"] for r in per_seed[method]]),
            "grad_var_final_mean": _safe_nanmean([r["grad_var_final"] for r in per_seed[method]]),
            "grad_raw_init_mean": _safe_nanmean([r["grad_raw_init"] for r in per_seed[method]]),
            "grad_raw_final_mean": _safe_nanmean([r["grad_raw_final"] for r in per_seed[method]]),
            "gpu_peak_train_mb_mean": _safe_nanmean([r["gpu_peak_train_mb"] for r in per_seed[method]]),
            "runtime_pretrain_sec_mean": _safe_nanmean(
                [r["runtime_pretrain_sec"] for r in per_seed[method]]
            ),
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
            "runtime_eval_test_sec_mean": _safe_nanmean(
                [r.get("runtime_eval_test_sec", float("nan")) for r in per_seed[method]]
            ),
        }
        for method in methods
    }


def _run_all_methods_for_seed(
    seed: int,
    methods: List[str],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dpf_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    experiment_name: str,
    out_root: str,
    save_traces: bool,
) -> Dict[str, Any]:
    """Top-level worker: run every method for one seed in a subprocess.

    Must be a module-level function so ProcessPoolExecutor can pickle it.
    Each subprocess gets its own TF global state (no tf.random.set_seed races)
    and builds its own fit_ssm via _run_single_seed -> _build_seed_context.
    Because the same seed value is used, sim data (x_true, y_obs) is identical
    across methods — the common-random-number property is preserved.
    """
    import sys as _sys

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"

    # The NUMA / GPU-device C++ messages bypass Python logging and write
    # directly to fd 2.  Redirect fd 2 to /dev/null while TF initialises
    # the GPU, then restore it so that real stderr still works afterwards.
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_fd2 = os.dup(2)
    _saved_stderr = _sys.stderr
    os.dup2(_devnull_fd, 2)
    _sys.stderr = open(os.devnull, "w")
    os.close(_devnull_fd)
    try:
        import tensorflow as _tf
        from absl import logging as _absl_logging
        _absl_logging.set_verbosity(_absl_logging.ERROR)
        _absl_logging.set_stderrthreshold("error")
        for _gpu in _tf.config.list_physical_devices("GPU"):
            _tf.config.experimental.set_memory_growth(_gpu, True)
        _tf.constant(0)  # trigger GPU context init now, while fd 2 is muted
    finally:
        _sys.stderr.close()
        _sys.stderr = _saved_stderr
        os.dup2(_saved_fd2, 2)
        os.close(_saved_fd2)

    out_root_path = Path(out_root)
    _init_log_file(out_root_path / f"seed{seed}.log")
    seed_results: Dict[str, Dict[str, Any]] = {}
    _log(f"{_seed_tag(seed)} started with methods={methods}")
    for method in methods:
        _log(f"{_seed_tag(seed, method)} starting")
        result = _run_single_seed(
            seed,
            exp_cfg,
            model_cfg,
            train_cfg,
            dpf_cfg,
            proposal_cfg,
            method=method,
            experiment_name=experiment_name,
            experiment_dir=out_root_path,
        )
        seed_results[method] = result
        if save_traces:
            _persist_trace(
                out_root=out_root_path,
                experiment_name=experiment_name,
                seed=seed,
                method=method,
                result=result,
            )
        _log(f"{_seed_tag(seed, method)} finished")
    _log(f"{_seed_tag(seed)} all methods finished")
    return {"seed": seed, "results": seed_results}


def run_lgssm_dpf_backprop(
    cfg: Dict[str, Any],
    config_path: Path,
    tag: str,
    *,
    exp_name: str | None = None,
    num_workers: int = 1,
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
    _init_log_file(out_root / "run.log")
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]
    save_traces = bool(exp_cfg.get("save_traces", True))
    methods = _resolve_dpf_methods(dpf_cfg)

    n_workers = max(1, int(num_workers))
    _log(f"[{tag}] config={config_path}")
    _log(f"[{tag}] exp_name={experiment_name}")
    _log(f"[{tag}] output_root={out_root}")
    _log(f"[{tag}] seeds={seeds}")
    _log(f"[{tag}] methods={methods}")
    _log(f"[{tag}] num_workers={n_workers}")

    per_seed: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}
    metrics_across_seeds: Dict[str, List[Dict[str, Any]]] = {
        method: [] for method in methods
    }

    def _collect_seed_results(seed: int, seed_result_map: Dict[str, Dict[str, Any]]) -> None:
        """Accumulate results for one seed into per_seed / metrics_across_seeds."""
        per_seed_train: Dict[str, Dict[str, Any]] = {}
        per_seed_test: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            result = seed_result_map[method]
            per_seed[method].append(result)
            final_metrics = _result_stage_metrics(result, "final")
            per_seed_train[method] = final_metrics
            per_seed_test[method] = _result_test_metrics(result)
            metrics_across_seeds[method].append(final_metrics)
            _log_method_result(tag=tag, seed=seed, method=method, result=result)

        print_separator(f"{tag} seed{seed} TRAIN summary")
        print_method_summary_table(
            per_seed_train,
            method_order=tuple(methods),
            keys=SUMMARY_KEYS,
        )
        print_separator(f"{tag} seed{seed} TRAIN compare")
        print_metrics_compare(
            per_seed_train,
            method_order=tuple(methods),
        )
        print_separator(f"{tag} seed{seed} TEST summary")
        print_method_summary_table(
            per_seed_test,
            method_order=tuple(methods),
            keys=TEST_SUMMARY_KEYS,
        )
        print_separator(f"{tag} seed{seed} TEST compare")
        print_metrics_compare(
            per_seed_test,
            method_order=tuple(methods),
        )

    if n_workers <= 1:
        # Serial path.
        for seed in seeds:
            seed_result_map: Dict[str, Dict[str, Any]] = {}
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
                seed_result_map[method] = result
                if save_traces:
                    _persist_trace(
                        out_root=out_root,
                        experiment_name=experiment_name,
                        seed=seed,
                        method=method,
                        result=result,
                    )
            _collect_seed_results(seed, seed_result_map)
    else:
        # Parallel path: one subprocess per seed, each seed runs all methods
        # sequentially inside its own process.
        # Each process has its own TF global state, so tf.random.set_seed()
        # calls cannot race across seeds.
        # Force all child processes to use memory-growth mode so n_workers
        # simultaneous CUDA contexts don't collectively exceed GPU memory.
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        _log(f"[{tag}] launching {len(seeds)} seeds across {n_workers} workers ...")
        worker_kwargs = dict(
            methods=methods,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dpf_cfg=dpf_cfg,
            proposal_cfg=proposal_cfg,
            experiment_name=experiment_name,
            out_root=str(out_root),
            save_traces=save_traces,
        )
        seed_outputs: Dict[int, Dict[str, Dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_all_methods_for_seed, seed, **worker_kwargs): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    worker_out = fut.result()
                except Exception as exc:
                    _log(f"[{tag}] seed={seed} FAILED: {exc}")
                    raise
                seed_outputs[seed] = worker_out["results"]
                _log(f"[{tag}] seed={seed} finished.")

        # Reassemble in original seed order for reproducible summaries.
        for seed in seeds:
            _collect_seed_results(seed, seed_outputs[seed])

    method_summary_new = _build_method_summary(
        per_seed=per_seed,
        methods=methods,
    )
    summary_path = out_root / "summary.json"
    # Merge with existing summary so that results from previous runs (different
    # methods) are preserved; only the methods in the current run are updated.
    import json as _json
    existing_methods: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            existing = _json.loads(summary_path.read_text(encoding="utf-8"))
            existing_methods = existing.get("methods", {})
        except Exception as exc:
            _log(
                f"[{tag}] WARNING: failed to read existing summary.json, "
                f"previous methods may be lost: {exc}"
            )
    if existing_methods:
        _log(f"[{tag}] merging with existing methods: {sorted(existing_methods.keys())}")
    merged_methods = {**existing_methods, **method_summary_new}
    summary = {
        "experiment_name": experiment_name,
        "num_seeds": int(len(seeds)),
        "methods": merged_methods,
    }
    save_json(summary_path, summary)
    _log(f"[{tag}] summary saved: {summary_path} (methods: {sorted(merged_methods.keys())})")
    for method in methods:
        method_summary = method_summary_new[method]
        pretrain_sec = method_summary.get("runtime_pretrain_sec_mean", 0.0)
        pretrain_part = f", mean pretrain_sec {pretrain_sec:.3f}" if pretrain_sec > 0.0 else ""
        _log(
            f"[{tag}] {method} "
            f"TRAIN: mean nll {method_summary['nll_init_mean']:.6f}->{method_summary['nll_final_mean']:.6f}, "
            f"mean rmse {method_summary['rmse_init_mean']:.6f}->{method_summary['rmse_final_mean']:.6f} | "
            f"TEST: mean nll {method_summary['nll_test_mean']:.6f}, "
            f"mean rmse {method_summary['rmse_test_mean']:.6f} | "
            f"mean phi_rmse {method_summary['rmse_phi_init_mean']:.6f}->{method_summary['rmse_phi_final_mean']:.6f}, "
            f"mean ess {method_summary['ess_mean_init_mean']:.2f}->{method_summary['ess_mean_final_mean']:.2f}, "
            f"mean grad_snr {method_summary['grad_snr_init_mean']:.3f}->{method_summary['grad_snr_final_mean']:.3f}, "
            f"mean grad_var {method_summary['grad_var_init_mean']:.3e}->{method_summary['grad_var_final_mean']:.3e}, "
            f"mean train_sec {method_summary['runtime_train_sec_mean']:.3f}"
            f"{pretrain_part}"
        )

    # Build per-method dicts for the final avg tables.
    merged_all_methods = merged_methods
    _TRAIN_KEY_MAP = {
        "loss": "loss_final_mean",
        "nll": "nll_final_mean",
        "rmse": "rmse_final_mean",
        "rmse_phi": "rmse_phi_final_mean",
        "ess_mean": "ess_mean_final_mean",
        "grad_snr": "grad_snr_final_mean",
        "grad_var": "grad_var_final_mean",
        "grad_raw": "grad_raw_final_mean",
        "gpu_peak_mb": "gpu_peak_train_mb_mean",
        "train_sec": "runtime_train_sec_mean",
    }
    _TEST_KEY_MAP = {
        "loss": "loss_test_mean",
        "nll": "nll_test_mean",
        "rmse": "rmse_test_mean",
        "ess_mean": "ess_mean_test_mean",
    }
    train_method_metrics: Dict[str, Dict[str, float]] = {}
    test_method_metrics: Dict[str, Dict[str, float]] = {}
    for m, ms in merged_all_methods.items():
        train_method_metrics[m] = {sk: ms.get(lk, float("nan")) for sk, lk in _TRAIN_KEY_MAP.items()}
        test_method_metrics[m] = {sk: ms.get(lk, float("nan")) for sk, lk in _TEST_KEY_MAP.items()}

    print_separator(f"{tag} avg TRAIN summary")
    print_method_summary_table(
        train_method_metrics,
        method_order=tuple(merged_all_methods.keys()),
        keys=SUMMARY_KEYS,
    )
    print_separator(f"{tag} avg TRAIN compare")
    print_metrics_compare(
        train_method_metrics,
        method_order=tuple(merged_all_methods.keys()),
    )
    print_separator(f"{tag} avg TEST summary")
    print_method_summary_table(
        test_method_metrics,
        method_order=tuple(merged_all_methods.keys()),
        keys=TEST_SUMMARY_KEYS,
    )
    print_separator(f"{tag} avg TEST compare")
    print_metrics_compare(
        test_method_metrics,
        method_order=tuple(merged_all_methods.keys()),
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
