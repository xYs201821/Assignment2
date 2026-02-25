from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.exp3.exp3_lgssm_dpf import reset_filter_rng
from experiments.exp3.exp3_model import LinearGaussianProposalPhi, build_exp3_linear_ssm_pair
from experiments.common.exp_utils import as_list, cfg_section, ensure_dir, load_config, save_json, set_seed
from src.filters.dpf import DPFBase
from src.filters.particle_transformer import ParticleTransformerResampler
from src.ssm import LinearGaussianSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp3_transformer_pretrain_config.yaml")


class StandardResamplingDPF(DPFBase):
    """DPF with standard systematic resampling."""

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor,
                      training: bool | None = None):
        log_w = tf.convert_to_tensor(log_w, dtype=tf.float32)
        w = tf.exp(log_w)
        w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))

        parent_indices = self.systematic_resample(w, self.ssm.rng)
        x_new = self.resample_particles(x, parent_indices)

        log_uniform = -tf.math.log(tf.cast(self.num_particles, log_w.dtype))
        log_w_new = log_uniform * tf.ones_like(log_w)
        return x_new, log_w_new, parent_indices


def _collect_resampler_examples(
    fit_ssm: LinearGaussianSSM,
    y_obs: tf.Tensor,
    proposal: LinearGaussianProposalPhi,
    num_particles: int,
    ess_threshold: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pf = StandardResamplingDPF(
        fit_ssm,
        num_particles=int(num_particles),
        ess_threshold=float(ess_threshold),
        resample="always",
        proposal=proposal,
    )
    reset_filter_rng(fit_ssm, seed + 21)
    x_seq, _, diagnostics, _ = pf.filter(y_obs, resample="always")

    x_in = tf.convert_to_tensor(diagnostics["x_pre"], dtype=tf.float32)  # [B, T, N, dx]
    w_in = tf.exp(tf.convert_to_tensor(diagnostics["log_w_pre"], dtype=tf.float32))  # [B, T, N]
    x_out = tf.convert_to_tensor(x_seq, dtype=tf.float32)  # [B, T, N, dx]

    # Ensure the supervised pairs are formed from valid normalized weights.
    w_in = tf.math.divide_no_nan(w_in, tf.reduce_sum(w_in, axis=-1, keepdims=True))
    tf.debugging.assert_all_finite(w_in, "w_pre contains NaN/Inf before log transform.")

    eps = tf.constant(1e-12, dtype=tf.float32)
    log_w_in = tf.math.log(tf.clip_by_value(w_in, eps, 1.0))

    x_in_f = tf.reshape(x_in, [-1, num_particles, tf.shape(x_in)[-1]])
    log_w_f = tf.reshape(log_w_in, [-1, num_particles])
    x_out_f = tf.reshape(x_out, [-1, num_particles, tf.shape(x_out)[-1]])
    return np.asarray(x_in_f), np.asarray(log_w_f), np.asarray(x_out_f)


def _energy_distance_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Unbiased mini-batch energy distance for particle sets."""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    eps = tf.constant(1e-12, dtype=x.dtype)
    n = tf.shape(x)[1]
    n_f = tf.cast(n, x.dtype)

    xx_sq = tf.reduce_sum(tf.square(x[:, :, None, :] - x[:, None, :, :]), axis=-1)
    yy_sq = tf.reduce_sum(tf.square(y[:, :, None, :] - y[:, None, :, :]), axis=-1)
    xy_sq = tf.reduce_sum(tf.square(x[:, :, None, :] - y[:, None, :, :]), axis=-1)

    xx_dist = tf.sqrt(xx_sq + eps)
    yy_dist = tf.sqrt(yy_sq + eps)
    xy_dist = tf.sqrt(xy_sq + eps)

    mask = tf.ones([n, n], dtype=x.dtype) - tf.eye(n, dtype=x.dtype)
    denom = n_f * tf.maximum(n_f - 1.0, 1.0)
    exx = tf.reduce_sum(xx_dist * mask[None, :, :], axis=[1, 2]) / denom
    eyy = tf.reduce_sum(yy_dist * mask[None, :, :], axis=[1, 2]) / denom
    exy = tf.reduce_mean(xy_dist, axis=[1, 2])

    loss = 2.0 * exy - exx - eyy
    return tf.reduce_mean(loss)


def _gmm_nll_loss(query: tf.Tensor, means: tf.Tensor, sigma: float) -> tf.Tensor:
    """Negative log-likelihood under an equally weighted isotropic Gaussian mixture."""
    query = tf.convert_to_tensor(query, dtype=tf.float32)  # [B, Nq, dx]
    means = tf.convert_to_tensor(means, dtype=tf.float32)  # [B, Nm, dx]

    sigma_t = tf.convert_to_tensor(float(sigma), dtype=query.dtype)
    sigma_t = tf.maximum(sigma_t, tf.constant(1e-6, dtype=query.dtype))
    sigma2 = tf.square(sigma_t)

    dx = tf.cast(tf.shape(query)[-1], query.dtype)
    nm = tf.cast(tf.shape(means)[1], query.dtype)
    log_2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=query.dtype))

    sq = tf.reduce_sum(tf.square(query[:, :, None, :] - means[:, None, :, :]), axis=-1)
    log_component = -0.5 * (dx * (log_2pi + tf.math.log(sigma2)) + sq / sigma2)
    log_prob = tf.reduce_logsumexp(log_component, axis=-1) - tf.math.log(tf.maximum(nm, 1.0))
    return -tf.reduce_mean(log_prob)


def _gmm_set_loss(x: tf.Tensor, y: tf.Tensor, sigma: float, symmetric: bool) -> tf.Tensor:
    """Set loss via GMM NLL, optionally symmetric in both directions."""
    loss_xy = _gmm_nll_loss(x, y, sigma=sigma)
    if not bool(symmetric):
        return loss_xy
    loss_yx = _gmm_nll_loss(y, x, sigma=sigma)
    return 0.5 * (loss_xy + loss_yx)


def _train_resampler_supervised(
    x_in: np.ndarray,
    log_w_in: np.ndarray,
    x_target: np.ndarray,
    *,
    d_model: int,
    hidden: int,
    num_heads: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dropout_rate: float,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    seed: int,
    loss_name: str = "energy",
    gmm_sigma: float = 0.25,
    gmm_symmetric: bool = True,
    loss_mix_alpha: float = 0.5,
) -> Tuple[ParticleTransformerResampler, List[float]]:
    num_particles = int(x_in.shape[1])
    layer = ParticleTransformerResampler(
        num_particles=num_particles,
        d_model=int(d_model),
        hidden=int(hidden),
        num_heads=int(num_heads),
        num_encoder_layers=int(num_encoder_layers),
        num_decoder_layers=int(num_decoder_layers),
        dropout_rate=float(dropout_rate),
    )
    x_np = np.asarray(x_in, dtype=np.float32)
    lw_np = np.asarray(log_w_in, dtype=np.float32)
    y_np = np.asarray(x_target, dtype=np.float32)
    _ = layer(
        tf.convert_to_tensor(x_np[:1], dtype=tf.float32),
        tf.convert_to_tensor(lw_np[:1], dtype=tf.float32),
        training=False,
    )

    opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
    rng = np.random.default_rng(seed + 33)
    num_samples = int(x_np.shape[0])

    loss_name_norm = str(loss_name).strip().lower()
    if loss_name_norm == "energy":
        loss_fn = lambda y_hat, y_b: _energy_distance_loss(y_hat, y_b)
    elif loss_name_norm == "gmm":
        loss_fn = lambda y_hat, y_b: _gmm_set_loss(
            y_hat,
            y_b,
            sigma=float(gmm_sigma),
            symmetric=bool(gmm_symmetric),
        )
    elif loss_name_norm in ("hybrid", "energy+gmm", "gmm+energy"):
        alpha = float(loss_mix_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        loss_fn = lambda y_hat, y_b: (
            alpha * _energy_distance_loss(y_hat, y_b)
            + (1.0 - alpha)
            * _gmm_set_loss(
                y_hat,
                y_b,
                sigma=float(gmm_sigma),
                symmetric=bool(gmm_symmetric),
            )
        )
    else:
        raise ValueError(
            "train.loss must be one of: 'energy', 'gmm', 'hybrid' (or aliases energy+gmm/gmm+energy)"
        )

    @tf.function(reduce_retracing=True)
    def train_step(x_b: tf.Tensor, lw_b: tf.Tensor, y_b: tf.Tensor):
        with tf.GradientTape() as tape:
            y_hat, _ = layer(x_b, lw_b, training=True)
            loss = loss_fn(y_hat, y_b)
        grads = tape.gradient(loss, layer.trainable_variables)
        valid = [(g, v) for g, v in zip(grads, layer.trainable_variables) if g is not None]
        if valid:
            opt.apply_gradients(valid)
        return loss

    hist: List[float] = []
    for step in range(int(steps)):
        k = min(int(batch_size), num_samples)
        idx = rng.integers(0, num_samples, size=k, endpoint=False)
        x_b = tf.convert_to_tensor(x_np[idx], dtype=tf.float32)
        lw_b = tf.convert_to_tensor(lw_np[idx], dtype=tf.float32)
        y_b = tf.convert_to_tensor(y_np[idx], dtype=tf.float32)

        loss = train_step(x_b, lw_b, y_b)
        hist.append(float(loss.numpy()))
        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0 or (step + 1) == steps):
            print(f"[pretrain] step {step + 1:4d}/{steps} loss={hist[-1]:.6f}")
    return layer, hist


def _save_resampler_weights(layer: ParticleTransformerResampler, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"w{i:03d}": w for i, w in enumerate(layer.get_weights())}
    np.savez_compressed(out_path, **payload)


def pretrain_transformer_resampler_for_seed(
    *,
    seed: int,
    model_cfg: Dict[str, Any],
    proposal_cfg: Dict[str, Any],
    num_particles: int,
    ess_threshold: float,
    d_model: int,
    hidden: int,
    num_heads: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dropout_rate: float,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    loss_name: str,
    gmm_sigma: float,
    gmm_symmetric: bool,
    loss_mix_alpha: float,
    sim_T: int,
    sim_batch_size: int,
    output_root: str | Path,
) -> Dict[str, Any]:
    """Run one-seed pretraining and save weights to output_root."""
    set_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    true_ssm, fit_ssm = build_exp3_linear_ssm_pair(model_cfg, int(seed), fit_trainable=False)
    _, y_obs = true_ssm.simulate(T=int(sim_T), shape=[int(sim_batch_size)])

    state_dim = int(fit_ssm.state_dim)
    obs_dim = int(fit_ssm.obs_dim)
    proposal = LinearGaussianProposalPhi(
        A=fit_ssm.A,
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_delta=proposal_cfg.get("init_delta", 1.5),
        init_gamma=proposal_cfg.get("init_gamma", 1.5),
        init_noise_std=float(proposal_cfg.get("init_noise_std", 0.05)),
        rng=rng,
    )

    print(f"[seed={seed}] collect standard-resampler examples")
    x_in, lw_in, x_tgt = _collect_resampler_examples(
        fit_ssm=fit_ssm,
        y_obs=y_obs,
        proposal=proposal,
        num_particles=int(num_particles),
        ess_threshold=float(ess_threshold),
        seed=int(seed),
    )
    print(f"[seed={seed}] collected {x_in.shape[0]} examples with N={x_in.shape[1]}, dx={x_in.shape[2]}")

    print(f"[seed={seed}] pretrain particle transformer resampler")
    resampler, loss_hist = _train_resampler_supervised(
        x_in,
        lw_in,
        x_tgt,
        d_model=int(d_model),
        hidden=int(hidden),
        num_heads=int(num_heads),
        num_encoder_layers=int(num_encoder_layers),
        num_decoder_layers=int(num_decoder_layers),
        dropout_rate=float(dropout_rate),
        steps=int(steps),
        batch_size=int(batch_size),
        lr=float(lr),
        log_every=int(log_every),
        seed=int(seed),
        loss_name=str(loss_name),
        gmm_sigma=float(gmm_sigma),
        gmm_symmetric=bool(gmm_symmetric),
        loss_mix_alpha=float(loss_mix_alpha),
    )

    out_root = Path(output_root)
    ensure_dir(out_root)
    weights_path = out_root / f"seed{seed}_resampler.weights.npz"
    _save_resampler_weights(resampler, weights_path)

    return {
        "seed": int(seed),
        "examples": int(x_in.shape[0]),
        "resampler_loss_init": float(loss_hist[0]) if loss_hist else float("nan"),
        "resampler_loss_final": float(loss_hist[-1]) if loss_hist else float("nan"),
        "weights_path": str(weights_path),
    }


def _parse_args(default_config: Path = DEFAULT_CONFIG_PATH) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain particle transformer resampler for exp3.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--seed", dest="seeds", type=int, action="append", default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--steps-train", type=int, default=None)
    return parser.parse_args()


def run_experiment(cfg: Dict[str, Any], config_path: Path, tag: str) -> None:
    exp_cfg = cfg_section(cfg, "experiment")
    model_cfg = cfg_section(cfg, "model")
    proposal_cfg = cfg_section(cfg, "proposal")
    collect_cfg = cfg_section(cfg, "collect")
    train_cfg = cfg_section(cfg, "train")

    out_root = Path(exp_cfg.get("output_root", f"results/{tag}"))
    ensure_dir(out_root)
    seeds = [int(s) for s in as_list(exp_cfg.get("seeds", [0]))]

    print(f"[{tag}] config={config_path}")
    print(f"[{tag}] seeds={seeds}")

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = pretrain_transformer_resampler_for_seed(
            seed=seed,
            model_cfg=model_cfg,
            proposal_cfg=proposal_cfg,
            num_particles=int(collect_cfg.get("num_particles", 25)),
            ess_threshold=float(collect_cfg.get("ess_threshold", 0.5)),
            d_model=int(train_cfg.get("d_model", 128)),
            hidden=int(train_cfg.get("hidden", 128)),
            num_heads=int(train_cfg.get("num_heads", 4)),
            num_encoder_layers=int(train_cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(train_cfg.get("num_decoder_layers", 1)),
            dropout_rate=float(train_cfg.get("dropout_rate", 0.0)),
            steps=int(train_cfg.get("steps", 300)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            lr=float(train_cfg.get("lr", 1e-3)),
            log_every=int(train_cfg.get("log_every", 50)),
            loss_name=str(train_cfg.get("loss", "energy")),
            gmm_sigma=float(train_cfg.get("gmm_sigma", 0.25)),
            gmm_symmetric=bool(train_cfg.get("gmm_symmetric", True)),
            loss_mix_alpha=float(train_cfg.get("loss_mix_alpha", 0.5)),
            sim_T=int(exp_cfg.get("T", 80)),
            sim_batch_size=int(exp_cfg.get("batch_size", 100)),
            output_root=out_root,
        )
        results.append(result)
        print(
            f"[{tag}] seed={seed} "
            f"resampler {result['resampler_loss_init']:.4f}->{result['resampler_loss_final']:.4f} "
            f"| weights={result['weights_path']}"
        )

    summary = {
        "num_seeds": len(results),
        "results": results,
        "mean": {
            "resampler_loss_final": float(np.mean([r["resampler_loss_final"] for r in results]))
            if results
            else float("nan")
        },
    }
    save_json(out_root / "summary.json", summary)
    print(f"[{tag}] summary saved: {out_root / 'summary.json'}")


def main() -> None:
    args = _parse_args(DEFAULT_CONFIG_PATH)
    cfg = load_config(args.config, [])
    if args.seeds:
        cfg.setdefault("experiment", {})["seeds"] = args.seeds
    if args.output_root is not None:
        cfg.setdefault("experiment", {})["output_root"] = args.output_root
    if args.steps_train is not None:
        cfg.setdefault("train", {})["steps"] = int(args.steps_train)

    run_experiment(cfg=cfg, config_path=args.config, tag="exp3_transformer_pretrain")


if __name__ == "__main__":
    main()
