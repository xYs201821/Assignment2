from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import sys
from pathlib import Path

# Comment this out if you want TensorFlow INFO logs again.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def _preparse_device(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    args, _ = parser.parse_known_args(argv)
    return str(args.device).strip().lower()


_PRESELECTED_DEVICE = _preparse_device(sys.argv[1:])

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("WARNING")
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.WARNING)
    absl_logging.set_stderrthreshold("warning")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_tf_device(device: str) -> str:
    selected = str(device).strip().lower()
    gpus = tf.config.list_physical_devices("GPU")
    if selected == "auto":
        return "gpu" if gpus else "cpu"
    if selected == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to switch TensorFlow to CPU before runtime initialization."
            ) from exc
        return "cpu"
    if selected == "gpu":
        if not gpus:
            raise ValueError("Requested --device gpu, but no GPU is available.")
        return "gpu"
    raise ValueError("device must be one of {'auto', 'cpu', 'gpu'}.")


_ACTIVE_DEVICE = _configure_tf_device(_PRESELECTED_DEVICE)

from experiments.common.exp_utils import load_config
from experiments.hmc.hmc_runner import run_hmc, run_nuts
from experiments.hmc.pmmh_runner import run_pmmh
from src.ssm.ADH_NonlinearSSM import ADHNonlinearSSM

DEFAULT_CONFIG_PATH = Path(__file__).with_name("exp_hmc_config.yaml")


class _TeeTextIO(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _build_log_path(out_dir: Path, sampler: str, cfg: dict, T: int) -> Path:
    return out_dir / (
        f"{sampler}_{cfg['inner_pf']}_{cfg['proposal_kind']}_"
        f"{T}_{cfg['num_particles']}_{cfg['num_steps']}_log.txt"
    )


def _build_tensorboard_run_dir(tb_root: Path, sampler: str, cfg: dict, T: int) -> Path:
    return tb_root / (
        f"{sampler}_{cfg['inner_pf']}_{cfg['proposal_kind']}_"
        f"{T}_{cfg['num_particles']}_{cfg['num_steps']}"
    )


@contextlib.contextmanager
def _mirror_python_loggers(log_file: io.TextIOBase):
    handler = logging.StreamHandler(log_file)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    loggers = [tf.get_logger(), logging.getLogger("absl")]
    attached = []
    try:
        for logger in loggers:
            logger.addHandler(handler)
            attached.append(logger)
        yield
    finally:
        for logger in attached:
            logger.removeHandler(handler)
        handler.flush()


def _make_tensorboard_progress_logger(
    writer: tf.summary.SummaryWriter,
    *,
    flush_every: int,
):
    flush_every = max(1, int(flush_every))

    def _log(step: int, metrics: dict[str, float], message: str | None = None) -> None:
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f"progress/{name}", float(value), step=step)
            if message is not None:
                tf.summary.text("progress/console", tf.convert_to_tensor(message), step=step)
        if step == 1 or step % flush_every == 0:
            writer.flush()

    return _log


def _write_tensorboard_run_header(
    writer: tf.summary.SummaryWriter,
    *,
    args: argparse.Namespace,
    sampler: str,
    T: int,
    cfg: dict,
    active_device: str,
    log_path: Path,
) -> None:
    meta = {
        "config": str(args.config),
        "sampler": sampler,
        "T": int(T),
        "requested_device": str(args.device),
        "active_device": str(active_device),
        "cfg": {key: value for key, value in cfg.items() if not callable(value)},
    }
    with writer.as_default():
        tf.summary.text("run/meta", tf.convert_to_tensor(json.dumps(meta, indent=2, sort_keys=True)), step=0)
        tf.summary.text("run/log_path", tf.convert_to_tensor(str(log_path)), step=0)
        tf.summary.scalar("config/T", float(T), step=0)
        tf.summary.scalar("config/num_steps", float(cfg["num_steps"]), step=0)
        tf.summary.scalar("config/num_particles", float(cfg["num_particles"]), step=0)
    writer.flush()


def _write_tensorboard_final_summary(
    writer: tf.summary.SummaryWriter,
    *,
    step: int,
    result: dict,
    sigma2_chain_post_thinned: np.ndarray,
    sigma_chain_post_thinned: np.ndarray,
    burnin: int,
    drop: int,
    data_path: Path,
    plot_path: Path,
    posterior_plot_path: Path,
    log_path: Path,
    tb_run_dir: Path,
) -> None:
    with writer.as_default():
        tf.summary.scalar("summary/accept_rate", float(result["accept_rate"]), step=step)
        tf.summary.scalar("summary/runtime_sec", float(result["runtime_sec"]), step=step)
        tf.summary.scalar("summary/burnin", float(burnin), step=step)
        tf.summary.scalar("summary/drop", float(drop), step=step)
        tf.summary.scalar("summary/kept_samples", float(sigma_chain_post_thinned.shape[0]), step=step)
        tf.summary.scalar("posterior/mean_sigma_v2", float(np.mean(sigma2_chain_post_thinned[:, 0])), step=step)
        tf.summary.scalar("posterior/mean_sigma_w2", float(np.mean(sigma2_chain_post_thinned[:, 1])), step=step)
        tf.summary.scalar("posterior/std_sigma_v2", float(np.std(sigma2_chain_post_thinned[:, 0])), step=step)
        tf.summary.scalar("posterior/std_sigma_w2", float(np.std(sigma2_chain_post_thinned[:, 1])), step=step)
        tf.summary.scalar("posterior/mean_sigma_v", float(np.mean(sigma_chain_post_thinned[:, 0])), step=step)
        tf.summary.scalar("posterior/mean_sigma_w", float(np.mean(sigma_chain_post_thinned[:, 1])), step=step)
        tf.summary.scalar("posterior/std_sigma_v", float(np.std(sigma_chain_post_thinned[:, 0])), step=step)
        tf.summary.scalar("posterior/std_sigma_w", float(np.std(sigma_chain_post_thinned[:, 1])), step=step)
        tf.summary.histogram("posterior/sigma_v2", sigma2_chain_post_thinned[:, 0], step=step)
        tf.summary.histogram("posterior/sigma_w2", sigma2_chain_post_thinned[:, 1], step=step)
        if "chain_ess" in result:
            chain_ess = np.asarray(result["chain_ess"], dtype=np.float64)
            tf.summary.scalar("summary/chain_ess_sigma_v2", float(chain_ess[0]), step=step)
            tf.summary.scalar("summary/chain_ess_sigma_w2", float(chain_ess[1]), step=step)
            tf.summary.scalar("summary/chain_ess_min", float(result["chain_ess_min"]), step=step)
        for key in (
            "step_size_final",
            "mean_leapfrogs",
            "max_depth_hit_rate",
            "divergence_rate",
            "pf_calls",
            "target_accept_prob",
        ):
            if key in result:
                tf.summary.scalar(f"summary/{key}", float(result[key]), step=step)
        if "num_leapfrog_steps" in result:
            tf.summary.scalar("summary/num_leapfrog_steps", float(result["num_leapfrog_steps"]), step=step)
        if "max_tree_depth" in result:
            tf.summary.scalar("summary/max_tree_depth", float(result["max_tree_depth"]), step=step)
        artifacts = {
            "result_npz": str(data_path),
            "summary_plot": str(plot_path),
            "posterior_plot": str(posterior_plot_path),
            "text_log": str(log_path),
            "tensorboard_dir": str(tb_run_dir),
        }
        tf.summary.text("run/artifacts", tf.convert_to_tensor(json.dumps(artifacts, indent=2, sort_keys=True)), step=step)
    writer.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PMMH/HMC/NUTS posterior inference for ADH nonlinear SSM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--sampler", type=str, choices=["hmc", "nuts", "pmmh"], default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--mcmc-seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--drop", type=int, default=None, help="Keep one sample every `drop` steps after burn-in.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default=_PRESELECTED_DEVICE,
        help="Execution device. 'auto' keeps TensorFlow default device selection.",
    )
    parser.add_argument(
        "--inner-pf",
        type=str,
        choices=["standard", "soft", "ot"],
        default=None,
    )
    parser.add_argument("--soft-lam", type=float, default=None, help="Soft resampling λ).")
    parser.add_argument("--proposal-kind", type=str, choices=["bootstrap", "ledh", "edh"], default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--num-lambda", type=int, default=None)
    parser.add_argument(
        "--proposal-std-v",
        type=float,
        default=None,
        help="PMMH proposal SD interpreted on sigma_v^2 scale; internally mapped to log-space.",
    )
    parser.add_argument(
        "--proposal-std-w",
        type=float,
        default=None,
        help="PMMH proposal SD interpreted on sigma_w^2 scale; internally mapped to log-space.",
    )
    parser.add_argument("--hmc-step-size", dest="hmc_step_size", type=float, default=None)
    parser.add_argument("--hmc-leapfrog-steps", dest="hmc_leapfrog_steps", type=int, default=None)
    parser.add_argument("--nuts-step-size", dest="nuts_step_size", type=float, default=None)
    parser.add_argument("--nuts-max-tree-depth", dest="nuts_max_tree_depth", type=int, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=None)
    parser.add_argument("--adaptation-rate", type=float, default=None)
    parser.add_argument("--adaptation-steps", type=int, default=None)
    parser.add_argument("--frozen-pf-seed", type=int, default=None)
    parser.add_argument("--ess-threshold", type=float, default=None)
    parser.add_argument("--resample", type=str, default=None)
    parser.add_argument("--ot-epsilon", type=float, default=None)
    parser.add_argument("--ot-num-iters", type=int, default=None)
    parser.add_argument("--ot-jitter", type=float, default=None)
    parser.add_argument("--prior-alpha", type=float, default=None)
    parser.add_argument("--prior-beta", type=float, default=None)
    parser.add_argument("--init-sigma-v2", type=float, default=None)
    parser.add_argument("--init-sigma-w2", type=float, default=None)
    parser.add_argument("--true-sigma-v2", type=float, default=None)
    parser.add_argument("--true-sigma-w2", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output tag. Default: {sampler}_T{T}_N{N}_S{steps}",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TensorBoard event logging.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log root. Default: <output_dir>/tensorboard",
    )
    parser.add_argument("--show", action="store_true", default=None)
    return parser.parse_args()


def _plot_summary(
    x_true: np.ndarray,
    y_obs: np.ndarray,
    sigma2_chain: np.ndarray,
    burnin: int,
    plot_path: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(x_true.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0, 0].plot(t, x_true, color="tab:blue", linewidth=1.1)
    axes[0, 0].set_title("latent x_t")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, linestyle=":")

    axes[0, 1].plot(t, y_obs, color="tab:orange", linewidth=1.1)
    axes[0, 1].set_title("observed y_t")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, linestyle=":")

    burnin = int(max(0, min(int(burnin), int(sigma2_chain.shape[0]))))
    sigma2_post = np.asarray(sigma2_chain[burnin:], dtype=np.float64)
    if sigma2_post.shape[0] == 0:
        sigma2_post = np.asarray(sigma2_chain, dtype=np.float64)
        trace_idx = np.arange(sigma2_post.shape[0])
    else:
        trace_idx = np.arange(burnin, burnin + sigma2_post.shape[0])

    axes[1, 0].plot(trace_idx, sigma2_post[:, 0], color="tab:green", linewidth=1.0)
    axes[1, 0].set_title(r"trace (post burn-in): $\sigma_V^2$")
    axes[1, 0].set_xlabel("MCMC iter")
    axes[1, 0].grid(True, linestyle=":")

    axes[1, 1].plot(trace_idx, sigma2_post[:, 1], color="tab:red", linewidth=1.0)
    axes[1, 1].set_title(r"trace (post burn-in): $\sigma_W^2$")
    axes[1, 1].set_xlabel("MCMC iter")
    axes[1, 1].grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_posterior_after_burnin(
    sigma_samples: np.ndarray,
    true_sigma_v: float,
    true_sigma_w: float,
    plot_path: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    samples = np.asarray(sigma_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sigma_samples must have shape [num_samples, 2].")
    if samples.shape[0] == 0:
        raise ValueError("sigma_samples is empty after burn-in/drop thinning.")
    sigma_v = samples[:, 0]
    sigma_w = samples[:, 1]
    bins = max(12, min(50, int(np.sqrt(max(20, samples.shape[0])))))

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    axes[0, 0].hist(sigma_v, bins=bins, histtype="step", color="k", linewidth=1.8)
    axes[0, 0].axvline(float(true_sigma_v), color="k", linestyle="--", linewidth=1.2)
    axes[0, 0].set_ylabel(r"$\sigma_V$")

    axes[0, 1].plot(sigma_w, sigma_v, "k+", markersize=6, alpha=0.75)
    axes[0, 1].set_ylabel(r"$\sigma_V$")

    axes[1, 0].plot(sigma_v, sigma_w, "k+", markersize=6, alpha=0.75)
    axes[1, 0].set_xlabel(r"$\sigma_V$")
    axes[1, 0].set_ylabel(r"$\sigma_W$")

    axes[1, 1].hist(sigma_w, bins=bins, histtype="step", color="k", linewidth=1.8)
    axes[1, 1].axvline(float(true_sigma_w), color="k", linestyle="--", linewidth=1.2)
    axes[1, 1].set_xlabel(r"$\sigma_W$")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _cv(cli_val, yaml_val, fallback):
    """CLI > yaml > fallback priority resolution."""
    if cli_val is not None:
        return cli_val
    if yaml_val is not None:
        return yaml_val
    return fallback

def _build_cfg(args, yaml_cfg: dict, sampler: str) -> dict:
    """Build the runner config dict from yaml + CLI overrides."""
    filter_cfg = yaml_cfg.get("filter", {})
    prior_cfg = yaml_cfg.get("prior", {})
    sampler_yaml = yaml_cfg.get(sampler, {})

    num_steps = int(_cv(args.num_steps, sampler_yaml.get("num_steps"), 10000))
    mcmc_seed = _cv(args.mcmc_seed, yaml_cfg.get("experiment", {}).get("mcmc_seed"), 0)

    shared = dict(
        num_steps      = num_steps,
        inner_pf       = str(_cv(args.inner_pf,      filter_cfg.get("inner_pf"),      "ot")),
        proposal_kind  = str(_cv(args.proposal_kind, filter_cfg.get("proposal_kind"), "bootstrap")),
        num_particles  = int(_cv(args.num_particles, filter_cfg.get("num_particles"),  1000)),
        num_lambda     = int(_cv(args.num_lambda,    filter_cfg.get("num_lambda"),     20)),
        ess_threshold  = float(_cv(args.ess_threshold, filter_cfg.get("ess_threshold"), 0.5)),
        soft_lam       = float(_cv(args.soft_lam,    filter_cfg.get("soft_lam"),      0.95)),
        ot_epsilon     = float(_cv(args.ot_epsilon,  filter_cfg.get("ot_epsilon"),    0.1)),
        ot_num_iters   = int(_cv(args.ot_num_iters,  filter_cfg.get("ot_num_iters"),  25)),
        ot_jitter      = float(_cv(args.ot_jitter,   filter_cfg.get("ot_jitter"),     1e-6)),
        prior_alpha    = float(_cv(args.prior_alpha, prior_cfg.get("alpha"),          0.01)),
        prior_beta     = float(_cv(args.prior_beta,  prior_cfg.get("beta"),           0.01)),
        init_sigma2_v  = float(_cv(args.init_sigma_v2, sampler_yaml.get("init_sigma_v2"), 10.0)),
        init_sigma2_w  = float(_cv(args.init_sigma_w2, sampler_yaml.get("init_sigma_w2"), 10.0)),
        seed           = int(mcmc_seed),
        proposal       = None,
        x0_mean        = 0.0,
        x0_var         = 5.0,
        t0             = 0.0,
        t0_var         = 1e-9,
        verbose        = True,
        print_every    = max(1, int(_cv(args.print_every, yaml_cfg.get("experiment", {}).get("print_every"), num_steps // 10))),
    )

    if sampler == "hmc":
        resample = str(_cv(args.resample, filter_cfg.get("resample"), "always"))
        burnin_raw = _cv(args.burnin, sampler_yaml.get("burnin"), None)
        return dict(
            **shared,
            resample            = resample,
            burnin              = None if burnin_raw is None else int(burnin_raw),
            step_size           = float(_cv(args.hmc_step_size, sampler_yaml.get("step_size"), 0.05)),
            num_leapfrog_steps  = int(_cv(args.hmc_leapfrog_steps, sampler_yaml.get("num_leapfrog_steps"), 5)),
            target_accept_prob  = float(_cv(args.target_accept_prob, sampler_yaml.get("target_accept_prob"), 0.6)),
            adaptation_rate     = float(_cv(args.adaptation_rate,    sampler_yaml.get("adaptation_rate"),    0.01)),
            adaptation_steps    = _cv(args.adaptation_steps, sampler_yaml.get("adaptation_steps"), None),
            frozen_pf_seed      = _cv(args.frozen_pf_seed,   sampler_yaml.get("frozen_pf_seed"),   None),
        )
    elif sampler == "nuts":
        resample = str(_cv(args.resample, filter_cfg.get("resample"), "always"))
        burnin_raw = _cv(args.burnin, sampler_yaml.get("burnin"), None)
        return dict(
            **shared,
            resample            = resample,
            burnin              = None if burnin_raw is None else int(burnin_raw),
            step_size           = float(_cv(args.nuts_step_size, sampler_yaml.get("step_size"), 0.05)),
            max_tree_depth      = int(_cv(args.nuts_max_tree_depth, sampler_yaml.get("max_tree_depth"), 10)),
            target_accept_prob  = float(_cv(args.target_accept_prob, sampler_yaml.get("target_accept_prob"), 0.6)),
            adaptation_rate     = float(_cv(args.adaptation_rate,    sampler_yaml.get("adaptation_rate"),    0.01)),
            adaptation_steps    = _cv(args.adaptation_steps, sampler_yaml.get("adaptation_steps"), None),
            frozen_pf_seed      = _cv(args.frozen_pf_seed,   sampler_yaml.get("frozen_pf_seed"),   None),
        )
    else:
        resample = str(_cv(args.resample, filter_cfg.get("resample"), "auto"))
        return dict(
            **shared,
            resample       = resample,
            proposal_std_v = float(_cv(args.proposal_std_v, sampler_yaml.get("proposal_std_v"), 0.15)),
            proposal_std_w = float(_cv(args.proposal_std_w, sampler_yaml.get("proposal_std_w"), 0.08)),
        )


def main() -> None:
    args = _parse_args()
    yaml_cfg = load_config(args.config, [])
    exp_cfg = yaml_cfg.get("experiment", {})
    true_cfg = yaml_cfg.get("true_params", {})

    sampler = str(_cv(args.sampler, exp_cfg.get("sampler"), "hmc")).strip().lower()
    T          = int(_cv(args.T,          exp_cfg.get("T"),         100))
    data_seed  = int(_cv(args.data_seed,  exp_cfg.get("data_seed"), 123))
    drop       = max(1, int(_cv(args.drop, exp_cfg.get("drop"),     10)))
    true_sv2   = float(_cv(args.true_sigma_v2, true_cfg.get("sigma_v2"), 10.0))
    true_sw2   = float(_cv(args.true_sigma_w2, true_cfg.get("sigma_w2"), 1.0))
    out_dir    = Path(_cv(args.output_dir, exp_cfg.get("output_dir"), "results/exp_hmc"))
    show       = bool(_cv(args.show,       exp_cfg.get("show"),      False))
    tag_override = args.tag or exp_cfg.get("tag")
    tensorboard_enabled = bool(_cv(args.tensorboard, exp_cfg.get("tensorboard"), True))

    cfg = _build_cfg(args, yaml_cfg, sampler)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag_override or f"{sampler}_T{T}_N{cfg['num_particles']}_S{cfg['num_steps']}"
    log_path = _build_log_path(out_dir, sampler, cfg, T)
    tb_root_cfg = _cv(args.tensorboard_dir, exp_cfg.get("tensorboard_dir"), None)
    tb_root = Path(tb_root_cfg) if tb_root_cfg is not None else out_dir / "tensorboard"
    tb_run_dir = _build_tensorboard_run_dir(tb_root, sampler, cfg, T) if tensorboard_enabled else None
    tb_writer = None
    progress_callback = None
    if tensorboard_enabled and tb_run_dir is not None:
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = tf.summary.create_file_writer(str(tb_run_dir))
        progress_callback = _make_tensorboard_progress_logger(
            tb_writer,
            flush_every=max(1, int(cfg["print_every"])),
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeTextIO(sys.stdout, log_file)
        tee_stderr = _TeeTextIO(sys.stderr, log_file)
        try:
            with _mirror_python_loggers(log_file):
                with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                    print(
                        f"[config] {args.config}"
                    )
                    print(f"[device] requested={args.device} active={_ACTIVE_DEVICE}")
                    print(f"[log] {log_path}")
                    if tb_run_dir is not None:
                        print(f"[tensorboard] {tb_run_dir}")
                    tf.random.set_seed(data_seed)

                    if tb_writer is not None and tb_run_dir is not None:
                        _write_tensorboard_run_header(
                            tb_writer,
                            args=args,
                            sampler=sampler,
                            T=T,
                            cfg=cfg,
                            active_device=_ACTIVE_DEVICE,
                            log_path=log_path,
                        )

                    true_ssm = ADHNonlinearSSM(
                        sigma_v=math.sqrt(true_sv2),
                        sigma_w=math.sqrt(true_sw2),
                        seed=data_seed,
                    )
                    x_true, y_obs = true_ssm.simulate(T=T, shape=[1])

                    if sampler == "hmc":
                        result = run_hmc(y_obs, cfg, progress_callback=progress_callback)
                    elif sampler == "nuts":
                        result = run_nuts(y_obs, cfg, progress_callback=progress_callback)
                    else:
                        result = run_pmmh(y_obs, cfg, progress_callback=progress_callback)
                    inner_pf = str(result.get("inner_pf", cfg["inner_pf"])).strip().lower()
                    if inner_pf == "standard":
                        print(
                            f"[inner] standard proposal={cfg['proposal_kind']} num_particles={cfg['num_particles']} "
                            f"num_lambda={cfg['num_lambda']} resample={cfg['resample']}"
                        )
                    elif inner_pf == "ot":
                        print(
                            f"[inner] ot proposal={cfg['proposal_kind']} num_particles={cfg['num_particles']} "
                            f"num_lambda={cfg['num_lambda']} resample={cfg['resample']} "
                            f"ot_eps={cfg['ot_epsilon']} ot_iters={cfg['ot_num_iters']}"
                        )

                    x_np = x_true.numpy()[0, :, 0]
                    y_np = y_obs.numpy()[0, :, 0]
                    sigma2_chain = np.asarray(result["sigma2_chain"], dtype=np.float64)
                    sigma_chain = np.sqrt(np.maximum(sigma2_chain, 0.0))
                    burnin = int(result["burnin"])
                    sigma_chain_post = sigma_chain[burnin:]
                    sigma_chain_post_thinned = sigma_chain_post[::drop]
                    sigma2_chain_post_thinned = np.square(sigma_chain_post_thinned)

                    plot_path = out_dir / f"{tag}_summary.png"
                    posterior_plot_path = out_dir / f"{tag}_posterior_after_burnin.png"
                    data_path = out_dir / f"{tag}_result.npz"
                    _plot_summary(
                        x_np,
                        y_np,
                        sigma2_chain,
                        burnin=burnin,
                        plot_path=plot_path,
                        show=show,
                    )
                    _plot_posterior_after_burnin(
                        sigma_samples=sigma_chain_post_thinned,
                        true_sigma_v=float(np.sqrt(true_sv2)),
                        true_sigma_w=float(np.sqrt(true_sw2)),
                        plot_path=posterior_plot_path,
                        show=show,
                    )

                    save_payload = {
                        "x_true": x_true.numpy(),
                        "y_obs": y_obs.numpy(),
                        "burnin": np.int32(burnin),
                        "drop": np.int32(drop),
                        "sigma2_chain": sigma2_chain,
                        "sigma_chain": sigma_chain,
                        "sigma_chain_post": sigma_chain_post,
                        "sigma_chain_post_thinned": sigma_chain_post_thinned,
                        "sigma2_chain_post_thinned": sigma2_chain_post_thinned,
                        "accept": np.asarray(result["accept"]),
                        "loglik_chain": np.asarray(result["loglik_chain"]),
                        "logpost_chain": np.asarray(result["logpost_chain"]),
                    }
                    if "log_sigma2_chain" in result:
                        save_payload["log_sigma2_chain"] = np.asarray(result["log_sigma2_chain"])
                    if "logtarget_chain" in result:
                        save_payload["logtarget_chain"] = np.asarray(result["logtarget_chain"])
                    if "leapfrogs_chain" in result:
                        save_payload["leapfrogs_chain"] = np.asarray(result["leapfrogs_chain"])
                    if "step_size_final" in result:
                        save_payload["step_size_final"] = np.float64(result["step_size_final"])
                    if "num_leapfrog_steps" in result:
                        save_payload["num_leapfrog_steps"] = np.int32(result["num_leapfrog_steps"])
                    if "max_tree_depth" in result:
                        save_payload["max_tree_depth"] = np.int32(result["max_tree_depth"])
                    if "mean_leapfrogs" in result:
                        save_payload["mean_leapfrogs"] = np.float64(result["mean_leapfrogs"])
                    if "max_depth_hit_rate" in result:
                        save_payload["max_depth_hit_rate"] = np.float64(result["max_depth_hit_rate"])
                    if "divergence_rate" in result:
                        save_payload["divergence_rate"] = np.float64(result["divergence_rate"])
                    if "pf_seed_trace" in result:
                        save_payload["pf_seed_trace"] = np.asarray(result["pf_seed_trace"], dtype=np.int32)
                    if "chain_ess" in result:
                        save_payload["chain_ess_sigma2"] = np.asarray(result["chain_ess"], dtype=np.float64)
                        save_payload["chain_ess_min_sigma2"] = np.float64(result["chain_ess_min"])
                        save_payload["chain_ess_num_samples"] = np.int32(result["chain_ess_num_samples"])
                        save_payload["chain_ess_burnin_used"] = np.int32(result["chain_ess_burnin_used"])
                    if tb_run_dir is not None:
                        save_payload["tensorboard_run_dir"] = np.asarray(str(tb_run_dir))
                    np.savez(data_path, **save_payload)

                    if tb_writer is not None and tb_run_dir is not None:
                        _write_tensorboard_final_summary(
                            tb_writer,
                            step=int(result["num_steps"]),
                            result=result,
                            sigma2_chain_post_thinned=sigma2_chain_post_thinned,
                            sigma_chain_post_thinned=sigma_chain_post_thinned,
                            burnin=burnin,
                            drop=drop,
                            data_path=data_path,
                            plot_path=plot_path,
                            posterior_plot_path=posterior_plot_path,
                            log_path=log_path,
                            tb_run_dir=tb_run_dir,
                        )

                    mean_v2, mean_w2 = np.mean(sigma2_chain_post_thinned, axis=0)
                    std_v2, std_w2 = np.std(sigma2_chain_post_thinned, axis=0)
                    mean_v, mean_w = np.mean(sigma_chain_post_thinned, axis=0)
                    std_v, std_w = np.std(sigma_chain_post_thinned, axis=0)
                    print(f"[done] result saved: {data_path}")
                    print(f"[done] plot saved:   {plot_path}")
                    print(f"[done] plot saved:   {posterior_plot_path}")
                    print(f"[done] log saved:    {log_path}")
                    if tb_run_dir is not None:
                        print(f"[done] tensorboard:  {tb_run_dir}")
                    print(f"[post] burnin={burnin} drop={drop} kept={sigma_chain_post_thinned.shape[0]}")
                    print(f"[{sampler.upper()}] accept_rate={result['accept_rate']:.3f} runtime_sec={result['runtime_sec']:.2f}")
                    if "chain_ess" in result:
                        chain_ess = np.asarray(result["chain_ess"], dtype=np.float64)
                        ess_labels = ("sigma_v2", "sigma_w2")
                        ess_text = " ".join(
                            f"{label}={value:.2f}" for label, value in zip(ess_labels, chain_ess, strict=False)
                        )
                        print(
                            f"[post] chain_ess(sigma^2) {ess_text} "
                            f"min={float(result['chain_ess_min']):.2f} "
                            f"draws={int(result['chain_ess_num_samples'])}"
                        )
                    print(f"[post] sigma_v2 mean={mean_v2:.4f} std={std_v2:.4f}")
                    print(f"[post] sigma_w2 mean={mean_w2:.4f} std={std_w2:.4f}")
                    print(f"[post] sigma_v  mean={mean_v:.4f} std={std_v:.4f}")
                    print(f"[post] sigma_w  mean={mean_w:.4f} std={std_w:.4f}")
        finally:
            if tb_writer is not None:
                tb_writer.close()


if __name__ == "__main__":
    main()
