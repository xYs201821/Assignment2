from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
_EXPERIMENTS = {
    "exp1": "experiments.exp1.exp1_linear_gaussian",
    "exp2a": "experiments.exp2a.exp2a_stochastic_vol",
    "exp2b": "experiments.exp2b.exp2b_range_bearing",
    "exp3": "experiments.exp3.exp3_lgssm_dpf",
    "exp3tune": "experiments.exp3.exp3_tune",
    "exp3pt": "experiments.exp3.exp3_transformer_pretrain",
    "exp3plot": "experiments.exp3.plot_exp3_dpf_diagnostics",
    "dai22": "experiments.dai22.exp_dai22",
}

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Run experiment modules by short name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Execution device selector applied before importing the experiment. "
            "Use 'cpu' to disable GPU, 'gpu' to keep default GPU visibility, "
            "or pass CUDA_VISIBLE_DEVICES values like '0' or '0,1'."
        ),
    )
    parser.add_argument(
        "experiment",
        choices=sorted(_EXPERIMENTS),
        help="Experiment to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the experiment module.",
    )
    return parser.parse_args(argv)


def _pop_device_from_args(args: List[str]) -> tuple[str | None, List[str]]:
    device: str | None = None
    out: List[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            out.extend(args[i:])
            break
        if token == "--device":
            if i + 1 >= len(args) or args[i + 1] == "--":
                raise SystemExit("error: --device requires a value")
            device = args[i + 1]
            i += 2
            continue
        if token.startswith("--device="):
            value = token.split("=", 1)[1].strip()
            if not value:
                raise SystemExit("error: --device requires a value")
            device = value
            i += 1
            continue
        out.append(token)
        i += 1
    return device, out


def _apply_device(device: str | None) -> None:
    if device is None:
        return
    value = str(device).strip()
    if not value:
        return
    lowered = value.lower()
    if lowered == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    if lowered == "gpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    if lowered.startswith("cuda:"):
        value = value.split(":", 1)[1].strip()
    elif lowered.startswith("gpu:"):
        value = value.split(":", 1)[1].strip()
    if not value:
        raise SystemExit("--device expects a non-empty GPU index list (e.g. '0' or '0,1').")
    os.environ["CUDA_VISIBLE_DEVICES"] = value


def _run_experiment(experiment: str, args: List[str]) -> None:
    module_name = _EXPERIMENTS[experiment]
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise SystemExit(f"{module_name} does not define main()")
    sys.argv = [module.__file__ or module_name] + args
    main()


def main() -> None:
    parsed = _parse_args(sys.argv[1:])
    tail_device, forwarded = _pop_device_from_args(parsed.args)
    device = tail_device if tail_device is not None else parsed.device
    _apply_device(device)
    _run_experiment(parsed.experiment, forwarded)


if __name__ == "__main__":
    main()
