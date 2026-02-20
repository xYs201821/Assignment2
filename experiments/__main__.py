from __future__ import annotations

import argparse
import importlib
import sys
from typing import List

_EXPERIMENTS = {
    "exp1": "experiments.exp1.exp1_linear_gaussian",
    "exp2a": "experiments.exp2a.exp2a_stochastic_vol",
    "exp2b": "experiments.exp2b.exp2b_range_bearing",
    "exp3": "experiments.exp3.exp3_lgssm_dpf",
    "exp3tune": "experiments.exp3.exp3_tune",
    "exp3pt": "experiments.exp3.exp3_transformer_pretrain",
    "exp3plot": "experiments.exp3.plot_exp3_dpf_diagnostics",
}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Run experiment modules by short name.",
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
    _run_experiment(parsed.experiment, parsed.args)


if __name__ == "__main__":
    main()
