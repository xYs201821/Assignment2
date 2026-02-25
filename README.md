# Assignment2: Particle Filtering and Differentiable Particle Filter Experiments

This repository contains a TensorFlow-based research codebase for nonlinear state estimation and differentiable particle filtering.

It includes:
- classical Gaussian filters (`KF`, `EKF`, `UKF`)
- bootstrap particle filtering
- particle-flow methods (`EDH`, `LEDH`, kernel flow, stochastic particle flow)
- differentiable particle filters with multiple resampling schemes (soft, OT, diffusion, transformer)
- experiment pipelines for multiple state-space models

## Requirements

- Python `3.11`
- Linux/macOS environment recommended
- Optional GPU support via TensorFlow (CPU-only runs are supported)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

Run the experiment router help:

```bash
python -m experiments --help
```

Run a default experiment:

```bash
python -m experiments exp3 --seed 0 --methods kalman --steps 1
```

Run unit tests:

```bash
pytest -m unit
```

## Experiment Entry Points

Primary entry command:

```bash
python -m experiments <name> [experiment args...]
```

Available `<name>` values:

| Name | Module | Default config |
| --- | --- | --- |
| `exp1` | `experiments.exp1.exp1_linear_gaussian` | `experiments/exp1/exp1_config.yaml` |
| `exp2a` | `experiments.exp2a.exp2a_stochastic_vol` | `experiments/exp2a/exp2a_config.yaml` |
| `exp2b` | `experiments.exp2b.exp2b_range_bearing` | `experiments/exp2b/exp2b_config.yaml` |
| `exp3` | `experiments.exp3.exp3_lgssm_dpf` | `experiments/exp3/exp3_config.yaml` |
| `exp3tune` | `experiments.exp3.exp3_tune` | `experiments/exp3/exp3_config.yaml` + `experiments/exp3/exp3_tuning.yaml` |
| `exp3pt` | `experiments.exp3.exp3_transformer_pretrain` | `experiments/exp3/exp3_transformer_pretrain_config.yaml` |
| `exp3plot` | `experiments.exp3.plot_exp3_dpf_diagnostics` | plot utility (reads saved traces) |
| `dai22` | `experiments.dai22.exp_dai22` | CLI-only (no YAML required) |

Examples:

```bash
# Run exp2b with defaults
python -m experiments exp2b

# Force device selection on router
python -m experiments --device cpu exp3 --steps 100 --num-workers 2

# Override YAML values (supported in exp1/exp2a/exp2b via --set).
# In zsh, quote values containing [] to avoid shell glob expansion.
python -m experiments exp2a \
  --set 'experiment.seeds=[0,1,2]' \
  --set 'filters.flow.num_particles=[100]'
```
## Testing

Pytest is configured with unit and integration markers.

```bash
# all tests
pytest

# only unit tests
pytest -m unit

# only integration tests
pytest -m integration

# specific test file
pytest tests/unit/test_ukf_sigma_points.py -q
```

## Results and Outputs

Outputs are written under `results/` by default (overridable in configs/CLI).

Typical artifacts include:
- per-seed traces (`*.npz`)
- diagnostics and metrics
- experiment summaries (`summary.json`)
- diagnostic plots (`*.png`)

For `exp3`, the default structure is:

```text
results/exp3_lgssm_dpf/<experiment_name>/
```

By default `experiment_name` is `exp3`, so plotting usually targets:

```bash
python -m experiments exp3plot --input-root results/exp3_lgssm_dpf/exp3
```

This plotting command requires existing trace files under the input directory
(run `exp3` first).

## Repository Layout

```text
src/
  filters/        # KF/EKF/UKF, PF, DPF variants, resampling layers
  flows/          # EDH/LEDH, kernel flow, stochastic PF, beta schedules
  ssm/            # state-space models (linear Gaussian, SV, range-bearing, VRNN)
experiments/
  exp1/ exp2a/ exp2b/ exp3/ exp4/ dai22/
  common/         # shared config, runner, metrics, plotting utilities
tests/
  unit/
  integration/
```

## Notes

- Run commands from repository root so relative config paths resolve correctly.
- Large particle counts and long horizons are memory-intensive; reduce `num_particles`, `batch_size`, or training `steps` if needed.
- Seeds are configurable in each experiment YAML under `experiment.seeds` for reproducibility.
