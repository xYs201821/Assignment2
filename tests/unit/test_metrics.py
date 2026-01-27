import numpy as np
import pytest

from experiments.exp3_multitarget_acoustic import _multitarget_metrics
from experiments.exp4_lorenz96 import _lorenz96_extra_metrics
from src.ssm.lorenz96 import Lorenz96SSM

pytestmark = pytest.mark.unit


def test_multitarget_metrics_zero_error():
    batch_size = 1
    T = 3
    num_targets = 2
    state_dim = 4 * num_targets
    x_true = np.zeros((batch_size, T, state_dim), dtype=np.float32)
    mean = np.zeros_like(x_true)

    metrics = _multitarget_metrics(
        x_true=x_true,
        mean=mean,
        num_targets=num_targets,
        ospa_cutoff=20.0,
        ospa_p=1.0,
        ospa_percentiles=[50.0, 90.0],
    )

    assert metrics["diverged"] == 0
    assert metrics["ospa"] == 0.0
    assert metrics["ospa_final"] == 0.0
    assert metrics["best_match_rmse"] == 0.0
    assert metrics["best_match_rmse_final"] == 0.0
    assert metrics["ospa_p50"] == 0.0


def test_multitarget_metrics_detects_divergence():
    batch_size = 1
    T = 2
    num_targets = 1
    state_dim = 4 * num_targets
    x_true = np.zeros((batch_size, T, state_dim), dtype=np.float32)
    mean = np.full_like(x_true, np.nan, dtype=np.float32)

    metrics = _multitarget_metrics(
        x_true=x_true,
        mean=mean,
        num_targets=num_targets,
        ospa_cutoff=20.0,
        ospa_p=1.0,
    )

    assert metrics["diverged"] == 1


def test_lorenz96_extra_metrics_zero_error():
    ssm = Lorenz96SSM(
        state_dim=8,
        obs_stride=2,
        dt=0.05,
        F=8.0,
        obs_op="linear",
        q_scale=0.1,
        r_scale=0.5,
        seed=0,
    )
    x_true = np.zeros((1, 3, 8), dtype=np.float32)
    mean = np.zeros_like(x_true)

    metrics = _lorenz96_extra_metrics(
        ssm,
        x_true=x_true,
        mean=mean,
        include_energy=True,
        include_final=True,
    )

    assert metrics["diverged"] == 0
    assert metrics["rmse_energy"] == 0.0
    assert metrics["rmse_final"] == 0.0
    assert metrics["rmse_obs_final"] == 0.0
    assert metrics["rmse_unobs_final"] == 0.0
