import numpy as np
import pytest

from src.metrics import multitarget_metrics

pytestmark = pytest.mark.unit


def test_multitarget_metrics_zero_error():
    batch_size = 1
    T = 3
    num_targets = 2
    state_dim = 4 * num_targets
    x_true = np.zeros((batch_size, T, state_dim), dtype=np.float32)
    mean = np.zeros_like(x_true)

    metrics = multitarget_metrics(
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

    metrics = multitarget_metrics(
        x_true=x_true,
        mean=mean,
        num_targets=num_targets,
        ospa_cutoff=20.0,
        ospa_p=1.0,
    )

    assert metrics["diverged"] == 1
