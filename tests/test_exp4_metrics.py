import numpy as np

from experiments.exp4_lorenz96 import _lorenz96_extra_metrics
from src.ssm.lorenz96 import Lorenz96SSM


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
