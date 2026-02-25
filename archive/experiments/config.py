from dataclasses import dataclass, field
import numpy as np


@dataclass
class CommonConfig:
    T: int = 40
    batch_size: int = 1
    seed: tuple[int, ...] = (42,)
    out_dir: str = "runs"
    save: bool = True
    show: bool = True


@dataclass
class SVConfig:
    alpha: float = 0.99
    sigma: float = 1.2
    beta: float = 2.5
    noise_scale_func: bool = False
    obs_mode: str = "logy2"
    obs_eps: float = 1e-16

    x0_true: float = 0.0
    m0_est: float = 0.0
    P0_scale: float = 5.0

    ukf_alpha: float = 5e-1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    pf_particles: int = 800
    pf_ess_threshold: float = 0.5
    pf_reweight: str = "always"
    edh_particles: int = 40
    edh_num_lambda: int = 20
    edh_ess_threshold: float = 0.5
    edh_reweight: str = 1
    kflow_particles: int = 100
    kflow_num_lambda: int = 20
    kflow_alpha: float = 1.0
    kflow_localization_radius: float | None = 4.0
    kflow_kernel_types: tuple[str, ...] = ("diag", "scalar")

    filters: tuple[str, ...] = ("ekf", "ukf", "pf", "edh", "ledh", "kflow")


@dataclass
class RBConfig:
    dt: float = 0.1
    motion_model: str = "cv"
    q_scale_pos: float = 1.0 / 40.0
    q_scale_v: float = 1.0 / 20.0
    q_scale_psi: float = 0.0
    q_scale_omega: float = 0.003
    r_range: float = 0.2
    r_bearing: float = 0.1

    x0_true: tuple[float, ...] = (-0.2, -0.2, 0.3, np.pi / 4, 0.0)
    m0_est: tuple[float, ...] = (-0.25, -0.25, 0.2, np.pi / 4, 0.0)
    P0_scale: tuple[float, ...] = (1.0, 1.0, 0.3, 1.0, 0.1)

    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    pf_particles: int = 10000
    pf_ess_threshold: float = 0.5
    pf_reweight: str = "always"
    edh_particles: int = 100
    edh_num_lambda: int = 10
    edh_ess_threshold: float = 0.5
    edh_reweight: str = "never"
    kflow_particles: int = 100
    kflow_num_lambda: int = 40
    kflow_ds_init: float | None = None
    kflow_alpha: float = 0.1
    kflow_localization_radius: float | None = None
    kflow_kernel_types: tuple[str, ...] = ("scalar", "diag")
    kflow_debug: bool = False
    kflow_debug_every: int = 0
    kflow_max_flow_norm: float | None = 10.0
    kflow_adaptive_step: bool = True
    kflow_adaptive_window: int = 20
    kflow_adaptive_factor: float = 1.4
    kflow_adaptive_min: float | None = None
    kflow_adaptive_max: float | None = None

    filters: tuple[str, ...] = ("ekf", "ukf", "pf", "edh", "ledh", "kflow")


@dataclass
class KernelFlowConfig:
    state_dim: int = 40
    obs_stride: int = 4
    dt: float = 0.01
    F: float = 8.0
    obs_op: str = "linear"
    q_scale: float = 0.1
    r_scale: float = 0.5
    x0_noise: float = 0.01
    T: int = 40
    batch_size: int = 1
    num_particles: int = 200
    num_lambda: int = 60
    alpha: float = 0.02
    localization_radius: float | None = None
    seed: int = 42
    out_dir: str = "runs/kernel_embedded"
    plot_dims: tuple[int, ...] = (0, 1, 2)
    t_plot: int = -1


@dataclass
class MultiTargetConfig:
    num_targets: int = 4
    area_size: float = 40.0
    grid_size: int = 5
    Psi: float = 10.0
    d0: float = 1.0
    sigma_w: float = 0.1
    seed: int = 42
    out_dir: str = "runs/multi_target"
    max_plot_sensors: int = 6
    x0: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                12.0,
                6.0,
                0.001,
                0.001,
                32.0,
                32.0,
                -0.001,
                -0.005,
                20.0,
                13.0,
                -0.1,
                0.01,
                15.0,
                35.0,
                0.002,
                -0.02,
            ],
            dtype=np.float32,
        )
    )
