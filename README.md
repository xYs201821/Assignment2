# Filters and SSM Interfaces

This README focuses on the SSM and filter interfaces used in exp1-2 and how the experiment code wires them together (generated with the help of ChatGPT).

## Experiment map
- exp1: LinearGaussianSSM + kf, pf, edh(pfpf), ledh(pfpf)
- exp2a: StochasticVolatilitySSM + ekf, ukf, pf, edh, ledh, edh(pfpf), kflow_diag, kflow_scalar
- exp2b: RangeBearingSSM + ekf, ukf, pf, edh

## SSM base interface (src/ssm/base.py)
Required properties:
- state_dim, obs_dim, q_dim, r_dim
- m0, P0 (initial mean/cov)
- cov_eps_x, cov_eps_y (process/obs noise cov)

Required methods:
- initial_state_dist(shape) -> tfd.Distribution
- transition_dist(x_prev) -> tfd.Distribution
- observation_dist(x) -> tfd.Distribution
- f(x) and h(x) deterministic dynamics/observation

Optional overrides:
- f_with_noise(x, q) / h_with_noise(x, r) for non-additive noise
- innovation(y, y_pred) for wrapped angles
- measurement_mean, measurement_residual for non-Euclidean obs

Common helpers:
- sample_initial_state, sample_transition, sample_observation
- simulate(T, shape, x0=None) -> x_traj [batch, T, dx], y_traj [batch, T, dy]

Shape conventions:
- state x: [batch, dx] or [batch, N, dx] for particles
- observation y: [batch, dy] or [batch, T, dy]

## SSMs used in exp1-2

### LinearGaussianSSM (exp1)
- file: src/ssm/base.py
- init: A, B, C, D, m0, P0, jitter, seed
- f(x) = A x, h(x) = C x
- cov_eps_x = B B^T, cov_eps_y = D D^T
- used in experiments/exp1_linear_gaussian.py 

### StochasticVolatilitySSM (exp2a)
- file: src/ssm/stochastic_volatility.py
- init: alpha, sigma, beta, mu, noise_scale_func, obs_mode, obs_eps, seed
- nonlinear obs; obs_mode in config is "logy2"
- overrides f_with_noise and h_with_noise (non-additive noise)
- used in experiments/exp2a_stochastic_vol.py 

### RangeBearingSSM (exp2b)
- file: src/ssm/range_bearing.py
- init: motion_model (ConstantVelocityMotionModel), cov_eps_y, jitter, seed
- observation returns [range, bearing] with angle wrapping
- overrides innovation and measurement_mean/residual for bearing
- used in experiments/exp2b_range_bearing.py 

## Filter interfaces

### Gaussian filters: KF, EKF, UKF
Classes:
- src/filters/kalman.py: KalmanFilter
- src/filters/ekf.py: ExtendedKalmanFilter
- src/filters/ukf.py: UnscentedKalmanFilter

Constructor: KalmanFilter(ssm) / ExtendedKalmanFilter(ssm, joseph=True) /
UnscentedKalmanFilter(ssm, alpha, beta, kappa, jitter, joseph)

Call pattern:
- filt.warmup(batch_size)   # used in experiments to trace tf.function
- out = filt.filter(y, m0=None, P0=None, memory_sampler=None)

Outputs (dict):
- m_filt: [batch, T, dx]
- P_filt: [batch, T, dx, dx]
- m_pred, P_pred: one-step predictions
- cond_P: condition numbers
- step_time_s, memory_rss, memory_gpu (optional)

SSM requirements:
- KF expects LinearGaussianSSM with A, C and cov_eps_x/cov_eps_y
- EKF/UKF use f/h and f_with_noise/h_with_noise for Jacobians or sigma points

### Particle and flow filters: PF, EDH, LEDH, KernelFlow
Classes:
- src/filters/pf_bootstrap.py: BootstrapParticleFilter (method name "pf")
- src/flows/edh.py: EDHFlow
- src/flows/ledh.py: LEDHFlow
- src/flows/kernel_embedded.py: KernelParticleFlow (method names "kflow_diag"/"kflow_scalar")

Common call pattern:
- filt.warmup(batch_size, ...)
- x_seq, w_seq, diagnostics, parents = filt.filter(
      y,
      num_particles=None,
      ess_threshold=None,
      reweight="always", # 'never' for kflow, edh/ledh
      resample="auto", # 'auto': only resample when below threshold; 'always': always resample; 'never' never resample
      init_dist=None,
      init_seed=None,
      init_particles=None, 
      memory_sampler=None,
  )

Outputs:
- x_seq: [batch, T, N, dx]
- w_seq: [batch, T, N]
- diagnostics: step_time_s, m_pred, P_pred, x_pred, w_pre, plus flow-specific stats
- parents: resampling parent indices [batch, T, N]

Notes:
- reweight/resample can be "never", "auto", "always" (or bool/int)
- init_dist is a callable: init_dist(shape) -> tfd.Distribution (see build_init_dist)
- init_particles can be used to seed PF/flows (used in exp2b)
- either provide init_particles or init_dist

KernelParticleFlow specific args:
- kernel_type: "diag" or "scalar" 
- ll_grad_mode: "linearized" or "dist" # whether or not use linearization to approximate likelihood
- optimizer: "fixed", "adagrad", "adam" # 'none' stands for fixed step size
- alpha, alpha_update_every, ds_init, max_flow_norm, localization_radius

## Experiment wiring (exp1-2)

All experiments build filters via experiments/filter_cfg.py and run them via
experiments/runner.py: run_filter(ssm, y_obs, method, **cfg).

Key config blocks in exp*_config.yaml:
- filters.methods: list of method names (kf, ekf, ukf, pf, edh, ledh, edh(pfpf), ledh(pfpf), kflow_diag, kflow_scalar)
- filters.ukf: alpha, beta, kappa, jitter
- filters.pf: num_particles, ess_threshold, reweight
- filters.flow: num_particles, num_lambda, ess_threshold, reweight
- filters.kflow: num_particles, num_lambda, alpha, ds_init, optimizer, ll_grad_mode, max_flow_norm, localization_radius

Method name behavior:
- "edh(pfpf)" / "ledh(pfpf)" force reweight="always" and resample="auto"
- "kflow_diag" / "kflow_scalar" select kernel_type
- kflow grid params are expanded into method suffixes via tag_from_cfg
