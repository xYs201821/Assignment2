"""Base filter classes and common helper routines."""

import tensorflow as tf

from src.utility import cholesky_solve, quadratic_matmul
import src.dtype_config as _dc

FILTER_PRINT_INFO = {
    "BaseFilter": {
        "title": "BaseFilter",
        "fields": ("state_dim", "obs_dim", "q_dim", "r_dim"),
    },
    "GaussianFilter": {
        "title": "GaussianFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "KalmanFilter": {
        "title": "KalmanFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "ExtendedKalmanFilter": {
        "title": "ExtendedKalmanFilter",
        "fields": ("state_dim", "obs_dim"),
    },
    "UnscentedKalmanFilter": {
        "title": "UnscentedKalmanFilter",
        "fields": ("state_dim", "obs_dim", "alpha", "beta", "kappa"),
    },
    "ParticleFilter": {
        "title": "ParticleFilter",
        "fields": ("num_particles", "ess_threshold"),
    },
    "BootstrapParticleFilter": {
        "title": "BootstrapParticleFilter",
        "fields": ("num_particles", "ess_threshold"),
    },
}


class BaseFilter(tf.Module):
    """Base class for filters"""

    def __init__(self, ssm, debug=False, print=False, **kwargs):
        """Initialize with an SSM and debug/print flags."""
        super().__init__()
        self.ssm = ssm
        self.state_dim = ssm.state_dim
        self.obs_dim = ssm.obs_dim
        self.q_dim = ssm.q_dim
        self.r_dim = ssm.r_dim
        self.debug = bool(debug)
        self.print = bool(print)
        if type(self) is BaseFilter:
            self._maybe_print()

    def filter(self, y, **kwargs):
        """Run the filter over observations y.

        Shapes:
          y: [T, dy] or [B, T, dy]
        Returns:
          implementation-specific outputs per time step.
        """
        raise NotImplementedError

    def _maybe_print(self):
        """Print filter metadata when requested."""
        if self.print:
            self._print_debug_info()

    def _print_debug_info(self):
        """Emit a structured debug line with key configuration fields."""
        info = FILTER_PRINT_INFO.get(type(self).__name__)
        if not info:
            return
        title = info.get("title", type(self).__name__)
        fields = info.get("fields", ())
        parts = [title]
        for field in fields:
            if hasattr(self, field):
                parts.append(f"{field}=")
                parts.append(getattr(self, field))
        tf.print("[filter]", *parts)

    @staticmethod
    def _stack_and_permute(ta, tail_dims=1):
        """Stack a TensorArray and move time to the first non-batch axis.

        Shapes:
          ta.element: [B, ...] with tail_dims trailing dimensions
        Returns:
          stacked: [B, T, ...]
        """
        seq = ta.stack()
        rank = tf.rank(seq)
        batch_rank = rank - (1 + tail_dims)
        prefix = tf.range(1, 1 + batch_rank)
        perm = tf.concat([prefix, [0], tf.range(1 + batch_rank, rank)], axis=0)
        return tf.transpose(seq, perm)

    @staticmethod
    def _init_memory_traces(size, memory_sampler):
        """Initialize memory trace TensorArrays when enabled."""
        if memory_sampler is None:
            return None, None
        return (
            tf.TensorArray(dtype=_dc.DTYPE, size=size),
            tf.TensorArray(dtype=_dc.DTYPE, size=size),
        )

    @staticmethod
    def _parse_memory_sample(sample):
        """Normalize memory sampler output into (rss, gpu)."""
        rss = None
        gpu = None
        if isinstance(sample, dict):
            rss = sample.get("rss")
            gpu = sample.get("gpu")
        elif isinstance(sample, (tuple, list)):
            if len(sample) > 0:
                rss = sample[0]
            if len(sample) > 1:
                gpu = sample[1]
        else:
            rss = sample
        return rss, gpu

    def _record_memory(self, step, memory_sampler, mem_rss_ta, mem_gpu_ta):
        """Record a memory sample into TensorArrays."""
        if memory_sampler is None:
            return mem_rss_ta, mem_gpu_ta
        sample = memory_sampler()
        rss, gpu = self._parse_memory_sample(sample)
        if rss is not None:
            mem_rss_ta = mem_rss_ta.write(step, tf.cast(rss, _dc.DTYPE))
        if mem_gpu_ta is not None:
            gpu_val = 0.0 if gpu is None else gpu
            mem_gpu_ta = mem_gpu_ta.write(step, tf.cast(gpu_val, _dc.DTYPE))
        return mem_rss_ta, mem_gpu_ta

    def _finalize_memory(self, out, mem_rss_ta, mem_gpu_ta):
        """Attach memory traces to output dict."""
        if mem_rss_ta is not None:
            out["memory_rss"] = self._stack_and_permute(mem_rss_ta, tail_dims=0)
        if mem_gpu_ta is not None:
            out["memory_gpu"] = self._stack_and_permute(mem_gpu_ta, tail_dims=0)
        return out


class GaussianFilter(BaseFilter):
    """Base class for Gaussian filters with predict/update structure."""

    def __init__(self, ssm, joseph=True, debug=False, print=False):
        """Initialize Gaussian filter with process/observation noise covariances."""
        super().__init__(ssm, debug=debug, print=print)
        self.cov_eps_x = tf.convert_to_tensor(ssm.cov_eps_x, dtype=_dc.DTYPE)
        self.cov_eps_y = tf.convert_to_tensor(ssm.cov_eps_y, dtype=_dc.DTYPE)
        self.update = self.update_joseph if joseph else self.update_naive
        if type(self) is GaussianFilter:
            self._maybe_print()

    def warmup(self, batch_size=1, T=2, y=None):
        """Trace and compile the filter function to avoid first-call overhead."""
        dy = int(self.ssm.obs_dim)

        if y is None:
            y = tf.zeros([batch_size, T, dy], _dc.DTYPE)
        else:
            y = tf.convert_to_tensor(y, dtype=_dc.DTYPE)
            if len(y.shape) == 2:
                y = y[tf.newaxis, :]
        _ = self.filter(y)

    def predict(self, m_prev, P_prev):
        """Predict next-step mean/covariance.

        Shapes:
          m_prev: [B, dx]
          P_prev: [B, dx, dx]
        Returns:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
        """
        raise NotImplementedError

    def update_joseph(self, m_pred, P_pred, y):
        """Measurement update using Joseph stabilized covariance update.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        raise NotImplementedError

    def update_naive(self, m_pred, P_pred, y):
        """Measurement update using the naive covariance formula.

        Shapes:
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
        """
        raise NotImplementedError

    def step(self, m_prev, P_prev, y):
        """Apply update then predict to advance one time step.

        Shapes:
          m_prev: [B, dx]
          P_prev: [B, dx, dx]
          y: [B, dy]
        Returns:
          m_filt: [B, dx]
          P_filt: [B, dx, dx]
          m_pred: [B, dx]
          P_pred: [B, dx, dx]
        """
        m_filt, P_filt = self.update(m_prev, P_prev, y)
        m_pred, P_pred = self.predict(m_filt, P_filt)
        return m_filt, P_filt, m_pred, P_pred

    @tf.function(reduce_retracing=True)
    def filter(self, y, m0=None, P0=None):
        """Filter a full observation sequence.

        Shapes:
          y: [T, dy] or [B, T, dy]
          m0: [dx] or [B, dx]
          P0: [dx, dx] or [B, dx, dx]
        Returns:
          m_filt: [B, T, dx]
          P_filt: [B, T, dx, dx]
          m_pred: [B, T, dx]
          P_pred: [B, T, dx, dx]
        """
        y = tf.convert_to_tensor(y, dtype=_dc.DTYPE)
        if len(y.shape) == 2:
            y = y[tf.newaxis, :]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]
        m_init = tf.convert_to_tensor(m0 if m0 is not None else self.ssm.m0, dtype=_dc.DTYPE)
        if len(m_init.shape) == 1:
            m_init = m_init[tf.newaxis, :]
            m_init = tf.tile(m_init, [batch_size, 1])
        P_init = tf.convert_to_tensor(P0 if P0 is not None else self.ssm.P0, dtype=_dc.DTYPE)
        if len(P_init.shape) == 2:
            P_init = P_init[tf.newaxis, :, :]
            P_init = tf.tile(P_init, [batch_size, 1, 1])

        m_filt_ta = tf.TensorArray(dtype=_dc.DTYPE, size=T, dynamic_size=False)
        P_filt_ta = tf.TensorArray(dtype=_dc.DTYPE, size=T, dynamic_size=False)
        m_pred_ta = tf.TensorArray(dtype=_dc.DTYPE, size=T, dynamic_size=False)
        P_pred_ta = tf.TensorArray(dtype=_dc.DTYPE, size=T, dynamic_size=False)

        m_invar = tf.TensorShape(None)
        P_invar = tf.TensorShape(None)

        def _cond(t, _state):
            return t < T

        def _body(t, state):
            m_pred, P_pred, m_filt_ta, P_filt_ta, m_pred_ta, P_pred_ta = state
            y_t = y[:, t, :]

            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)

            m_filt, P_filt, m_pred_next, P_pred_next = self.step(m_pred, P_pred, y_t)
            m_filt_ta = m_filt_ta.write(t, m_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)

            return t + 1, (m_pred_next, P_pred_next, m_filt_ta, P_filt_ta, m_pred_ta, P_pred_ta)

        _, (_, _, m_filt_ta, P_filt_ta, m_pred_ta, P_pred_ta) = tf.while_loop(
            _cond,
            _body,
            (tf.constant(0), (m_init, P_init, m_filt_ta, P_filt_ta, m_pred_ta, P_pred_ta)),
            shape_invariants=(
                tf.TensorShape([]),
                (m_invar, P_invar, None, None, None, None),
            ),
        )

        return {
            "m_filt": self._stack_and_permute(m_filt_ta, tail_dims=1),
            "P_filt": self._stack_and_permute(P_filt_ta, tail_dims=2),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
        }

    @staticmethod
    def _kalman_gain(C, P, R):
        """Compute Kalman gain and innovation covariance.

        Shapes:
          C: [..., dy, dx]
          P: [..., dx, dx]
          R: [..., dy, dy]
        Returns:
          K: [..., dx, dy]
          S: [..., dy, dy]
        """
        # S = C P C^T + R and K = P C^T S^{-1}.
        S = quadratic_matmul(C, P, C) + R
        RHS = tf.linalg.matmul(C, P, transpose_b=True)
        K_transpose = cholesky_solve(S, RHS)
        K = tf.linalg.matrix_transpose(K_transpose)
        return K, S

    @staticmethod
    def _joseph_update(C, P, K, R):
        """Joseph-form covariance update for symmetry and PSD stability.

        Shapes:
          C: [..., dy, dx]
          P: [..., dx, dx]
          K: [..., dx, dy]
          R: [..., dy, dy]
        Returns:
          P_new: [..., dx, dx]
        """
        I = tf.eye(tf.shape(P)[-1], batch_shape=tf.shape(P)[:-2], dtype=_dc.DTYPE)
        I_KC = I - tf.linalg.matmul(K, C)
        return quadratic_matmul(I_KC, P, I_KC) + quadratic_matmul(K, R, K)
