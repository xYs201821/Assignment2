"""Diagnostic updaters for particle flow filters."""

from __future__ import annotations

import tensorflow as tf


def _uniform_weights(x: tf.Tensor) -> tf.Tensor:
    """Return uniform weights matching particle batch shape."""
    n = tf.cast(tf.shape(x)[-2], x.dtype)
    return tf.ones(tf.shape(x)[:-1], dtype=x.dtype) / tf.maximum(n, tf.cast(1.0, x.dtype))


def _percentile(values: tf.Tensor, q: float) -> tf.Tensor:
    """Compute elementwise percentile along the last axis."""
    values = tf.sort(values, axis=-1)
    n = tf.shape(values)[-1]
    q = tf.cast(q, values.dtype) / tf.cast(100.0, values.dtype)
    idx = tf.cast(tf.floor(q * tf.cast(n - 1, values.dtype)), tf.int32)
    idx = tf.clip_by_value(idx, 0, tf.maximum(n - 1, 0))
    return tf.gather(values, idx, axis=-1)


def _cond_from_matrix(mat: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """Condition number estimate for symmetric matrices."""
    mat = 0.5 * (mat + tf.linalg.matrix_transpose(mat))
    eigvals = tf.linalg.eigvalsh(mat)
    eig_min = tf.reduce_min(eigvals, axis=-1)
    eig_max = tf.reduce_max(eigvals, axis=-1)
    eig_min = tf.maximum(eig_min, eps)
    return eig_max / (eig_min + eps)


def _cond_from_rect(mat: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """Condition number estimate for rectangular matrices via SVD."""
    s = tf.linalg.svd(mat, compute_uv=False)
    s_max = tf.reduce_max(s, axis=-1)
    s_min = tf.reduce_min(s, axis=-1)
    return s_max / (s_min + eps)


class _FlowDiagnosticsBase:
    """Shared diagnostics utilities for particle flows."""

    def __init__(self, flow, batch_shape, eps, log10_base):
        self.flow = flow
        self.eps = eps
        self.log10_base = log10_base
        self.metrics: dict[str, tf.Tensor] = {}

    def _init_metrics(self, batch_shape, keys, extra_keys=None):
        self.metrics = {key: tf.zeros(batch_shape, dtype=tf.float32) for key in keys}
        if extra_keys:
            for key in extra_keys:
                self.metrics[key] = tf.zeros(batch_shape, dtype=tf.float32)

    def _update_max(self, key: str, value: tf.Tensor) -> None:
        self.metrics[key] = tf.maximum(self.metrics[key], value)

    def _finalize_cov(self, x: tf.Tensor, eps_float: tf.Tensor | float):
        w_uniform = _uniform_weights(x)
        cov = self.flow.ssm.state_cov(x, w_uniform)
        cond_cov = _cond_from_matrix(cov, self.eps)
        cond_cov_log10 = tf.math.log(cond_cov + self.eps) / self.log10_base
        eigvals = tf.linalg.eigvalsh(cov)
        eigvals = tf.maximum(eigvals, tf.cast(eps_float, eigvals.dtype))
        logdet_cov = tf.reduce_sum(tf.math.log(eigvals), axis=-1)
        return cond_cov_log10, logdet_cov

    def finalize(self, x: tf.Tensor, eps_float: tf.Tensor | float):
        """Finalize diagnostics with covariance-based metrics."""
        cond_cov_log10, logdet_cov = self._finalize_cov(x, eps_float)
        diagnostics = dict(self.metrics)
        diagnostics["condCov_log10"] = cond_cov_log10
        diagnostics["logdet_cov"] = logdet_cov
        return diagnostics


class EDHDiagnostics(_FlowDiagnosticsBase):
    """Diagnostics updater for EDH flow."""

    def __init__(self, flow, batch_shape, eps_t, log10_base):
        super().__init__(flow, batch_shape, eps_t, log10_base)
        self._init_metrics(
            batch_shape,
            keys=(
                "dx_p95_max",
                "condS_log10_max",
                "condH_log10_max",
                "condJ_log10_max",
                "flow_norm_mean_max",
            ),
        )

    def update(self, H, J, S, flow_vec, dx):
        """Update running diagnostic maxima for EDH flow."""
        condH = _cond_from_rect(H, self.eps)
        condH_log10 = tf.math.log(condH + self.eps) / self.log10_base
        self._update_max("condH_log10_max", condH_log10)

        condJ = _cond_from_rect(J, self.eps)
        condJ_log10 = tf.math.log(condJ + self.eps) / self.log10_base
        self._update_max("condJ_log10_max", condJ_log10)

        condS = _cond_from_matrix(S, self.eps)
        condS_log10 = tf.math.log(condS + self.eps) / self.log10_base
        self._update_max("condS_log10_max", condS_log10)

        flow_norm_mean = tf.reduce_mean(tf.norm(flow_vec, axis=-1), axis=-1)
        self._update_max("flow_norm_mean_max", flow_norm_mean)

        dx_norm = tf.norm(dx, axis=-1)
        dx_p95 = _percentile(dx_norm, 95.0)
        self._update_max("dx_p95_max", dx_p95)


class LEDHDiagnostics(_FlowDiagnosticsBase):
    """Diagnostics updater for LEDH flow."""

    def __init__(self, flow, batch_shape, eps_t, log10_base):
        super().__init__(flow, batch_shape, eps_t, log10_base)
        self._init_metrics(
            batch_shape,
            keys=(
                "dx_p95_max",
                "condS_log10_max",
                "condH_log10_max",
                "condJ_log10_max",
                "flow_norm_mean_max",
            ),
        )

    def update(self, H, J, S_mean, flow_vec, dx):
        """Update running diagnostic maxima for LEDH flow."""
        condH_particles = _cond_from_rect(H, self.eps)
        condH_mean = tf.reduce_mean(condH_particles, axis=-1)
        condH_log10 = tf.math.log(condH_mean + self.eps) / self.log10_base
        self._update_max("condH_log10_max", condH_log10)

        condS = _cond_from_matrix(S_mean, self.eps)
        condS_log10 = tf.math.log(condS + self.eps) / self.log10_base
        self._update_max("condS_log10_max", condS_log10)

        condJ_particles = _cond_from_rect(J, self.eps)
        condJ_mean = tf.reduce_mean(condJ_particles, axis=-1)
        condJ_log10 = tf.math.log(condJ_mean + self.eps) / self.log10_base
        self._update_max("condJ_log10_max", condJ_log10)

        flow_norm_mean = tf.reduce_mean(tf.norm(flow_vec, axis=-1), axis=-1)
        self._update_max("flow_norm_mean_max", flow_norm_mean)

        dx_norm = tf.norm(dx, axis=-1)
        dx_p95 = _percentile(dx_norm, 95.0)
        self._update_max("dx_p95_max", dx_p95)


class KernelDiagnostics(_FlowDiagnosticsBase):
    """Diagnostics updater for kernel particle flow."""

    def __init__(self, flow, batch_shape, eps_val, log10_base, track_max_flow_frac=False):
        super().__init__(flow, batch_shape, eps_val, log10_base)
        extra_keys = ("max_flow_frac",) if track_max_flow_frac else None
        self._init_metrics(
            batch_shape,
            keys=("dx_p95_max", "condK_log10_max", "flow_norm_mean_max"),
            extra_keys=extra_keys,
        )

    def update(self, K, dx_norm, flow_norm, jitter_val, clipped_frac=None):
        """Update running diagnostic maxima for kernel flow."""
        dx_p95 = _percentile(dx_norm, 95.0)
        self._update_max("dx_p95_max", dx_p95)
        self._update_max("flow_norm_mean_max", flow_norm)

        if self.flow.kernel_type == "scalar":
            K_sym = 0.5 * (K + tf.linalg.matrix_transpose(K))
            eye_k = tf.eye(
                tf.shape(K_sym)[-1],
                batch_shape=tf.shape(K_sym)[:-2],
                dtype=K_sym.dtype,
            )
            K_sym = K_sym + jitter_val * eye_k
            condK = _cond_from_matrix(K_sym, self.eps)
        else:
            rank = tf.rank(K)
            perm_swap = tf.concat(
                [
                    tf.range(rank - 3),
                    tf.stack([rank - 2, rank - 3, rank - 1]),
                ],
                axis=0,
            )
            K_t = tf.transpose(K, perm=perm_swap)
            K_sym = 0.5 * (K + K_t)
            perm_dx = tf.concat(
                [
                    tf.range(rank - 3),
                    tf.stack([rank - 1, rank - 3, rank - 2]),
                ],
                axis=0,
            )
            K_sym_dx = tf.transpose(K_sym, perm=perm_dx)
            eye_k = tf.eye(
                tf.shape(K_sym_dx)[-1],
                batch_shape=tf.shape(K_sym_dx)[:-2],
                dtype=K_sym_dx.dtype,
            )
            K_sym_dx = K_sym_dx + jitter_val * eye_k
            cond_per_dim = _cond_from_matrix(K_sym_dx, self.eps)
            condK = tf.reduce_max(cond_per_dim, axis=-1)

        condK_log10 = tf.math.log(condK + self.eps) / self.log10_base
        self._update_max("condK_log10_max", condK_log10)

        if clipped_frac is not None and "max_flow_frac" in self.metrics:
            self._update_max("max_flow_frac", clipped_frac)
