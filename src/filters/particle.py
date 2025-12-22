import tensorflow as tf

DEBUG_RECORD = {
    "print_weights": False,
    "print_weights_max_t": 3,
    "print_weights_classes": {"EDHFlow", "LEDHFlow"},
    "record": {
        "classes": {"EDHFlow", "LEDHFlow"},
        "keys": ["loglik", "logtrans", "log_q", "log_q0", "log_det", "log_w"],
    },
}

from src.filters.base import BaseFilter


class ParticleFilter(BaseFilter):
    def __init__(self, ssm, num_particles=100, ess_threshold=0.5, debug=False, print=False):
        super().__init__(ssm, debug=debug, print=print)
        self.num_particles = int(num_particles)
        self.ess_threshold = tf.convert_to_tensor(ess_threshold, tf.float32)
        self._maybe_print()

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        raise NotImplementedError
        
    @staticmethod
    def _log_normalize(log_w):
        logZ = tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
        log_w_norm = log_w - logZ
        w = tf.exp(log_w_norm)
        return log_w_norm, w, tf.squeeze(logZ, axis=-1)

    @staticmethod
    def ess(w):
        return 1.0 / tf.reduce_sum(tf.square(w), axis=-1)

    def update_params(self, num_particles=None, ess_threshold=None):
        if num_particles is not None:
            self.num_particles = int(num_particles)
        if ess_threshold is not None:
            self.ess_threshold = tf.convert_to_tensor(ess_threshold, tf.float32)

    @staticmethod
    def systematic_resample(w, rng):
        shape = tf.shape(w)
        N = shape[-1]
        B = tf.reduce_prod(shape[:-1])
        w2 = tf.reshape(w, [B, N])
        cdf = tf.cumsum(w2, axis=-1)

        u0 = rng.uniform([B, 1], 0.0, 1.0 / tf.cast(N, tf.float32), dtype=tf.float32)
        js = tf.cast(tf.range(N)[tf.newaxis, :], tf.float32)
        u = u0 + js / tf.cast(N, tf.float32)

        idx = tf.searchsorted(cdf, u, side="left")
        idx = tf.clip_by_value(idx, 0, N - 1)
        return tf.reshape(idx, shape)

    @staticmethod
    def resample_particles(x, idx):
        shape = tf.shape(x)
        B = tf.reduce_prod(shape[:-2])
        x_flatten = tf.reshape(x, [B, shape[-2], shape[-1]])
        idx_flatten = tf.reshape(idx, [B, shape[-2]])
        out_flatten = tf.gather(x_flatten, idx_flatten, batch_dims=1)
        return tf.reshape(out_flatten, shape)

    def step(self, x_prev, log_w_prev, y_t, reweight="auto", **kwargs):
        x_t, log_q = self.sample(
            self.ssm,
            x_prev,
            y_t,
            seed=self.ssm._tfp_seed(),
            **kwargs,
        )

        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])
        logtrans = self.ssm.transition_dist(x_prev).log_prob(x_t)

        log_w = log_w_prev + tf.cast(loglik + logtrans - log_q, tf.float32)

        log_w_norm, w, logZ = self._log_normalize(log_w)
        ess = self.ess(w)

        if reweight in (1, 2):
            N_float = tf.cast(self.num_particles, tf.float32)
            if reweight == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w, self.ssm.rng)

            batch_shape = tf.shape(x_t)[:-2]
            no_rs_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            mask_do_rs = mask_do_rs[..., tf.newaxis]
            parent_indices = tf.where(mask_do_rs, rs_indices, no_rs_indices)

            x_t = self.resample_particles(x_t, parent_indices)
            log_w_reset = -tf.math.log(N_float) * tf.ones_like(log_w_norm)
            log_w_final = tf.where(mask_do_rs, log_w_reset, log_w_norm)
        else:
            batch_shape = tf.shape(x_t)[:-2]
            parent_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            log_w_final = log_w_norm
        w_final = tf.exp(log_w_final)
        flow_info = {
            "ess": ess,
            "logZ": logZ,
            "parent_indices": parent_indices,
        }
        return x_t, log_w_final, w_final, None, flow_info

    @staticmethod
    def _normalize_reweight(reweight):
        if isinstance(reweight, bool):
            return 1 if reweight else 0
        if isinstance(reweight, str):
            mode_map = {"never": 0, "auto": 1, "always": 2}
            if reweight not in mode_map:
                raise ValueError("reweight must be True/False or one of 'auto', 'never', 'always'")
            return mode_map[reweight]
        if isinstance(reweight, int):
            if reweight not in (0, 1, 2):
                raise ValueError("reweight int must be 0 (never), 1 (auto), or 2 (always)")
            return reweight
        raise ValueError("reweight must be bool, int, or one of 'auto', 'never', 'always'")

    # Backward-compatible alias for callers still using the old name.
    _normalize_resample = _normalize_reweight

    def _normalize_y(self, y):
        y = tf.convert_to_tensor(y, tf.float32)
        assert_rank = tf.debugging.assert_greater_equal(
            tf.rank(y),
            2,
            message="y must have shape [..., T, dy] or [T, dy]",
        )
        with tf.control_dependencies([assert_rank]):
            y = tf.identity(y)
        y = tf.cond(tf.equal(tf.rank(y), 2), lambda: y[tf.newaxis, ...], lambda: y)
        if y.shape.rank is not None:
            batch_dim = y.shape[0] if y.shape.rank >= 1 else None
            time_dim = y.shape[1] if y.shape.rank >= 2 else None
            y = tf.ensure_shape(y, [batch_dim, time_dim, self.ssm.obs_dim])
        dy = tf.cast(self.ssm.obs_dim, tf.int32)
        tf.debugging.assert_equal(
            tf.shape(y)[-1],
            dy,
            message="Last dimension of y must match ssm.obs_dim",
        )
        return y

    def _init_state(self, y, init_dist):
        batch_shape = tf.shape(y)[:-2]
        N = self.num_particles
        if init_dist is None:
            x = self.ssm.sample_initial_state(tf.concat([batch_shape, [N]], axis=0), return_log_prob=False)
        else:
            if not callable(init_dist):
                raise TypeError("init_dist must be a callable: init_dist(shape) -> tfd.Distribution")
            dist = init_dist(tf.concat([batch_shape, [N]], axis=0))
            x = tf.cast(dist.sample(seed=self.ssm._tfp_seed()), dtype=tf.float32)
        log_w = -tf.math.log(tf.cast(N, tf.float32)) * tf.ones(tf.concat([batch_shape, [N]], axis=0), tf.float32)
        parent_indices = tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape, [self.num_particles]], axis=0),
        )
        state_dim = tf.cast(self.ssm.state_dim, tf.int32)
        init_mode = getattr(self, "init_from_particles", False)
        if init_mode:
            if init_mode == "sample":
                w0 = tf.exp(log_w)
                m0 = self.ssm.state_mean(x, w0)
                P0 = self.ssm.state_cov(x, w0)
            else:
                m0 = x
                if hasattr(self.ssm, "P0"):
                    P0 = tf.convert_to_tensor(self.ssm.P0, dtype=tf.float32)
                else:
                    P0 = tf.eye(state_dim, dtype=tf.float32)
                P0 = tf.broadcast_to(
                    P0,
                    tf.concat([batch_shape, [self.num_particles, state_dim, state_dim]], axis=0),
                )
        else:
            if hasattr(self.ssm, "m0") and hasattr(self.ssm, "P0"):
                m0 = tf.convert_to_tensor(self.ssm.m0, dtype=tf.float32)
                P0 = tf.convert_to_tensor(self.ssm.P0, dtype=tf.float32)
            else:
                m0 = tf.zeros([state_dim], dtype=tf.float32)
                P0 = tf.eye(state_dim, dtype=tf.float32)
        tracker = {
            "m": m0,
            "P": P0,
        }
        return x, log_w, tracker, parent_indices

    def filter(self, y, num_particles=None, ess_threshold=None, reweight="auto", resample=None, init_dist=None):
        self.update_params(num_particles, ess_threshold)
        y = self._normalize_y(y)
        if resample is not None:
            reweight = resample
        reweight = self._normalize_reweight(reweight)

        batch_shape = tf.shape(y)[:-2]
        T = tf.shape(y)[-2]
        x, log_w, tracker, parent_indices = self._init_state(y, init_dist)

        ess_ta = tf.TensorArray(tf.float32, size=T)
        logZ_ta = tf.TensorArray(tf.float32, size=T)
        x_ta = tf.TensorArray(tf.float32, size=T)
        w_ta = tf.TensorArray(tf.float32, size=T)
        log_w_ta = tf.TensorArray(tf.float32, size=T)
        parent_ta = tf.TensorArray(tf.int32, size=T)
        step_time_ta = tf.TensorArray(tf.float32, size=T)
        class_name = type(self).__name__
        record_cfg = DEBUG_RECORD.get("record", {})
        record_keys = list(record_cfg.get("keys", []))
        record_enabled = class_name in record_cfg.get("classes", set())
        record_ta = tf.TensorArray(tf.float32, size=T)

        for t in tf.range(T):
            t_start = tf.timestamp()
            y_t = y[..., t, :]
            step_out = self.step(x, log_w, y_t, reweight=reweight, tracker=tracker)
            flow_info = None
            parent_out = parent_indices
            x, log_w, w, tracker_out, flow_info = step_out
            if tracker_out is None:
                tracker_out = tracker
            tracker = tracker_out
            if flow_info is not None:
                ess = flow_info.get("ess", None)
                logZ = flow_info.get("logZ", None)
                parent_out = flow_info.get("parent_indices", parent_out)
            else:
                ess = None
                logZ = None
            parent_indices = parent_out
            if ess is None:
                ess = self.ess(w)
            if logZ is None:
                log_w_norm, _, logZ = self._log_normalize(log_w)
            if (
                DEBUG_RECORD.get("print_weights", False)
                and class_name in DEBUG_RECORD.get("print_weights_classes", set())
                and t <= DEBUG_RECORD.get("print_weights_max_t", 10)
            ):
                tf.print("t", t, "weights", w, summarize=-1)
                tf.print(
                    "t",
                    t,
                    "sum_w",
                    tf.reduce_sum(w, axis=-1),
                    "min_w",
                    tf.reduce_min(w, axis=-1),
                    "max_w",
                    tf.reduce_max(w, axis=-1),
                    summarize=-1,
                )
            if record_enabled:
                if flow_info is None:
                    raise ValueError("record is enabled but step did not provide flow_info.")
                values = []
                base = None
                for key in record_keys:
                    value = flow_info.get(key, None)
                    if value is None:
                        raise ValueError(f"record is enabled but step did not provide '{key}'.")
                    if base is None:
                        base = tf.shape(value)
                    value = tf.broadcast_to(value, base)
                    values.append(value)
                record_ta = record_ta.write(t, tf.stack(values, axis=-1))
            ess_ta = ess_ta.write(t, ess)
            logZ_ta = logZ_ta.write(t, logZ)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            log_w_ta = log_w_ta.write(t, log_w)
            parent_ta = parent_ta.write(t, parent_indices)
            step_time = tf.cast(tf.timestamp() - t_start, tf.float32)
            step_time_ta = step_time_ta.write(t, step_time)

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        log_w_seq = self._stack_and_permute(log_w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)

        diagnostics = {
            "ess": self._stack_and_permute(ess_ta, tail_dims=0),
            "logZ": self._stack_and_permute(logZ_ta, tail_dims=0),
            "log_w": log_w_seq,
            "step_time_s": self._stack_and_permute(step_time_ta, tail_dims=0),
        }
        if record_enabled:
            record_seq = self._stack_and_permute(record_ta, tail_dims=2)
            for idx, key in enumerate(record_keys):
                diagnostics[key] = record_seq[..., idx]
        if tracker is not None:
            diagnostics["tracker"] = tracker
        return x_seq, w_seq, diagnostics, parent_seq

class BootstrapParticleFilter(ParticleFilter):
    def __init__(self, ssm, num_particles=100, ess_threshold=0.5, debug=False, print=False):
        super().__init__(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            debug=debug,
            print=print,
        )

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        x_t = ssm.sample_transition(x_prev, seed=seed)
        return x_t, ssm.transition_dist(x_prev).log_prob(x_t)

    def log_prob(self, ssm, x, x_prev, y_t):
        return ssm.transition_dist(x_prev).log_prob(x)
