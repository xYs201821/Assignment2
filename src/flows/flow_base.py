import tensorflow as tf

from src.filters.particle import ParticleFilter
from src.filters.mixins import LinearizationMixin


class FlowBase(ParticleFilter, LinearizationMixin):
    """Shared scaffolding for EDH/LEDH flows."""

    def __init__(
        self,
        ssm,
        num_lambda=20,
        num_particles=100,
        ess_threshold=0.5,
        reweight="never",
        init_from_particles="sample",
        debug=False,
        jitter=1e-12,
    ):
        super().__init__(ssm, num_particles=num_particles, ess_threshold=ess_threshold)
        self.num_lambda = int(num_lambda)
        self.default_reweight = reweight
        self.init_from_particles = init_from_particles
        self.debug = bool(debug)
        self.jitter = jitter

    # --- Prior builders (can be overridden by subclasses) ---
    def _prior_from_sample(self, mu_tilde, w, batch_shape, state_dim):
        m = tf.einsum("...n,...ni->...i", w, mu_tilde)
        mu_resid = mu_tilde - m[..., tf.newaxis, :]
        P = tf.einsum("...n,...ni,...nj->...ij", w, mu_resid, mu_resid)
        return m, P

    # --- Abstract flow transport ---
    def _flow_transport(self, mu_tilde, y, m0, P):
        raise NotImplementedError

    def _propagate_particles(self, x_prev, seed=None):
        mu_tilde = self.ssm.sample_transition(x_prev, seed=seed)
        return mu_tilde, self.ssm.transition_dist(x_prev).log_prob(mu_tilde)

    def sample(self, x_prev, y_t, w=None, seed=None):
        batch_shape = tf.shape(x_prev)[:-2]
        state_dim = tf.shape(x_prev)[-1]

        mu_tilde, log_q0 = self._propagate_particles(x_prev, seed=seed)
        w = tf.convert_to_tensor(w)
        w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w)
        P_pred = self.ssm.state_cov(mu_tilde, w)
        m, P = self._prior_from_sample(mu_tilde, w, batch_shape, state_dim)

        x_next, log_det, _ = self._flow_transport(mu_tilde, y_t, m, P)
        return x_next, log_q0 - log_det, m_pred, P_pred

    def warmup(self, batch_size=1, reweight="auto"):
        x_spec = tf.TensorSpec(
            shape=[None, self.num_particles, self.ssm.state_dim],
            dtype=tf.float32,
        )
        y_spec = tf.TensorSpec(shape=[None, self.ssm.obs_dim], dtype=tf.float32)
        log_w_spec = tf.TensorSpec(shape=[None, self.num_particles], dtype=tf.float32)
        if reweight is None:
            reweight = self.default_reweight
        if reweight is None:
            reweight = self.default_reweight
        reweight = self._normalize_reweight(reweight)
        _ = self.step.get_concrete_function(
            x_spec,
            log_w_spec,
            y_spec,
            reweight=reweight,
        )

        x = tf.zeros([batch_size, self.num_particles, self.ssm.state_dim], dtype=tf.float32)
        log_w = tf.zeros([batch_size, self.num_particles], dtype=tf.float32)
        y = tf.zeros([batch_size, self.ssm.obs_dim], dtype=tf.float32)
        _ = self.step(x, log_w, y, reweight=reweight)

    @tf.function
    def step(
        self,
        x_prev,
        log_w_prev,
        y_t,
        reweight="auto",
        **kwargs,
    ):
        if reweight is None:
            reweight = self.default_reweight

        mu_tilde = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        trans_dist = self.ssm.transition_dist(x_prev)
        log_q0 = trans_dist.log_prob(mu_tilde)

        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w_prev)
        P_pred = self.ssm.state_cov(mu_tilde, w_prev)

        batch_shape = tf.shape(mu_tilde)[:-2]
        state_dim = tf.shape(mu_tilde)[-1]
        m, P = self._prior_from_sample(mu_tilde, w_prev, batch_shape, state_dim)
        x_t, log_det, _ = self._flow_transport(mu_tilde, y_t, m, P)

        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])

        update_weights = reweight != 0
        if update_weights:
            log_prior = trans_dist.log_prob(x_t)
            log_det = tf.convert_to_tensor(log_det, dtype=log_q0.dtype)
            rank_diff = tf.rank(log_q0) - tf.rank(log_det)

            def _pad_log_det():
                new_shape = tf.concat(
                    [tf.shape(log_det), tf.ones(rank_diff, dtype=tf.int32)],
                    axis=0,
                )
                return tf.reshape(log_det, new_shape)

            log_det = tf.cond(rank_diff > 0, _pad_log_det, lambda: log_det)
            log_q = log_q0 - log_det
            log_w = log_w_prev + tf.cast(loglik + log_prior - log_q, tf.float32)
        else:
            log_w = log_w_prev

        log_w_norm, w, _ = self._log_normalize(log_w)

        batch_shape_out = tf.shape(x_t)[:-2]
        parent_indices = tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape_out, [self.num_particles]], axis=0),
        )
        return mu_tilde, x_t, log_w_norm, w, parent_indices, m_pred, P_pred

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        reweight="auto",
        init_dist=None,
        init_seed=None,
        memory_sampler=None,
    ):
        self.update_params(num_particles, ess_threshold)
        y = self._normalize_y(y)
        reweight = self._normalize_reweight(reweight)

        T = int(y.shape[1])
        x_prev, log_w, parent_indices = self._init_particles(y, init_dist, init_seed=init_seed)

        x_ta = tf.TensorArray(tf.float32, size=T)
        x_pred_ta = tf.TensorArray(tf.float32, size=T)
        w_ta = tf.TensorArray(tf.float32, size=T)
        m_pred_ta = tf.TensorArray(tf.float32, size=T)
        P_pred_ta = tf.TensorArray(tf.float32, size=T)
        parent_ta = tf.TensorArray(tf.int32, size=T)
        step_time_ta = tf.TensorArray(tf.float32, size=T)
        mem_rss_ta = tf.TensorArray(tf.float32, size=T) if memory_sampler is not None else None
        mem_gpu_ta = tf.TensorArray(tf.float32, size=T) if memory_sampler is not None else None
        for t in range(T):
            t_start = tf.timestamp()
            y_t = y[..., t, :]
            x_pred, x, log_w, w, parent_indices, m_pred, P_pred = self.step(
                x_prev,
                log_w,
                y_t,
                reweight=reweight,
            )
            x_prev = x
            x_pred_ta = x_pred_ta.write(t, x_pred)
            x_ta = x_ta.write(t, x)
            w_ta = w_ta.write(t, w)
            m_pred_ta = m_pred_ta.write(t, m_pred)
            P_pred_ta = P_pred_ta.write(t, P_pred)
            parent_ta = parent_ta.write(t, parent_indices)
            step_time = tf.cast(tf.timestamp() - t_start, tf.float32)
            step_time_ta = step_time_ta.write(t, step_time)
            if memory_sampler is not None:
                sample = memory_sampler()
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
                if rss is not None:
                    mem_rss_ta = mem_rss_ta.write(t, tf.cast(rss, tf.float32))
                if mem_gpu_ta is not None:
                    gpu_val = 0.0 if gpu is None else gpu
                    mem_gpu_ta = mem_gpu_ta.write(t, tf.cast(gpu_val, tf.float32))

        x_seq = self._stack_and_permute(x_ta, tail_dims=2)
        w_seq = self._stack_and_permute(w_ta, tail_dims=1)
        parent_seq = self._stack_and_permute(parent_ta, tail_dims=1)

        diagnostics = {
            "step_time_s": self._stack_and_permute(step_time_ta, tail_dims=0),
            "m_pred": self._stack_and_permute(m_pred_ta, tail_dims=1),
            "P_pred": self._stack_and_permute(P_pred_ta, tail_dims=2),
            "x_pred": self._stack_and_permute(x_pred_ta, tail_dims=2),
        }
        if mem_rss_ta is not None:
            diagnostics["memory_rss"] = self._stack_and_permute(mem_rss_ta, tail_dims=0)
        if mem_gpu_ta is not None:
            diagnostics["memory_gpu"] = self._stack_and_permute(mem_gpu_ta, tail_dims=0)
        return x_seq, w_seq, diagnostics, parent_seq
