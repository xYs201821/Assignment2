from __future__ import annotations
import tensorflow as tf
from src.filters.particle import ParticleFilter


class BootstrapParticleFilter(ParticleFilter):
    def __init__(self, ssm, num_particles=100, ess_threshold=0.5, resample="auto", debug=False, print=False):
        super().__init__(ssm, num_particles=num_particles, ess_threshold=ess_threshold, debug=debug, print=print)
        self.resample = self._normalize_reweight(resample)

    def update_params(self, num_particles=None, ess_threshold=None, resample=None):
        if num_particles is not None:
            self.num_particles = int(num_particles)
        if ess_threshold is not None:
            self.ess_threshold = tf.convert_to_tensor(ess_threshold, tf.float32)
        if resample is not None:
            self.resample = self._normalize_reweight(resample)

    def warmup(self, batch_size=1, resample="auto"):
        x_spec = tf.TensorSpec(shape=[batch_size, self.num_particles, self.ssm.state_dim], dtype=tf.float32)
        y_spec = tf.TensorSpec(shape=[batch_size, self.ssm.obs_dim], dtype=tf.float32)
        log_w_spec = tf.TensorSpec(shape=[batch_size, self.num_particles], dtype=tf.float32)
        resample = self._normalize_reweight(resample)
        _ = self.step.get_concrete_function(x_spec, log_w_spec, y_spec, resample=resample)

        x = tf.zeros([batch_size, self.num_particles, self.ssm.state_dim], dtype=tf.float32)
        log_w = tf.zeros([batch_size, self.num_particles], dtype=tf.float32)
        y = tf.zeros([batch_size, self.ssm.obs_dim], dtype=tf.float32)
        _ = self.step(x, log_w, y, resample=resample)

    @tf.function
    def step(self, x_prev, log_w_prev, y_t, resample="auto"):
        x_pred = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(x_pred, w_prev)
        P_pred = self.ssm.state_cov(x_pred, w_prev)
        loglik = self.ssm.observation_dist(x_pred).log_prob(y_t[..., tf.newaxis, :])
        log_w = log_w_prev + tf.cast(loglik, tf.float32)

        log_w_norm, w, _ = self._log_normalize(log_w)
        ess = self.ess(w)

        if resample in (1, 2):
            N_float = tf.cast(self.num_particles, tf.float32)
            if resample == 2:
                mask_do_rs = tf.ones_like(ess, dtype=tf.bool)
            else:
                mask_do_rs = ess < (self.ess_threshold * N_float)

            rs_indices = self.systematic_resample(w, self.ssm.rng)
            batch_shape = tf.shape(x_pred)[:-2]
            no_rs_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            mask_do_rs = mask_do_rs[..., tf.newaxis]
            parent_indices = tf.where(mask_do_rs, rs_indices, no_rs_indices)

            x_t = self.resample_particles(x_pred, parent_indices)
            log_w_reset = -tf.math.log(N_float) * tf.ones_like(log_w_norm)
            log_w_final = tf.where(mask_do_rs, log_w_reset, log_w_norm)
        else:
            batch_shape = tf.shape(x_pred)[:-2]
            parent_indices = tf.broadcast_to(
                tf.range(self.num_particles, dtype=tf.int32),
                tf.concat([batch_shape, [self.num_particles]], axis=0),
            )
            log_w_final = log_w_norm
            x_t = x_pred

        w_final = tf.exp(log_w_final)
        return x_pred, x_t, log_w_final, w_final, parent_indices, m_pred, P_pred

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample="auto",
        init_dist=None,
        memory_sampler=None,
    ):
        self.update_params(num_particles, ess_threshold)
        y = self._normalize_y(y)
        resample = self._normalize_reweight(resample)

        T = int(y.shape[1])
        x_prev, log_w, parent_indices = self._init_particles(y, init_dist)

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
            y_t = y[:, t, :]
            x_pred, x, log_w, w, parent_indices, m_pred, P_pred = self.step(
                x_prev,
                log_w,
                y_t,
                resample=resample,
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
