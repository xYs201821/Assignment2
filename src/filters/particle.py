import tensorflow as tf

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
    def _normalize_init_seed(seed):
        if seed is None:
            return None
        if isinstance(seed, (tuple, list)) and len(seed) == 2:
            return tf.convert_to_tensor(seed, dtype=tf.int32)
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        if seed.shape.rank == 1 and seed.shape[0] == 2:
            return seed
        if seed.shape.rank == 0:
            return tf.stack([seed, tf.constant(0, dtype=seed.dtype)])
        seed = tf.reshape(seed, [-1])
        return tf.stack([seed[0], tf.constant(0, dtype=seed.dtype)])
    def _init_particles(self, y, init_dist, init_seed=None):
        batch_shape = tf.shape(y)[:-2]
        N = self.num_particles
        seed = self._normalize_init_seed(init_seed)
        if init_dist is None:
            shape = tf.concat([batch_shape, [N]], axis=0)
            if seed is None:
                x = self.ssm.sample_initial_state(shape, return_log_prob=False)
            else:
                dist = self.ssm.initial_state_dist(shape)
                x = tf.cast(dist.sample(seed=seed), dtype=tf.float32)
        else:
            if not callable(init_dist):
                raise TypeError("init_dist must be a callable: init_dist(shape) -> tfd.Distribution")
            dist = init_dist(tf.concat([batch_shape, [N]], axis=0))
            sample_seed = self.ssm._tfp_seed() if seed is None else seed
            x = tf.cast(dist.sample(seed=sample_seed), dtype=tf.float32)
        log_w = -tf.math.log(tf.cast(N, tf.float32)) * tf.ones(tf.concat([batch_shape, [N]], axis=0), tf.float32)
        parent_indices = tf.broadcast_to(
            tf.range(N, dtype=tf.int32),
            tf.concat([batch_shape, [N]], axis=0),
        )
        return x, log_w, parent_indices

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

    def _normalize_y(self, y):
        y = tf.convert_to_tensor(y, tf.float32)
        if y.shape.rank == 2:
            y = y[tf.newaxis, ...]
        return y

    def step(self, x_prev, log_w_prev, y_t, resample="auto"):
        raise NotImplementedError
    
    def filter(self, y, num_particles=None, ess_threshold=None, resample="auto", init_dist=None, memory_sampler=None):
        raise NotImplementedError
