from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.hmc.pure_resamplers import (
    OTResampler,
    SoftResampler,
    StandardResampler,
    normalize_log_weights,
    split_seed,
    to_stateless_seed,
)


@dataclass
class PurePFConfig:
    num_particles: int
    ess_threshold: float
    resample: str  # {'never','auto','always'}


class PureParticleFilter:
    def __init__(self, ssm, proposal, resampler, cfg: PurePFConfig):
        self.ssm = ssm
        self.proposal = proposal
        self.resampler = resampler
        self.cfg = cfg
        self._resampler_needs_seed = isinstance(resampler, (StandardResampler, SoftResampler))

    @staticmethod
    def _log_normalize(log_w: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return normalize_log_weights(log_w)

    @staticmethod
    def _ess(w: tf.Tensor) -> tf.Tensor:
        return 1.0 / tf.reduce_sum(tf.square(w), axis=-1)

    def _should_resample(self, w: tf.Tensor, mode: str) -> tf.Tensor:
        mode = str(mode).strip().lower()
        if mode == "always":
            return tf.ones(tf.shape(w)[:-1], dtype=tf.bool)
        if mode == "never":
            return tf.zeros(tf.shape(w)[:-1], dtype=tf.bool)
        ess = self._ess(w)
        return ess < (float(self.cfg.ess_threshold) * tf.cast(self.cfg.num_particles, tf.float32))

    @tf.function(reduce_retracing=True)
    def filter(
        self,
        y: tf.Tensor,
        *,
        params: dict[str, tf.Tensor],
        seed: tf.Tensor | int,
        resample: str | None = None,
    ):
        y = tf.convert_to_tensor(y, tf.float32)
        b = tf.shape(y)[0]
        t_int = tf.shape(y)[1]
        n = int(self.cfg.num_particles)
        resample_mode = self.cfg.resample if resample is None else str(resample)

        seeds = split_seed(to_stateless_seed(seed), t_int + 1)
        init_dist = self.ssm.initial_state_dist(tf.stack([b, tf.constant(n, tf.int32)]))
        x_prev0 = tf.cast(init_dist.sample(seed=seeds[0]), tf.float32)
        log_w_prev0 = -tf.math.log(tf.cast(n, tf.float32)) * tf.ones([b, n], tf.float32)

        logz_ta = tf.TensorArray(tf.float32, size=t_int)
        x_ta = tf.TensorArray(tf.float32, size=t_int)
        w_ta = tf.TensorArray(tf.float32, size=t_int)
        parent_ta = tf.TensorArray(tf.int32, size=t_int)

        def cond(k, *_):
            return k < t_int

        def body(k, x_prev, log_w_prev, logz_acc, x_acc, w_acc, parent_acc):
            y_t = y[:, k, :]
            y_prev = tf.cond(
                tf.equal(k, 0),
                lambda: tf.zeros_like(y_t),
                lambda: y[:, k - 1, :],
            )
            x_pred, log_q = self.proposal.sample(
                self.ssm,
                x_prev,
                y_t,
                params=params,
                seed=seeds[k + 1],
                time_index=k,
                y_prev=y_prev,
                log_w_prev=log_w_prev,
            )
            # Restore static shape info lost by proposals with dynamic ops (e.g. LEDH).
            x_pred = tf.ensure_shape(x_pred, x_prev.shape)
            log_q  = tf.ensure_shape(log_q,  log_w_prev.shape)

            loglik = tf.cast(
                self.ssm.observation_dist(x_pred, params=params).log_prob(y_t[:, tf.newaxis, :]),
                tf.float32,
            )
            logf = tf.cast(
                self.ssm.transition_dist(x_prev, y_prev=y_prev, params=params, time_index=k).log_prob(x_pred),
                tf.float32,
            )
            log_w = log_w_prev + loglik + (logf - log_q)
            log_w_n, w_pre, logz_t = self._log_normalize(log_w)
            do_rs = self._should_resample(w_pre, resample_mode)

            if self._resampler_needs_seed:
                rs_seed = seeds[k + 1] + tf.constant([42, 1024], tf.int32)
                x_rs, log_w_rs, parent_idx = self.resampler.resample(x_pred, log_w_n, rs_seed)
            else:
                x_rs, log_w_rs, parent_idx = self.resampler.resample(x_pred, log_w_n)

            no_parent = tf.broadcast_to(tf.range(n, dtype=tf.int32), [b, n])
            x_t = tf.where(do_rs[:, tf.newaxis, tf.newaxis], x_rs, x_pred)
            log_w_next = tf.where(do_rs[:, tf.newaxis], log_w_rs, log_w_n)
            parent = tf.where(do_rs[:, tf.newaxis], parent_idx, no_parent)

            logz_acc = logz_acc.write(k, logz_t)
            x_acc = x_acc.write(k, x_t)
            w_acc = w_acc.write(k, tf.exp(log_w_next))
            parent_acc = parent_acc.write(k, parent)
            return k + 1, x_t, log_w_next, logz_acc, x_acc, w_acc, parent_acc

        _, _, _, logz_ta, x_ta, w_ta, parent_ta = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                x_prev0,
                log_w_prev0,
                logz_ta,
                x_ta,
                w_ta,
                parent_ta,
            ),
            shape_invariants=(
                tf.TensorShape([]),
                x_prev0.shape,
                log_w_prev0.shape,
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
            ),
            parallel_iterations=1,
        )

        x_seq = tf.transpose(x_ta.stack(), perm=[1, 0, 2, 3])
        w_seq = tf.transpose(w_ta.stack(), perm=[1, 0, 2])
        parent_seq = tf.transpose(parent_ta.stack(), perm=[1, 0, 2])
        logz_seq = tf.transpose(logz_ta.stack(), perm=[1, 0])
        diagnostics = {
            "x": x_seq,
            "w": w_seq,
            "log_z": logz_seq,
            "parent_index": parent_seq,
        }
        return x_seq, w_seq, diagnostics, parent_seq


def build_resampler(kind: str, *, ot_epsilon: float, ot_num_iters: int, ot_jitter: float, soft_lam: float = 0.95):
    k = str(kind).strip().lower()
    if k == "standard":
        return StandardResampler()
    if k == "soft":
        return SoftResampler(lam=float(soft_lam))
    if k == "ot":
        return OTResampler(epsilon=float(ot_epsilon), num_iters=int(ot_num_iters), jitter=float(ot_jitter))
    raise ValueError("inner_pf must be one of {'standard', 'soft', 'ot'}")
