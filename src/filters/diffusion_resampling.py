"""Diffusion-based differentiable resampling and DPF variant."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.filters.dpf import DPFBase


class DiffusionResamplingDPF(DPFBase):
    """Differentiable PF using diffusion-based resampling."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        diff_a: float = -1.0,
        diff_T: float = 1.0,
        diff_steps: int = 8,
        diff_ode: bool = True,
        diff_eps: float = 1e-6,
        resample: str | int | bool = "auto",
        debug: bool = False,
        print: bool = False,
        proposal=None,
    ) -> None:
        super().__init__(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            debug=debug,
            print=print,
            proposal=proposal,
        )
        self.diff_a = float(diff_a)
        self.diff_T = float(diff_T)
        self.diff_steps = int(diff_steps)
        self.diff_ode = bool(diff_ode)
        self.diff_eps = float(diff_eps)
        self._validate_diffusion_cfg()

    def _validate_diffusion_cfg(self) -> None:
        if self.diff_a >= 0.0:
            raise ValueError("diff_a must be negative.")
        if self.diff_T <= 0.0:
            raise ValueError("diff_T must be positive.")
        if self.diff_steps <= 0:
            raise ValueError("diff_steps must be a positive integer.")
        if self.diff_eps <= 0.0:
            raise ValueError("diff_eps must be positive.")

    def _time_grid(self, dtype: tf.DType) -> tf.Tensor:
        return tf.linspace(tf.cast(0.0, dtype), tf.cast(self.diff_T, dtype), self.diff_steps + 1)

    def update_params(
        self,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        diff_a=None,
        diff_T=None,
        diff_steps=None,
        diff_ode=None,
        diff_eps=None,
    ):
        super().update_params(
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )
        if diff_a is not None:
            self.diff_a = float(diff_a)
        if diff_T is not None:
            self.diff_T = float(diff_T)
        if diff_steps is not None:
            self.diff_steps = int(diff_steps)
        if diff_ode is not None:
            self.diff_ode = bool(diff_ode)
        if diff_eps is not None:
            self.diff_eps = float(diff_eps)
        self._validate_diffusion_cfg()

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor):
        """Diffusion differentiable resampling core implementation."""
        ts = self._time_grid(x.dtype)
        eps_t = tf.cast(self.diff_eps, x.dtype)
        a_t = tf.cast(self.diff_a, x.dtype)
        ode = self.diff_ode
        #log_w = tf.stop_gradient(log_w)
        log_w = log_w - tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)
        w = tf.exp(log_w)

        mu_base = self.ssm.state_mean(x, w)
        mu = mu_base[:, tf.newaxis, :]
        centered = self.ssm.state_residual(x, mu_base)
        stat_vars = tf.reduce_sum(w[..., tf.newaxis] * tf.square(centered), axis=-2, keepdims=True)
        stat_vars = tf.maximum(stat_vars, eps_t)
        b2 = tf.maximum(-stat_vars * (2.0 * a_t), eps_t)

        t0 = ts[0]
        T = ts[-1]
        nsteps = tf.shape(ts)[0] - 1
        log2pi = tf.cast(tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32)), x.dtype)

        seed = self.ssm._tfp_seed()
        split = tf.random.experimental.stateless_split(seed, num=3)
        eps0 = tf.random.stateless_normal(tf.shape(x), seed=split[0], dtype=x.dtype)
        if not ode:
            rnd_shape = tf.concat([tf.reshape(nsteps, [1]), tf.shape(x)], axis=0)
            rnds = tf.random.stateless_normal(rnd_shape, seed=split[1], dtype=x.dtype)

        x_t = mu + tf.sqrt(stat_vars) * eps0
        final_alps = tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1]], dtype=x.dtype)

        def _ensemble_score(curr_x: tf.Tensor, t_fwd: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            delta = t_fwd - t0
            sg = tf.exp(a_t * delta)
            sig2 = stat_vars * (1.0 - tf.exp(2.0 * a_t * delta))
            sig2 = tf.maximum(sig2, eps_t)
            inv_sig2 = 1.0 / sig2

            mu_t = x * sg + mu * (1.0 - sg)

            x_inv = curr_x * inv_sig2
            m_inv = mu_t * inv_sig2

            quad_a = tf.reduce_sum(curr_x * x_inv, axis=-1)
            quad_b = tf.reduce_sum(mu_t * m_inv, axis=-1)
            cross = tf.matmul(curr_x, m_inv, transpose_b=True)
            quad = quad_a[..., :, tf.newaxis] - 2.0 * cross + quad_b[..., tf.newaxis, :] # quadratic term of q(x_t | x_0)

            log_det = tf.reduce_sum(log2pi + tf.math.log(sig2), axis=-1)[..., tf.newaxis] # determinant term of q(x_t | x_0)
            logpdf = -0.5 * (quad + log_det)
            log_alps = logpdf + log_w[:, tf.newaxis, :]
            
            alps = tf.nn.softmax(log_alps, axis=-1)
            score = tf.matmul(alps, m_inv) - x_inv

            return score, alps

        for k in tf.range(nsteps):
            t_prev = ts[k]
            dt = ts[k + 1] - t_prev
            score, alps = _ensemble_score(x_t, T - t_prev)
            alps = tf.ensure_shape(alps, final_alps.shape)
            final_alps = alps

            if ode:
                f = a_t * mu + 0.5 * (b2 * score)
            else:
                f = a_t * mu + (b2 * score)
            drift = -a_t * x_t + f
            x_t = x_t + drift * dt

            if not ode:
                x_t = x_t + tf.sqrt(dt) * tf.sqrt(b2) * rnds[k]

        n_particles = tf.shape(x_t)[-2]
        log_uniform = -tf.math.log(tf.cast(n_particles, x.dtype))
        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        parent_indices = tf.argmax(final_alps, axis=-1, output_type=tf.int32)
        x_t = tf.ensure_shape(x_t, x.shape)
        log_w_new = tf.ensure_shape(log_w_new, log_w.shape)
        parent_indices = tf.ensure_shape(parent_indices, log_w.shape)
        return x_t, log_w_new, parent_indices

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        diff_a=None,
        diff_T=None,
        diff_steps=None,
        diff_ode=None,
        diff_eps=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        if any(
            v is not None
            for v in (
                num_particles,
                ess_threshold,
                resample,
                proposal,
                diff_a,
                diff_T,
                diff_steps,
                diff_ode,
                diff_eps,
            )
        ):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                resample=resample,
                proposal=proposal,
                diff_a=diff_a,
                diff_T=diff_T,
                diff_steps=diff_steps,
                diff_ode=diff_ode,
                diff_eps=diff_eps,
            )
        return super().filter(
            y,
            num_particles=None,
            ess_threshold=None,
            resample=None,
            proposal=None,
            init_dist=init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )


__all__ = [
    "DiffusionResamplingDPF",
]
