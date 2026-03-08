"""Proposal interfaces shared by particle filters."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Proposal(tf.Module):
    """Base protocol for proposal q(x_t | x_{t-1}, y_t)."""

    def __init__(self) -> None:
        super().__init__()

    def dist(self, ssm, x_prev, y_t, **kwargs):
        raise NotImplementedError

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        z = self.dist(ssm, x_prev, y_t, **kwargs).sample(seed=seed)
        return z, self.log_prob(ssm, z, x_prev, y_t, **kwargs)

    def log_prob(self, ssm, x, x_prev, y_t, **kwargs):
        return self.dist(ssm, x_prev, y_t, **kwargs).log_prob(x)


class BootstrapProposal(Proposal):
    """Default prior proposal q(x_t | x_{t-1}) from model transition."""

    def dist(self, ssm, x_prev, y_t, **kwargs):
        del y_t
        y_prev = kwargs.get("y_prev")
        time_index = kwargs.get("time_index")
        return ssm.transition_dist(x_prev, y_prev=y_prev, time_index=time_index)

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        y_prev = kwargs.get("y_prev")
        time_index = kwargs.get("time_index")
        x = ssm.sample_transition(x_prev, seed=seed, y_prev=y_prev, time_index=time_index)
        return x, self.log_prob(ssm, x, x_prev, y_t, **kwargs)


class FlowTransportProposal(Proposal):
    """Proposal built from invertible particle-flow transport (LEDH/EDH)."""

    def __init__(
        self,
        ssm=None,
        flow_kind: str = "ledh",
        num_lambda: int = 20,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        reweight: str | int | bool = "always",
        jitter: float = 1e-5,
        beta_schedule=None,
    ) -> None:
        super().__init__()
        del ssm
        kind = str(flow_kind).strip().lower()
        if kind not in ("ledh", "edh"):
            raise ValueError("flow_kind must be one of {'ledh', 'edh'}.")
        self.flow_kind = kind
        self.num_lambda = int(num_lambda)
        self.num_particles = int(num_particles)
        self.ess_threshold = float(ess_threshold)
        self.reweight = reweight
        self.jitter = float(jitter)
        self.beta_schedule = beta_schedule

    def _build_flow(self, ssm):
        if self.flow_kind == "ledh":
            from src.flows.ledh import LEDHFlow

            return LEDHFlow(
                ssm,
                num_lambda=self.num_lambda,
                num_particles=self.num_particles,
                ess_threshold=self.ess_threshold,
                reweight=self.reweight,
                jitter=self.jitter,
                beta_schedule=self.beta_schedule,
            )
        from src.flows.edh import EDHFlow

        return EDHFlow(
            ssm,
            num_lambda=self.num_lambda,
            num_particles=self.num_particles,
            ess_threshold=self.ess_threshold,
            reweight=self.reweight,
            jitter=self.jitter,
            beta_schedule=self.beta_schedule,
        )

    def dist(self, ssm, x_prev, y_t, **kwargs):
        raise NotImplementedError(
            "FlowTransportProposal has no explicit dist(...); use sample(...) to obtain (x, log_q)."
        )

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        x_prev = tf.convert_to_tensor(x_prev, dtype=tf.float32)
        y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
        w_prev = None
        log_w_prev = kwargs.get("log_w_prev")
        if log_w_prev is not None:
            log_w_prev = tf.convert_to_tensor(log_w_prev, dtype=tf.float32)
            w_prev = tf.exp(log_w_prev)
            w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        flow = self._build_flow(ssm)
        x_next, log_q = flow.sample(x_prev=x_prev, y_t=y_t, w=w_prev, seed=seed)
        x_next = tf.ensure_shape(tf.cast(x_next, tf.float32), x_prev.shape)
        log_q = tf.ensure_shape(tf.cast(log_q, tf.float32), x_prev.shape[:-1])
        return x_next, log_q

    def log_prob(self, ssm, x, x_prev, y_t, **kwargs):
        raise NotImplementedError(
            "FlowTransportProposal.log_prob is not implemented; use sample(...) that returns log_q."
        )


__all__ = [
    "Proposal",
    "BootstrapProposal",
    "FlowTransportProposal",
]
