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
        return ssm.transition_dist(x_prev, y_prev=y_prev)

    def sample(self, ssm, x_prev, y_t, seed=None, **kwargs):
        y_prev = kwargs.get("y_prev")
        x = ssm.sample_transition(x_prev, seed=seed, y_prev=y_prev)
        return x, self.log_prob(ssm, x, x_prev, y_t, **kwargs)


__all__ = ["Proposal", "BootstrapProposal"]
