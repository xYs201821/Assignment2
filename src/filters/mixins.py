"""Mixins for filter linearization utilities."""

import tensorflow as tf


class LinearizationMixin:
    """Provides Jacobian helpers for nonlinear models."""

    @tf.function(reduce_retracing=True)
    def jacobian_f_x(self, x, q):
        """Jacobian of transition function with respect to state.

        Shapes:
          x: [B, dx]
          q: [B, dq]
        Returns:
          J_x(f): [B, dx, dx]
          f(x,q): [B, dx]
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ssm.f_with_noise(x, q)
        return tape.batch_jacobian(y, x), y

    @tf.function(reduce_retracing=True)
    def jacobian_f_q(self, x, q):
        """Jacobian of transition function with respect to process noise.

        Shapes:
          x: [B, dx]
          q: [B, dq]
        Returns:
          J_q(f): [B, dx, dq]
          f(x,q): [B, dx]
        """
        with tf.GradientTape() as tape:
            tape.watch(q)
            y = self.ssm.f_with_noise(x, q)
        return tape.batch_jacobian(y, q), y

    @tf.function(reduce_retracing=True)
    def jacobian_h_x(self, x, r):
        """Jacobian of observation function with respect to state.

        Shapes:
          x: [B, dx]
          r: [B, dr]
        Returns:
          J_x(h): [B, dy, dx]
          h(x,r): [B, dy]
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ssm.h_with_noise(x, r)
        return tape.batch_jacobian(y, x), y

    @tf.function(reduce_retracing=True)
    def jacobian_h_r(self, x, r):
        """Jacobian of observation function with respect to observation noise.

        Shapes:
          x: [B, dx]
          r: [B, dr]
        Returns:
          J_r(h): [B, dy, dr]
          h(x,r): [B, dy]
        """
        with tf.GradientTape() as tape:
            tape.watch(r)
            y = self.ssm.h_with_noise(x, r)
        return tape.batch_jacobian(y, r), y

    def _jacobian(self, func, x):
        """Generic Jacobian helper for custom functions."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.batch_jacobian(y, x), y
