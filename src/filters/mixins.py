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
        ssm_jac = getattr(self.ssm, "jacobian_f_x", None)
        if callable(ssm_jac):
            return ssm_jac(x, q)
        x = tf.convert_to_tensor(x)
        q = tf.convert_to_tensor(q)
        batch_shape = tf.shape(x)[:-1]
        x_dim = tf.shape(x)[-1]
        q_dim = tf.shape(q)[-1]
        x_flat = tf.reshape(x, tf.stack([-1, x_dim]))
        q_flat = tf.reshape(q, tf.stack([-1, q_dim]))
        with tf.GradientTape() as tape:
            tape.watch(x_flat)
            y_flat = self.ssm.f_with_noise(x_flat, q_flat)
        J_flat = tape.batch_jacobian(y_flat, x_flat)
        y_dim = tf.shape(y_flat)[-1]
        y = tf.reshape(y_flat, tf.concat([batch_shape, [y_dim]], axis=0))
        J = tf.reshape(J_flat, tf.concat([batch_shape, [y_dim, x_dim]], axis=0))
        return J, y

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
        ssm_jac = getattr(self.ssm, "jacobian_f_q", None)
        if callable(ssm_jac):
            return ssm_jac(x, q)
        x = tf.convert_to_tensor(x)
        q = tf.convert_to_tensor(q)
        batch_shape = tf.shape(x)[:-1]
        x_dim = tf.shape(x)[-1]
        q_dim = tf.shape(q)[-1]
        x_flat = tf.reshape(x, tf.stack([-1, x_dim]))
        q_flat = tf.reshape(q, tf.stack([-1, q_dim]))
        with tf.GradientTape() as tape:
            tape.watch(q_flat)
            y_flat = self.ssm.f_with_noise(x_flat, q_flat)
        J_flat = tape.batch_jacobian(y_flat, q_flat)
        y_dim = tf.shape(y_flat)[-1]
        y = tf.reshape(y_flat, tf.concat([batch_shape, [y_dim]], axis=0))
        J = tf.reshape(J_flat, tf.concat([batch_shape, [y_dim, q_dim]], axis=0))
        return J, y

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
        ssm_jac = getattr(self.ssm, "jacobian_h_x", None)
        if callable(ssm_jac):
            return ssm_jac(x, r)
        x = tf.convert_to_tensor(x)
        r = tf.convert_to_tensor(r)
        batch_shape = tf.shape(x)[:-1]
        x_dim = tf.shape(x)[-1]
        r_dim = tf.shape(r)[-1]
        x_flat = tf.reshape(x, tf.stack([-1, x_dim]))
        r_flat = tf.reshape(r, tf.stack([-1, r_dim]))
        with tf.GradientTape() as tape:
            tape.watch(x_flat)
            y_flat = self.ssm.h_with_noise(x_flat, r_flat)
        J_flat = tape.batch_jacobian(y_flat, x_flat)
        y_dim = tf.shape(y_flat)[-1]
        y = tf.reshape(y_flat, tf.concat([batch_shape, [y_dim]], axis=0))
        J = tf.reshape(J_flat, tf.concat([batch_shape, [y_dim, x_dim]], axis=0))
        return J, y

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
        ssm_jac = getattr(self.ssm, "jacobian_h_r", None)
        if callable(ssm_jac):
            return ssm_jac(x, r)
        x = tf.convert_to_tensor(x)
        r = tf.convert_to_tensor(r)
        batch_shape = tf.shape(x)[:-1]
        x_dim = tf.shape(x)[-1]
        r_dim = tf.shape(r)[-1]
        x_flat = tf.reshape(x, tf.stack([-1, x_dim]))
        r_flat = tf.reshape(r, tf.stack([-1, r_dim]))
        with tf.GradientTape() as tape:
            tape.watch(r_flat)
            y_flat = self.ssm.h_with_noise(x_flat, r_flat)
        J_flat = tape.batch_jacobian(y_flat, r_flat)
        y_dim = tf.shape(y_flat)[-1]
        y = tf.reshape(y_flat, tf.concat([batch_shape, [y_dim]], axis=0))
        J = tf.reshape(J_flat, tf.concat([batch_shape, [y_dim, r_dim]], axis=0))
        return J, y

    def _jacobian(self, func, x):
        """Generic Jacobian helper for custom functions."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.batch_jacobian(y, x), y
