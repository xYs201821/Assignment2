import tensorflow as tf


class LinearizationMixin:
    @tf.function(reduce_retracing=True)
    def jacobian_f_x(self, x, q):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ssm.f_with_noise(x, q)
        return tape.batch_jacobian(y, x), y

    @tf.function(reduce_retracing=True)
    def jacobian_f_q(self, x, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            y = self.ssm.f_with_noise(x, q)
        return tape.batch_jacobian(y, q), y

    @tf.function(reduce_retracing=True)
    def jacobian_h_x(self, x, r):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ssm.h_with_noise(x, r)
        return tape.batch_jacobian(y, x), y

    @tf.function(reduce_retracing=True)
    def jacobian_h_r(self, x, r):
        with tf.GradientTape() as tape:
            tape.watch(r)
            y = self.ssm.h_with_noise(x, r)
        return tape.batch_jacobian(y, r), y

    def _jacobian(self, func, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.batch_jacobian(y, x), y
