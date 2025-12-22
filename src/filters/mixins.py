import tensorflow as tf


class LinearizationMixin:
    def _jacobian(self, func, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.batch_jacobian(y, x), y
