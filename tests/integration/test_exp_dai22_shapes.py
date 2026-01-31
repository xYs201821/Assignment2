import numpy as np
import tensorflow as tf

from experiments.exp_dai22 import BearingOnlySSM


def test_bearingonly_ssm_output_shapes():
    sensors = np.array([[-3.5, 0.0], [3.5, 0.0]], dtype=np.float32)
    R = np.diag([0.04, 0.04]).astype(np.float32)
    ssm = BearingOnlySSM(sensors=sensors, R=R, seed=0)

    x = tf.constant([[4.0, 4.0], [3.0, 5.0]], dtype=tf.float32)  # [N, 2]
    y_pred = ssm.h(x)
    tf.debugging.assert_shapes(
        [(x, ("N", 2)), (y_pred, ("N", 2))]
    )

    x_b = tf.stack([x, x + 1.0], axis=0)  # [B, N, 2]
    y_pred_b = ssm.h(x_b)
    tf.debugging.assert_shapes(
        [(x_b, ("B", "N", 2)), (y_pred_b, ("B", "N", 2))]
    )

    y = tf.constant([0.4754, 1.1868], dtype=tf.float32)
    innov = ssm.innovation(y, y_pred)
    tf.debugging.assert_shapes([(innov, ("N", 2))])
