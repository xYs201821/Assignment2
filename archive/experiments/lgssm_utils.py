from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from src.ssm import LinearGaussianSSM


def build_lgssm(seed: int = 42) -> Tuple[LinearGaussianSSM, int, int]:
    dx, dy = 3, 2
    dtype = tf.float32

    A = tf.constant(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.9],
        ],
        dtype=dtype,
    )
    B = tf.eye(dx, dtype=dtype)
    C = tf.constant(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=dtype,
    )
    D = 0.3 * tf.eye(dy, dtype=dtype)

    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.eye(dx, dtype=np.float32)

    return LinearGaussianSSM(A=A, B=B, C=C, D=D, m0=m0, P0=P0, seed=seed), dx, dy
