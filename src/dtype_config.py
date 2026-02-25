"""Central dtype configuration for the entire codebase."""

import tensorflow as tf
import numpy as np

DTYPE = tf.float32
NP_DTYPE = np.float32

JITTER = 1e-6
JITTER_SMALL = 1e-8
EPS = 1e-6
EPS_SMALL = 1e-8


def set_dtype(tf_dtype):
    """Set global floating-point dtype and keep NP_DTYPE in sync.

    Call this at the very start of an experiment, before importing or
    instantiating any filters / SSMs::

        import src.dtype_config as cfg
        cfg.set_dtype(tf.float16)
    """
    global DTYPE, NP_DTYPE
    DTYPE = tf.as_dtype(tf_dtype)
    NP_DTYPE = DTYPE.as_numpy_dtype


def to_dtype(value):
    """Convert any value to a tensor with the global DTYPE, casting if needed."""
    return tf.cast(tf.convert_to_tensor(value), DTYPE)
