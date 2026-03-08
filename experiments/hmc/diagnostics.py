from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def compute_chain_ess(chain: np.ndarray | tf.Tensor, burnin: int = 0) -> Dict[str, Any]:
    """Compute post-burnin effective sample size for a single MCMC chain."""
    chain_np = np.asarray(chain, dtype=np.float64)

    burnin = int(max(0, min(int(burnin), int(chain_np.shape[0]))))
    post = chain_np[burnin:]
    burnin_used = burnin

    # TFP ESS needs at least a short series; if burn-in removes too much, fall back.
    if post.shape[0] < 2:
        post = chain_np
        burnin_used = 0

    ess_t = tfp.mcmc.effective_sample_size(
        tf.convert_to_tensor(post, dtype=tf.float32),
        filter_beyond_positive_pairs=True,
    )
    ess_raw = np.asarray(ess_t.numpy(), dtype=np.float64)
    if ess_raw.ndim == 0:
        ess_raw = ess_raw[np.newaxis]

    num_post = float(post.shape[0])
    # TFP can return inf or values above the chain length for very short chains.
    ess = np.where(np.isfinite(ess_raw), ess_raw, num_post)
    ess = np.clip(ess, 0.0, num_post)
    ess_min = float(np.min(ess))

    return {
        "chain_ess": ess,
        "chain_ess_min": ess_min,
        "chain_ess_num_samples": int(post.shape[0]),
        "chain_ess_burnin_used": int(burnin_used),
    }
