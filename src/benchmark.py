from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import psutil
import tensorflow as tf


def now() -> float:
    return time.perf_counter()


class MemorySampler:
    """Sample per-process memory usage (RSS) and optional GPU memory."""
    def __init__(self, sample_gpu: bool = False):
        self._proc = psutil.Process(os.getpid())
        self._sample_gpu = bool(sample_gpu)

    def sample(self) -> Tuple[int, Optional[int]]:
        """Return (rss_bytes, gpu_bytes_or_none) for the current process."""
        rss = int(self._proc.memory_info().rss)
        gpu = None
        if self._sample_gpu:
            try:
                info = tf.config.experimental.get_memory_info("GPU:0")
                gpu = int(info.get("current", 0))
            except Exception:
                gpu = None
        return rss, gpu


def summarize_step_times(step_times: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(step_times, dtype=np.float32)
    if arr.size == 0:
        return {"total_s": 0.0, "mean_s": 0.0, "p95_s": 0.0}
    return {
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "p95_s": float(np.percentile(arr, 95)),
    }


def summarize_rss(rss_bytes: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(rss_bytes, dtype=np.float64)
    if arr.size == 0:
        return {"peak_rss_mb": 0.0}
    return {"peak_rss_mb": float(np.max(arr) / (1024.0 * 1024.0))}


def summarize_gpu(gpu_bytes: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(gpu_bytes, dtype=np.float64)
    if arr.size == 0:
        return {"peak_gpu_mb": 0.0}
    return {"peak_gpu_mb": float(np.max(arr) / (1024.0 * 1024.0))}
