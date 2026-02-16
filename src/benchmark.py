from __future__ import annotations

import os
from typing import Optional, Tuple

import psutil
import tensorflow as tf


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
