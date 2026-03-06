"""Compatibility wrapper for soft-resampling DPF."""

from src.filters.dpf.soft_resampling import SoftResamplingDPF, categorical_sample

__all__ = [
    "categorical_sample",
    "SoftResamplingDPF",
]
