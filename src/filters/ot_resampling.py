"""Compatibility wrapper for OT-resampling DPF."""

from src.filters.dpf.ot_resampling import (
    OTResamplingDPF,
    ot_resample_barycentric,
    pairwise_distance,
    sinkhorn_log_plan,
)

__all__ = [
    "pairwise_distance",
    "sinkhorn_log_plan",
    "ot_resample_barycentric",
    "OTResamplingDPF",
]
