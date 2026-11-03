"""Synthetic observers for S2. Coverage semantics live in Agent A's modules."""

from coopx.observers.lidar_synth import (
    CoverageContent,
    GridSpec,
    LidarObserver,
    Rect,
    SynthWorld,
    cluster_occupied,
    S2_GRID,
)

__all__ = (
    "CoverageContent",
    "GridSpec",
    "LidarObserver",
    "Rect",
    "SynthWorld",
    "cluster_occupied",
    "S2_GRID",
)
