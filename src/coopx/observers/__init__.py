"""Observers. Grid and coverage types come from coopx.coverage."""

from coopx.coverage import CoverageContent, GridSpec
from coopx.observers.camera import CameraObserver
from coopx.observers.lidar_synth import (
    LidarObserver,
    Rect,
    SynthWorld,
    cluster_occupied,
    S2_GRID,
)

__all__ = (
    "CameraObserver",
    "CoverageContent",
    "GridSpec",
    "LidarObserver",
    "Rect",
    "SynthWorld",
    "cluster_occupied",
    "S2_GRID",
)
