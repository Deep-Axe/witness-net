"""GT-free box-to-LiDAR lifting through the release's calibrated transforms.

The anchor is a camera-facing SURFACE cluster, not an inferred object centre.
Spread is a robust proxy with a 0.2m XY floor for observed pedestrian skew;
it is not a calibrated one-sigma position posterior. Fast movers can exceed it.
"""

from dataclasses import dataclass
import math

import numpy as np

from coopx.dataset.v2xreal import AgentFrame, project_camera, transform_points
from coopx.domain import PositionAnchor


@dataclass(frozen=True, slots=True)
class Lifted:
    xyz_site: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    std_m: tuple[float, float, float]
    n_points: int
    anchor: PositionAnchor


def lift_box(box_xyxy, frame: AgentFrame, camera_name: str, *,
             points_site=None, shrink=0.15, min_points=5) -> Lifted | None:
    box = np.asarray(box_xyxy, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("expected finite positive-area xyxy pixel box")
    if (type(shrink) not in (int, float) or not math.isfinite(shrink) or not 0 <= shrink < .5
            or type(min_points) is not int or min_points < 1):
        raise ValueError("invalid erosion or minimum support")
    if points_site is None:
        local = frame.lidar()[:, :3]
        local = local[np.linalg.norm(local, axis=1) > .001]
        points = transform_points(local, frame.lidar_to_site)
    else:
        points = np.asarray(points_site, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_site must be Nx3 site-frame points from this observer")
        points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_points:
        return None
    camera = frame.to_camera(points, camera_name)
    uv, depth, valid = project_camera(camera, frame.cameras[camera_name].intrinsic)
    dx, dy = (box[2:] - box[:2]) * shrink
    selected = (valid & (uv[:, 0] >= box[0]+dx) & (uv[:, 0] < box[2]-dx)
                & (uv[:, 1] >= box[1]+dy) & (uv[:, 1] < box[3]-dy))
    indices = np.flatnonzero(selected)
    if len(indices) < min_points:
        return None
    order = indices[np.argsort(depth[indices], kind='stable')]
    # Perspective sparse range returns spread more with distance. A separated
    # background wall must not win merely because it has more points.
    gaps = np.diff(depth[order])
    thresholds = np.maximum(.35, .015 * depth[order[:-1]])
    clusters = np.split(order, np.flatnonzero(gaps > thresholds) + 1)
    for cluster in clusters:  # nearest supported depth component first
        if len(cluster) < min_points:
            continue
        values = depth[cluster]
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        cluster = cluster[np.abs(values - median) <= max(.15, 3 * 1.4826 * mad)]
        if len(cluster) < min_points:
            continue
        foreground = points[cluster]
        xyz = np.median(foreground, axis=0)
        extent = np.maximum(.05, np.quantile(foreground, .9, axis=0) - np.quantile(foreground, .1, axis=0))
        std = np.maximum([.2, .2, .1], 1.4826 * np.median(np.abs(foreground - xyz), axis=0))
        return Lifted(tuple(map(float, xyz)), tuple(map(float, extent)), tuple(map(float, std)),
                      len(foreground), PositionAnchor.SURFACE)
    return None
