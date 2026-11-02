"""Legacy synthetic/CARLA depth math, independent of the simulator runtime.

Every projection call must explicitly select axial depth or ray range. Live
CARLA calibration must establish this convention; synthetic examples prove only
the math under the chosen convention.

V2X-Real optical-camera/LiDAR projection uses dataset.v2xreal, not these CARLA
camera-axis helpers. Its release contains no depth images.
"""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from coopx.domain import Vec3


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    horizontal_fov_deg: float

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int
                or self.width <= 0 or self.height <= 0
                or not 0 < self.horizontal_fov_deg < 180):
            raise ValueError("invalid camera intrinsics")

    @property
    def focal_px(self) -> float:
        return self.width / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))


def decode_depth_bgra(raw) -> np.ndarray:
    """Decode raw BGRA bytes to metres; promote before channel multiplication."""
    pixels = np.asarray(raw)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("depth input must be an H x W x 4 uint8 BGRA array")
    channels = pixels.astype(np.uint32)
    packed = channels[..., 2] + 256 * channels[..., 1] + 65536 * channels[..., 0]
    return packed.astype(np.float64) * (1000 / (256**3 - 1))


def backproject(pixels_uv, depths_m, intrinsics: CameraIntrinsics,
                camera_to_world, *, depth_mode: Literal["axial", "range"]) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != (len(pixels),):
        raise ValueError("expected N x 2 pixels and N depths")
    if (not np.isfinite(pixels).all() or not np.isfinite(depths).all()
            or np.any(depths <= 0)):
        raise ValueError("pixels/depths must be finite and depth positive")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)):
        raise ValueError("camera_to_world must be a rigid 4 x 4 transform")
    f = intrinsics.focal_px
    # CARLA local camera coordinates: forward, right, up.
    rays = np.column_stack((np.ones(len(pixels)),
                            (pixels[:, 0] - intrinsics.width / 2) / f,
                            -(pixels[:, 1] - intrinsics.height / 2) / f))
    if depth_mode == "range":
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    elif depth_mode != "axial":
        raise ValueError("depth_mode must explicitly be 'axial' or 'range'")
    camera_points = rays * depths[:, None]
    return camera_points @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True, slots=True)
class SurfaceEstimate:
    xyz_m: Vec3
    spread_proxy_m: Vec3
    supporting_pixels: int


def lift_mask_surface(depth_m, mask, intrinsics: CameraIntrinsics, camera_to_world,
                      *, depth_mode: Literal["axial", "range"],
                      max_range_m: float = 100, depth_gap_m: float = 0.5,
                      min_pixels: int = 9, spread_floor_m: float = 0.1) -> SurfaceEstimate:
    """Erode a mask and lift its largest supported depth cluster.

    Returns a visible SURFACE anchor, not an inferred hidden object centre.
    spread_proxy_m is robust point spread with a floor, not a calibrated position
    posterior. Largest-cluster selection assumes the mask mostly covers target.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    region = np.asarray(mask)
    shape = (intrinsics.height, intrinsics.width)
    if depth.shape != shape or region.shape != shape or region.dtype != np.bool_:
        raise ValueError("depth and boolean mask must match camera dimensions")
    if (not math.isfinite(max_range_m) or max_range_m <= 0
            or not math.isfinite(depth_gap_m) or depth_gap_m <= 0
            or type(min_pixels) is not int or min_pixels < 1
            or not math.isfinite(spread_floor_m) or spread_floor_m <= 0):
        raise ValueError("invalid lift thresholds")
    padded = np.pad(region, 1, constant_values=False)
    eroded = np.ones(shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + shape[0], dx:dx + shape[1]]
    selected = eroded & np.isfinite(depth) & (depth > 0) & (depth < max_range_m)
    rows, cols = np.nonzero(selected)
    values = depth[rows, cols]
    if len(values) < min_pixels:
        raise ValueError("insufficient valid foreground depth")
    order = np.argsort(values)
    splits = np.flatnonzero(np.diff(values[order]) > depth_gap_m) + 1
    clusters = np.split(order, splits)
    cluster = max(clusters, key=len)
    if len(cluster) < min_pixels:
        raise ValueError("insufficient depth cluster support")
    points = backproject(np.column_stack((cols[cluster], rows[cluster])),
                         values[cluster], intrinsics, camera_to_world, depth_mode=depth_mode)
    centre = np.median(points, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(points - centre), axis=0), spread_floor_m)
    return SurfaceEstimate(tuple(float(v) for v in centre),
                           tuple(float(v) for v in spread), len(cluster))
