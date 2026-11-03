"""Minimal V2X-Real reader and explicit projection conventions.

Points are little-endian float32 (x, y, z, intensity). Stored camera extrinsics
map optical camera coordinates (right, down, forward) TO the local LiDAR frame.
Poses are [x, y, z, roll, yaw, pitch], metres/degrees; the release's OPV2V
rotation convention is Rz(yaw) @ Ry(-pitch) @ Rx(-roll). Camera ``cords`` are
retained as metadata, not used to reconstruct the calibrated camera transform.

Frame numbers provide a nominal replay clock, NOT measured capture timestamps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import yaml


def pose_to_site(pose) -> np.ndarray:
    """Local-to-site rigid transform for the release's pose-angle convention."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("pose must contain six finite values")
    roll, yaw, pitch = np.radians(values[3:])
    cr, sr = np.cos(-roll), np.sin(-roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = values[:3]
    return result


def rigid_matrix(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError("expected a proper rigid 4x4 transform")
    return matrix


def transform_points(points, transform) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("expected finite Nx3 points")
    transform = rigid_matrix(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class Camera:
    name: str
    image_path: Path
    intrinsic: np.ndarray
    camera_to_lidar: np.ndarray
    cords: tuple[float, ...]

    def image(self) -> np.ndarray:
        """RGB image, at the native resolution expected by the intrinsics."""
        with Image.open(self.image_path) as image:
            return np.asarray(image.convert("RGB"))


@dataclass(frozen=True)
class GroundTruthBox:
    object_id: str
    label: str
    location: tuple[float, ...]
    center: tuple[float, ...]
    extent: tuple[float, ...]  # half-lengths, not full dimensions
    angle: tuple[float, ...]  # roll, yaw, pitch in degrees

    def corners_site(self) -> np.ndarray:
        signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        local = signs * np.asarray(self.extent) + np.asarray(self.center)
        return transform_points(local, pose_to_site((*self.location, *self.angle)))

    def center_site(self) -> np.ndarray:
        return transform_points(np.asarray([self.center]),
                                pose_to_site((*self.location, *self.angle)))[0]

    def contains_site(self, point) -> bool:
        local = transform_points(np.asarray(point)[None],
                                 np.linalg.inv(pose_to_site((*self.location, *self.angle))))[0]
        return bool(np.all(np.abs(local - self.center) <= np.asarray(self.extent)))


@dataclass(frozen=True)
class AgentFrame:
    scenario: str
    agent_id: str
    frame_id: str
    lidar_path: Path
    lidar_to_site: np.ndarray
    cameras: Mapping[str, Camera]
    boxes: Mapping[str, GroundTruthBox]
    metadata: dict
    nominal_time_us: int
    timestamp_basis: str = "frame_index_at_assumed_10Hz; capture time unavailable"

    def lidar(self) -> np.ndarray:
        if self.lidar_path.stat().st_size % 16:
            raise ValueError(f"LiDAR file is not packed float32 XYZI: {self.lidar_path}")
        points = np.fromfile(self.lidar_path, dtype="<f4").reshape(-1, 4)
        if not np.isfinite(points).all():
            raise ValueError("nonfinite LiDAR data")
        return points

    def camera_to_site(self, camera_name: str) -> np.ndarray:
        return self.lidar_to_site @ self.cameras[camera_name].camera_to_lidar

    def to_camera(self, points, camera_name: str, *, source_to_site=None) -> np.ndarray:
        """Input points are in site_map, or in the explicitly supplied source frame."""
        transform = np.linalg.inv(self.camera_to_site(camera_name))
        if source_to_site is not None:
            transform = transform @ rigid_matrix(source_to_site)
        return transform_points(points, transform)


def project_camera(points_camera, intrinsic, *, near_m: float = 0.1):
    """Return pixels, optical-axis depth, valid mask; behind-camera pixels are NaN."""
    points = np.asarray(points_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected Nx3 camera points")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("invalid intrinsic matrix")
    if not np.isfinite(near_m) or near_m <= 0:
        raise ValueError("near_m must be positive")
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > near_m)
    uv = np.full((len(points), 2), np.nan)
    homogeneous = points[valid] @ intrinsic.T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    return uv, depth, valid


def camera_image_size(camera: Camera) -> tuple[int, int]:
    with Image.open(camera.image_path) as image:
        return image.size


def count_lidar_in_box(frame: AgentFrame, box: GroundTruthBox) -> int:
    points = transform_points(frame.lidar()[:, :3], frame.lidar_to_site)
    inverse = np.linalg.inv(pose_to_site((*box.location, *box.angle)))
    local = transform_points(points, inverse)
    inside = np.all(np.abs(local - np.asarray(box.center)) <= np.asarray(box.extent), axis=1)
    return int(inside.sum())


def box_fully_in_camera(frame: AgentFrame, box: GroundTruthBox, camera_name: str) -> bool:
    camera = frame.cameras[camera_name]
    width, height = camera_image_size(camera)
    uv, _depth, valid = project_camera(frame.to_camera(box.corners_site(), camera_name),
                                       camera.intrinsic)
    return bool(valid.all() and np.all((uv[:, 0] >= 0) & (uv[:, 0] < width)
                                       & (uv[:, 1] >= 0) & (uv[:, 1] < height)))


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Inclusive ray-crossing test for a simple polygon."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 0.0) + x0):
            inside = not inside
        if abs((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) <= 1e-9:
            lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
            lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
            if lo_x - 1e-9 <= x <= hi_x + 1e-9 and lo_y - 1e-9 <= y <= hi_y + 1e-9:
                return True
        x0, y0 = x1, y1
    return inside


def camera_footprint_xy(frame: AgentFrame, camera_name: str, *,
                        max_range_m: float = 40.0,
                        ground_z: float | None = None) -> tuple[tuple[float, float], ...]:
    """Ground-plane footprint of the camera frustum, clipped at max_range_m."""
    camera = frame.cameras[camera_name]
    width, height = camera_image_size(camera)
    cam_to_site = frame.camera_to_site(camera_name)
    origin = cam_to_site[:3, 3]
    plane_z = float(frame.lidar_to_site[2, 3]) if ground_z is None else float(ground_z)
    k_inv = np.linalg.inv(camera.intrinsic)
    pixels = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    corners = []
    for u, v in pixels:
        ray_cam = k_inv @ np.array([u, v, 1.0])
        if ray_cam[2] <= 1e-9:
            direction = cam_to_site[:3, :3] @ (ray_cam / (np.linalg.norm(ray_cam) or 1.0))
            point = origin + direction * max_range_m
            corners.append((float(point[0]), float(point[1])))
            continue
        # Optical (right, down, forward) ray to site.
        direction = cam_to_site[:3, :3] @ ray_cam
        if abs(direction[2]) < 1e-9:
            unit = direction / (np.linalg.norm(direction) or 1.0)
            point = origin + unit * max_range_m
        else:
            t = (plane_z - origin[2]) / direction[2]
            point = origin + direction * t
            if t <= 0:
                unit = direction / (np.linalg.norm(direction) or 1.0)
                point = origin + unit * max_range_m
        offset = point[:2] - origin[:2]
        dist = float(np.linalg.norm(offset))
        if dist > max_range_m:
            point = np.array([origin[0], origin[1], plane_z]) 
            if dist > 0:
                point[:2] = origin[:2] + offset * (max_range_m / dist)
        corners.append((float(point[0]), float(point[1])))
    return tuple(corners)


class V2XReal:
    def __init__(self, root: str | Path, *, nominal_hz: float = 10):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not np.isfinite(nominal_hz) or nominal_hz <= 0:
            raise ValueError("nominal_hz must be positive")
        self.nominal_hz = nominal_hz

    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in self.root.iterdir() if path.is_dir()))

    def agents(self, scenario: str) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in (self.root / scenario).iterdir()
                            if path.is_dir()))

    def frame_ids(self, scenario: str, agent: str) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in (self.root / scenario / agent).glob("*.yaml")))

    def aligned_frame_ids(self, scenario: str, agents) -> tuple[str, ...]:
        """Common file IDs only: this does not certify capture synchronization."""
        sets = [set(self.frame_ids(scenario, str(agent))) for agent in agents]
        return tuple(sorted(set.intersection(*sets))) if sets else ()

    def read(self, scenario: str, agent: str, frame_id: str) -> AgentFrame:
        directory = self.root / scenario / str(agent)
        path = directory / f"{frame_id}.yaml"
        with path.open() as stream:
            metadata = yaml.safe_load(stream)
        cameras = {}
        for name, value in metadata.items():
            if name.startswith("cam") and isinstance(value, dict) and "intrinsic" in value:
                intrinsic = np.asarray(value["intrinsic"], dtype=np.float64)
                if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                        or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                        or not np.allclose(intrinsic[2], [0, 0, 1])):
                    raise ValueError(f"invalid intrinsics: {path} {name}")
                cameras[name] = Camera(name, directory / f"{frame_id}_{name}.jpeg",
                                       intrinsic, rigid_matrix(value["extrinsic"]),
                                       tuple(value.get("cords", ())))
        boxes = {}
        for object_id, value in metadata.get("vehicles", {}).items():
            box = GroundTruthBox(str(object_id), value["obj_type"], tuple(value["location"]),
                                 tuple(value.get("center", (0, 0, 0))),
                                 tuple(value["extent"]), tuple(value["angle"]))
            if (any(len(v) != 3 or not np.isfinite(v).all()
                    for v in (box.location, box.center, box.extent, box.angle))
                    or any(v <= 0 for v in box.extent)):
                raise ValueError(f"invalid box: {path} {object_id}")
            boxes[box.object_id] = box
        return AgentFrame(scenario, str(agent), frame_id, directory / f"{frame_id}.bin",
                          pose_to_site(metadata["lidar_pose"]), cameras, boxes, metadata,
                          round(int(frame_id) * 1_000_000 / self.nominal_hz))
