"""GT-free calibrated LiDAR/mask lifting and opt-in ground-contact fallback.

LiDAR anchors describe a camera-facing SURFACE, never an inferred centre.
Multiple supported depth components inflate the association uncertainty; these
robust spread/hypothesis bounds are not calibrated one-sigma posteriors.
Ground rays require an explicit grounded-object assumption and a supported
local plane from this observer's own LiDAR. Their anchor is GROUND, not SURFACE;
therefore they cannot silently corroborate a surface measurement.
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image

from coopx.dataset.v2xreal import AgentFrame, project_camera, transform_points
from coopx.domain import PositionAnchor


@dataclass(frozen=True, slots=True)
class Lifted:
    xyz_site: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    std_m: tuple[float, float, float]
    n_points: int
    anchor: PositionAnchor
    method: str = "lidar"
    association_ambiguous: bool = False
    ground_support: int = 0


def _box(box_xyxy):
    box = np.asarray(box_xyxy, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("expected finite positive-area xyxy pixel box")
    return box


def _points(frame, points_site):
    if points_site is None:
        local = frame.lidar()[:, :3]
        local = local[np.linalg.norm(local, axis=1) > .001]
        return transform_points(local, frame.lidar_to_site)
    points = np.asarray(points_site, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_site must be Nx3 site-frame points from this observer")
    return points[np.isfinite(points).all(axis=1)]


def _image_size(frame, camera_name):
    with Image.open(frame.cameras[camera_name].image_path) as image:
        return image.size


def lift_box(box_xyxy, frame: AgentFrame, camera_name: str, *, mask=None,
             points_site=None, shrink=0.15, min_points=5,
             ground_fallback=False) -> Lifted | None:
    """Lift mask-selected points, else an eroded box; nearest supported cluster.

    mask must be an original-image boolean HxW raster. It replaces box erosion,
    not camera calibration; no resize or GT is used here. ground_fallback=True
    is ONLY for objects explicitly assumed to contact the visible ground at
    their box bottom. It runs only if no supported LiDAR component survives.
    """
    box = _box(box_xyxy)
    if (type(shrink) not in (int, float) or not math.isfinite(shrink) or not 0 <= shrink < .5
            or type(min_points) is not int or min_points < 1 or type(ground_fallback) is not bool):
        raise ValueError("invalid erosion, minimum support or ground fallback")
    points = _points(frame, points_site)
    camera = frame.to_camera(points, camera_name)
    uv, depth, valid = project_camera(camera, frame.cameras[camera_name].intrinsic)
    if mask is not None:
        width, height = _image_size(frame, camera_name)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (height, width):
            raise ValueError("mask must be an original-image boolean HxW raster")
        selected = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        indices = np.flatnonzero(selected)
        selected[indices] &= mask[np.floor(uv[indices, 1]).astype(int), np.floor(uv[indices, 0]).astype(int)]
    else:
        dx, dy = (box[2:] - box[:2]) * shrink
        selected = (valid & (uv[:, 0] >= box[0]+dx) & (uv[:, 0] < box[2]-dx)
                    & (uv[:, 1] >= box[1]+dy) & (uv[:, 1] < box[3]-dy))
    indices = np.flatnonzero(selected)
    order = indices[np.argsort(depth[indices], kind='stable')]
    gaps = np.diff(depth[order])
    thresholds = np.maximum(.35, .015 * depth[order[:-1]])
    clusters = np.split(order, np.flatnonzero(gaps > thresholds) + 1)
    supported = []
    for cluster in clusters:
        if len(cluster) < min_points:
            continue
        values = depth[cluster]
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        cluster = cluster[np.abs(values - median) <= max(.15, 3 * 1.4826 * mad)]
        if len(cluster) >= min_points:
            supported.append(cluster)
    if supported:
        foreground = points[supported[0]]
        xyz = np.median(foreground, axis=0)
        extent = np.maximum(.05, np.quantile(foreground, .9, axis=0) - np.quantile(foreground, .1, axis=0))
        std = np.maximum([.2, .2, .1], 1.4826 * np.median(np.abs(foreground - xyz), axis=0))
        # A box alone cannot identify which of several supported surfaces is
        # the semantic object. Keep the nearest return, but expose rather than
        # hide association ambiguity. A verified object mask can resolve it.
        ambiguous = len(supported) > 1
        if ambiguous:
            hypotheses = np.array([np.median(points[c], axis=0) for c in supported])
            std = np.maximum(std, np.max(np.abs(hypotheses-xyz), axis=0))
        return Lifted(tuple(map(float, xyz)), tuple(map(float, extent)), tuple(map(float, std)),
                      len(foreground), PositionAnchor.SURFACE, association_ambiguous=ambiguous)
    if ground_fallback:
        # An empty mask supplies no evidence for a visible grounded object.
        if mask is not None and not mask.any():
            return None
        return ground_ray_lift(box, frame, camera_name, points_site=points)
    return None


def _fit_plane(points):
    """Deterministic lower-envelope/RANSAC road fit, independent of object GT."""
    if len(points) < 40:
        return None
    # One low return per 1m tile avoids a wall's point count dominating ground.
    cells = np.floor(points[:, :2]).astype(np.int64)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((points[:, 2], group))
    _, first = np.unique(group[order], return_index=True)
    samples = points[order[first]]
    if len(samples) < 30:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(96):
        triple = samples[rng.choice(len(samples), 3, replace=False)]
        design = np.column_stack((triple[:, :2], np.ones(3)))
        if abs(np.linalg.det(design)) < .1:
            continue
        plane = np.linalg.solve(design, triple[:, 2])
        if np.linalg.norm(plane[:2]) > math.tan(math.radians(15)):
            continue
        residual = np.abs(samples[:, 2] - np.column_stack((samples[:, :2], np.ones(len(samples)))) @ plane)
        inliers = residual <= .12
        score = int(inliers.sum())
        if score >= 30 and (best is None or score > best[0]):
            best = (score, inliers)
    if best is None:
        return None
    ground = samples[best[1]]
    design = np.column_stack((ground[:, :2], np.ones(len(ground))))
    plane = np.linalg.lstsq(design, ground[:, 2], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > math.tan(math.radians(15)):
        return None
    residual = ground[:, 2]-design @ plane
    sigma = max(.15, 1.4826 * float(np.median(np.abs(residual-np.median(residual)))))
    return plane, sigma, ground


def ground_ray_lift(box_xyxy, frame: AgentFrame, camera_name: str, *, points_site=None) -> Lifted | None:
    """Conditional ground contact estimate, not an object centre or range hit.

    Caller asserts a fully visible grounded object with a trustworthy box
    bottom. Require downward ray, a supported near-horizontal own-LiDAR plane,
    nearby 2-D plane support and <=80m range. Occluded feet, elevated objects,
    clipped boxes, weak/missing ground, grazing rays or >10m XY uncertainty
    have no valid fallback. Own-LiDAR ground inliers are reported separately
    from n_points=0 (no object returns). XY uncertainty includes plane height,
    two pixels at the box bottom, and a range-dependent model-error floor.
    """
    box = _box(box_xyxy)
    width, height = _image_size(frame, camera_name)
    if box[0] < 0 or box[1] < 0 or box[2] >= width or box[3] >= height:
        return None
    points = _points(frame, points_site)
    camera_to_site = frame.camera_to_site(camera_name)
    origin = camera_to_site[:3, 3]
    inverse_k = np.linalg.inv(frame.cameras[camera_name].intrinsic)
    pixel = np.array([(box[0]+box[2])/2, box[3], 1.])
    direction = camera_to_site[:3, :3] @ (inverse_k @ pixel)
    direction /= np.linalg.norm(direction)
    horizontal = np.linalg.norm(direction[:2])
    if direction[2] >= -.05 or horizontal < .05:
        return None
    axis = direction[:2]/horizontal
    relative = points-origin
    along = relative[:, :2] @ axis
    lateral = np.abs(relative[:, 0]*axis[1]-relative[:, 1]*axis[0])
    corridor = points[(along > 1) & (along < 85) & (lateral < 6) & (points[:, 2] < origin[2]-.5)]
    fitted = _fit_plane(corridor)
    if fitted is None:
        return None
    plane, sigma, ground = fitted
    def intersect(ray, plane):
        denominator = ray[2]-plane[:2] @ ray[:2]
        if denominator >= -.05:
            return None
        t = (plane[:2] @ origin[:2]+plane[2]-origin[2])/denominator
        if t <= 0 or t*np.linalg.norm(ray[:2]) > 80:
            return None
        return origin+t*ray, denominator
    hit = intersect(direction, plane)
    if hit is None:
        return None
    point, _ = hit
    local = corridor[np.linalg.norm(corridor[:, :2]-point[:2], axis=1) < 6]
    # Refit local road so a distant dominant road surface cannot be extrapolated.
    fitted = _fit_plane(local)
    if fitted is None:
        return None
    plane, sigma, ground = fitted
    hit = intersect(direction, plane)
    if hit is None:
        return None
    point, denominator = hit
    if (np.linalg.norm(np.ptp(ground[:, :2], axis=0)) < 2
            or (point[:2] < ground[:, :2].min(0)-.5).any()
            or (point[:2] > ground[:, :2].max(0)+.5).any()
            or not .5 < origin[2]-point[2] < 12):
        return None
    height_error = sigma*np.abs(direction[:2]/denominator)
    pixel_error = np.zeros(2)
    for du, dv in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        ray = camera_to_site[:3, :3] @ (inverse_k @ (pixel+[du,dv,0]))
        ray /= np.linalg.norm(ray)
        shifted = intersect(ray, plane)
        if shifted is None:
            return None
        pixel_error = np.maximum(pixel_error, np.abs(shifted[0][:2]-point[:2]))
    distance = np.linalg.norm(point[:2]-origin[:2])
    xy_std = np.maximum(.5+.02*distance, height_error+pixel_error)
    if (xy_std > 10).any():
        return None
    std = (*map(float, xy_std), float(sigma+np.linalg.norm(plane[:2]*xy_std)))
    return Lifted(tuple(map(float, point)), (.05,.05,.05), std, 0,
                  PositionAnchor.GROUND, method="ground_ray", ground_support=len(ground))
