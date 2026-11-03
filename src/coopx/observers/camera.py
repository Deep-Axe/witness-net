"""Camera observer: SAM → lift → signed CX2 observations (contract 8/9 stubs).

`Capability`, `Query`, `Answer` and `answer_query` here are local stubs so this
branch does not import Agent A's modules. The reviewer swaps the seams.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import re
from typing import Callable

import numpy as np
from PIL import Image

from coopx.dataset.v2xreal import (AgentFrame, camera_footprint_xy, camera_image_size,
                                  point_in_polygon, project_camera, transform_points)
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, Kind, Lineage, ObservationMessage, sign
from coopx.identity import NodeKey
from coopx.meta.client import MetaClient, ModelResult, model_call_event
from coopx.meta.sam import Detection, segment

CONCEPT_CLASS = {
    "person": ObjectClass.PEDESTRIAN,
    "pedestrian": ObjectClass.PEDESTRIAN,
    "car": ObjectClass.VEHICLE,
    "vehicle": ObjectClass.VEHICLE,
    "cyclist": ObjectClass.CYCLIST,
}
NOUN_PHRASE = re.compile(r"^[A-Za-z][A-Za-z -]{0,63}$")
QUERY_TIERS = ("T0", "T1", "T2", "T3")


@dataclass(frozen=True)
class Lifted:
    xyz_site: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    std_m: tuple[float, float, float]
    n_points: int
    anchor: PositionAnchor


@dataclass(frozen=True)
class Capability:
    observer_id: str
    modalities: tuple[str, ...]
    query_tiers: tuple[str, ...]
    vocabulary: str
    coverage_xy: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if not self.observer_id or not self.modalities or not self.query_tiers:
            raise ValueError("capability needs an observer, modalities and tiers")
        if any(t not in QUERY_TIERS for t in self.query_tiers):
            raise ValueError("query_tiers must be a subset of T0–T3")
        if self.vocabulary not in ("fixed", "geometric", "open"):
            raise ValueError("vocabulary must be fixed, geometric or open")
        if len(self.coverage_xy) < 3:
            raise ValueError("coverage_xy must be a polygon")


@dataclass(frozen=True)
class Query:
    query_id: str
    tier: str
    observer_id: str
    center_xy: tuple[float, float]
    radius_m: float
    time_us: int
    concept: str | None
    reason: str

    def __post_init__(self):
        if not self.query_id or not self.observer_id:
            raise ValueError("query_id and observer_id required")
        if self.tier not in QUERY_TIERS:
            raise ValueError("tier must be T0–T3")
        if (len(self.center_xy) != 2 or not all(math.isfinite(v) for v in self.center_xy)
                or not math.isfinite(self.radius_m) or self.radius_m <= 0 or self.radius_m > 200):
            raise ValueError("region must be finite with 0 < radius_m ≤ 200")
        if type(self.time_us) is not int or self.time_us < 0:
            raise ValueError("time_us must be a nonnegative int")
        if self.concept is not None and (
                not isinstance(self.concept, str) or not NOUN_PHRASE.match(self.concept.strip())):
            raise ValueError("concept must be a simple noun phrase")


@dataclass(frozen=True)
class Answer:
    query_id: str
    observer_id: str
    result: str
    observed_time_us: int | None
    coverage_ref: str | None
    evidence_refs: tuple[str, ...]


def stub_lift_box(box_xyxy, frame: AgentFrame, camera_name: str, *,
                  points_site=None, shrink=0.15, min_points=5) -> Lifted | None:
    """GT-free centroid of projected LiDAR points inside the (shrunk) box."""
    if points_site is None:
        points = transform_points(frame.lidar()[:, :3], frame.lidar_to_site)
    else:
        points = np.asarray(points_site, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_site must be Nx3")
    camera_pts = frame.to_camera(points, camera_name)
    uv, _depth, valid = project_camera(camera_pts, frame.cameras[camera_name].intrinsic)
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    pad_x, pad_y = (x2 - x1) * shrink, (y2 - y1) * shrink
    inside = (valid
              & (uv[:, 0] >= x1 + pad_x) & (uv[:, 0] <= x2 - pad_x)
              & (uv[:, 1] >= y1 + pad_y) & (uv[:, 1] <= y2 - pad_y))
    if int(inside.sum()) < min_points:
        return None
    chosen = points[inside]
    xyz = tuple(float(v) for v in chosen.mean(axis=0))
    extent = tuple(max(0.1, float(v)) for v in np.ptp(chosen, axis=0))
    std = tuple(float(v) for v in chosen.std(axis=0))
    return Lifted(xyz, extent, std, int(inside.sum()), PositionAnchor.SURFACE)


def stub_answer_query(query: Query, capability: Capability,
                      observations: list[ObservationMessage],
                      coverage: list, now_us: int) -> Answer:
    if query.tier not in capability.query_tiers or (
            query.tier == "T3" and capability.vocabulary != "open"):
        return Answer(query.query_id, capability.observer_id, "unsupported",
                      None, None, ())
    if not _circle_hits_polygon(query.center_xy, query.radius_m, capability.coverage_xy):
        return Answer(query.query_id, capability.observer_id, "not_covered",
                      None, None, ())
    wanted = CONCEPT_CLASS.get((query.concept or "").strip().lower())
    refs = []
    observed_at = None
    for message in observations:
        obs = message.observation
        if obs.source != capability.observer_id:
            continue
        if wanted is not None and obs.cls != wanted:
            continue
        if math.hypot(obs.xyz_m[0] - query.center_xy[0],
                      obs.xyz_m[1] - query.center_xy[1]) > query.radius_m:
            continue
        from coopx.envelope import evidence_ref
        refs.append(evidence_ref(message))
        observed_at = obs.captured_sim_us
    if refs:
        return Answer(query.query_id, capability.observer_id, "detected",
                      observed_at, None, tuple(dict.fromkeys(refs)))
    if not observations and not coverage:
        return Answer(query.query_id, capability.observer_id, "unavailable",
                      None, None, ())
    return Answer(query.query_id, capability.observer_id, "not_detected",
                  None, None, ())


def _circle_hits_polygon(center, radius, polygon) -> bool:
    if point_in_polygon(center, polygon):
        return True
    cx, cy = center
    pts = list(polygon)
    pts.append(pts[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        t = max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / (length * length)))
        if math.hypot(cx - (x0 + t * dx), cy - (y0 + t * dy)) <= radius:
            return True
    return False


def rsu_camera_capability(observer_id: str, frame: AgentFrame, camera_name: str) -> Capability:
    return Capability(
        observer_id, ("camera_rgb",), ("T0", "T1", "T2", "T3"), "open",
        camera_footprint_xy(frame, camera_name, max_range_m=40.0))


@dataclass
class CameraFrame:
    image_png: bytes
    image_size: tuple[int, int]
    capture_us: int
    agent_frame: AgentFrame
    camera_name: str


def png_from_jpeg_path(path) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue(), size


class CameraObserver:
    """Latest-frame-wins camera. Fast-path detections never wait on Spark."""

    def __init__(self, observer_id: str, frame_source: Callable[[], CameraFrame | None],
                 camera_name: str, client: MetaClient, lift_fn: Callable, key: NodeKey,
                 *, session: str = "s3", spatial_frame: str = "site_map",
                 ttl_us: int = 1_000_000, concepts: tuple[str, ...] = ("person",),
                 answer_fn=stub_answer_query):
        self.observer_id = observer_id
        self.frame_source = frame_source
        self.camera_name = camera_name
        self.client = client
        self.lift_fn = lift_fn
        self.key = key
        self.session = session
        self.spatial_frame = spatial_frame
        self.ttl_us = ttl_us
        self.concepts = concepts
        self.answer_fn = answer_fn
        self._seq = 0
        self._latest: CameraFrame | None = None
        self._admitted: list[ObservationMessage] = []
        self._events: list[dict] = []
        self.last_detections: list[Detection] = []
        self.last_result: ModelResult | None = None

    def _next_seq(self) -> int:
        value = self._seq
        self._seq += 1
        return value

    def _sign(self, xyz, cls: ObjectClass, capture_us: int, object_id: int,
              publish_us: int, std=None, dimensions=None) -> ObservationMessage:
        sequence = self._next_seq()
        observation = Observation(
            self.observer_id, object_id, sequence, capture_us, cls, xyz,
            PositionAnchor.SURFACE, dimensions_m=dimensions, confidence=0.8,
            position_std_m=std)
        message_id = f"{self.observer_id}-{sequence:04d}"
        envelope = Envelope(
            2, "observation", message_id, self.observer_id, self.session,
            self.spatial_frame, capture_us, publish_us, publish_us + self.ttl_us,
            Auth(f"{self.observer_id}-key"), sequence)
        lineage = Lineage(self.observer_id, self.camera_name, message_id, capture_us,
                          derivation="direct")
        return sign(ObservationMessage(envelope, observation, lineage, Kind.PERCEIVED),
                    self.key)

    def _detect(self, frame: CameraFrame, concept: str, query_time_us: int):
        detections, result = segment(
            self.client, frame.image_png, concept,
            image_size=frame.image_size, query_time_us=query_time_us)
        self.last_detections = detections
        self.last_result = result
        self._events.append(model_call_event(result, self.observer_id))
        messages = []
        for index, detection in enumerate(detections):
            lifted = self.lift_fn(detection.box_xyxy, frame.agent_frame, frame.camera_name)
            if lifted is None:
                continue
            cls = CONCEPT_CLASS.get(concept.strip().lower(), ObjectClass.UNKNOWN)
            message = self._sign(
                lifted.xyz_site, cls, frame.capture_us, 200 + index,
                result.availability_time_us, lifted.std_m, lifted.extent_m)
            messages.append(message)
            self._admitted.append(message)
        return messages, result

    def drain_events(self) -> list[dict]:
        events = self._events
        self._events = []
        return events

    def observe(self) -> list[ObservationMessage]:
        frame = self.frame_source()
        if frame is None:
            return []
        self._latest = frame
        published = []
        for concept in self.concepts:
            messages, _result = self._detect(frame, concept, frame.capture_us)
            published.extend(messages)
        return published

    def on_query(self, query: Query, capability: Capability) -> Answer:
        frame = self._latest or self.frame_source()
        if frame is None:
            return Answer(query.query_id, self.observer_id, "unavailable", None, None, ())
        self._latest = frame
        if query.tier == "T3" and query.concept:
            self._detect(frame, query.concept.strip(), query.time_us)
        return self.answer_fn(query, capability, list(self._admitted), [], query.time_us)


def jpeg_to_camera_frame(agent_frame: AgentFrame, camera_name: str) -> CameraFrame:
    png, size = png_from_jpeg_path(agent_frame.cameras[camera_name].image_path)
    return CameraFrame(png, size, agent_frame.nominal_time_us, agent_frame, camera_name)
