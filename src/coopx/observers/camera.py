"""Camera observer: SAM → lift → signed CX2 observations; answers typed queries.

Query/answer types and rules come from `coopx.query`; lifting from
`coopx.observers.camera_lift`. SAM detections are published at the SAM result's
availability time, never earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
from typing import Callable

from PIL import Image

from coopx.dataset.v2xreal import AgentFrame, camera_footprint_xy
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, Kind, Lineage, ObservationMessage, sign
from coopx.identity import NodeKey
from coopx.meta.client import MetaClient, ModelResult, model_call_event
from coopx.meta.sam import Detection, segment
from coopx.observers.camera_lift import Lifted, lift_box
from coopx.query import (CLASS_CONCEPTS, Answer, Capability, Query, answer_query,
                        normalize_concept)

MAX_LABELS = 64

def declared_camera_footprint(frame: AgentFrame, camera_name: str):
    """A's dense footprint when present; the 4-ray 40 m polygon until then."""
    try:
        from coopx.observers.footprint import camera_footprint
        return camera_footprint(frame, camera_name)
    except ImportError:
        return camera_footprint_xy(frame, camera_name, max_range_m=40.0)


def rsu_camera_capability(observer_id: str, frame: AgentFrame, camera_name: str) -> Capability:
    return Capability(
        observer_id, ("camera_rgb",), ("T0", "T1", "T2", "T3"), "open",
        declared_camera_footprint(frame, camera_name))


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
                 answer_fn=answer_query, period_us: int = 0):
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
        self.period_us = int(period_us)
        self._last_observe_us: int | None = None
        self._seq = 0
        self._latest: CameraFrame | None = None
        # What THIS observer signed and sent. Not "admitted": only the
        # receiving consumer's checker decides that.
        self.published: list[ObservationMessage] = []
        self._events: list[dict] = []
        self.last_detections: list[Detection] = []
        self.fast_path_detections: list[Detection] = []
        self.last_result: ModelResult | None = None
        # Bounded dynamic label table for open-vocabulary concepts: label_id is
        # signed into each observation so "trash can" stays distinct from any
        # other UNKNOWN object (contract: concepts are data, never instructions).
        self.labels: dict[int, str] = {}
        self._label_ids: dict[str, int] = {}
        # (frame capture, concept) already segmented: reuse, never re-sign.
        self._segmented: dict[tuple[int, str], ModelResult] = {}

    def _next_object_id(self) -> int:
        self._object_seq = getattr(self, "_object_seq", 199) + 1
        return self._object_seq

    def _next_seq(self) -> int:
        value = self._seq
        self._seq += 1
        return value

    def label_for(self, concept: str) -> int | None:
        """Fixed classes need no label; other phrases get a bounded label id."""
        normalized = normalize_concept(concept.strip())
        if normalized in CLASS_CONCEPTS:
            return None
        if normalized not in self._label_ids:
            if len(self._label_ids) >= MAX_LABELS:
                raise ValueError("label table full")
            label_id = len(self._label_ids) + 1
            self._label_ids[normalized] = label_id
            self.labels[label_id] = normalized
        return self._label_ids[normalized]

    def _sign(self, xyz, cls: ObjectClass, capture_us: int, object_id: int,
              publish_us: int, std=None, dimensions=None,
              label_id: int | None = None,
              anchor: PositionAnchor = PositionAnchor.SURFACE) -> ObservationMessage:
        sequence = self._next_seq()
        observation = Observation(
            self.observer_id, object_id, sequence, capture_us, cls, xyz,
            anchor, dimensions_m=dimensions, confidence=0.8,
            position_std_m=std)
        message_id = f"{self.observer_id}-{sequence:04d}"
        envelope = Envelope(
            2, "observation", message_id, self.observer_id, self.session,
            self.spatial_frame, capture_us, publish_us, publish_us + self.ttl_us,
            Auth(f"{self.observer_id}-key"), sequence)
        lineage = Lineage(self.observer_id, self.camera_name, message_id, capture_us,
                          derivation="direct")
        return sign(ObservationMessage(envelope, observation, lineage, Kind.PERCEIVED,
                                       label_id=label_id), self.key)

    def _lift(self, detection: Detection, frame: CameraFrame):
        """Lift with the SAM mask when decoded (contract 12), else the box."""
        return self.lift_fn(detection.box_xyxy, frame.agent_frame, frame.camera_name,
                            mask=detection.mask)

    def _detect(self, frame: CameraFrame, concept: str, query_time_us: int):
        key = (frame.capture_us, normalize_concept(concept.strip()))
        if key in self._segmented:
            # Same frame, same concept: the detections are already signed and sent.
            return [], self._segmented[key]
        label_id = self.label_for(concept)
        detections, result = segment(
            self.client, frame.image_png, concept,
            image_size=frame.image_size, query_time_us=query_time_us)
        self.last_detections = detections
        self.last_result = result
        self._segmented[key] = result
        self._events.append(model_call_event(result, self.observer_id))
        messages = []
        for index, detection in enumerate(detections):
            lifted = self._lift(detection, frame)
            if lifted is None:
                continue
            cls = CLASS_CONCEPTS.get(normalize_concept(concept.strip()), ObjectClass.UNKNOWN)
            # A ground-ray contact point has no measured body size (contract 8 notes).
            dimensions = None if getattr(lifted, "method", "lidar") != "lidar" else lifted.extent_m
            message = self._sign(
                lifted.xyz_site, cls, frame.capture_us, self._next_object_id(),
                result.availability_time_us, lifted.std_m, dimensions, label_id,
                anchor=lifted.anchor)
            messages.append(message)
            self.published.append(message)
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
        if (self.period_us > 0 and self._last_observe_us is not None
                and frame.capture_us - self._last_observe_us < self.period_us):
            return []
        self._last_observe_us = frame.capture_us
        published = []
        for concept in self.concepts:
            messages, _result = self._detect(frame, concept, frame.capture_us)
            published.extend(messages)
        self.fast_path_detections = list(self.last_detections)
        return published

    def on_query(self, query: Query, capability: Capability, now_us: int) -> tuple[Answer, int]:
        """Answer a query received at `now_us`; returns (answer, time it exists).

        A T3 query runs SAM with its concept when received; new detections and
        the answer exist only once that SAM result is available. A concept
        already segmented on this frame reuses that result (no second call).
        """
        frame = self._latest or self.frame_source()
        if frame is None:
            return Answer(query.query_id, self.observer_id, "unavailable", None, None, ()), now_us
        self._latest = frame
        answered_at = now_us
        if query.tier == "T3" and query.concept:
            key = (frame.capture_us, normalize_concept(query.concept.strip()))
            reused = key in self._segmented
            _messages, result = self._detect(frame, query.concept.strip(), now_us)
            if not reused:
                # The answer exists only once SAM has replied, even when it
                # found nothing ("not_detected" is a result, not an instant).
                answered_at = max(answered_at, result.availability_time_us)
        answer = self.answer_fn(query, capability, list(self.published), [], answered_at,
                                labels=dict(self.labels))
        return answer, answered_at


def jpeg_to_camera_frame(agent_frame: AgentFrame, camera_name: str) -> CameraFrame:
    """Send the camera's original JPEG (re-encoding to PNG made uploads ~10x larger)."""
    path = agent_frame.cameras[camera_name].image_path
    with Image.open(path) as image:
        size = image.size
    return CameraFrame(Path(path).read_bytes(), size, agent_frame.nominal_time_us,
                       agent_frame, camera_name)
