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
                 answer_fn=answer_query):
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
            cls = CLASS_CONCEPTS.get(normalize_concept(concept.strip()), ObjectClass.UNKNOWN)
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

    def on_query(self, query: Query, capability: Capability, now_us: int) -> tuple[Answer, int]:
        """Answer a query received at `now_us`; returns (answer, time it exists).

        A T3 query runs SAM with its concept when received; new detections and
        the answer exist only once that SAM result is available.
        """
        frame = self._latest or self.frame_source()
        if frame is None:
            return Answer(query.query_id, self.observer_id, "unavailable", None, None, ()), now_us
        self._latest = frame
        answered_at = now_us
        if query.tier == "T3" and query.concept:
            _messages, result = self._detect(frame, query.concept.strip(), now_us)
            answered_at = max(answered_at, result.availability_time_us)
        answer = self.answer_fn(query, capability, list(self._admitted), [], answered_at)
        return answer, answered_at


def jpeg_to_camera_frame(agent_frame: AgentFrame, camera_name: str) -> CameraFrame:
    """Send the camera's original JPEG (re-encoding to PNG made uploads ~10x larger)."""
    path = agent_frame.cameras[camera_name].image_path
    with Image.open(path) as image:
        size = image.size
    return CameraFrame(Path(path).read_bytes(), size, agent_frame.nominal_time_us,
                       agent_frame, camera_name)
