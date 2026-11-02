"""S1 deterministic admission checks. Accepted means admissible, not true.

Instances are single-consumer/run state. Rejected packets never mutate replay
or latest-object state. Source/session changes do not reset replay protection;
authorized restarts require a new Checker until session rollover is specified.
"""

from dataclasses import dataclass, field
import math
import struct

from coopx.envelope import ObservationMessage, decode, verify
from coopx.identity import TrustAnchors


@dataclass(frozen=True, slots=True)
class CheckResult:
    codes: tuple[str, ...]
    state: str
    message: ObservationMessage | None


@dataclass(slots=True)
class _ReplayState:
    highest: int | None = None
    sequences: set[int] = field(default_factory=set)
    message_ids: set[str] = field(default_factory=set)


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class Checker:
    def __init__(self, anchors: TrustAnchors, *, ttl_us: int = 1_000_000,
                 future_tolerance_us: int = 200_000, replay_window: int = 64,
                 roi: tuple[float, float, float, float, float, float] | None = None):
        _nonnegative_int(ttl_us, "ttl_us")
        _nonnegative_int(future_tolerance_us, "future_tolerance_us")
        _nonnegative_int(replay_window, "replay_window")
        if replay_window == 0:
            raise ValueError("replay_window must be positive")
        if roi is not None:
            if (len(roi) != 6 or not all(math.isfinite(v) for v in roi)
                    or any(roi[i] > roi[i + 3] for i in range(3))):
                raise ValueError("roi must be finite xmin,ymin,zmin,xmax,ymax,zmax")
            roi = tuple(roi)
        self.anchors = dict(anchors)
        self.ttl_us = ttl_us
        self.future_tolerance_us = future_tolerance_us
        self.replay_window = replay_window
        self.roi = roi
        self._sources: dict[str, _ReplayState] = {}
        self._latest: dict[tuple[str, int], ObservationMessage] = {}

    def latest(self, source_id: str, object_id: int) -> ObservationMessage | None:
        """Newest-by-capture accepted message; first accepted wins capture ties.

        This is history, not a currently fresh track: consumers must expire it.
        """
        return self._latest.get((source_id, object_id))

    def check(self, raw: bytes, now_us: int) -> CheckResult:
        _nonnegative_int(now_us, "now_us")
        try:
            message = decode(raw)
        except (ValueError, TypeError, OverflowError, struct.error):
            return CheckResult(("DECODE_ERROR",), "rejected", None)
        envelope, observation = message.envelope, message.observation
        anchor = self.anchors.get(envelope.source_id)
        if anchor is None:
            return CheckResult(("UNKNOWN_SOURCE",), "rejected", message)
        if anchor.key_ref != envelope.auth.key_ref:
            return CheckResult(("KEY_MISMATCH",), "rejected", message)
        if not verify(message, self.anchors):
            return CheckResult(("BAD_SIGNATURE",), "rejected", message)

        codes = []
        if now_us - envelope.capture_time_us > self.ttl_us:
            codes.append("STALE")
        if now_us > envelope.valid_until_us:
            codes.append("EXPIRED")
        if max(envelope.capture_time_us, envelope.generation_time_us) > now_us + self.future_tolerance_us:
            codes.append("FUTURE_TIME")

        state = self._sources.get(envelope.source_id)
        sequence = envelope.sensor_seq
        if state is not None:
            if (envelope.message_id in state.message_ids
                    or (sequence is not None and sequence in state.sequences)):
                codes.append("DUPLICATE")
            if (sequence is not None and state.highest is not None
                    and sequence <= state.highest - self.replay_window):
                codes.append("OUTSIDE_REPLAY_WINDOW")

        xyz = observation.xyz_m
        if (not all(math.isfinite(value) for value in xyz)
                or (self.roi is not None and any(
                    xyz[i] < self.roi[i] or xyz[i] > self.roi[i + 3] for i in range(3)))):
            codes.append("OUT_OF_BOUNDS")
        if not anchor.permits(xyz):
            codes.append("OUTSIDE_PERMITTED_REGION")
        if codes:
            return CheckResult(tuple(codes), "rejected", message)

        # Commit only after every gate passes. Sequence storage stays bounded;
        # IDs are retained for this run so refreshing an old ID cannot revive it.
        state = self._sources.setdefault(envelope.source_id, _ReplayState())
        state.message_ids.add(envelope.message_id)
        if sequence is not None:
            state.highest = sequence if state.highest is None else max(state.highest, sequence)
            state.sequences.add(sequence)
            state.sequences = {seq for seq in state.sequences
                               if seq > state.highest - self.replay_window}
        key = observation.key
        previous = self._latest.get(key)
        if previous is None or envelope.capture_time_us > previous.envelope.capture_time_us:
            self._latest[key] = message
        return CheckResult(("OK",), "accepted", message)
