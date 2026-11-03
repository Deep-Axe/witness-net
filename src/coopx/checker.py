"""S1 deterministic admission checks. Accepted means admissible, not true.

Instances are single-consumer/run state. Rejected packets never mutate replay
or latest-object state. Source/session changes do not reset replay protection;
authorized restarts require a new Checker until session rollover is specified.

sensor_seq remains optional for CX2/S1 compatibility: packets without it get
ID and time checks but cannot advance a sequence window. S2 publishers should
allocate one increasing per-source sequence across coverage and observations.
"""

from dataclasses import dataclass, field
import math
import struct

from coopx.envelope import ObservationMessage, decode, verify
from coopx.coverage import CoverageMessage, coverage_bounds
from coopx.identity import TrustAnchors


@dataclass(frozen=True, slots=True)
class CheckResult:
    codes: tuple[str, ...]
    state: str
    message: ObservationMessage | CoverageMessage | None
    mode: str = "live"


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

    def check(self, raw: bytes, now_us: int, *, scene_time_us: int | None = None) -> CheckResult:
        """Admit live traffic or explicitly requested historical evidence.

        Live admission is unchanged. Snapshot admission requires capture within
        300 ms of scene_time_us, generation no older than ttl_us at receipt,
        unexpired validity at receipt, and the usual future/signature/identity/
        replay/bounds checks. The requested scene itself must not be future.

        This API has no question-dispatch time or signed query nonce: it CANNOT
        prove a packet was produced in response to a particular question. A
        previously generated but recently signed, first-delivered observation
        is deliberately allowed; an already accepted packet is a duplicate in
        either mode. Reuse previously admitted evidence by reference, never by
        re-admitting its bytes. Old archived packets with old generation times
        fail STALE even when the requested scene matches. Recent generation
        authenticates an observer's assertion about a historical frame, not
        current occupancy or the truth of the assertion.

        Both modes share replay state; snapshot packets never update latest().
        The caller MUST label mode='snapshot' and route these packets only to
        snapshot Q&A, NEVER to Assessor, coverage updates or hazard warnings.
        Consumer-chosen scene_time_us is a trusted request, not packet input.
        """
        _nonnegative_int(now_us, "now_us")
        mode = "snapshot" if scene_time_us is not None else "live"
        if scene_time_us is not None:
            _nonnegative_int(scene_time_us, "scene_time_us")
            if scene_time_us > now_us:
                raise ValueError("requested scene must not be in the future")
        def result(codes, state, message):
            return CheckResult(codes, state, message, mode)
        try:
            message = decode(raw)
        except (ValueError, TypeError, OverflowError, struct.error):
            return result(("DECODE_ERROR",), "rejected", None)
        envelope = message.envelope
        anchor = self.anchors.get(envelope.source_id)
        if anchor is None:
            return result(("UNKNOWN_SOURCE",), "rejected", message)
        if anchor.key_ref != envelope.auth.key_ref:
            return result(("KEY_MISMATCH",), "rejected", message)
        if not verify(message, self.anchors):
            return result(("BAD_SIGNATURE",), "rejected", message)

        codes = []
        if scene_time_us is None:
            stale = now_us - envelope.capture_time_us > self.ttl_us
        else:
            stale = (abs(envelope.capture_time_us - scene_time_us) > 300_000
                     or now_us - envelope.generation_time_us > self.ttl_us)
        if stale:
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

        points = (coverage_bounds(message.content) if isinstance(message, CoverageMessage)
                  else (message.observation.xyz_m,))
        if any(not all(math.isfinite(value) for value in xyz)
               or (self.roi is not None and any(
                   xyz[i] < self.roi[i] or xyz[i] > self.roi[i + 3] for i in range(3)))
               for xyz in points):
            codes.append("OUT_OF_BOUNDS")
        if any(not anchor.permits(xyz) for xyz in points):
            codes.append("OUTSIDE_PERMITTED_REGION")
        if codes:
            return result(tuple(codes), "rejected", message)

        # Commit only after every gate passes. Sequence storage stays bounded;
        # IDs are retained for this run so refreshing an old ID cannot revive it.
        state = self._sources.setdefault(envelope.source_id, _ReplayState())
        state.message_ids.add(envelope.message_id)
        if sequence is not None:
            state.highest = sequence if state.highest is None else max(state.highest, sequence)
            state.sequences.add(sequence)
            state.sequences = {seq for seq in state.sequences
                               if seq > state.highest - self.replay_window}
        if mode == "live" and isinstance(message, ObservationMessage):
            key = message.observation.key
            previous = self._latest.get(key)
            if previous is None or envelope.capture_time_us > previous.envelope.capture_time_us:
                self._latest[key] = message
        return result(("OK",), "accepted", message)
