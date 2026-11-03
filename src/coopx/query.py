"""Typed snapshot queries and evidence-linked answers; all free text is data.

Inputs to answer_query are admitted packets. Their capture/expiry is evaluated
at the requested scene time; generation must already exist by now_us. This
supports explicitly labelled historical/snapshot Q&A without refreshing capture
timestamps. T0 is a snapshot read here, not a persistent subscription service.
"""

from dataclasses import dataclass
import math
import re

from coopx.coverage import CoverageMessage, RANGE_MODALITIES
from coopx.domain import ObjectClass
from coopx.envelope import ObservationMessage, evidence_ref
from coopx.events import Event, SCHEMA

TIERS = ("T0", "T1", "T2", "T3")
RESULTS = ("detected", "not_detected", "observed_free", "not_covered", "unsupported", "unavailable")
STATES = ("corroborated", "contradicted", "unverified", "contested")
MAX_RADIUS_M = 100.0
MAX_QUERIES = 4
CLASS_CONCEPTS = {
    "person": ObjectClass.PEDESTRIAN, "people": ObjectClass.PEDESTRIAN,
    "pedestrian": ObjectClass.PEDESTRIAN, "pedestrians": ObjectClass.PEDESTRIAN,
    "car": ObjectClass.VEHICLE, "vehicle": ObjectClass.VEHICLE,
    "bicycle": ObjectClass.CYCLIST, "bike": ObjectClass.CYCLIST,
    "cyclist": ObjectClass.CYCLIST,
}


def _id(value, name="id"):
    if type(value) is not str or not value.strip() or len(value) > 512 or '\x00' in value:
        raise ValueError(f"invalid {name}")


def _time(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("time must be a nonnegative uint64 integer")


def _xy(value):
    if (type(value) is not tuple or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("expected a finite coordinate pair")


def normalize_concept(value):
    if (type(value) is not str or not 0 < len(value) <= 64
            or re.fullmatch(r'[A-Za-z]+(?:[ -][A-Za-z]+)*', value) is None):
        raise ValueError("concept must be a simple noun phrase of at most 64 characters")
    return value.lower().replace('-', ' ')


def _refs(values):
    if (type(values) is not tuple
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("evidence refs must be unique lowercase SHA-256 strings")


def _distance_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = 0 if not length else max(0, min(1, ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / length))
    return math.hypot(point[0] - a[0] - t*dx, point[1] - a[1] - t*dy)


def inside_polygon(point, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _distance_segment(point, a, b) <= 1e-9:
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            if point[0] < (b[0]-a[0]) * (point[1]-a[1]) / (b[1]-a[1]) + a[0]:
                inside = not inside
    return inside


def circle_intersects(center, radius, polygon):
    return inside_polygon(center, polygon) or any(
        _distance_segment(center, a, b) <= radius for a, b in zip(polygon, polygon[1:] + polygon[:1]))


@dataclass(frozen=True, slots=True)
class Capability:
    observer_id: str
    modalities: tuple[str, ...]
    query_tiers: tuple[str, ...]
    vocabulary: str
    coverage_xy: tuple[tuple[float, float], ...]

    def __post_init__(self):
        _id(self.observer_id, "observer_id")
        for values in (self.modalities, self.query_tiers):
            if type(values) is not tuple or not values or any(type(v) is not str for v in values) or len(set(values)) != len(values):
                raise ValueError("modalities and tiers must be unique nonempty tuples")
        if any(re.fullmatch('[a-z][a-z0-9_]{0,31}', m) is None for m in self.modalities):
            raise ValueError("invalid modality")
        if any(t not in TIERS for t in self.query_tiers) or self.vocabulary not in ('fixed', 'geometric', 'open'):
            raise ValueError("unsupported tiers or vocabulary")
        if type(self.coverage_xy) is not tuple or not 3 <= len(self.coverage_xy) <= 64:
            raise ValueError("coverage needs a bounded footprint polygon")
        for point in self.coverage_xy:
            _xy(point)
        area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(self.coverage_xy, self.coverage_xy[1:]+self.coverage_xy[:1]))
        if not math.isfinite(area) or abs(area) < 1e-9:
            raise ValueError("coverage polygon must have nonzero area")


@dataclass(frozen=True, slots=True)
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
        _id(self.query_id, "query_id")
        _id(self.observer_id, "observer_id")
        _xy(self.center_xy)
        _time(self.time_us)
        if self.tier not in TIERS or type(self.radius_m) not in (int, float) or not 0 < self.radius_m <= MAX_RADIUS_M:
            raise ValueError("invalid tier or radius")
        if self.concept is not None:
            normalize_concept(self.concept)
        if self.tier == 'T3' and self.concept is None:
            raise ValueError("T3 needs a concept")
        if type(self.reason) is not str or len(self.reason) > 1024:
            raise ValueError("reason must be bounded text data")

    def to_event(self, now_us, source_id="reasoner"):
        return _event(now_us, 'query', source_id, self.observer_id, self.query_id, 'query', None,
                      {'tier': self.tier, 'region': {'center': list(self.center_xy), 'radius_m': self.radius_m},
                       'time_us': self.time_us, 'concept': self.concept, 'reason': self.reason})


@dataclass(frozen=True, slots=True)
class Answer:
    query_id: str
    observer_id: str
    result: str
    observed_time_us: int | None
    coverage_ref: str | None
    evidence_refs: tuple[str, ...]

    def __post_init__(self):
        _id(self.query_id)
        _id(self.observer_id)
        if self.result not in RESULTS:
            raise ValueError("invalid answer result")
        if self.observed_time_us is not None:
            _time(self.observed_time_us)
        _refs(self.evidence_refs)
        if self.coverage_ref is not None:
            _refs((self.coverage_ref,))
        if (self.result == 'detected' and not self.evidence_refs) or (self.result == 'observed_free' and self.coverage_ref is None):
            raise ValueError("answer lacks required evidence")

    def to_event(self, now_us, dest_id="reasoner"):
        return _event(now_us, 'answer', self.observer_id, dest_id, self.query_id, 'answer', self.result,
                      {'observed_time_us': self.observed_time_us, 'coverage_ref': self.coverage_ref,
                       'evidence_refs': list(self.evidence_refs)})


@dataclass(frozen=True, slots=True)
class Verdict:
    verdict_id: str
    subject: str
    state: str
    evidence_refs: tuple[str, ...]
    reason: str
    model_call_ids: tuple[str, ...]

    def __post_init__(self):
        _id(self.verdict_id)
        _id(self.subject, 'subject')
        if self.state not in STATES:
            raise ValueError("invalid verdict state")
        _refs(self.evidence_refs)
        if self.state != 'unverified' and not self.evidence_refs:
            raise ValueError("non-unverified verdict needs evidence")
        if type(self.reason) is not str or len(self.reason) > 1024:
            raise ValueError("reason must be bounded text data")
        if type(self.model_call_ids) is not tuple:
            raise ValueError("model_call_ids must be a unique tuple")
        for call in self.model_call_ids:
            _id(call, 'model_call_id')
        if len(set(self.model_call_ids)) != len(self.model_call_ids):
            raise ValueError("duplicate model_call_ids")

    def to_event(self, now_us):
        return _event(now_us, 'verdict', 'reasoner', None, self.verdict_id, 'verdict', self.state,
                      {'subject': self.subject, 'evidence_refs': list(self.evidence_refs),
                       'reason': self.reason, 'model_call_ids': list(self.model_call_ids)})


def _event(now, kind, source, dest, mid, message_type, state, detail):
    value = dict(schema=SCHEMA, t_replay_us=now, type=kind, source_id=source,
                 dest_id=dest, message_id=mid, message_type=message_type, evidence_ref=None,
                 bytes=None, check_codes=[], state=state, detail=detail)
    Event(**value)  # Keep the frozen event contract authoritative.
    return value


def _matches(message, concept, labels):
    if concept is None:
        return True
    normalized = normalize_concept(concept)
    if message.label_id is not None and message.label_id in labels:
        return normalize_concept(labels[message.label_id]) == normalized
    return CLASS_CONCEPTS.get(normalized) == message.observation.cls


def _circle_cells(query, spec):
    x, y = query.center_xy
    r = query.radius_m
    x0, y0, x1, y1 = spec.bounds_xy
    if x-r < x0 or y-r < y0 or x+r >= x1 or y+r >= y1:
        return None
    ix0, iy0 = spec.cell_at(x-r, y-r)
    ix1 = min(spec.nx-1, spec.cell_at(x+r, y)[0])
    iy1 = min(spec.ny-1, spec.cell_at(x, y+r)[1])
    cells = set()
    for ix in range(max(ix0, 0), ix1+1):
        for iy in range(max(iy0, 0), iy1+1):
            lo_x, lo_y, hi_x, hi_y = spec.cell_bounds((ix, iy))
            closest = (max(lo_x, min(x, hi_x)), max(lo_y, min(y, hi_y)))
            if math.dist(closest, query.center_xy) <= r:
                cells.add((ix, iy))
    return cells


def answer_query(query, capability, observations, coverage, now_us, *, labels=None):
    """Answer about admitted data at query.time_us; never infer empty from silence.

    Optional labels maps shared label_id values to concepts for open vocabulary.
    No semantic meaning is guessed for UNKNOWN or unregistered dynamic labels.
    """
    _time(now_us)
    if query.observer_id != capability.observer_id:
        raise ValueError("capability does not describe the queried observer")
    def answer(result, observed=None, cov=None, refs=()):
        return Answer(query.query_id, capability.observer_id, result, observed, cov, refs)
    if query.tier not in capability.query_tiers or (query.tier == 'T3' and capability.vocabulary != 'open'):
        return answer('unsupported')
    if not circle_intersects(query.center_xy, query.radius_m, capability.coverage_xy):
        return answer('not_covered')
    def fresh(message):
        e = message.envelope
        return (e.source_id == capability.observer_id and e.generation_time_us <= now_us
                and e.capture_time_us <= query.time_us <= min(e.valid_until_us, now_us))
    observations = [m for m in observations if isinstance(m, ObservationMessage) and fresh(m)]
    coverage = [m for m in coverage if isinstance(m, CoverageMessage) and fresh(m)]
    if not observations and not coverage:
        return answer('unavailable')
    matches = [m for m in observations if _matches(m, query.concept, labels or {})
               and math.dist(m.observation.xyz_m[:2], query.center_xy) <= query.radius_m
               and inside_polygon(m.observation.xyz_m[:2], capability.coverage_xy)]
    if matches:
        return answer('detected', max(m.envelope.capture_time_us for m in matches),
                      refs=tuple(sorted({evidence_ref(m) for m in matches})))
    full_nominal = inside_polygon(query.center_xy, capability.coverage_xy) and all(
        _distance_segment(query.center_xy, a, b) >= query.radius_m
        for a, b in zip(capability.coverage_xy, capability.coverage_xy[1:]+capability.coverage_xy[:1]))
    for message in sorted(coverage, key=lambda m: m.envelope.capture_time_us, reverse=True):
        c = message.content
        if (full_nominal and c.modality in RANGE_MODALITIES and c.modality in capability.modalities
                and query.time_us - message.envelope.capture_time_us <= 300_000):
            cells = _circle_cells(query, c.spec)
            if cells and cells <= c.free:
                return answer('observed_free', message.envelope.capture_time_us, evidence_ref(message))
    observed = max(m.envelope.capture_time_us for m in observations+coverage)
    return answer('not_detected', observed)
