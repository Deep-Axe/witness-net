"""Object-report fusion, Muse off, same admitted inputs for every policy.

Geo is a deterministic object-level baseline inspired by ROBOSAC consensus
(https://arxiv.org/abs/2303.09495) and CAD collaborative anomaly detection
(https://www.usenix.org/conference/usenixsecurity24/presentation/zhang-qingzhao),
NOT a reproduction of either feature-level method. For each report, count
physical observers among class/anchor/frame-compatible reports within 1.5m
(pedestrians), 3m (vehicles), and 300ms; two are confirmation. At least half
its footprint in eligible recent range free space contradicts it, taking
precedence over consensus. Unverified reports still create bounded caution.
Geometric grouping uses configured physical bodies, not lineage/common-cause
proofs. Witness uses authenticated roots and the existing Assessor, retains
contested evidence, and protects every local detection. No model is called.
"""

import math
from dataclasses import dataclass

from coopx.coverage import evidence_eligible
from coopx.domain import ObjectClass
from coopx.envelope import ObservationMessage, evidence_ref
from coopx.observers.tracker import PARAMETERS

FUSION_MODES = ("ego", "union", "geo", "witness")
PED_GATE_M = 1.5  # project plan: conservative cross-view pedestrian association
VEHICLE_GATE_M = 3.0  # surface anchors can differ across opposite vehicle faces
COMPATIBLE_US = 300_000  # three nominal 10Hz frames; fixed across all modes
MIN_WITNESSES = 2  # single-source reports remain caution, not confirmation
MAX_PREDICTION_US = 4_000_000  # five-second freshness budget minus a one-second guard
ACCELERATION_STD_MPS2 = (
    1.5  # declared constant-velocity model process noise, not a GT fit
)
FREE_FRACTION = 0.5  # same full-footprint contradiction threshold as Assessor


@dataclass(frozen=True)
class Belief:
    xy: tuple[float, float]
    cls: str
    local: bool
    state: str
    refs: tuple[str, ...]
    track_id: int | None = None
    velocity_mps: tuple | None = None
    predicted_position: tuple | None = None
    position_std_m: tuple | None = None
    capture_time_us: int | None = None
    captured_position: tuple | None = None
    source_id: str | None = None
    velocity_std_mps: tuple | None = None
    observation_ref: str | None = None


def _belief(message, local, state, refs, now_us, tracking):
    o = message.observation
    age = (now_us - o.captured_sim_us) / 1e6
    velocity = o.velocity_mps if tracking else None
    xyz = tuple(x + age * v for x, v in zip(o.xyz_m, velocity or (0.0, 0.0, 0.0)))
    std = o.position_std_m or (1.0, 1.0, 1.0)
    if tracking:
        speed_bound = PARAMETERS["speed_limits_mps"].get(int(o.cls), 20.0)
        vstd = message.velocity_std_mps or (speed_bound,) * 3
        std = tuple(
            math.hypot(p, age * v, 0.5 * ACCELERATION_STD_MPS2 * age * age)
            for p, v in zip(std, vstd)
        )
    return Belief(
        xyz[:2],
        o.cls.name,
        local,
        state,
        tuple(sorted(set(refs))),
        o.object_id if tracking else None,
        velocity,
        xyz if tracking else None,
        tuple(std),
        o.captured_sim_us,
        tuple(o.xyz_m),
        message.envelope.source_id,
        message.velocity_std_mps,
        evidence_ref(message),
    )


def _capture_assessor(message, store, coverage_map, default, now_us, memory):
    """Only already-admitted scans, selected at capture time. Prediction never
    changes the claim being tested. Cache maps per capture to avoid rebuilding
    dense LiDAR coverage for every box on every decision tick.
    """
    if memory is None or not memory.get("coverage_history"):
        return default
    from coopx.assess import Assessor
    from coopx.coverage import CoverageMap

    capture = message.envelope.capture_time_us
    caches = memory.setdefault("capture_maps", {})
    for old in list(caches):
        if now_us - old > MAX_PREDICTION_US:
            del caches[old]
    if capture not in caches:
        grid = CoverageMap(
            coverage_map.spec,
            ttl_us=coverage_map.ttl_us,
            spatial_frame_id=coverage_map.spatial_frame_id,
            store=store,
        )
        caches[capture] = (grid, Assessor(store, grid), set())
    grid, assessor, seen = caches[capture]
    for coverage in memory["coverage_history"]:
        e = coverage.envelope
        if (
            abs(e.capture_time_us - capture) > COMPATIBLE_US
            or e.generation_time_us > now_us
            or not e.capture_time_us
            <= now_us
            <= min(e.valid_until_us, e.capture_time_us + grid.ttl_us)
        ):
            continue
        ref = evidence_ref(coverage)
        if ref not in seen:
            grid.add(coverage, now_us)
            seen.add(ref)
    return assessor


def fuse(
    mode,
    *,
    ego_observations,
    remote_admitted,
    store,
    coverage_map,
    assessor,
    now_us,
    memory=None,
    tracking=True,
):
    """`memory` (optional dict, kept by the caller across decisions): a remote
    claim once contradicted by time-compatible free space stays contradicted.
    The map keeps only each slot's newest scan, so without memory the refuting
    scan is superseded and a stale claim would look merely unverified again.
    Local (ego) beliefs are never contradicted or removed."""
    if mode not in FUSION_MODES:
        raise ValueError("unknown fusion mode")
    if type(now_us) is not int or now_us < 0:
        raise ValueError("invalid fusion time")
    if assessor is None:
        from coopx.assess import Assessor

        assessor = Assessor(store, coverage_map)
    elif assessor.store is not store or assessor.coverage_map is not coverage_map:
        raise ValueError("fusion evidence store and assessment map must match")

    def active(message):
        if not isinstance(message, ObservationMessage):
            return False
        e = message.envelope
        return (
            e.spatial_frame_id == coverage_map.spatial_frame_id
            and e.generation_time_us <= now_us
            and e.capture_time_us
            <= now_us
            <= min(e.valid_until_us, e.capture_time_us + coverage_map.ttl_us)
            and (not tracking or now_us - e.capture_time_us < MAX_PREDICTION_US)
        )

    local = [m for m in ego_observations if active(m)]
    remote = [m for m in remote_admitted if active(m)]

    # Select the latest update per object, not a stream of repeated votes.
    def latest(messages):
        result = {}
        for m in messages:
            old = result.get(m.observation.key)
            if old is None or m.envelope.capture_time_us > old.envelope.capture_time_us:
                result[m.observation.key] = m
        return list(result.values())

    # A per-frame detector has no tracker: object ids are not stable across
    # frames, so a source's newer frame supersedes its older detections (the
    # camera's current view). Same rule for ego and remote sources.
    def current_frame(messages):
        newest = {}
        for m in messages:
            src = m.envelope.source_id
            newest[src] = max(newest.get(src, -1), m.envelope.capture_time_us)
        return [
            m
            for m in messages
            if m.envelope.capture_time_us == newest[m.envelope.source_id]
        ]

    local, remote = latest(local), latest(remote)
    if not tracking:
        local, remote = current_frame(local), current_frame(remote)
    if mode == "ego":
        return [
            _belief(m, True, "confirmed", [evidence_ref(m)], now_us, tracking)
            for m in local
        ]
    messages = local + remote
    for message in messages:
        store.add(message)
    # All admitted observations must exist before assessing the first one, so
    # corroboration cannot depend on input iteration order.
    history = [
        m
        for m in store.recent_observations(now_us - coverage_map.ttl_us, now_us)
        if active(m)
    ]
    result = []
    for m in messages:
        is_local = m in local
        ref = evidence_ref(m)
        capture_assessor = (
            _capture_assessor(m, store, coverage_map, assessor, now_us, memory)
            if not is_local and mode in ("geo", "witness")
            else assessor
        )
        if mode == "union":
            state = "confirmed" if is_local else "unverified"
            refs = (ref,)
        elif mode == "witness":
            if is_local:
                state, refs = "confirmed", (ref,)
            else:
                assessment = capture_assessor.assess(m, now_us)
                state = (
                    "confirmed"
                    if assessment.state == "corroborated"
                    else assessment.state
                )
                refs = (ref, *assessment.supporting, *assessment.contradicting)
        else:

            def agrees(c, m=m):
                a, b = c.observation, m.observation
                gate = VEHICLE_GATE_M if b.cls == ObjectClass.VEHICLE else PED_GATE_M
                return (
                    a.cls == b.cls
                    and a.anchor == b.anchor
                    and c.label_id == m.label_id
                    and c.envelope.spatial_frame_id == m.envelope.spatial_frame_id
                    and abs(a.captured_sim_us - b.captured_sim_us) <= COMPATIBLE_US
                    and math.dist(a.xyz_m, b.xyz_m) <= gate
                )

            agreeing = [c for c in history if agrees(c)]
            bodies = {
                a.physical_observer
                for c in agreeing
                if (a := store.anchors.get(c.envelope.source_id)) is not None
                and a.physical_observer is not None
                and evidence_eligible(a)
            }
            refs = tuple(evidence_ref(c) for c in agreeing)
            # Same verified geometry as Witness, but disagreement rejects a
            # report even when geometric consensus supports it (three states).
            from coopx.assess import footprint

            cells, denominator = footprint(m, coverage_map.spec)
            free, free_refs, _ = capture_assessor._range_evidence(m, cells, now_us)
            conflict = bool(cells) and len(free) / denominator >= FREE_FRACTION
            state = (
                "contradicted"
                if conflict
                else "confirmed"
                if len(bodies) >= MIN_WITNESSES
                else "unverified"
            )
            refs = (*refs, *free_refs)
        if memory is not None and not is_local and mode in ("geo", "witness"):
            refuted = memory.setdefault("refuted", {})
            # Object IDs are SOURCE-local (domain.Observation.key), including
            # across sensors. Renaming a sensor must not reset a refutation.
            track = (
                m.envelope.source_id,
                m.envelope.session_id,
                m.observation.object_id,
            )
            if state == "contradicted":
                refuted[track] = tuple(sorted({*refuted.get(track, ()), *refs}))
            elif track in refuted:
                state = "contradicted"
                refs = (*refs, *refuted[track])
        result.append(_belief(m, is_local, state, refs, now_us, tracking))
    return result
