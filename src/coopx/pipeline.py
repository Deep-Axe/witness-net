"""S1/S2 runs: signed observations and signed coverage through PacketTransport.

Default checker is the deterministic `coopx.checker.Checker` built from the
run's trust anchors. `PassThroughChecker` remains available for transport-only
debugging (`checker=PassThroughChecker()`).

S2 admits every packet (observation or CX2C coverage) with the real Checker,
feeds admitted coverage into `coopx.coverage.CoverageMap`, and judges admitted
observations with `coopx.assess.Assessor`. No LLM.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from math import floor
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from coopx.assess import Assessment, Assessor
from coopx.checker import Checker, CheckResult
from coopx.coverage import CoverageContent, CoverageMap, CoverageMessage, GridSpec
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, EvidenceStore, Lineage, ObservationMessage,
                            decode, encode, sign)
from coopx.identity import NodeKey, TrustAnchor, TrustClass
from coopx.observers import lidar_synth
from coopx.observers.lidar_synth import LidarObserver, default_s2_world
from coopx.transport import PacketDelivery, PacketTransport, make_event, packet_hash

S1_NODES = ("pole1", "car1", "ego")
S1_DT_US = 100_000
S1_DURATION_US = 5_000_000
S1_TTL_US = 1_000_000
S1_FRAME = "site_map"
S1_SESSION = "s1"

S2_NODES = ("pole1", "lidar1", "ghost1", "ghost2", "ego")
S2_DT_US = 100_000
S2_DURATION_US = 5_000_000
S2_TTL_US = 1_000_000
S2_FRAME = "site_map"
S2_SESSION = "s2"
S2_GHOST_START_US = 2_000_000
S2_LIDAR_POSE = (0.0, 0.0)
S2_POLE_POSE = (0.0, -4.0)
S2_GHOST_FREE_XYZ = (5.5, -8.5, 0.5)
S2_GHOST_SILENT_XYZ = (24.0, 15.2, 0.5)  # behind the parked truck along a lidar ray

# Wire grid (coopx.coverage.GridSpec) mirroring the observer's geometry grid.
S2_GRID = GridSpec(float(lidar_synth.S2_GRID.origin_x), float(lidar_synth.S2_GRID.origin_y),
                   float(lidar_synth.S2_GRID.cell_m), lidar_synth.S2_GRID.nx,
                   lidar_synth.S2_GRID.ny)


class PassThroughChecker:
    """Accepts any decodable packet; transport debugging only. Same shape as Checker."""

    def check(self, raw: bytes, now_us: int) -> CheckResult:
        try:
            message = decode(raw)
        except (ValueError, TypeError):
            return CheckResult(("DECODE_ERROR",), "rejected", None)
        return CheckResult(("OK",), "accepted", message)


def demo_node_key(key_seed: int, name: str) -> NodeKey:
    """Demo-only deterministic Ed25519. Not a production identity.

    material = sha256(f"{key_seed}|{name}"); wrapped in NodeKey.
    """
    if type(key_seed) is not int:
        raise ValueError("key_seed must be an int")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    material = hashlib.sha256(f"{key_seed}|{name}".encode("utf-8")).digest()
    return NodeKey(Ed25519PrivateKey.from_private_bytes(material))


def _anchors_and_keys(nodes: tuple[str, ...] = S1_NODES, *, key_seed: int | None = None,
                      trust: dict[str, TrustClass] | None = None,
                      anchored: dict[str, bool] | None = None):
    keys = {
        name: demo_node_key(key_seed, name) if key_seed is not None else NodeKey.generate()
        for name in nodes
    }
    region = (-200.0, -200.0, -10.0, 200.0, 200.0, 20.0)
    trust = trust or {name: TrustClass.A for name in nodes}
    anchored = anchored or {name: True for name in nodes}
    anchors = {}
    for name in nodes:
        cls = trust[name]
        is_anchored = anchored[name]
        anchors[name] = TrustAnchor(
            keys[name].public_key, f"{name}-key", cls,
            physical_observer=name if is_anchored else None,
            permitted_region=region, anchored=is_anchored)
    return keys, anchors


def _signed_observation(source: str, key: NodeKey, *, object_id: int, frame: int,
                        sim_us: int, cls: ObjectClass, xyz: tuple[float, float, float],
                        velocity: tuple[float, float, float],
                        session: str = S1_SESSION,
                        dimensions: tuple[float, float, float] | None = None,
                        message_id: str | None = None,
                        sensor_seq: int | None = None) -> ObservationMessage:
    sequence = frame if sensor_seq is None else sensor_seq
    observation = Observation(
        source, object_id, sequence, sim_us, cls, xyz, PositionAnchor.SURFACE,
        velocity_mps=velocity, dimensions_m=dimensions, confidence=0.9)
    message_id = message_id or f"{source}-{frame:04d}"
    envelope = Envelope(
        2, "observation", message_id, source, session, S1_FRAME,
        sim_us, sim_us, sim_us + S1_TTL_US, Auth(f"{source}-key"), sequence)
    lineage = Lineage(source, "synthetic", message_id, sim_us, derivation="direct")
    return sign(ObservationMessage(envelope, observation, lineage), key)


def _pose(source: str, frame: int) -> tuple[ObjectClass, tuple[float, float, float], tuple[float, float, float]]:
    t_s = frame * S1_DT_US / 1_000_000
    if source == "pole1":
        return ObjectClass.PEDESTRIAN, (8.0 + 1.4 * t_s, 1.0, 0.5), (1.4, 0.0, 0.0)
    return ObjectClass.VEHICLE, (12.0 + 6.0 * t_s, 4.0, 0.8), (6.0, 0.0, 0.0)


def _check_event(delivery: PacketDelivery, result: CheckResult, now_us: int) -> dict:
    message = result.message
    envelope = None if message is None else message.envelope
    detail = {}
    if isinstance(message, ObservationMessage):
        detail = {
            "xyz": list(message.observation.xyz_m),
            "cls": message.observation.cls.name,
            "sensor_seq": envelope.sensor_seq,
        }
    elif message is not None:
        detail = {"sensor_seq": envelope.sensor_seq}
    if delivery.tx_id is not None:
        detail["tx_id"] = delivery.tx_id
    return make_event(
        t_replay_us=now_us, type="check",
        source_id=delivery.source, dest_id=delivery.receiver,
        message_id=None if envelope is None else envelope.message_id,
        message_type=delivery.message_type if envelope is None else envelope.message_type,
        evidence_ref=delivery.evidence_ref or packet_hash(delivery.packet),
        bytes=len(delivery.packet), check_codes=list(result.codes),
        state=result.state, detail=detail)


def _ordered_events(events: list[dict]) -> list[dict]:
    """Nondecreasing t_replay_us; a check stays immediately after its delivery."""
    tagged: list[tuple[int, int, int, dict]] = []
    i = 0
    seq = 0
    while i < len(events):
        event = events[i]
        nxt = events[i + 1] if i + 1 < len(events) else None
        if (event["type"] == "packet_delivered" and nxt is not None
                and nxt["type"] == "check"
                and nxt.get("evidence_ref") == event.get("evidence_ref")
                and nxt.get("dest_id") == event.get("dest_id")):
            tagged.append((event["t_replay_us"], seq, 0, event))
            tagged.append((nxt["t_replay_us"], seq, 1, nxt))
            seq += 1
            i += 2
            continue
        tagged.append((event["t_replay_us"], seq, 0, event))
        seq += 1
        i += 1
    tagged.sort()
    return [item[3] for item in tagged]


def _write_events(path: Path, events: list[dict]) -> list[dict]:
    ordered = _ordered_events(events)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for event in ordered:
            handle.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
    return ordered


# ---------------------------------------------------------------------------
# S2 helpers: signed coverage, events
# ---------------------------------------------------------------------------

def wire_coverage(content: lidar_synth.CoverageContent, capture_us: int) -> CoverageContent:
    """Observer output (contract 2) -> coopx.coverage.CoverageContent.

    The capture window ends at the envelope capture time (coverage invariant).
    """
    spec = content.spec
    grid = GridSpec(float(spec.origin_x), float(spec.origin_y), float(spec.cell_m),
                    spec.nx, spec.ny)
    return CoverageContent(
        grid, (max(0, capture_us - S2_DT_US), capture_us), content.modality,
        tuple(float(v) for v in content.height_band_m), float(content.min_size_m),
        frozenset(content.observed), frozenset(content.free), frozenset(content.occupied))


def _signed_coverage(source: str, key: NodeKey, content: CoverageContent, *, sim_us: int,
                     sensor_seq: int, message_id: str) -> CoverageMessage:
    envelope = Envelope(
        2, "coverage", message_id, source, S2_SESSION, S2_FRAME,
        sim_us, sim_us, sim_us + S2_TTL_US, Auth(f"{source}-key"), sensor_seq)
    lineage = Lineage(source, "synthetic_lidar", message_id, sim_us, derivation="direct")
    return sign(CoverageMessage(envelope, content, lineage), key)


def _coverage_published_event(message: CoverageMessage, raw: bytes, sim_us: int) -> dict:
    content = message.content
    return make_event(
        t_replay_us=sim_us, type="coverage_published", source_id=message.envelope.source_id,
        message_id=message.envelope.message_id, message_type="coverage",
        evidence_ref=packet_hash(raw), bytes=len(raw),
        detail={
            "grid": {
                "origin": [content.spec.origin_x, content.spec.origin_y],
                "cell_m": content.spec.cell_m,
                "nx": content.spec.nx,
                "ny": content.spec.ny,
            },
            "modality": content.modality,
            "window_us": list(content.window_us),
            "free": [list(c) for c in sorted(content.free)],
            "occupied": [list(c) for c in sorted(content.occupied)],
            "observed_count": len(content.observed),
        })


def _assessment_event(message: ObservationMessage, assessment: Assessment,
                      evidence_ref: str, now_us: int, nbytes: int) -> dict:
    return make_event(
        t_replay_us=now_us, type="assessment",
        source_id=message.envelope.source_id, dest_id="ego",
        message_id=message.envelope.message_id, message_type="observation",
        evidence_ref=evidence_ref, bytes=nbytes,
        check_codes=list(assessment.codes), state=assessment.state,
        detail={
            "cells": [list(c) for c in assessment.cells],
            "supporting": list(assessment.supporting),
            "contradicting": list(assessment.contradicting),
            "reason": assessment.reason,
        })


def _map_snapshot_event(coverage_map: CoverageMap, now_us: int) -> dict:
    """Ego map at a tick, plus the per-source claims behind every contested cell."""
    detail = dict(coverage_map.snapshot(now_us))
    detail["contested_claims"] = [
        {"cell": cell, "claims": [[ref, kind] for ref, kind in
                                  coverage_map.claims_at(tuple(cell), now_us)]}
        for cell in detail["contested"]]
    return make_event(t_replay_us=now_us, type="map_snapshot", source_id="ego", detail=detail)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_s1(output: Path, *, seed: int = 0, loss: float = 0.0, delay_us: int = 100_000,
           jitter_us: int = 50_000, disconnect_at_us: int | None = None,
           checker=None, key_seed: int | None = None) -> dict:
    """Replay ~5 s at 10 Hz. `checker=` defaults to the real Checker over this run's anchors."""
    if disconnect_at_us is not None and (type(disconnect_at_us) is not int or disconnect_at_us < 0):
        raise ValueError("disconnect_at_us must be a nonnegative int")
    keys, anchors = _anchors_and_keys(key_seed=key_seed)
    if checker is None:
        checker = Checker(anchors, ttl_us=S1_TTL_US)
    events: list[dict] = []
    accepted = rejected = 0

    def on_delivery(delivery: PacketDelivery) -> None:
        nonlocal accepted, rejected
        if delivery.receiver != "ego":
            return
        result = checker.check(delivery.packet, delivery.available_sim_us)
        events.append(_check_event(delivery, result, delivery.available_sim_us))
        if result.state == "accepted":
            accepted += 1
        else:
            rejected += 1

    transport = PacketTransport(
        S1_NODES, delay_us=delay_us, jitter_us=jitter_us, loss_probability=loss,
        seed=seed, event_callback=events.append, delivery_callback=on_delivery)
    disconnected = False
    publishers = (("pole1", 1), ("car1", 2))

    for frame, sim_us in enumerate(range(0, S1_DURATION_US, S1_DT_US)):
        transport.receive_until(sim_us)
        if disconnect_at_us is not None and not disconnected and sim_us >= disconnect_at_us:
            transport.disconnect("pole1", sim_us=sim_us)
            disconnected = True
        for source, object_id in publishers:
            cls, xyz, velocity = _pose(source, frame)
            message = _signed_observation(
                source, keys[source], object_id=object_id, frame=frame,
                sim_us=sim_us, cls=cls, xyz=xyz, velocity=velocity)
            raw = encode(message)
            events.append(make_event(
                t_replay_us=sim_us, type="observation_published", source_id=source,
                message_id=message.envelope.message_id, message_type="observation",
                evidence_ref=packet_hash(raw), bytes=len(raw),
                detail={"xyz": list(xyz), "cls": cls.name, "sensor_seq": frame}))
            transport.send(source, raw, sim_us=sim_us,
                           message_id=message.envelope.message_id, message_type="observation")
    drain_us = (S1_DURATION_US - S1_DT_US) + delay_us + jitter_us
    transport.receive_until(drain_us)
    events[:] = _write_events(output, events)

    by_type = Counter(event["type"] for event in events)
    return {
        "events": str(output),
        "events_by_type": dict(by_type),
        "sent_bytes": transport.sent_bytes,
        "received_bytes": transport.received_bytes,
        "drops": dict(transport.drops),
        "accepted": accepted,
        "rejected": rejected,
        "checker": type(checker).__name__,
        "n_events": len(events),
    }


def run_s2(output: Path, *, seed: int = 0, loss: float = 0.0, delay_us: int = 100_000,
           jitter_us: int = 0, key_seed: int = 0, checker=None) -> dict:
    """Pole camera + LiDAR observer + two camera ghosts, judged with no LLM.

    Every packet goes through the real Checker. Admitted coverage feeds the
    CoverageMap; admitted observations are judged by the Assessor.
    """
    trust = {
        "pole1": TrustClass.A, "lidar1": TrustClass.A, "ego": TrustClass.A,
        "ghost1": TrustClass.B, "ghost2": TrustClass.B,
    }
    anchored = {
        "pole1": True, "lidar1": True, "ego": True,
        "ghost1": False, "ghost2": False,
    }
    keys, anchors = _anchors_and_keys(
        S2_NODES, key_seed=key_seed, trust=trust, anchored=anchored)
    if checker is None:
        checker = Checker(anchors, ttl_us=S2_TTL_US)
    store = EvidenceStore(anchors)
    coverage_map = CoverageMap(S2_GRID, ttl_us=S2_TTL_US, spatial_frame_id=S2_FRAME, store=store)
    assessor = Assessor(store, coverage_map)

    world = default_s2_world()
    lidar = LidarObserver(lidar_synth.S2_GRID, pose=S2_LIDAR_POSE, seed=seed)
    events: list[dict] = []
    accepted = rejected = 0
    outcomes: Counter = Counter()
    sequences: Counter = Counter()

    def next_seq(source: str) -> int:
        # One increasing stream per source across coverage and object packets.
        value = sequences[source]
        sequences[source] += 1
        return value

    def on_delivery(delivery: PacketDelivery) -> None:
        nonlocal accepted, rejected
        if delivery.receiver != "ego":
            return
        now_us = delivery.available_sim_us
        result = checker.check(delivery.packet, now_us)
        events.append(_check_event(delivery, result, now_us))
        if result.state != "accepted":
            rejected += 1
            return
        accepted += 1
        if isinstance(result.message, CoverageMessage):
            coverage_map.add(result.message, now_us)
            return
        assessment = assessor.assess(result.message, now_us)
        events.append(_assessment_event(
            result.message, assessment, delivery.evidence_ref, now_us, len(delivery.packet)))
        outcomes[(result.message.envelope.source_id, assessment.codes[0])] += 1

    transport = PacketTransport(
        S2_NODES, delay_us=delay_us, jitter_us=jitter_us, loss_probability=loss,
        seed=seed, event_callback=events.append, delivery_callback=on_delivery)

    def publish_observation(source: str, object_id: int, frame: int, sim_us: int,
                            cls: ObjectClass, xyz, velocity, dimensions=None,
                            message_id: str | None = None) -> None:
        sensor_seq = next_seq(source)
        message = _signed_observation(
            source, keys[source], object_id=object_id, frame=frame,
            sim_us=sim_us, cls=cls, xyz=xyz, velocity=velocity,
            session=S2_SESSION, dimensions=dimensions,
            message_id=message_id, sensor_seq=sensor_seq)
        raw = encode(message)
        events.append(make_event(
            t_replay_us=sim_us, type="observation_published", source_id=source,
            message_id=message.envelope.message_id, message_type="observation",
            evidence_ref=packet_hash(raw), bytes=len(raw),
            detail={"xyz": list(xyz), "cls": cls.name, "sensor_seq": sensor_seq}))
        transport.send(source, raw, sim_us=sim_us,
                       message_id=message.envelope.message_id, message_type="observation")

    for frame, sim_us in enumerate(range(0, S2_DURATION_US, S2_DT_US)):
        transport.receive_until(sim_us)
        events.append(_map_snapshot_event(coverage_map, sim_us))

        cls, xyz, velocity = _pose("pole1", frame)
        publish_observation("pole1", 1, frame, sim_us, cls, xyz, velocity)

        scan = lidar.scan(world, t_us=sim_us)
        cov_id = f"lidar1-cov-{frame:04d}"
        coverage = _signed_coverage(
            "lidar1", keys["lidar1"], wire_coverage(scan, sim_us), sim_us=sim_us,
            sensor_seq=next_seq("lidar1"), message_id=cov_id)
        raw_cov = encode(coverage)
        events.append(_coverage_published_event(coverage, raw_cov, sim_us))
        transport.send("lidar1", raw_cov, sim_us=sim_us,
                       message_id=cov_id, message_type="coverage")

        for index, geom in enumerate(lidar.objects(scan)):
            publish_observation(
                "lidar1", 100 + index, frame, sim_us, ObjectClass.UNKNOWN,
                geom["xyz"], (0.0, 0.0, 0.0), geom["dimensions_m"],
                message_id=f"lidar1-{frame:04d}-{index}")

        if sim_us >= S2_GHOST_START_US:
            publish_observation(
                "ghost1", 91, frame, sim_us, ObjectClass.PEDESTRIAN,
                S2_GHOST_FREE_XYZ, (0.0, 0.0, 0.0))
            publish_observation(
                "ghost2", 92, frame, sim_us, ObjectClass.PEDESTRIAN,
                S2_GHOST_SILENT_XYZ, (0.0, 0.0, 0.0))

    drain_us = (S2_DURATION_US - S2_DT_US) + delay_us + jitter_us
    transport.receive_until(drain_us)
    events.append(_map_snapshot_event(coverage_map, drain_us))
    events[:] = _write_events(output, events)

    by_type = Counter(event["type"] for event in events)
    return {
        "events": str(output),
        "events_by_type": dict(by_type),
        "sent_bytes": transport.sent_bytes,
        "received_bytes": transport.received_bytes,
        "drops": dict(transport.drops),
        "accepted": accepted,
        "rejected": rejected,
        "assessments": sum(outcomes.values()),
        "assessment_outcomes": {f"{src}:{code}": n for (src, code), n in sorted(outcomes.items())},
        "checker": type(checker).__name__,
        "n_events": len(events),
        "grid": {
            "origin": [S2_GRID.origin_x, S2_GRID.origin_y],
            "cell_m": S2_GRID.cell_m, "nx": S2_GRID.nx, "ny": S2_GRID.ny,
        },
    }
