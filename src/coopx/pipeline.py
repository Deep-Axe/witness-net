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
import math
import os
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

S2_NODES = ("pole1", "pole2", "pole3", "lidar1", "ghost1", "ghost2", "ego")
S2_DT_US = 100_000
S2_DURATION_US = 5_000_000
S2_TTL_US = 1_000_000
S2_FRAME = "site_map"
S2_SESSION = "s2"
S2_GHOST_START_US = 2_000_000
S2_REFLECTION_START_US = 3_000_000
S2_LIDAR_POSE = (0.0, 0.0)
S2_POLE_POSE = (0.0, -4.0)
S2_GHOST_FREE_XYZ = (5.5, -8.5, 0.5)
S2_GHOST_SILENT_XYZ = (24.0, 15.2, 0.5)  # behind the parked truck along a lidar ray
# Honest-but-wrong cameras: a reflection in glass, >1.5 m from ghost1 so they
# cannot corroborate the class-B ghost (unanchored → never a witness).
S2_REFLECTION_XYZ = (3.5, -12.5, 0.7)
S2_REFLECTION_SOURCES = ("pole2", "pole3")
S2_GRID = lidar_synth.S2_GRID
# Calibrated S2 map poses (site frame). Viewer falls back to its table for old logs.
S2_MAP_OBSERVERS = [
    {"id": "lidar1", "xy": [float(S2_LIDAR_POSE[0]), float(S2_LIDAR_POSE[1])]},
    {"id": "pole1", "xy": [float(S2_POLE_POSE[0]), float(S2_POLE_POSE[1])]},
    {"id": "pole2", "xy": [-2.0, -8.0]},
    {"id": "pole3", "xy": [2.0, -8.0]},
    {"id": "ego", "xy": [-2.0, -2.0]},
]


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
                      anchored: dict[str, bool] | None = None,
                      region: tuple[float, float, float, float, float, float] | None = None):
    keys = {
        name: demo_node_key(key_seed, name) if key_seed is not None else NodeKey.generate()
        for name in nodes
    }
    region = region or (-200.0, -200.0, -10.0, 200.0, 200.0, 20.0)
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


def _result_mode(result, fallback: str) -> str:
    mode = getattr(result, "mode", None)
    return mode if mode in ("live", "snapshot") else fallback


def admit_packet(checker, raw: bytes, now_us: int, *, scene_time_us: int | None = None):
    """Live admission, or snapshot admission for Q&A packets (contract 11).

    Snapshot admissions (Checker docstring is the rule) must never feed the
    Assessor, coverage map or any hazard path: Reasoner evidence only.
    """
    result = checker.check(raw, now_us, scene_time_us=scene_time_us)
    return result, result.mode


def _check_event(delivery: PacketDelivery, result: CheckResult, now_us: int,
                 *, mode: str = "live") -> dict:
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
    detail["mode"] = mode if mode in ("live", "snapshot") else "live"
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


def _require_output(path: Path, overwrite: bool) -> None:
    """Fail before any dataset, model or file work if the log would be refused."""
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")


def _write_events(path: Path, events: list[dict], *, overwrite: bool = False) -> list[dict]:
    """Write via a same-directory temp file; replace only on success."""
    ordered = _ordered_events(events)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            for event in ordered:
                handle.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return ordered


# ---------------------------------------------------------------------------
# S2 helpers: signed coverage, events
# ---------------------------------------------------------------------------

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


def _grid_dict(spec) -> dict:
    return {"origin": [spec.origin_x, spec.origin_y], "cell_m": spec.cell_m,
            "nx": spec.nx, "ny": spec.ny}


def _is_default_s2_grid(spec) -> bool:
    return (spec.origin_x, spec.origin_y, spec.cell_m, spec.nx, spec.ny) == (
        -10.0, -30.0, 1.0, 60, 60)


def _assessment_event(message: ObservationMessage, assessment: Assessment,
                      evidence_ref: str, now_us: int, nbytes: int, *, spec=None) -> dict:
    detail = {
        "cells": [list(c) for c in assessment.cells],
        "supporting": list(assessment.supporting),
        "contradicting": list(assessment.contradicting),
        "reason": assessment.reason,
    }
    if spec is not None and not _is_default_s2_grid(spec):
        detail["grid"] = _grid_dict(spec)
    return make_event(
        t_replay_us=now_us, type="assessment",
        source_id=message.envelope.source_id, dest_id="ego",
        message_id=message.envelope.message_id, message_type="observation",
        evidence_ref=evidence_ref, bytes=nbytes,
        check_codes=list(assessment.codes), state=assessment.state,
        detail=detail)


def _map_snapshot_event(coverage_map: CoverageMap, now_us: int,
                        observers: list[dict] | None = None) -> dict:
    """Ego map at a tick, plus the per-source claims behind every contested cell."""
    detail = dict(coverage_map.snapshot(now_us))
    detail["contested_claims"] = [
        {"cell": cell, "claims": [[ref, kind] for ref, kind in
                                  coverage_map.claims_at(tuple(cell), now_us)]}
        for cell in detail["contested"]]
    if not _is_default_s2_grid(coverage_map.spec):
        detail["grid"] = _grid_dict(coverage_map.spec)
    if observers:
        detail["observers"] = observers
    return make_event(t_replay_us=now_us, type="map_snapshot", source_id="ego", detail=detail)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_s1(output: Path, *, seed: int = 0, loss: float = 0.0, delay_us: int = 100_000,
           jitter_us: int = 50_000, disconnect_at_us: int | None = None,
           checker=None, key_seed: int | None = None, overwrite: bool = False) -> dict:
    """Replay ~5 s at 10 Hz. `checker=` defaults to the real Checker over this run's anchors."""
    _require_output(output, overwrite)
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
    events[:] = _write_events(output, events, overwrite=overwrite)

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
           jitter_us: int = 0, key_seed: int = 0, checker=None, overwrite: bool = False) -> dict:
    """Pole camera + LiDAR observer + two camera ghosts, judged with no LLM.

    Every packet goes through the real Checker. Admitted coverage feeds the
    CoverageMap; admitted observations are judged by the Assessor.
    """
    _require_output(output, overwrite)
    trust = {
        "pole1": TrustClass.A, "pole2": TrustClass.A, "pole3": TrustClass.A,
        "lidar1": TrustClass.A, "ego": TrustClass.A,
        "ghost1": TrustClass.B, "ghost2": TrustClass.B,
    }
    anchored = {
        "pole1": True, "pole2": True, "pole3": True,
        "lidar1": True, "ego": True,
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
    lidar = LidarObserver(S2_GRID, pose=S2_LIDAR_POSE, seed=seed)
    events: list[dict] = []
    accepted = rejected = 0
    outcomes: Counter = Counter()
    sequences: Counter = Counter()
    reflection_latest: dict[str, tuple] = {}

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
        source = result.message.envelope.source_id
        if source in S2_REFLECTION_SOURCES:
            reflection_latest[source] = (
                result.message, delivery.evidence_ref, len(delivery.packet))
            # Wait until both honest-but-wrong cameras are admitted so each
            # assessment sees two independent witnesses + LiDAR free space.
            if len(reflection_latest) < 2:
                return
            to_judge = list(reflection_latest.values())
        else:
            to_judge = [(result.message, delivery.evidence_ref, len(delivery.packet))]
        # Both cameras must be in the store before either is judged, otherwise
        # the first looks like a lone ghost (contradicted, not contested).
        for message, _ref, _nbytes in to_judge:
            store.add(message)
        for message, ref, nbytes in to_judge:
            assessment = assessor.assess(message, now_us)
            events.append(_assessment_event(message, assessment, ref, now_us, nbytes))
            outcomes[(message.envelope.source_id, assessment.codes[0])] += 1

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
        events.append(_map_snapshot_event(coverage_map, sim_us, S2_MAP_OBSERVERS))

        cls, xyz, velocity = _pose("pole1", frame)
        publish_observation("pole1", 1, frame, sim_us, cls, xyz, velocity)

        scan = lidar.scan(world, t_us=sim_us)
        cov_id = f"lidar1-cov-{frame:04d}"
        coverage = _signed_coverage(
            "lidar1", keys["lidar1"], scan, sim_us=sim_us,
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
        if sim_us >= S2_REFLECTION_START_US:
            for source, object_id in (("pole2", 81), ("pole3", 82)):
                publish_observation(
                    source, object_id, frame, sim_us, ObjectClass.PEDESTRIAN,
                    S2_REFLECTION_XYZ, (0.0, 0.0, 0.0))

    drain_us = (S2_DURATION_US - S2_DT_US) + delay_us + jitter_us
    transport.receive_until(drain_us)
    events.append(_map_snapshot_event(coverage_map, drain_us, S2_MAP_OBSERVERS))
    events[:] = _write_events(output, events, overwrite=overwrite)

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


# ---------------------------------------------------------------------------
# S3: one dataset frame, SAM observer, Muse Spark Reasoner (plan -> execute -> judge)
# ---------------------------------------------------------------------------

S3_SCENARIO = "2023-04-04-14-28-53_45_0"
S3_RSU = "-2"
S3_RSU_NEAR = "-1"
S3_EGO = "2"
S3_NODES = ("rsu-2", "rsu-1", "ego")
S3_SESSION = "s3"
S3_TTL_US = 5_000_000
def _default_s3_dataset() -> Path:
    here = Path(__file__).resolve().parents[2] / "data" / "v2x-real" / "val"
    shared = Path("/home/deep/meta_hackathon/data/v2x-real/val")
    return here if here.is_dir() else shared


S3_DATASET = _default_s3_dataset()
S3_OVERLAY = Path("data/s3/overlay.png")
S3_PICK_PATH = Path("data/s3/pick.json")
S3_REGION = (-1000.0, -1000.0, -50.0, 1000.0, 1000.0, 80.0)
_S3_PICK_CACHE = {}


@dataclass(frozen=True)
class S3FramePick:
    scenario: str
    frame_id: str
    rsu_id: str
    ego_id: str
    camera_name: str
    pedestrian_id: str
    other_camera: str
    other_rsu_id: str


# Verified: frame 000096, pedestrian 53 in RSU -2 cam2 and RSU -1 cam1.
S3_FIXED_PICK = S3FramePick(
    S3_SCENARIO, "000096", S3_RSU, S3_EGO, "cam2", "53", "cam1", S3_RSU_NEAR)


def _pick_fingerprint(dataset_root: Path, scenario: str) -> str:
    root = Path(dataset_root).resolve()
    sentinel = root / scenario / S3_RSU / "000096.yaml"
    if sentinel.is_file():
        st = sentinel.stat()
        return f"{root}|{scenario}|{st.st_size}|{int(st.st_mtime)}"
    return f"{root}|{scenario}"


def _verify_s3_pick(dataset, pick: S3FramePick) -> bool:
    from coopx.dataset.v2xreal import box_fully_in_camera
    try:
        rsu = dataset.read(pick.scenario, pick.rsu_id, pick.frame_id)
        box = rsu.boxes[pick.pedestrian_id]
    except (KeyError, FileNotFoundError, OSError, ValueError):
        return False
    return (box.label.lower() == "pedestrian"
            and pick.camera_name in rsu.cameras
            and box_fully_in_camera(rsu, box, pick.camera_name))


def _store_s3_pick(path: Path, fingerprint: str, pick: S3FramePick) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({
        "fingerprint": fingerprint,
        "pick": {
            "scenario": pick.scenario, "frame_id": pick.frame_id, "rsu_id": pick.rsu_id,
            "ego_id": pick.ego_id, "camera_name": pick.camera_name,
            "pedestrian_id": pick.pedestrian_id, "other_camera": pick.other_camera,
            "other_rsu_id": pick.other_rsu_id,
        },
    }, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def pick_s3_frame(dataset_root: Path = S3_DATASET, *,
                  scenario: str = S3_SCENARIO, rsu_id: str = S3_RSU,
                  ego_id: str = S3_EGO) -> S3FramePick:
    """Deterministic first frame with a fully-visible RSU pedestrian.

    Cheap path: in-process cache, then data/s3/pick.json, then the verified
    fixed frame. Full dataset scan only if those fail.
    """
    cache_key = (str(Path(dataset_root).resolve()), scenario, rsu_id, ego_id)
    if cache_key in _S3_PICK_CACHE:
        return _S3_PICK_CACHE[cache_key]
    from coopx.dataset.v2xreal import (V2XReal, box_fully_in_camera, count_lidar_in_box)
    dataset = V2XReal(dataset_root)
    fingerprint = _pick_fingerprint(dataset_root, scenario)
    if S3_PICK_PATH.is_file():
        try:
            data = json.loads(S3_PICK_PATH.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint:
                pick = S3FramePick(**data["pick"])
                if _verify_s3_pick(dataset, pick):
                    _S3_PICK_CACHE[cache_key] = pick
                    return pick
        except (ValueError, TypeError, KeyError, OSError):
            pass
    if (scenario, rsu_id, ego_id) == (S3_SCENARIO, S3_RSU, S3_EGO) and _verify_s3_pick(
            dataset, S3_FIXED_PICK):
        _store_s3_pick(S3_PICK_PATH, fingerprint, S3_FIXED_PICK)
        _S3_PICK_CACHE[cache_key] = S3_FIXED_PICK
        return S3_FIXED_PICK
    frame_ids = dataset.aligned_frame_ids(scenario, (rsu_id, ego_id, S3_RSU_NEAR))
    if not frame_ids:
        frame_ids = dataset.aligned_frame_ids(scenario, (rsu_id, ego_id))
    preferred = None
    fallback = None
    for frame_id in frame_ids:
        rsu = dataset.read(scenario, rsu_id, frame_id)
        ego = dataset.read(scenario, ego_id, frame_id)
        pedestrians = [b for b in rsu.boxes.values() if b.label.lower() == "pedestrian"]
        ego_xy = (float(ego.lidar_to_site[0, 3]), float(ego.lidar_to_site[1, 3]))
        for box in pedestrians:
            center = box.center_site()
            if math.hypot(center[0] - ego_xy[0], center[1] - ego_xy[1]) > 40.0:
                continue
            for camera_name in sorted(rsu.cameras):
                if not box_fully_in_camera(rsu, box, camera_name):
                    continue
                if count_lidar_in_box(rsu, box) < 20:
                    continue
                hidden_from_ego = not any(
                    box_fully_in_camera(ego, box, ego_cam) for ego_cam in ego.cameras)
                pick = S3FramePick(
                    scenario, frame_id, rsu_id, ego_id, camera_name, box.object_id,
                    "cam1", S3_RSU_NEAR)
                if hidden_from_ego:
                    preferred = pick
                    break
                if fallback is None:
                    fallback = pick
            if preferred is not None:
                break
        if preferred is not None:
            break
    pick = preferred or fallback
    if pick is None:
        raise FileNotFoundError(
            f"no S3 pedestrian frame in {scenario} rsu={rsu_id} ego={ego_id}")
    _store_s3_pick(S3_PICK_PATH, fingerprint, pick)
    _S3_PICK_CACHE[cache_key] = pick
    return pick


def _scene_grid(points_xy, *, cell_m=1.0, pad=15.0) -> GridSpec:
    xs, ys = zip(*points_xy)
    x0 = math.floor(min(xs) - pad)
    y0 = math.floor(min(ys) - pad)
    x1 = math.ceil(max(xs) + pad)
    y1 = math.ceil(max(ys) + pad)
    nx = max(8, min(200, math.ceil((x1 - x0) / cell_m)))
    ny = max(8, min(200, math.ceil((y1 - y0) / cell_m)))
    return GridSpec(float(x0), float(y0), float(cell_m), nx, ny)


def _write_s3_overlay(path: Path, frame, camera_name: str, detections, gt_box) -> None:
    from PIL import Image, ImageDraw
    from coopx.dataset.v2xreal import project_camera
    image = Image.open(frame.cameras[camera_name].image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        if getattr(detection, "mask", None) is not None:
            ys, xs = detection.mask.nonzero()
            if len(xs):
                draw.point(list(zip(xs.tolist(), ys.tolist())), fill=(0, 220, 80))
        draw.rectangle(detection.box_xyxy, outline=(0, 220, 80), width=3)
    if gt_box is not None:
        uv, _depth, valid = project_camera(
            frame.to_camera(gt_box.corners_site(), camera_name),
            frame.cameras[camera_name].intrinsic)
        if valid.all():
            xs, ys = uv[:, 0], uv[:, 1]
            draw.rectangle((float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())),
                           outline=(220, 40, 40), width=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_s3(output: Path, *, offline: bool = False, key_seed: int = 0,
           budget_usd: float = 3.0, overwrite: bool = False,
           dataset_root: Path = S3_DATASET, client=None,
           delay_us: int = 100_000, overlay: Path = S3_OVERLAY) -> dict:
    """One V2X-Real frame: SAM fast path, then a Spark-planned, Spark-judged question.

    The fast path (SAM -> lift -> checker -> assessment) never waits on Spark;
    every model result enters the log at its availability time.
    """
    _require_output(output, overwrite)
    from coopx.dataset.v2xreal import (V2XReal, box_fully_in_camera,
                                       camera_footprint_xy, point_in_polygon)
    from coopx.envelope import encode, evidence_ref
    from coopx.meta.client import (BudgetExceeded, CacheMiss, MetaAPIError, MetaClient,
                                   model_call_event)
    from coopx.meta.reasoner import ObserverUnavailable, Reasoner, execute
    from coopx.observers.camera import (CameraObserver, jpeg_to_camera_frame,
                                       rsu_camera_capability)
    from coopx.observers.camera_lift import lift_box
    from coopx.query import Answer, answer_query

    pick = pick_s3_frame(dataset_root)
    dataset = V2XReal(dataset_root)
    rsu = dataset.read(pick.scenario, pick.rsu_id, pick.frame_id)
    other = dataset.read(pick.scenario, pick.other_rsu_id, pick.frame_id)
    ego = dataset.read(pick.scenario, pick.ego_id, pick.frame_id)
    pedestrian = rsu.boxes[pick.pedestrian_id]
    center = tuple(float(v) for v in pedestrian.center_site())

    other_box = other.boxes.get(pick.pedestrian_id)
    covering_cam, gap_cam = None, None
    for name in sorted(other.cameras):
        if other_box is not None:
            inside = box_fully_in_camera(other, other_box, name)
        else:
            inside = point_in_polygon(center[:2], camera_footprint_xy(other, name))
        if inside and covering_cam is None:
            covering_cam = name
        if not inside and gap_cam is None:
            gap_cam = name
    if covering_cam is None:
        covering_cam = pick.other_camera if pick.other_camera in other.cameras else sorted(other.cameras)[0]
    if gap_cam is None:
        gap_cam = next((n for n in sorted(other.cameras) if n != covering_cam), covering_cam)

    cap_rsu2 = rsu_camera_capability("rsu-2", rsu, pick.camera_name)
    cap_rsu1 = rsu_camera_capability("rsu-1", other, covering_cam)
    cap_rsu1b = rsu_camera_capability("rsu-1b", other, gap_cam)
    capabilities = [cap_rsu2, cap_rsu1, cap_rsu1b]

    def _xy(frame) -> list[float]:
        return [round(float(frame.lidar_to_site[0, 3]), 3),
                round(float(frame.lidar_to_site[1, 3]), 3)]

    map_observers = [
        {"id": "rsu-2", "xy": _xy(rsu)},
        {"id": "rsu-1", "xy": _xy(other)},
        {"id": "ego", "xy": _xy(ego)},
    ]
    points = [tuple(o["xy"]) for o in map_observers] + [center[:2]]
    grid = _scene_grid(points)
    keys, anchors = _anchors_and_keys(
        S3_NODES, key_seed=key_seed,
        trust={name: TrustClass.A for name in S3_NODES},
        anchored={name: True for name in S3_NODES},
        region=S3_REGION)
    checker = Checker(anchors, ttl_us=S3_TTL_US)
    store = EvidenceStore(anchors)
    coverage_map = CoverageMap(grid, ttl_us=S3_TTL_US, spatial_frame_id=S2_FRAME, store=store)
    assessor = Assessor(store, coverage_map)

    if client is None:
        client = MetaClient(budget_usd=budget_usd, offline=offline)
    events: list[dict] = []
    accepted = rejected = 0
    admitted: list = []
    accepted_refs: dict[str, ObservationMessage] = {}
    snapshot_scene: dict[str, int] = {}

    def on_delivery(delivery: PacketDelivery) -> None:
        nonlocal accepted, rejected
        if delivery.receiver != "ego":
            return
        now_us = delivery.available_sim_us
        ref = delivery.evidence_ref or packet_hash(delivery.packet)
        result, mode = admit_packet(
            checker, delivery.packet, now_us, scene_time_us=snapshot_scene.get(ref))
        events.append(_check_event(delivery, result, now_us, mode=mode))
        if result.state != "accepted":
            rejected += 1
            return
        accepted += 1
        if isinstance(result.message, ObservationMessage):
            accepted_refs[evidence_ref(result.message)] = result.message
            if mode == "snapshot":
                return
            admitted.append(result.message)
            assessment = assessor.assess(result.message, now_us)
            events.append(_assessment_event(
                result.message, assessment, delivery.evidence_ref, now_us,
                len(delivery.packet), spec=grid))

    transport = PacketTransport(
        S3_NODES, delay_us=delay_us, jitter_us=0, loss_probability=0.0,
        seed=0, event_callback=events.append, delivery_callback=on_delivery)

    def make_observer(observer_id, agent_frame, camera_name):
        return CameraObserver(
            observer_id,
            lambda f=agent_frame, c=camera_name: jpeg_to_camera_frame(f, c),
            camera_name, client, lift_box, keys[observer_id],
            session=S3_SESSION, ttl_us=S3_TTL_US, answer_fn=answer_query)

    observer_rsu2 = make_observer("rsu-2", rsu, pick.camera_name)
    observer_rsu1 = make_observer("rsu-1", other, covering_cam)
    cameras = {"rsu-2": observer_rsu2, "rsu-1": observer_rsu1}
    published_refs: dict[str, ObservationMessage] = {}

    def publish_message(message, *, scene_time_us: int | None = None) -> None:
        raw = encode(message)
        ref = evidence_ref(message)
        published_refs[ref] = message
        if scene_time_us is not None:
            snapshot_scene[ref] = scene_time_us
        events.append(make_event(
            t_replay_us=message.envelope.generation_time_us, type="observation_published",
            source_id=message.envelope.source_id, message_id=message.envelope.message_id,
            message_type="observation", evidence_ref=ref, bytes=len(raw),
            detail={"xyz": list(message.observation.xyz_m),
                    "cls": message.observation.cls.name,
                    "sensor_seq": message.envelope.sensor_seq}))
        transport.send(message.envelope.source_id, raw,
                       sim_us=message.envelope.generation_time_us,
                       message_id=message.envelope.message_id, message_type="observation")

    # Fast path: both RSUs SAM -> lift -> signed observations -> live checker.
    capture_us = rsu.nominal_time_us
    publish_us = capture_us
    last_avail = capture_us
    for obs in (observer_rsu2, observer_rsu1):
        for message in obs.observe():
            publish_message(message)
            publish_us = max(publish_us, message.envelope.generation_time_us)
        events.extend(obs.drain_events())
        if obs.last_result:
            last_avail = max(last_avail, obs.last_result.availability_time_us)
    drain_us = max(publish_us, last_avail) + delay_us
    transport.receive_until(drain_us)
    events.append(_map_snapshot_event(coverage_map, drain_us, map_observers))

    # Supervisory path: question -> Spark plan -> execute -> answers -> Spark judge.
    t_q = drain_us
    question = ("Is anyone standing in the asked region (a crossing area near the ego)? "
                "Use the observers that can see it.")
    region = {"center": [round(center[0], 2), round(center[1], 2)], "radius_m": 4.0}
    ego_xy = [round(float(ego.lidar_to_site[0, 3]), 2), round(float(ego.lidar_to_site[1, 3]), 2)]
    context = {
        "time_us": capture_us,
        "asked_region": region,
        "ego_xy": ego_xy,
        "admitted_objects": [
            {"source_id": m.envelope.source_id, "class": m.observation.cls.name,
             "xy": [round(m.observation.xyz_m[0], 2), round(m.observation.xyz_m[1], 2)]}
            for m in admitted],
        # Which offered observers are the same physical body (from trust anchors,
        # not from model text): rsu-1b is a second camera on rsu-1's unit.
        "physical_observers": {"rsu-2": "rsu-2", "rsu-1": "rsu-1", "rsu-1b": "rsu-1"},
    }
    def independent(refs):
        # The deterministic witness rule (authenticated lineage, physical
        # observer, common causes), applied to accepted evidence only.
        messages = [accepted_refs[r] for r in refs if r in accepted_refs]
        if not messages:
            return 0
        return max(store.independent_witnesses(m, messages) for m in messages)

    reasoner = Reasoner(client, independent=independent)
    queries, plan_result = reasoner.plan(question, capabilities, context, t_q)
    plan_event = model_call_event(plan_result, "reasoner")
    plan_event["detail"]["question"] = question
    events.append(plan_event)
    plan_avail = plan_result.availability_time_us
    for query in queries:
        event = query.to_event(plan_avail)
        event["detail"]["question"] = question
        events.append(event)

    answer_times: dict[str, int] = {}

    caps = {"rsu-2": cap_rsu2, "rsu-1": cap_rsu1, "rsu-1b": cap_rsu1b}

    def call_camera(observer_id, query):
        observer = cameras[observer_id]
        before = len(observer.published)
        try:
            answer, at_us = observer.on_query(query, caps[observer_id], plan_avail)
        except (MetaAPIError, CacheMiss, BudgetExceeded) as exc:
            raise ObserverUnavailable(type(exc).__name__) from exc
        for message in observer.published[before:]:
            publish_message(message, scene_time_us=query.time_us)
        answer_times[query.query_id] = at_us
        return answer

    def call_rsu1b(query):
        # Same physical observer as rsu-1, a camera that does not cover the region.
        answer_times[query.query_id] = plan_avail
        return answer_query(query, cap_rsu1b, [], [], plan_avail)

    raw_answers = execute(queries, capabilities, {
        "rsu-2": lambda q: call_camera("rsu-2", q),
        "rsu-1": lambda q: call_camera("rsu-1", q),
        "rsu-1b": call_rsu1b,
    })
    events.extend(observer_rsu2.drain_events())
    events.extend(observer_rsu1.drain_events())
    answers_us = plan_avail
    for answer in raw_answers:
        answers_us = max(answers_us, answer_times.get(answer.query_id, plan_avail) + delay_us)
    # Finish transmission and checks before the asker uses any evidence.
    transport.receive_until(answers_us)

    # The Reasoner sees only evidence its own checker accepted. An observer's
    # claim backed by rejected or undelivered packets is not evidence here.
    answers = []
    for answer in raw_answers:
        kept = tuple(r for r in answer.evidence_refs if r in accepted_refs)
        withheld = [r for r in answer.evidence_refs if r not in accepted_refs]
        if answer.result == "detected" and not kept:
            accepted_answer = Answer(answer.query_id, answer.observer_id, "unavailable",
                                     None, None, ())
        else:
            accepted_answer = Answer(answer.query_id, answer.observer_id, answer.result,
                                     answer.observed_time_us, answer.coverage_ref, kept)
        answers.append(accepted_answer)
        event = accepted_answer.to_event(answers_us)
        event["detail"]["observer_result"] = answer.result
        event["detail"]["withheld_refs"] = withheld
        events.append(event)

    evidence = {}
    for answer in answers:
        for ref in answer.evidence_refs:
            message = accepted_refs[ref]
            o = message.observation
            evidence[ref] = {
                "source_id": message.envelope.source_id, "object_id": o.object_id,
                "message_id": message.envelope.message_id, "class": o.cls.name,
                "label": (cameras.get(message.envelope.source_id).labels.get(message.label_id)
                          if message.label_id and cameras.get(message.envelope.source_id) else None),
                "xy": [round(o.xyz_m[0], 2), round(o.xyz_m[1], 2)],
                "capture_time_us": message.envelope.capture_time_us,
                "availability_time_us": message.envelope.generation_time_us,
            }
    verdicts, judge_result = reasoner.judge(question, queries, answers, evidence, answers_us)
    events.append(model_call_event(judge_result, "reasoner"))
    for verdict in verdicts:
        events.append(verdict.to_event(judge_result.availability_time_us))

    try:
        _write_s3_overlay(overlay, rsu, pick.camera_name, observer_rsu2.fast_path_detections, pedestrian)
    except OSError:
        pass

    events[:] = _write_events(output, events, overwrite=overwrite)
    by_type = Counter(event["type"] for event in events)
    model_cost = sum(e["detail"].get("cost_usd", 0) for e in events if e["type"] == "model_call")
    return {
        "events": str(output),
        "events_by_type": dict(by_type),
        "scenario": pick.scenario,
        "frame_id": pick.frame_id,
        "rsu_id": pick.rsu_id,
        "ego_id": pick.ego_id,
        "camera_name": pick.camera_name,
        "pedestrian_id": pick.pedestrian_id,
        "accepted": accepted,
        "rejected": rejected,
        "n_events": len(events),
        "model_call_count": by_type.get("model_call", 0),
        "model_cost_usd": model_cost,
        "question": question,
        "withheld_refs": sum(len(e["detail"].get("withheld_refs", [])) for e in events
                             if e["type"] == "answer"),
        "overlay": str(overlay),
        "grid": _grid_dict(grid),
        "checker": type(checker).__name__,
        "reasoner": type(reasoner).__name__,
        "queries": [{"observer_id": q.observer_id, "tier": q.tier, "concept": q.concept}
                    for q in queries],
        "answers": [a.result for a in answers],
        "verdicts": [{"subject": v.subject, "state": v.state, "refs": len(v.evidence_refs)}
                     for v in verdicts],
        "reasoner_rejections": list(reasoner.rejections),
        "rsu2_published": len(observer_rsu2.published),
        "rsu1_published": len(observer_rsu1.published),
        "rsu1_camera": covering_cam,
        "rsu1b_camera": gap_cam,
        "snapshot_accepted": sum(
            1 for e in events if e["type"] == "check"
            and e["detail"].get("mode") == "snapshot" and e["state"] == "accepted"),
    }
