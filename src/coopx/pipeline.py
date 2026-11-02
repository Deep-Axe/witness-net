"""S1 run: signed observations through PacketTransport to an ego checker.

Default checker is the deterministic `coopx.checker.Checker` built from the
run's trust anchors. `PassThroughChecker` remains available for transport-only
debugging (`checker=PassThroughChecker()`).
"""

from collections import Counter
import json
from pathlib import Path

from coopx.checker import Checker, CheckResult
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, Lineage, ObservationMessage, decode, encode, sign
from coopx.identity import NodeKey, TrustAnchor, TrustClass
from coopx.transport import PacketTransport, make_event, packet_hash

S1_NODES = ("pole1", "car1", "ego")
S1_DT_US = 100_000
S1_DURATION_US = 5_000_000
S1_TTL_US = 1_000_000
S1_FRAME = "site_map"
S1_SESSION = "s1"


class PassThroughChecker:
    """Accepts any decodable packet; transport debugging only. Same shape as Checker."""

    def check(self, raw: bytes, now_us: int) -> CheckResult:
        try:
            message = decode(raw)
        except (ValueError, TypeError):
            return CheckResult(("DECODE_ERROR",), "rejected", None)
        return CheckResult(("OK",), "accepted", message)


def _anchors_and_keys(nodes: tuple[str, ...] = S1_NODES):
    keys = {name: NodeKey.generate() for name in nodes}
    region = (-200.0, -200.0, -10.0, 200.0, 200.0, 20.0)
    anchors = {
        name: TrustAnchor(keys[name].public_key, f"{name}-key", TrustClass.A,
                          physical_observer=name, permitted_region=region)
        for name in nodes
    }
    return keys, anchors


def _signed_observation(source: str, key: NodeKey, *, object_id: int, frame: int,
                        sim_us: int, cls: ObjectClass, xyz: tuple[float, float, float],
                        velocity: tuple[float, float, float]) -> ObservationMessage:
    observation = Observation(
        source, object_id, frame, sim_us, cls, xyz, PositionAnchor.SURFACE,
        velocity_mps=velocity, confidence=0.9)
    message_id = f"{source}-{frame:04d}"
    envelope = Envelope(
        2, "observation", message_id, source, S1_SESSION, S1_FRAME,
        sim_us, sim_us, sim_us + S1_TTL_US, Auth(f"{source}-key"), frame)
    lineage = Lineage(source, "synthetic", message_id, sim_us, derivation="direct")
    return sign(ObservationMessage(envelope, observation, lineage), key)


def _pose(source: str, frame: int) -> tuple[ObjectClass, tuple[float, float, float], tuple[float, float, float]]:
    t_s = frame * S1_DT_US / 1_000_000
    if source == "pole1":
        # Pedestrian walking +x at 1.4 m/s.
        return ObjectClass.PEDESTRIAN, (8.0 + 1.4 * t_s, 1.0, 0.5), (1.4, 0.0, 0.0)
    # Car rolling +x at 6 m/s, offset in y.
    return ObjectClass.VEHICLE, (12.0 + 6.0 * t_s, 4.0, 0.8), (6.0, 0.0, 0.0)


def _check_event(delivery, result: CheckResult, now_us: int) -> dict:
    message = result.message
    envelope = None if message is None else message.envelope
    detail = {}
    if message is not None:
        detail = {
            "xyz": list(message.observation.xyz_m),
            "cls": message.observation.cls.name,
            "sensor_seq": envelope.sensor_seq,
        }
    return make_event(
        t_replay_us=now_us, type="check",
        source_id=delivery.source, dest_id=delivery.receiver,
        message_id=None if envelope is None else envelope.message_id,
        message_type=None if envelope is None else envelope.message_type,
        evidence_ref=delivery.evidence_ref or packet_hash(delivery.packet),
        bytes=len(delivery.packet), check_codes=list(result.codes),
        state=result.state, detail=detail)


def _handle_deliveries(deliveries, checker, events: list[dict]) -> tuple[int, int]:
    """Insert a check event immediately after each ego packet_delivered."""
    accepted = rejected = 0
    if not deliveries:
        return accepted, rejected
    head = events[:-len(deliveries)]
    tail = events[-len(deliveries):]
    rebuilt = []
    for delivery, event in zip(deliveries, tail):
        rebuilt.append(event)
        if delivery.receiver != "ego":
            continue
        result = checker.check(delivery.packet, delivery.available_sim_us)
        rebuilt.append(_check_event(delivery, result, delivery.available_sim_us))
        if result.state == "accepted":
            accepted += 1
        else:
            rejected += 1
    events[:] = head + rebuilt
    return accepted, rejected


def run_s1(output: Path, *, seed: int = 0, loss: float = 0.0, delay_us: int = 100_000,
           jitter_us: int = 50_000, disconnect_at_us: int | None = None,
           checker=None) -> dict:
    """Replay ~5 s at 10 Hz. `checker=` defaults to the real Checker over this run's anchors."""
    if disconnect_at_us is not None and (type(disconnect_at_us) is not int or disconnect_at_us < 0):
        raise ValueError("disconnect_at_us must be a nonnegative int")
    keys, anchors = _anchors_and_keys()
    if checker is None:
        checker = Checker(anchors, ttl_us=S1_TTL_US)
    events: list[dict] = []
    transport = PacketTransport(
        S1_NODES, delay_us=delay_us, jitter_us=jitter_us, loss_probability=loss,
        seed=seed, event_callback=events.append)
    accepted = rejected = 0
    disconnected = False
    publishers = (("pole1", 1), ("car1", 2))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for frame, sim_us in enumerate(range(0, S1_DURATION_US, S1_DT_US)):
            if (disconnect_at_us is not None and not disconnected and sim_us >= disconnect_at_us):
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
            got = transport.receive_until(sim_us)
            a, r = _handle_deliveries(got, checker, events)
            accepted += a
            rejected += r
        drain_us = (S1_DURATION_US - S1_DT_US) + delay_us + jitter_us
        got = transport.receive_until(drain_us)
        a, r = _handle_deliveries(got, checker, events)
        accepted += a
        rejected += r
        for event in events:
            handle.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")

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
