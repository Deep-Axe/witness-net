"""Deterministic attack packets (contract 18). Signed with the attacker's keys.

A replay of captured bytes is the only kind that reuses someone else's signature.
Scene + kind + seed fix every byte. Free-space placement uses
`observers.lidar_real.lidar_coverage` when that module exists; otherwise the
synthetic S2 free-space point.
"""

from __future__ import annotations

from dataclasses import dataclass

from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, Lineage, ObservationMessage, decode,
                            encode, evidence_ref, sign)
from coopx.identity import TrustClass
from coopx.pipeline import S2_GHOST_FREE_XYZ, demo_node_key

ATTACK_KINDS = (
    "ghost_free", "ghost_unseen", "shift", "replay_old", "replay_refreshed",
    "class_flip", "sybil", "majority_veto", "remote_clear", "relay_dup",
    "loss_sweep", "disconnect",
)
SYBIL_N = 5
VETO_K = 3
RELAY_N = 5
TTL_US = 5_000_000
SESSION = "s4a"
FRAME = "site_map"


@dataclass(frozen=True)
class AttackNode:
    name: str
    trust: TrustClass
    anchored: bool


@dataclass(frozen=True)
class AttackPacket:
    source: str
    raw: bytes
    sim_us: int
    message_id: str
    message_type: str = "observation"


@dataclass(frozen=True)
class AttackPlan:
    kind: str
    seed: int
    packets: tuple[AttackPacket, ...]
    nodes: tuple[AttackNode, ...]
    loss: float = 0.0
    disconnect: tuple[str, int] | None = None


def loss_rate(seed: int) -> float:
    """0, 0.1, …, 0.5 from the seed. The sweep, not a tuned threshold."""
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    return (seed % 6) * 0.1


def plan_attack(kind: str, *, seed: int, scene: dict, now_us: int,
                captured: bytes | None = None) -> AttackPlan:
    if kind not in ATTACK_KINDS:
        raise ValueError(f"unknown attack {kind}")
    if type(seed) is not int or type(now_us) is not int or now_us < 0:
        raise ValueError("seed and now_us must be nonnegative ints")
    if kind == "loss_sweep":
        return AttackPlan(kind, seed, (), (), loss=loss_rate(seed))
    if kind == "disconnect":
        node = str(scene.get("disconnect_node") or "rsu-2")
        return AttackPlan(kind, seed, (), (), disconnect=(node, now_us + 1_000_000))
    if kind == "replay_old" and captured:
        return AttackPlan(kind, seed, (AttackPacket(
            str(scene.get("replay_source") or "rsu-2"), bytes(captured), now_us,
            str(scene.get("replay_message_id") or "replay-old")),), ())

    target = _xyz(scene.get("target_xyz"), (21.181, -179.669, -0.6))
    ego_local = _xyz(scene.get("ego_xyz"), (14.8, -171.72, 0.5))
    builders = {
        "ghost_free": lambda: _one("ghost", _free_xyz(scene), ObjectClass.PEDESTRIAN, now_us, seed),
        "ghost_unseen": lambda: _one("ghost", _xyz(scene.get("unseen_xyz"), (40.0, 15.0, 0.5)),
                                     ObjectClass.PEDESTRIAN, now_us, seed),
        "shift": lambda: _one("ghost", (target[0] + 5.0, target[1], target[2]),
                              ObjectClass.PEDESTRIAN, now_us, seed),
        "replay_old": lambda: _one("ghost", target, ObjectClass.PEDESTRIAN,
                                   max(0, now_us - 30_000_000), seed, at_us=now_us),
        "replay_refreshed": lambda: _one("ghost", target, ObjectClass.PEDESTRIAN, now_us, seed),
        "class_flip": lambda: _one("ghost", target, ObjectClass.VEHICLE, now_us, seed),
        "sybil": lambda: _many("sybil", SYBIL_N, target, ObjectClass.PEDESTRIAN, now_us, seed,
                               TrustClass.B, False),
        "majority_veto": lambda: _many("veto", VETO_K, target, ObjectClass.UNKNOWN, now_us, seed,
                                       TrustClass.B, False),
        "remote_clear": lambda: _one("clear", ego_local, ObjectClass.UNKNOWN, now_us, seed),
        "relay_dup": lambda: _relays(target, now_us, seed),
    }
    nodes, packets = builders[kind]()
    return AttackPlan(kind, seed, tuple(packets), tuple(nodes))


def apply_attack(transport, plan: AttackPlan) -> list:
    """Inject packets and an optional disconnect. Loss is a transport setting."""
    produced = []
    for packet in plan.packets:
        produced.extend(transport.send(
            packet.source, packet.raw, sim_us=packet.sim_us,
            message_id=packet.message_id, message_type=packet.message_type))
    if plan.disconnect is not None:
        node, when = plan.disconnect
        produced.extend(transport.disconnect(node, sim_us=when))
    return produced


def attack_anchors(plan: AttackPlan, *, region):
    from coopx.identity import TrustAnchor
    anchors = {}
    keys = {}
    for node in plan.nodes:
        key = demo_node_key(plan.seed, node.name)
        keys[node.name] = key
        anchors[node.name] = TrustAnchor(
            key.public_key, f"{node.name}-key", node.trust,
            physical_observer=node.name if node.anchored else None,
            permitted_region=region, anchored=node.anchored)
    return keys, anchors


def _free_xyz(scene: dict):
    given = scene.get("free_xyz")
    if given:
        return _xyz(given, S2_GHOST_FREE_XYZ)
    try:
        import coopx.observers.lidar_real  # noqa: F401  seam for A's real free space
    except ImportError:
        return S2_GHOST_FREE_XYZ
    return S2_GHOST_FREE_XYZ


def _xyz(value, default):
    if not value:
        return default
    nums = tuple(float(v) for v in value)
    if len(nums) == 2:
        return (nums[0], nums[1], 0.5)
    return nums[:3]


def _sign(source: str, seed: int, *, object_id: int, capture_us: int, publish_us: int,
          cls: ObjectClass, xyz, message_id: str, derivation="direct",
          origin=None, origin_id=None, origin_capture=None,
          parents=(), root_ref="") -> AttackPacket:
    key = demo_node_key(seed, source)
    observation = Observation(
        source, object_id, object_id, capture_us, cls, xyz, PositionAnchor.SURFACE,
        confidence=0.4)
    origin_name = source if origin is None else origin
    lineage = Lineage(
        origin_name, "attack", message_id if origin_id is None else origin_id,
        capture_us if origin_capture is None else origin_capture,
        derivation, tuple(parents), root_ref)
    envelope = Envelope(
        2, "observation", message_id, source, SESSION, FRAME,
        capture_us, publish_us, publish_us + TTL_US, Auth(f"{source}-key"), object_id)
    message = sign(ObservationMessage(envelope, observation, lineage), key)
    return AttackPacket(source, encode(message), publish_us, message_id)


def _one(name, xyz, cls, capture_us, seed, *, at_us=None):
    publish = capture_us if at_us is None else at_us
    node = AttackNode(name, TrustClass.A, True)
    packet = _sign(name, seed, object_id=400 + (seed % 1000), capture_us=capture_us,
                   publish_us=publish, cls=cls, xyz=xyz, message_id=f"{name}-{seed}")
    return (node,), (packet,)


def _many(prefix, count, xyz, cls, now_us, seed, trust, anchored):
    nodes, packets = [], []
    for index in range(count):
        name = f"{prefix}-{index}"
        nodes.append(AttackNode(name, trust, anchored))
        packets.append(_sign(
            name, seed, object_id=500 + index, capture_us=now_us, publish_us=now_us,
            cls=cls, xyz=xyz, message_id=f"{name}-{seed}"))
    return tuple(nodes), tuple(packets)


def _relays(xyz, now_us, seed):
    origin = _sign("origin", seed, object_id=700, capture_us=now_us, publish_us=now_us,
                   cls=ObjectClass.PEDESTRIAN, xyz=xyz, message_id=f"origin-{seed}")
    ref = evidence_ref(decode(origin.raw))
    nodes = [AttackNode("origin", TrustClass.A, True)]
    packets = [origin]
    for index in range(RELAY_N):
        name = f"relay-{index}"
        nodes.append(AttackNode(name, TrustClass.A, True))
        packets.append(_sign(
            name, seed, object_id=710 + index, capture_us=now_us, publish_us=now_us + index + 1,
            cls=ObjectClass.PEDESTRIAN, xyz=xyz, message_id=f"{name}-{seed}",
            derivation="relayed", origin="origin", origin_id=f"origin-{seed}",
            origin_capture=now_us, parents=(ref,), root_ref=ref))
    return tuple(nodes), tuple(packets)
