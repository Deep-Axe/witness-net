"""CX2 signed observation envelopes and authenticated evidence lineage.

CX2 has no CX1/CARLA 99-byte cap. Metadata, the 59-byte CX1 object body and
key reference are signed. SHA-256 refs address complete signed packets.
Freshness, replay, permissions and spatial relevance belong to the checker.
"""
from __future__ import annotations

import base64
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field, replace
from enum import IntEnum
import hashlib
import math
import struct
from typing import Callable, Iterable, TYPE_CHECKING
from types import MappingProxyType

if TYPE_CHECKING:
    from coopx.coverage import CoverageMessage

from coopx.domain import ObjectClass, Observation, uint
from coopx.identity import NodeKey, TrustAnchors, TrustClass
from coopx import radio_codec

MAGIC = b"CX2\x00"
MAX_PACKET_BYTES = 65536
MAX_TEXT_BYTES = 512
MAX_PARENTS = 32
MESSAGE_TYPES = ("capability", "observation", "coverage", "query", "query_ack", "answer", "verdict", "label_table")
DERIVATIONS = ("direct", "relayed", "tracked", "fused", "verdict")

class Kind(IntEnum):
    PERCEIVED = 0
    SELF = 1

class Geometry(IntEnum):
    POINT_STD = 0
    ORIENTED_BOX = 1
    # Regions require a future region-body codec, not a disguised point.

def _text(value: str, name: str):
    if not isinstance(value, str) or not value or len(value.encode("utf8")) > MAX_TEXT_BYTES or "\x00" in value:
        raise ValueError(f"invalid {name}")

def _ref(value: str, *, empty=False):
    if empty and value == "":
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("evidence reference must be lowercase SHA-256 hex")

@dataclass(frozen=True, slots=True)
class Auth:
    key_ref: str
    signature: bytes = b""
    scheme: str = "Ed25519"

    def __post_init__(self):
        _text(self.key_ref, "key_ref")
        if self.scheme != "Ed25519" or not isinstance(self.signature, bytes) or len(self.signature) not in (0, 64):
            raise ValueError("invalid Ed25519 authentication")

@dataclass(frozen=True, slots=True)
class Envelope:
    protocol_version: int
    message_type: str
    message_id: str
    source_id: str
    session_id: str
    spatial_frame_id: str
    capture_time_us: int
    generation_time_us: int
    valid_until_us: int
    auth: Auth
    sensor_seq: int | None = None

    def __post_init__(self):
        if self.protocol_version != 2 or self.message_type not in MESSAGE_TYPES:
            raise ValueError("unsupported protocol or message type")
        for field in ("message_id", "source_id", "session_id", "spatial_frame_id"):
            _text(getattr(self, field), field)
        for field in ("capture_time_us", "generation_time_us", "valid_until_us"):
            uint(getattr(self, field), 64, field)
        if self.generation_time_us < self.capture_time_us or self.valid_until_us < self.generation_time_us:
            raise ValueError("capture <= generation <= expiry required")
        if self.sensor_seq is not None:
            uint(self.sensor_seq, 32, "sensor_seq")
        if not isinstance(self.auth, Auth):
            raise ValueError("Auth required")

@dataclass(frozen=True, slots=True)
class Lineage:
    origin_observer: str
    origin_sensor: str
    origin_observation_id: str
    origin_capture_time_us: int
    derivation: str = "direct"
    parents: tuple[str, ...] = ()
    root_ref: str = ""

    def __post_init__(self):
        for field in ("origin_observer", "origin_sensor", "origin_observation_id"):
            _text(getattr(self, field), field)
        uint(self.origin_capture_time_us, 64, "origin_capture_time_us")
        if self.derivation not in DERIVATIONS or not isinstance(self.parents, tuple) or len(self.parents) > MAX_PARENTS or len(set(self.parents)) != len(self.parents):
            raise ValueError("invalid derivation or parent references")
        _ref(self.root_ref, empty=True)
        for ref in self.parents:
            _ref(ref)

@dataclass(frozen=True, slots=True)
class ObservationMessage:
    envelope: Envelope
    observation: Observation
    lineage: Lineage
    kind: Kind = Kind.PERCEIVED
    geometry: Geometry = Geometry.POINT_STD
    label_id: int | None = None
    _wire: bytes | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self):
        e, o, l = self.envelope, self.observation, self.lineage
        if e.message_type != "observation" or e.source_id != o.source or e.capture_time_us != o.captured_sim_us:
            raise ValueError("observation/envelope identity or capture mismatch")
        if e.sensor_seq is not None and e.sensor_seq != o.sequence:
            raise ValueError("observation/envelope sequence mismatch")
        if not isinstance(self.kind, Kind) or not isinstance(self.geometry, Geometry):
            raise ValueError("unsupported kind or geometry")
        if self.geometry == Geometry.ORIENTED_BOX and (o.dimensions_m is None or o.yaw_deg is None):
            raise ValueError("oriented box requires dimensions and yaw")
        if self.label_id is not None:
            uint(self.label_id, 32, "label_id")
        if l.derivation == "direct" and (l.origin_observer != e.source_id or l.origin_observation_id != e.message_id or l.origin_capture_time_us != e.capture_time_us or l.root_ref):
            raise ValueError("direct origin must be this observation")

class _Reader:
    def __init__(self, raw):
        self.raw, self.offset = raw, 0

    def take(self, size):
        if self.offset + size > len(self.raw):
            raise ValueError("truncated CX2 packet")
        out = self.raw[self.offset:self.offset+size]
        self.offset += size
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def text(self):
        size, = self.unpack("<H")
        if size > MAX_TEXT_BYTES:
            raise ValueError("CX2 text too long")
        try:
            return self.take(size).decode("utf8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid CX2 UTF-8") from exc

def _pack_text(value):
    raw = value.encode("utf8")
    return struct.pack("<H", len(raw)) + raw

def signing_bytes(message: ObservationMessage | CoverageMessage) -> bytes:
    if message._wire is not None:
        return message._wire[:-64]
    if not isinstance(message, ObservationMessage):
        from coopx.coverage import CoverageMessage, coverage_signing_bytes
        if not isinstance(message, CoverageMessage):
            raise ValueError("unsupported signed message body")
        return coverage_signing_bytes(message)
    e, l = message.envelope, message.lineage
    flags = (e.sensor_seq is not None) | ((message.label_id is not None) << 1) | (bool(l.root_ref) << 2)
    out = MAGIC + struct.pack("<BBB", e.protocol_version, MESSAGE_TYPES.index(e.message_type), flags)
    for value in (e.message_id, e.source_id, e.session_id, e.spatial_frame_id, e.auth.key_ref):
        out += _pack_text(value)
    out += struct.pack("<QQQ", e.capture_time_us, e.generation_time_us, e.valid_until_us)
    if e.sensor_seq is not None:
        out += struct.pack("<I", e.sensor_seq)
    out += struct.pack("<BBB", int(message.kind), int(message.geometry), DERIVATIONS.index(l.derivation))
    if message.label_id is not None:
        out += struct.pack("<I", message.label_id)
    for value in (l.origin_observer, l.origin_sensor, l.origin_observation_id):
        out += _pack_text(value)
    out += struct.pack("<QB", l.origin_capture_time_us, len(l.parents))
    if l.root_ref:
        out += bytes.fromhex(l.root_ref)
    out += b"".join(bytes.fromhex(ref) for ref in l.parents)
    out += base64.b64decode(radio_codec.encode(message.observation)[len(radio_codec.PREFIX):])
    if len(out) + 64 > MAX_PACKET_BYTES:
        raise ValueError("CX2 packet too large")
    return out

def sign(message: ObservationMessage | CoverageMessage, key: NodeKey):
    signature = key.sign(signing_bytes(message))
    return replace(message, envelope=replace(message.envelope, auth=replace(message.envelope.auth, signature=signature)))

def encode(message: ObservationMessage | CoverageMessage) -> bytes:
    if len(message.envelope.auth.signature) != 64:
        raise ValueError("signed message required")
    if message._wire is not None:
        return message._wire
    return signing_bytes(message) + message.envelope.auth.signature

def decode(raw: bytes) -> ObservationMessage | CoverageMessage:
    if not isinstance(raw, bytes) or not 64 <= len(raw) <= MAX_PACKET_BYTES:
        raise ValueError("invalid CX2 packet size/type")
    if raw[:4] == b"CX2C":
        from coopx.coverage import decode_coverage
        return decode_coverage(raw)
    r = _Reader(raw)
    if r.take(4) != MAGIC:
        raise ValueError("unsupported CX2 magic")
    version, msgtype, flags = r.unpack("<BBB")
    if msgtype >= len(MESSAGE_TYPES) or flags & ~7:
        raise ValueError("unsupported CX2 type/flags")
    mid, source, session, frame, key_ref = (r.text() for _ in range(5))
    capture, generation, expiry = r.unpack("<QQQ")
    seq = r.unpack("<I")[0] if flags & 1 else None
    kind, geometry, derivation = r.unpack("<BBB")
    if derivation >= len(DERIVATIONS):
        raise ValueError("unsupported CX2 derivation")
    label = r.unpack("<I")[0] if flags & 2 else None
    origin, sensor, oid = (r.text() for _ in range(3))
    origin_capture, count = r.unpack("<QB")
    if count > MAX_PARENTS:
        raise ValueError("too many parents")
    root = r.take(32).hex() if flags & 4 else ""
    parents = tuple(r.take(32).hex() for _ in range(count))
    body = r.take(radio_codec.RECORD.size)
    signature = r.take(64)
    if r.offset != len(raw):
        raise ValueError("trailing CX2 bytes")
    observation = radio_codec.decode(radio_codec.PREFIX + base64.b64encode(body).decode("ascii"), source=source)
    message = ObservationMessage(
        Envelope(version, MESSAGE_TYPES[msgtype], mid, source, session, frame, capture, generation, expiry, Auth(key_ref, signature), seq),
        observation, Lineage(origin, sensor, oid, origin_capture, DERIVATIONS[derivation], parents, root), Kind(kind), Geometry(geometry), label)
    # CX1 ignores absent slots; canonical checking ensures verification covers
    # exactly the bytes received, even for fields ignored by the CX1 decoder.
    if encode(message) != raw:
        raise ValueError("noncanonical CX2 encoding")
    object.__setattr__(message, "_wire", raw)
    return message

def verify(message: ObservationMessage | CoverageMessage, anchors: TrustAnchors) -> bool:
    anchor = anchors.get(message.envelope.source_id)
    return bool(anchor and anchor.key_ref == message.envelope.auth.key_ref and NodeKey.verify(
        anchor.public_key, signing_bytes(message), message.envelope.auth.signature))

def evidence_ref(message: ObservationMessage | CoverageMessage) -> str:
    return hashlib.sha256(encode(message)).hexdigest()

def byte_accounting(message: ObservationMessage | CoverageMessage) -> dict[str, int | str]:
    """Application-wire bytes; transport reports link framing separately."""
    total = len(encode(message))
    key_ref = len(_pack_text(message.envelope.auth.key_ref))
    if not isinstance(message, ObservationMessage):
        from coopx.coverage import coverage_body
        body_size = len(coverage_body(message.content))
        return {"message_type": "coverage", "total": total, "signature": 64,
                "key_ref": key_ref, "coverage_body": body_size,
                "envelope_and_lineage": total - 64 - key_ref - body_size}
    return {"message_type": message.envelope.message_type, "total": total,
            "signature": 64, "key_ref": key_ref, "object_body": radio_codec.RECORD.size,
            "envelope_and_lineage": total - 64 - key_ref - radio_codec.RECORD.size}


def default_agrees(root_message: ObservationMessage, claim: ObservationMessage) -> bool:
    """Conservative point agreement, not proof of truth or object identity.

    Positions must share a spatial frame and anchor; no surface/centre
    conversion is inferred. UNKNOWN cannot confirm a known semantic class.
    Dynamic labels must match (their table must be shared by the caller).
    Gates are inclusive: 300ms and 1.5m, or 3m for vehicles. Only the root's
    own velocity widens its gate: a claimant cannot invent speed to borrow an
    unrelated root. Callers may supply a geometry-specific agreement policy.
    """
    root, target = root_message.observation, claim.observation
    if (root_message.envelope.spatial_frame_id != claim.envelope.spatial_frame_id
            or root.anchor != target.anchor or root.cls != target.cls
            or root_message.label_id != claim.label_id):
        return False
    elapsed_us = abs(root.captured_sim_us - target.captured_sim_us)
    if elapsed_us > 300_000:
        return False
    gate_m = 3.0 if root.cls == ObjectClass.VEHICLE else 1.5
    if root.velocity_mps is not None:
        gate_m += math.hypot(*root.velocity_mps) * elapsed_us / 1_000_000
    return math.dist(root.xyz_m, target.xyz_m) <= gate_m

class EvidenceStore:
    """Content-addressed packets with cached verification and resolved lineage.

    Trust anchors are a deeply immutable construction-time snapshot. Positive
    signature/root caches are sound under that snapshot; replace_anchors()
    invalidates both. Missing/cyclic roots are never cached. discard() also
    invalidates lineage caches so pruned originals fail closed. Observation
    candidates are indexed by capture time; archived evidence need not be
    scanned or deleted to expire it from assessment.
    """
    def __init__(self, anchors: TrustAnchors):
        self._messages: dict[str, ObservationMessage | CoverageMessage] = {}
        self._identity_refs: dict[int, str] = {}
        self._verified: set[str] = set()
        self._roots: dict[str, frozenset[str]] = {}
        self._observation_index: list[tuple[int, str]] = []
        self.anchor_epoch = 0
        self.proof_epoch = 0
        self.replace_anchors(anchors)

    @property
    def anchors(self):
        return self._anchors

    def replace_anchors(self, anchors: TrustAnchors) -> None:
        self._anchors = MappingProxyType({source: replace(
            anchor, public_key=bytes(anchor.public_key), common_causes=tuple(anchor.common_causes),
            permitted_region=None if anchor.permitted_region is None else tuple(anchor.permitted_region))
            for source, anchor in anchors.items()})
        self._verified.clear()
        self._roots.clear()
        self.anchor_epoch += 1
        self.proof_epoch += 1

    def reference(self, message) -> str:
        ref = self._identity_refs.get(id(message))
        return ref if ref is not None and self._messages.get(ref) is message else evidence_ref(message)

    def _verify(self, message, ref):
        if ref not in self._verified:
            if not verify(message, self.anchors):
                return False
            self._verified.add(ref)
        return True

    def add(self, message: ObservationMessage | CoverageMessage | bytes) -> str:
        if isinstance(message, bytes):
            raw = message
            ref = hashlib.sha256(raw).hexdigest()
        else:
            ref = self.reference(message)
            raw = None
        existing = self._messages.get(ref)
        canonical = existing
        if canonical is None:
            # Only our decoder sets _wire, after canonical validation. Decoded
            # bodies are deeply immutable; dataclasses.replace resets _wire.
            canonical = (message if raw is None and message._wire is not None
                         else decode(raw if raw is not None else encode(message)))
        if not self._verify(canonical, ref):
            raise ValueError("unauthenticated evidence")
        if existing is None:
            self._messages[ref] = canonical
            self._identity_refs[id(canonical)] = ref
            if isinstance(canonical, ObservationMessage):
                insort(self._observation_index, (canonical.envelope.capture_time_us, ref))
        return ref

    def get(self, ref: str) -> ObservationMessage | CoverageMessage | None:
        return self._messages.get(ref)

    def messages(self) -> tuple[ObservationMessage | CoverageMessage, ...]:
        """Snapshot of authenticated packets; admission/freshness is separate."""
        return tuple(self._messages.values())

    def recent_observations(self, start_us: int, end_us: int) -> tuple[ObservationMessage, ...]:
        """Inclusive capture-time range, O(log archive + candidates), no full scan."""
        if type(start_us) is not int or type(end_us) is not int or start_us > end_us:
            raise ValueError("ordered integer capture-time bounds required")
        lo = bisect_left(self._observation_index, (start_us, ""))
        hi = bisect_right(self._observation_index, (end_us, "~"))
        return tuple(self._messages[ref] for _, ref in self._observation_index[lo:hi])

    def discard(self, ref: str) -> None:
        message = self._messages.pop(ref, None)
        if message is not None:
            self._identity_refs.pop(id(message), None)
            self._verified.discard(ref)
            if isinstance(message, ObservationMessage):
                entry = (message.envelope.capture_time_us, ref)
                self._observation_index.pop(bisect_left(self._observation_index, entry))
            self._roots.clear()
            self.proof_epoch += 1

    def authenticated_roots(self, message: ObservationMessage | CoverageMessage) -> frozenset[str]:
        """Fail closed on missing, fabricated, mismatched or cyclic provenance."""
        def resolve(current, visiting):
            ref = self.reference(current)
            if ref in visiting or len(visiting) >= 64 or not self._verify(current, ref):
                return None
            if ref in self._roots:
                return self._roots[ref]
            l, e = current.lineage, current.envelope
            visiting = visiting | {ref}
            dependencies = []
            for parent_ref in l.parents:
                parent = self.get(parent_ref)
                roots = resolve(parent, visiting) if parent is not None else None
                if roots is None:
                    return None
                dependencies.extend(roots)
            if l.derivation == "direct":
                roots = frozenset(dependencies) | {ref}
                self._roots[ref] = roots
                return roots
            original = self.get(l.root_ref)
            if original is None or original.lineage.derivation != "direct":
                return None
            oe, ol = original.envelope, original.lineage
            if (l.origin_observer, l.origin_sensor, l.origin_observation_id, l.origin_capture_time_us) != (oe.source_id, ol.origin_sensor, oe.message_id, oe.capture_time_us):
                return None
            if e.spatial_frame_id != oe.spatial_frame_id:
                return None
            if l.derivation == "relayed":
                if type(current) is not type(original):
                    return None
                if isinstance(current, ObservationMessage):
                    equal = ((radio_codec.encode(current.observation), current.kind, current.geometry, current.label_id)
                             == (radio_codec.encode(original.observation), original.kind, original.geometry, original.label_id))
                else:
                    equal = current.content == original.content
                if not equal:
                    return None
            roots = resolve(original, visiting)
            if roots is None:
                return None
            # Forwarding cannot manufacture extra witnesses by attaching other
            # packets. Only the original measurement supplies relay support.
            result = frozenset(roots) if l.derivation == "relayed" else frozenset(dependencies) | roots
            self._roots[ref] = result
            return result
        return frozenset(resolve(message, set()) or ())

    def independent_witnesses(
        self, claim: ObservationMessage, messages: Iterable[ObservationMessage],
        agrees: Callable[[ObservationMessage, ObservationMessage], bool] = default_agrees,
    ) -> int:
        """Count authenticated provenance groups whose OWN content supports claim.

        Same-body roots, shared ancestor roots and configured common causes
        collapse. No physical association or unresolved origin gives no credit.
        A lineage reference alone is not support. Class-B roots require a
        receiver-established physical anchor; D never earns credit. This
        still does not establish truth, coverage, permission or freshness
        against the consumer's clock; the checker handles those separately.
        """
        groups: list[set[str]] = []
        roots_to_messages = {}
        for message in messages:
            if not isinstance(message, ObservationMessage):
                continue  # A coverage root cannot confirm a semantic class.
            for ref in self.authenticated_roots(message):
                root = self.get(ref) or (message if ref == self.reference(message) else None)
                if isinstance(root, ObservationMessage):
                    roots_to_messages[ref] = root
        # Each authenticated direct root is a candidate witness. Dependency
        # edges tie a derived measurement to its ancestors; combining independent
        # measurements in a fusion/verdict must not erase their independence.
        for ref, root in roots_to_messages.items():
            anchor = self.anchors[root.envelope.source_id]
            if (anchor.trust_class == TrustClass.D or anchor.physical_observer is None
                    or (anchor.trust_class == TrustClass.B and not anchor.anchored)
                    or not agrees(root, claim)):
                continue
            tokens = {"body:" + anchor.physical_observer, "root:" + ref}
            tokens.update("cause:" + cause for cause in anchor.common_causes)
            tokens.update("root:" + ancestor for ancestor in self.authenticated_roots(root))
            merged = tokens
            while True:
                matched = [g for g in groups if g & merged]
                if not matched:
                    break
                for g in matched:
                    merged |= g
                    groups.remove(g)
            groups.append(merged)
        return len(groups)
