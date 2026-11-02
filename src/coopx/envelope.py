"""CX2 signed observation envelopes and authenticated evidence lineage.

CX2 has no CX1/CARLA 99-byte cap. Metadata, the 59-byte CX1 object body and
key reference are signed. SHA-256 refs address complete signed packets.
Freshness, replay, permissions and spatial relevance belong to the checker.
"""
import base64
from dataclasses import dataclass, replace
from enum import IntEnum
import hashlib
import struct
from typing import Iterable

from coopx.domain import Observation, uint
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

def signing_bytes(message: ObservationMessage) -> bytes:
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

def sign(message: ObservationMessage, key: NodeKey) -> ObservationMessage:
    signature = key.sign(signing_bytes(message))
    return replace(message, envelope=replace(message.envelope, auth=replace(message.envelope.auth, signature=signature)))

def encode(message: ObservationMessage) -> bytes:
    if len(message.envelope.auth.signature) != 64:
        raise ValueError("signed message required")
    return signing_bytes(message) + message.envelope.auth.signature

def decode(raw: bytes) -> ObservationMessage:
    if not isinstance(raw, bytes) or not 64 <= len(raw) <= MAX_PACKET_BYTES:
        raise ValueError("invalid CX2 packet size/type")
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
    return message

def verify(message: ObservationMessage, anchors: TrustAnchors) -> bool:
    anchor = anchors.get(message.envelope.source_id)
    return bool(anchor and anchor.key_ref == message.envelope.auth.key_ref and NodeKey.verify(
        anchor.public_key, signing_bytes(message), message.envelope.auth.signature))

def evidence_ref(message: ObservationMessage) -> str:
    return hashlib.sha256(encode(message)).hexdigest()

def byte_accounting(message: ObservationMessage) -> dict[str, int | str]:
    """Application-wire bytes; transport reports link framing separately."""
    total = len(encode(message))
    key_ref = len(_pack_text(message.envelope.auth.key_ref))
    return {"message_type": message.envelope.message_type, "total": total,
            "signature": 64, "key_ref": key_ref, "object_body": radio_codec.RECORD.size,
            "envelope_and_lineage": total - 64 - key_ref - radio_codec.RECORD.size}

class EvidenceStore:
    """Content-addressed authenticated packets, with a snapshot of trust anchors."""
    def __init__(self, anchors: TrustAnchors):
        self.anchors = dict(anchors)
        self._messages: dict[str, ObservationMessage] = {}

    def add(self, message: ObservationMessage | bytes) -> str:
        message = decode(message) if isinstance(message, bytes) else decode(encode(message))
        if not verify(message, self.anchors):
            raise ValueError("unauthenticated evidence")
        ref = evidence_ref(message)
        self._messages[ref] = message
        return ref

    def get(self, ref: str) -> ObservationMessage | None:
        return self._messages.get(ref)

    def authenticated_roots(self, message: ObservationMessage) -> frozenset[str]:
        """Fail closed on missing, fabricated, mismatched or cyclic provenance."""
        def resolve(current, visiting):
            if len(visiting) >= 64 or not verify(current, self.anchors):
                return None
            ref, l, e = evidence_ref(current), current.lineage, current.envelope
            if ref in visiting:
                return None
            visiting = visiting | {ref}
            dependencies = []
            for parent_ref in l.parents:
                parent = self.get(parent_ref)
                roots = resolve(parent, visiting) if parent is not None else None
                if roots is None:
                    return None
                dependencies.extend(roots)
            if l.derivation == "direct":
                return set(dependencies) | {ref}
            original = self.get(l.root_ref)
            if original is None or original.lineage.derivation != "direct":
                return None
            oe, ol = original.envelope, original.lineage
            if (l.origin_observer, l.origin_sensor, l.origin_observation_id, l.origin_capture_time_us) != (oe.source_id, ol.origin_sensor, oe.message_id, oe.capture_time_us):
                return None
            if e.spatial_frame_id != oe.spatial_frame_id:
                return None
            if l.derivation == "relayed":
                if (radio_codec.encode(current.observation), current.kind, current.geometry, current.label_id) != (radio_codec.encode(original.observation), original.kind, original.geometry, original.label_id):
                    return None
            roots = resolve(original, visiting)
            if roots is None:
                return None
            # Forwarding cannot manufacture extra witnesses by attaching other
            # packets. Only the original measurement supplies relay support.
            return set(roots) if l.derivation == "relayed" else set(dependencies) | set(roots)
        return frozenset(resolve(message, set()) or ())

    def independent_witnesses(self, messages: Iterable[ObservationMessage]) -> int:
        """Count independent provenance groups after caller's relevance checks.

        Same-body roots, shared ancestor roots and configured common causes
        collapse. No physical association or unresolved origin gives no credit.
        This does not establish agreement, truth or relevance to a claim.
        """
        groups: list[set[str]] = []
        roots_to_messages = {}
        for message in messages:
            for ref in self.authenticated_roots(message):
                root = self.get(ref) or (message if ref == evidence_ref(message) else None)
                if root is not None:
                    roots_to_messages[ref] = root
        # Each authenticated direct root is a candidate witness. Dependency
        # edges tie a derived measurement to its ancestors; combining independent
        # measurements in a fusion/verdict must not erase their independence.
        for ref, root in roots_to_messages.items():
            anchor = self.anchors[root.envelope.source_id]
            if anchor.trust_class == TrustClass.D or anchor.physical_observer is None:
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
