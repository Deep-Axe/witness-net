"""CX1 object packet: 59 binary bytes / 84 ASCII bytes, below CARLA's cap.

    <BBHIIQ3f3f3HhB3H (little endian)
    version, class, flags, local id, observation sequence, capture time µs,
    world xyz metres (float32), velocity m/s (float32), dimensions cm (uint16),
    yaw centidegrees (int16), confidence /255 (uint8), position std cm (uint16).

Source identity comes from an out-of-band station mapping. No simulator frame ID
or global actor identity is transmitted. Null optional values have zero slots;
flags distinguish missing from real zero. Flags bits 0..4 indicate velocity,
dimensions, yaw, confidence, std; bits 5..6 encode the position anchor.
"""

import base64
import binascii
import math
import struct

from coopx.domain import ObjectClass, Observation, PositionAnchor

PREFIX = "CX1:"
PAYLOAD_LIMIT = 99
RECORD = struct.Struct("<BBHIIQ3f3f3HhB3H")
KNOWN_FLAGS = 0x7F


def validate_payload(payload: str) -> int:
    if not isinstance(payload, str) or "\x00" in payload:
        raise ValueError("radio payload must be a NUL-free ASCII string")
    try:
        size = len(payload.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError("radio payload must be ASCII") from exc
    if not 0 < size <= PAYLOAD_LIMIT:
        raise ValueError(f"radio payload must contain 1..{PAYLOAD_LIMIT} bytes")
    return size


def _centimetres(values, *, strictly_positive=False):
    result = tuple(round(v * 100) for v in values)
    lower = 1 if strictly_positive else 0
    if any(not lower <= v <= 65535 for v in result):
        raise ValueError("dimension/std cannot be represented in centimetres")
    return result


def encode(observation: Observation) -> str:
    flags = int(observation.anchor) << 5
    optional = (observation.velocity_mps, observation.dimensions_m,
                observation.yaw_deg, observation.confidence, observation.position_std_m)
    for bit, value in enumerate(optional):
        if value is not None:
            flags |= 1 << bit
    dims = (0, 0, 0) if observation.dimensions_m is None else _centimetres(
        observation.dimensions_m, strictly_positive=True)
    std = (0, 0, 0) if observation.position_std_m is None else _centimetres(
        observation.position_std_m)
    velocity = observation.velocity_mps or (0, 0, 0)
    try:
        raw = RECORD.pack(
            1, int(observation.cls), flags, observation.object_id,
            observation.sequence, observation.captured_sim_us,
            *observation.xyz_m, *velocity, *dims,
            round((observation.yaw_deg or 0) * 100),
            round((observation.confidence or 0) * 255), *std,
        )
    except (OverflowError, struct.error) as exc:
        raise ValueError("observation exceeds radio field range") from exc
    # Reject float32 overflow even if a platform pack implementation emits inf.
    if not all(math.isfinite(v) for v in RECORD.unpack(raw)[6:12]):
        raise ValueError("position/velocity exceeds float32 range")
    payload = PREFIX + base64.b64encode(raw).decode("ascii")
    validate_payload(payload)
    return payload


def decode(payload: str, *, source: str) -> Observation:
    validate_payload(payload)
    if not payload.startswith(PREFIX):
        raise ValueError("unsupported radio prefix")
    try:
        raw = base64.b64decode(payload[len(PREFIX):], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 radio record") from exc
    if len(raw) != RECORD.size:
        raise ValueError("incorrect radio record size")
    data = RECORD.unpack(raw)
    version, cls, flags, object_id, sequence, captured_us = data[:6]
    if version != 1 or flags & ~KNOWN_FLAGS:
        raise ValueError("unsupported radio version or flags")
    try:
        anchor = PositionAnchor((flags >> 5) & 3)
        object_class = ObjectClass(cls)
    except ValueError as exc:
        raise ValueError("unknown class or position anchor") from exc
    return Observation(
        source=source, object_id=object_id, sequence=sequence,
        captured_sim_us=captured_us, cls=object_class,
        xyz_m=tuple(data[6:9]), anchor=anchor,
        velocity_mps=tuple(data[9:12]) if flags & 1 else None,
        dimensions_m=tuple(v / 100 for v in data[12:15]) if flags & 2 else None,
        yaw_deg=data[15] / 100 if flags & 4 else None,
        confidence=data[16] / 255 if flags & 8 else None,
        position_std_m=tuple(v / 100 for v in data[17:20]) if flags & 16 else None,
    )
