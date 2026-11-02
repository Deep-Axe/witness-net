"""Strict JSONL contract shared by the S1 checker, pipeline and UI."""

from dataclasses import dataclass
import json
import math

SCHEMA = "coopx.events/1"
EVENT_TYPES = frozenset({
    "observation_published", "packet_sent", "packet_dropped", "packet_delivered",
    "check", "observer_disconnected", "observer_reconnected",
})
CHECK_CODES = (
    "OK", "DECODE_ERROR", "UNKNOWN_SOURCE", "KEY_MISMATCH", "BAD_SIGNATURE",
    "STALE", "EXPIRED", "FUTURE_TIME", "DUPLICATE", "OUTSIDE_REPLAY_WINDOW",
    "OUT_OF_BOUNDS", "OUTSIDE_PERMITTED_REGION",
)
FIELDS = (
    "schema", "t_replay_us", "type", "source_id", "dest_id", "message_id",
    "message_type", "evidence_ref", "bytes", "check_codes", "state", "detail",
)


def _json_value(value, visiting: set[int]) -> None:
    """Reject non-JSON objects, nonfinite numbers, nonstring keys and cycles."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) not in (list, dict) or id(value) in visiting:
        raise ValueError("detail must contain finite, acyclic JSON values")
    visiting.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("detail object keys must be strings")
            values = value.values()
        else:
            values = value
        for child in values:
            _json_value(child, visiting)
    finally:
        visiting.remove(id(value))


def validate(d: dict) -> None:
    """Raise ValueError for malformed or inconsistent coopx.events/1 objects."""
    if type(d) is not dict or d.keys() != set(FIELDS):
        raise ValueError("event must have exactly the coopx.events/1 keys")
    if type(d["schema"]) is not str or d["schema"] != SCHEMA:
        raise ValueError("unsupported event schema")
    if type(d["t_replay_us"]) is not int or d["t_replay_us"] < 0:
        raise ValueError("t_replay_us must be a nonnegative integer")
    if type(d["type"]) is not str or d["type"] not in EVENT_TYPES:
        raise ValueError("unknown event type")
    if type(d["source_id"]) is not str or not d["source_id"].strip():
        raise ValueError("source_id must be a nonempty string")
    for name in ("dest_id", "message_id", "message_type"):
        if d[name] is not None and (type(d[name]) is not str or not d[name].strip()):
            raise ValueError(f"{name} must be a nonempty string or null")
    ref = d["evidence_ref"]
    if ref is not None and (type(ref) is not str or len(ref) != 64
                            or any(char not in "0123456789abcdef" for char in ref)):
        raise ValueError("evidence_ref must be lowercase SHA-256 hex or null")
    if d["bytes"] is not None and (type(d["bytes"]) is not int or d["bytes"] < 0):
        raise ValueError("bytes must be a nonnegative integer or null")
    codes = d["check_codes"]
    if type(codes) is not list or any(type(code) is not str or code not in CHECK_CODES for code in codes):
        raise ValueError("invalid check_codes")
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate check codes")
    if d["type"] == "check":
        if not codes or ("OK" in codes and codes != ["OK"]):
            raise ValueError("check needs OK alone or one or more failure codes")
        expected = "accepted" if codes == ["OK"] else "rejected"
        if type(d["state"]) is not str or d["state"] != expected:
            raise ValueError("check state does not match codes")
    elif codes or d["state"] is not None:
        raise ValueError("non-check event needs empty codes and null state")
    if type(d["detail"]) is not dict:
        raise ValueError("detail must be a JSON object")
    try:
        _json_value(d["detail"], set())
    except RecursionError as exc:
        raise ValueError("detail nesting too deep") from exc


@dataclass(frozen=True, slots=True)
class Event:
    schema: str
    t_replay_us: int
    type: str
    source_id: str
    dest_id: str | None
    message_id: str | None
    message_type: str | None
    evidence_ref: str | None
    bytes: int | None
    check_codes: list[str]
    state: str | None
    detail: dict

    def __post_init__(self) -> None:
        # Avoid asdict before validation: malformed cyclic detail must raise
        # ValueError, rather than recurse inside the dataclass copier.
        validate({name: getattr(self, name) for name in FIELDS})

    def to_json_line(self) -> str:
        return to_json_line({name: getattr(self, name) for name in FIELDS})


def to_json_line(event: Event | dict) -> str:
    """Serialize exactly one validated JSON object, including its final newline."""
    value = {name: getattr(event, name) for name in FIELDS} if isinstance(event, Event) else event
    validate(value)
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")) + "\n"
