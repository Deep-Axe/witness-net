"""Strict JSONL contract shared by the S1 checker, pipeline and UI."""

from dataclasses import dataclass
import json
import math

SCHEMA = "coopx.events/1"
EVENT_TYPES = frozenset({
    "observation_published", "packet_sent", "packet_dropped", "packet_delivered",
    "check", "observer_disconnected", "observer_reconnected",
    "coverage_published", "assessment", "map_snapshot",
})
CHECK_CODES = (
    "OK", "DECODE_ERROR", "UNKNOWN_SOURCE", "KEY_MISMATCH", "BAD_SIGNATURE",
    "STALE", "EXPIRED", "FUTURE_TIME", "DUPLICATE", "OUTSIDE_REPLAY_WINDOW",
    "OUT_OF_BOUNDS", "OUTSIDE_PERMITTED_REGION",
)
ASSESSMENT_CODES = (
    "CORROBORATED", "CONTRADICTED_BY_FREE_SPACE",
    "POSITION_CORROBORATED_CLASS_UNVERIFIED", "UNVERIFIED_SINGLE_SOURCE", "CONTESTED",
)
ASSESSMENT_STATES = {
    "CORROBORATED": "corroborated", "CONTRADICTED_BY_FREE_SPACE": "contradicted",
    "POSITION_CORROBORATED_CLASS_UNVERIFIED": "unverified",
    "UNVERIFIED_SINGLE_SOURCE": "unverified", "CONTESTED": "contested",
}
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


def _reference(ref):
    return (type(ref) is str and len(ref) == 64
            and all(char in "0123456789abcdef" for char in ref))


def _grid(detail):
    from coopx.coverage import GridSpec
    value = detail.get("grid", {"origin": [-10.0, -30.0], "cell_m": 1.0, "nx": 60, "ny": 60})
    if (type(value) is not dict or set(value) != {"origin", "cell_m", "nx", "ny"}
            or type(value["origin"]) is not list or len(value["origin"]) != 2):
        raise ValueError("grid must have origin:[x,y], cell_m,nx,ny")
    return GridSpec(*value["origin"], value["cell_m"], value["nx"], value["ny"])


def _cells(value, grid):
    if type(value) is not list or len(value) > grid.nx * grid.ny:
        raise ValueError("cells must be a bounded list")
    cells = set()
    for cell in value:
        if type(cell) is not list or not grid.contains(tuple(cell)):
            raise ValueError("cell must be an in-grid [int,int]")
        cells.add(tuple(cell))
    if len(cells) != len(value):
        raise ValueError("duplicate cells")
    return cells


def _s2_detail(d):
    detail, kind = d["detail"], d["type"]
    if kind == "coverage_published":
        if (d["message_type"] != "coverage" or d["message_id"] is None
                or not _reference(d["evidence_ref"]) or d["bytes"] is None or d["bytes"] <= 0
                or not {"grid", "modality", "window_us", "free", "occupied", "observed_count"} <= detail.keys()):
            raise ValueError("coverage event needs coverage identity, wire size and content")
        grid = _grid(detail)
        free, occupied = _cells(detail["free"], grid), _cells(detail["occupied"], grid)
        window = detail["window_us"]
        if (type(window) is not list or len(window) != 2
                or any(type(v) is not int or not 0 <= v < 2**64 for v in window)
                or window[0] > window[1]
                or detail["modality"] not in ("lidar", "camera_rgb", "camera_rgbd")
                or (detail["modality"] == "camera_rgb" and free) or free & occupied
                or type(detail["observed_count"]) is not int
                or not len(free | occupied) <= detail["observed_count"] <= grid.nx * grid.ny):
            raise ValueError("invalid coverage event content")
    elif kind == "assessment":
        if (d["message_type"] != "observation" or d["message_id"] is None
                or not _reference(d["evidence_ref"])
                or not {"cells", "supporting", "contradicting", "reason"} <= detail.keys()):
            raise ValueError("assessment needs subject and evidence detail")
        _cells(detail["cells"], _grid(detail))
        for name in ("supporting", "contradicting"):
            values = detail[name]
            if type(values) is not list or any(not _reference(ref) for ref in values) or len(set(values)) != len(values):
                raise ValueError("assessment evidence refs must be unique SHA-256 strings")
        if type(detail["reason"]) is not str or not detail["reason"].strip():
            raise ValueError("assessment needs a generated reason string")
        if d["state"] in ("corroborated", "contested") and not detail["supporting"]:
            raise ValueError("supporting evidence required")
        if d["state"] in ("contradicted", "contested") and not detail["contradicting"]:
            raise ValueError("contradicting evidence required")
        if d["check_codes"] == ["POSITION_CORROBORATED_CLASS_UNVERIFIED"] and not detail["supporting"]:
            raise ValueError("geometric support required")
    elif kind == "map_snapshot":
        if d["source_id"] != "ego" or not {"occupied", "free", "contested", "unknown_count"} <= detail.keys():
            raise ValueError("map snapshot must be an ego occupancy summary")
        grid = _grid(detail)
        occupied, free, contested = (_cells(detail[name], grid) for name in ("occupied", "free", "contested"))
        if (occupied & free or occupied & contested or free & contested
                or type(detail["unknown_count"]) is not int
                or detail["unknown_count"] != grid.nx * grid.ny - len(occupied | free | contested)):
            raise ValueError("map states must partition the grid")
        if "contested_claims" in detail:
            entries = detail["contested_claims"]
            if type(entries) is not list or len(entries) != len(contested):
                raise ValueError("contested_claims must list every contested cell once")
            seen = set()
            for entry in entries:
                if (type(entry) is not dict or set(entry) != {"cell", "claims"}
                        or type(entry["cell"]) is not list or tuple(entry["cell"]) not in contested
                        or tuple(entry["cell"]) in seen or type(entry["claims"]) is not list):
                    raise ValueError("contested_claims entry must name a contested cell")
                seen.add(tuple(entry["cell"]))
                kinds = set()
                for claim in entry["claims"]:
                    if (type(claim) is not list or len(claim) != 2 or not _reference(claim[0])
                            or claim[1] not in ("free", "occupied")):
                        raise ValueError("claim must be [evidence_ref, free|occupied]")
                    kinds.add(claim[1])
                if kinds != {"free", "occupied"}:
                    raise ValueError("a contested cell needs both a free and an occupied claim")


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
    allowed = ASSESSMENT_CODES if d["type"] == "assessment" else CHECK_CODES
    if type(codes) is not list or any(type(code) is not str or code not in allowed for code in codes):
        raise ValueError("invalid check_codes")
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate check codes")
    if d["type"] == "check":
        if not codes or ("OK" in codes and codes != ["OK"]):
            raise ValueError("check needs OK alone or one or more failure codes")
        expected = "accepted" if codes == ["OK"] else "rejected"
        if type(d["state"]) is not str or d["state"] != expected:
            raise ValueError("check state does not match codes")
    elif d["type"] == "assessment":
        if len(codes) != 1 or d["state"] != ASSESSMENT_STATES[codes[0]]:
            raise ValueError("assessment needs exactly one state-consistent assessment code")
    elif codes or d["state"] is not None:
        raise ValueError("non-check event needs empty codes and null state")
    if type(d["detail"]) is not dict:
        raise ValueError("detail must be a JSON object")
    try:
        _json_value(d["detail"], set())
    except RecursionError as exc:
        raise ValueError("detail nesting too deep") from exc
    _s2_detail(d)


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
