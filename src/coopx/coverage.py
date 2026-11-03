"""Signed coverage, compact bitmaps, and source-preserving occupancy evidence.

The map consumes admitted packets, not raw network input. Supply an evidence
store (also bound by Assessor) for signature/permission checks and trust-aware
free space. Snapshot lists are disjoint; is_occupied() includes contested cells.
"""

from dataclasses import dataclass, replace
import math
import struct

from coopx.domain import PositionAnchor, uint
from coopx.envelope import (Auth, DERIVATIONS, Envelope, Lineage, MAX_PARENTS,
                           MAX_PACKET_BYTES, MESSAGE_TYPES, ObservationMessage,
                           Geometry, _pack_text, _Reader, decode, encode, evidence_ref)
from coopx.identity import TrustClass

COVERAGE_MAGIC = b"CX2C"
MAX_GRID_CELLS = 65_536
COMPATIBLE_US = 300_000
MODALITIES = ("lidar", "camera_rgb", "camera_rgbd")
RANGE_MODALITIES = frozenset(("lidar", "camera_rgbd"))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class GridSpec:
    origin_x: float
    origin_y: float
    cell_m: float
    nx: int
    ny: int

    def __post_init__(self):
        if (not all(_finite(v) for v in (self.origin_x, self.origin_y, self.cell_m))
                or self.cell_m <= 0 or type(self.nx) is not int or type(self.ny) is not int
                or not 1 <= self.nx <= 65_535 or not 1 <= self.ny <= 65_535
                or self.nx * self.ny > MAX_GRID_CELLS
                or not all(math.isfinite(v) for v in self.bounds_xy)):
            raise ValueError("invalid or oversized grid")

    @property
    def bounds_xy(self):
        return (self.origin_x, self.origin_y,
                self.origin_x + self.nx * self.cell_m, self.origin_y + self.ny * self.cell_m)

    def contains(self, cell):
        return (type(cell) is tuple and len(cell) == 2 and all(type(i) is int for i in cell)
                and 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny)

    def cell_at(self, x, y):
        if not _finite(x) or not _finite(y):
            raise ValueError("coordinates must be finite")
        return (math.floor((x - self.origin_x) / self.cell_m),
                math.floor((y - self.origin_y) / self.cell_m))


@dataclass(frozen=True, slots=True)
class CoverageContent:
    spec: GridSpec
    window_us: tuple[int, int]
    modality: str
    height_band_m: tuple[float, float]
    min_size_m: float
    observed: frozenset[tuple[int, int]]
    free: frozenset[tuple[int, int]]
    occupied: frozenset[tuple[int, int]]

    def __post_init__(self):
        if not isinstance(self.spec, GridSpec):
            raise ValueError("GridSpec required")
        if type(self.window_us) is not tuple or len(self.window_us) != 2:
            raise ValueError("window_us must be a start/end tuple")
        for value in self.window_us:
            uint(value, 64, "window_us")
        if self.window_us[0] > self.window_us[1] or self.modality not in MODALITIES:
            raise ValueError("invalid coverage window or modality")
        if (type(self.height_band_m) is not tuple or len(self.height_band_m) != 2
                or not all(_finite(v) for v in self.height_band_m)
                or self.height_band_m[0] >= self.height_band_m[1]
                or not _finite(self.min_size_m) or self.min_size_m <= 0):
            raise ValueError("invalid height band or detectable size")
        for cells in (self.observed, self.free, self.occupied):
            if type(cells) is not frozenset or any(not self.spec.contains(cell) for cell in cells):
                raise ValueError("coverage cells must be in-grid frozensets of integer pairs")
        if (not self.free <= self.observed or not self.occupied <= self.observed
                or self.free & self.occupied or (self.modality == "camera_rgb" and self.free)):
            raise ValueError("invalid coverage state: absence of RGB detections is not free space")


@dataclass(frozen=True, slots=True)
class CoverageMessage:
    envelope: Envelope
    content: CoverageContent
    lineage: Lineage

    def __post_init__(self):
        e, c, l = self.envelope, self.content, self.lineage
        if not isinstance(c, CoverageContent) or e.message_type != "coverage":
            raise ValueError("coverage envelope and content required")
        if c.window_us[1] != e.capture_time_us:
            raise ValueError("coverage window must end at envelope capture time")
        if l.derivation == "direct" and (
                l.origin_observer != e.source_id or l.origin_observation_id != e.message_id
                or l.origin_capture_time_us != e.capture_time_us or l.root_ref):
            raise ValueError("direct coverage origin must be this report")


def coverage_body(content):
    """Three fixed-size row-major bitmaps, bit (iy*nx+ix), least significant first."""
    spec = content.spec
    out = struct.pack("<dddHHQQBddd", spec.origin_x, spec.origin_y, spec.cell_m,
                      spec.nx, spec.ny, *content.window_us, MODALITIES.index(content.modality),
                      *content.height_band_m, content.min_size_m)
    size = (spec.nx * spec.ny + 7) // 8
    for cells in (content.observed, content.free, content.occupied):
        bitmap = bytearray(size)
        for ix, iy in cells:
            bit = iy * spec.nx + ix
            bitmap[bit // 8] |= 1 << (bit % 8)
        out += bitmap
    return out


def coverage_signing_bytes(message):
    e, l = message.envelope, message.lineage
    out = COVERAGE_MAGIC + struct.pack("<BBB", e.protocol_version,
                                      MESSAGE_TYPES.index(e.message_type), e.sensor_seq is not None)
    for value in (e.message_id, e.source_id, e.session_id, e.spatial_frame_id, e.auth.key_ref):
        out += _pack_text(value)
    out += struct.pack("<QQQ", e.capture_time_us, e.generation_time_us, e.valid_until_us)
    if e.sensor_seq is not None:
        out += struct.pack("<I", e.sensor_seq)
    out += struct.pack("<B", DERIVATIONS.index(l.derivation))
    for value in (l.origin_observer, l.origin_sensor, l.origin_observation_id):
        out += _pack_text(value)
    out += struct.pack("<QBB", l.origin_capture_time_us, len(l.parents), bool(l.root_ref))
    if l.root_ref:
        out += bytes.fromhex(l.root_ref)
    out += b"".join(bytes.fromhex(ref) for ref in l.parents)
    out += coverage_body(message.content)
    if len(out) + 64 > MAX_PACKET_BYTES:
        raise ValueError("CX2 coverage packet exceeds 65536 bytes")
    return out


def decode_coverage(raw):
    r = _Reader(raw)
    if r.take(4) != COVERAGE_MAGIC:
        raise ValueError("unsupported coverage magic")
    version, msgtype, flags = r.unpack("<BBB")
    if msgtype != MESSAGE_TYPES.index("coverage") or flags & ~1:
        raise ValueError("unsupported coverage type/flags")
    mid, source, session, frame, key_ref = (r.text() for _ in range(5))
    capture, generation, expiry = r.unpack("<QQQ")
    seq = r.unpack("<I")[0] if flags & 1 else None
    derivation, = r.unpack("<B")
    if derivation >= len(DERIVATIONS):
        raise ValueError("invalid coverage derivation")
    origin, sensor, oid = (r.text() for _ in range(3))
    origin_capture, count, has_root = r.unpack("<QBB")
    if count > MAX_PARENTS or has_root not in (0, 1):
        raise ValueError("invalid coverage lineage")
    root = r.take(32).hex() if has_root else ""
    parents = tuple(r.take(32).hex() for _ in range(count))
    x, y, cell_m, nx, ny, start, end, modality, lo, hi, min_size = r.unpack("<dddHHQQBddd")
    spec = GridSpec(x, y, cell_m, nx, ny)
    if modality >= len(MODALITIES):
        raise ValueError("invalid modality")
    sets = []
    for _ in range(3):
        bitmap = r.take((nx * ny + 7) // 8)
        sets.append(frozenset((bit % nx, bit // nx) for bit in range(nx * ny)
                              if bitmap[bit // 8] & (1 << (bit % 8))))
    signature = r.take(64)
    if r.offset != len(raw):
        raise ValueError("trailing coverage bytes")
    message = CoverageMessage(
        Envelope(version, "coverage", mid, source, session, frame, capture, generation, expiry,
                 Auth(key_ref, signature), seq),
        CoverageContent(spec, (start, end), MODALITIES[modality], (lo, hi), min_size, *sets),
        Lineage(origin, sensor, oid, origin_capture, DERIVATIONS[derivation], parents, root))
    if encode(message) != raw:
        raise ValueError("noncanonical coverage encoding")
    return message


def coverage_bounds(content):
    x0, y0, x1, y1 = content.spec.bounds_xy
    z0, z1 = content.height_band_m
    return ((x0, y0, z0), (x1, y1, z1))


def evidence_eligible(anchor):
    return bool(anchor and anchor.trust_class != TrustClass.D and (
        anchor.trust_class != TrustClass.B or (anchor.anchored and anchor.physical_observer is not None)))


def vertical_bounds(message):
    if isinstance(message, CoverageMessage):
        return message.content.height_band_m
    o = message.observation
    dz = (o.position_std_m or (0, 0, 0))[2]
    if message.geometry == Geometry.ORIENTED_BOX and o.anchor == PositionAnchor.CENTRE:
        dz += o.dimensions_m[2] / 2
    return o.xyz_m[2] - dz, o.xyz_m[2] + dz


@dataclass(frozen=True, slots=True)
class CellClaim:
    evidence_ref: str
    source_id: str
    state: str
    capture_time_us: int
    expires_us: int
    message: CoverageMessage | ObservationMessage
    range_origin: CoverageMessage | None = None


class CoverageMap:
    """Forward-time map; old source/sensor claims cannot overwrite newer cells.

    A plain map trusts its caller's admission policy. Binding an EvidenceStore
    additionally authenticates every input and prevents B-unanchored/D free
    claims from establishing displayed free space. Assessor binds its store.
    """
    def __init__(self, spec, *, ttl_us=1_000_000, spatial_frame_id="site_map", store=None):
        if not isinstance(spec, GridSpec) or type(ttl_us) is not int or ttl_us < 0:
            raise ValueError("GridSpec and nonnegative TTL required")
        self.spec, self.ttl_us, self.spatial_frame_id = spec, ttl_us, spatial_frame_id
        self.store = store
        self._cells = {}
        self._observations = {}
        self._now_us = 0

    def bind_store(self, store):
        if self.store is not None and self.store is not store:
            raise ValueError("map is already bound to a different evidence store")
        unique = {claim.evidence_ref: claim.message for claims in self._cells.values()
                  for claim in claims.values()}
        for message in unique.values():
            store.add(message)
        self.store = store
        origins = {ref: self._range_origin(message) for ref, message in unique.items()}
        for claims in self._cells.values():
            for slot, claim in list(claims.items()):
                claims[slot] = replace(claim, range_origin=origins[claim.evidence_ref])

    def _range_origin(self, message):
        if not isinstance(message, CoverageMessage):
            return None
        if self.store is not None and not self.store.authenticated_roots(message):
            return None
        if message.lineage.derivation == "direct":
            return message
        if message.lineage.derivation == "relayed" and self.store is not None:
            original = self.store.get(message.lineage.root_ref)
            return original if isinstance(original, CoverageMessage) else None
        # A tracked/fused free-space claim needs a defined region-derivation
        # proof. Merely citing unrelated coverage never establishes emptiness.
        return None

    def _prune(self, now_us):
        if type(now_us) is not int or now_us < self._now_us:
            raise ValueError("map clock must be nonnegative and nondecreasing")
        if now_us == self._now_us:
            return
        self._now_us = now_us
        for cell in list(self._cells):
            self._cells[cell] = {slot: value for slot, value in self._cells[cell].items()
                                 if value.expires_us >= now_us}
            if not self._cells[cell]:
                del self._cells[cell]

    def _prepare(self, message, now_us):
        message = decode(encode(message))
        e = message.envelope
        if (type(now_us) is not int or now_us < self._now_us or e.spatial_frame_id != self.spatial_frame_id
                or not e.capture_time_us <= now_us <= min(e.valid_until_us, e.capture_time_us + self.ttl_us)):
            raise ValueError("map input is future, expired, stale or in a different frame")
        if self.store is not None:
            anchor = self.store.anchors.get(e.source_id)
            bounds = coverage_bounds(message.content) if isinstance(message, CoverageMessage) else (message.observation.xyz_m,)
            if anchor is None or any(not anchor.permits(point) for point in bounds):
                raise ValueError("map input outside permitted region")
            self.store.add(message)
        return message

    def add(self, coverage_message, now_us):
        if isinstance(coverage_message, ObservationMessage):
            return self.add_observation(coverage_message, now_us)
        if not isinstance(coverage_message, CoverageMessage) or coverage_message.content.spec != self.spec:
            raise ValueError("coverage must use this map's grid")
        message = self._prepare(coverage_message, now_us)
        e, c = message.envelope, message.content
        self._prune(now_us)
        ref = evidence_ref(message)
        origin = self._range_origin(message)
        slot = (e.source_id, message.lineage.origin_sensor, "coverage")
        for cell in c.observed:
            state = "free" if cell in c.free else "occupied" if cell in c.occupied else "unknown"
            old = self._cells.get(cell, {}).get(slot)
            if old is None or e.capture_time_us > old.capture_time_us:
                self._cells.setdefault(cell, {})[slot] = CellClaim(
                    ref, e.source_id, state, e.capture_time_us,
                    min(e.valid_until_us, e.capture_time_us + self.ttl_us), message, origin)

    def add_observation(self, observation_message, now_us):
        # Local import keeps coverage wire primitives independent of assessment.
        from coopx.assess import footprint
        message = self._prepare(observation_message, now_us)
        e, o = message.envelope, message.observation
        old = self._observations.get(o.key)
        if old is not None and old[0] >= e.capture_time_us:
            return
        self._prune(now_us)
        cells, _ = footprint(message, self.spec)
        slot = (e.source_id, "object", o.object_id)
        if old is not None:
            for cell in old[1]:
                self._cells.get(cell, {}).pop(slot, None)
        self._observations[o.key] = (e.capture_time_us, cells)
        ref = evidence_ref(message)
        for cell in cells:
            self._cells.setdefault(cell, {})[slot] = CellClaim(
                ref, e.source_id, "occupied", e.capture_time_us,
                min(e.valid_until_us, e.capture_time_us + self.ttl_us), message)

    def records_at(self, cell, now_us):
        if not self.spec.contains(cell):
            raise ValueError("cell outside map")
        self._prune(now_us)
        return tuple(sorted(self._cells.get(cell, {}).values(), key=lambda c: (c.evidence_ref, c.state)))

    def _free_usable(self, claim):
        if self.store is None:
            return True  # Only for callers that already applied admission/trust policy.
        origin = claim.range_origin
        if origin is None:
            return False
        anchor = self.store.anchors.get(origin.envelope.source_id)
        return (evidence_eligible(anchor)
                and all(anchor.permits(point) for point in coverage_bounds(origin.content))
                and self._now_us <= min(origin.envelope.valid_until_us,
                                        origin.envelope.capture_time_us + self.ttl_us))

    def claims_at(self, cell, now_us):
        return sorted({(c.evidence_ref, c.state) for c in self.records_at(cell, now_us)
                       if c.state == "occupied" or (c.state == "free" and self._free_usable(c))})

    def _state(self, records):
        occupied = [c for c in records if c.state == "occupied"]
        free = [c for c in records if c.state == "free" and self._free_usable(c)]
        def compatible(o, f):
            lo, hi = vertical_bounds(o.message)
            band = f.message.content.height_band_m
            small = (isinstance(o.message, ObservationMessage)
                     and o.message.observation.dimensions_m is not None
                     and min(o.message.observation.dimensions_m[:2]) < f.message.content.min_size_m)
            return (not small and band[0] <= lo <= hi <= band[1]
                    and abs(o.capture_time_us - f.capture_time_us) <= COMPATIBLE_US)
        if any(compatible(o, f) for o in occupied for f in free):
            return "contested"
        return "occupied" if occupied else "free" if free else "unknown"

    def is_occupied(self, cell, now_us):
        return self._state(self.records_at(cell, now_us)) in ("occupied", "contested")

    def snapshot(self, now_us):
        self._prune(now_us)
        result = {"occupied": [], "free": [], "contested": []}
        for cell, records in sorted(self._cells.items()):
            state = self._state(records.values())
            if state != "unknown":
                result[state].append(list(cell))
        result["unknown_count"] = self.spec.nx * self.spec.ny - sum(map(len, result.values()))
        return result
