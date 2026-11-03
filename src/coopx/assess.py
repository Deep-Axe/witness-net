"""Deterministic assessment of admitted observations; no LLM or deletion policy.

Semantic corroboration needs independent class-compatible roots. Range occupancy
supports position only. Free-space contradiction labels remote claims and never
removes a local observation. A contested map cell remains occupied for caution.
"""

from dataclasses import dataclass
import math

from coopx.coverage import (COMPATIBLE_US, RANGE_MODALITIES, CoverageMessage,
                           coverage_bounds, evidence_eligible, vertical_bounds)
from coopx.domain import PositionAnchor
from coopx.envelope import (Geometry, ObservationMessage, default_agrees,
                           evidence_ref)


@dataclass(frozen=True, slots=True)
class Assessment:
    state: str
    codes: tuple[str, ...]
    cells: tuple[tuple[int, int], ...]
    supporting: tuple[str, ...]
    contradicting: tuple[str, ...]
    reason: str


def footprint(message, spec):
    """Return in-map footprint cells and a conservative full-footprint count.

    Points use +/- one supplied position std (not a calibrated confidence
    region). Oriented boxes use full dimensions only with a CENTRE anchor;
    a surface point is never silently turned into a hidden box centre.
    Out-of-map footprint stays unknown in the free-space fraction. For a box
    crossing the map edge its full AABB count conservatively bounds that
    denominator; no loop enumerates an unbounded out-of-map footprint.
    """
    o = message.observation
    x, y, _ = o.xyz_m
    sx, sy, _ = o.position_std_m or (0, 0, 0)
    corners = None
    axes = None
    if message.geometry == Geometry.ORIENTED_BOX and o.anchor == PositionAnchor.CENTRE:
        length, width, _ = o.dimensions_m
        yaw = math.radians(o.yaw_deg)
        # Exact quarter-turn angles must not add a cell through roundoff.
        forward = tuple(0.0 if abs(v) < 1e-15 else v for v in (math.cos(yaw), math.sin(yaw)))
        lateral = (-forward[1], forward[0])
        # Bounding local expansion is conservative for global x/y uncertainty.
        margin = max(sx, sy)
        hx, hy = length / 2 + margin, width / 2 + margin
        corners = [(x + a * hx * forward[0] + b * hy * lateral[0],
                    y + a * hx * forward[1] + b * hy * lateral[1])
                   for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        axes = ((1, 0), (0, 1), forward, lateral)
        lo_x, hi_x = min(p[0] for p in corners), max(p[0] for p in corners)
        lo_y, hi_y = min(p[1] for p in corners), max(p[1] for p in corners)
    else:
        lo_x, hi_x, lo_y, hi_y = x - sx, x + sx, y - sy, y + sy
    ix0, iy0 = spec.cell_at(lo_x, lo_y)
    # nextafter in grid coordinates avoids floating cancellation at boundaries.
    gx = (hi_x - spec.origin_x) / spec.cell_m
    gy = (hi_y - spec.origin_y) / spec.cell_m
    ix1 = math.floor(math.nextafter(gx, -math.inf)) if hi_x > lo_x else ix0
    iy1 = math.floor(math.nextafter(gy, -math.inf)) if hi_y > lo_y else iy0
    cells = []
    for ix in range(max(0, ix0), min(spec.nx - 1, ix1) + 1):
        for iy in range(max(0, iy0), min(spec.ny - 1, iy1) + 1):
            if corners is not None:
                px = spec.origin_x + ix * spec.cell_m
                py = spec.origin_y + iy * spec.cell_m
                square = ((px, py), (px + spec.cell_m, py),
                          (px + spec.cell_m, py + spec.cell_m), (px, py + spec.cell_m))
                separated = False
                for ax, ay in axes:
                    a = [cx * ax + cy * ay for cx, cy in corners]
                    b = [cx * ax + cy * ay for cx, cy in square]
                    if max(a) <= min(b) or max(b) <= min(a):
                        separated = True
                        break
                if separated:
                    continue
            cells.append((ix, iy))
    full_count = len(cells)
    if ix0 < 0 or iy0 < 0 or ix1 >= spec.nx or iy1 >= spec.ny:
        full_count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    return tuple(sorted(cells)), max(full_count, 1)


class Assessor:
    def __init__(self, store, coverage_map, *, agrees=default_agrees, local_source_id="ego"):
        self.store = store
        self.coverage_map = coverage_map
        self.agrees = agrees
        self.local_source_id = local_source_id
        coverage_map.bind_store(store)

    def _active(self, message, now_us):
        e = message.envelope
        return (e.spatial_frame_id == self.coverage_map.spatial_frame_id
                and e.capture_time_us <= now_us <= min(
                    e.valid_until_us, e.capture_time_us + self.coverage_map.ttl_us))

    def _eligible(self, message):
        anchor = self.store.anchors.get(message.envelope.source_id)
        points = coverage_bounds(message.content) if isinstance(message, CoverageMessage) else (message.observation.xyz_m,)
        return (evidence_eligible(anchor) and self.store.get(self.store.reference(message)) is not None
                and bool(self.store.authenticated_roots(message))
                and all(anchor.permits(point) for point in points))

    def _independent_range(self, coverage, subject):
        anchor = self.store.anchors.get(coverage.envelope.source_id)
        if not self._eligible(coverage) or anchor.physical_observer is None:
            return False
        roots = self.store.authenticated_roots(subject)
        if evidence_ref(coverage) in roots:
            return False  # This range report already supplied the subject geometry.
        for ref in roots:
            original = self.store.get(ref)
            if original is None:
                continue
            parent_anchor = self.store.anchors.get(original.envelope.source_id)
            if parent_anchor and (parent_anchor.physical_observer == anchor.physical_observer
                                  or set(parent_anchor.common_causes) & set(anchor.common_causes)):
                return False
        return True

    def _range_applies(self, coverage, subject):
        o = subject.observation
        if abs(coverage.envelope.capture_time_us - subject.envelope.capture_time_us) > COMPATIBLE_US:
            return False
        c = coverage.content
        if c.modality not in RANGE_MODALITIES:
            return False
        lo, hi = vertical_bounds(subject)
        if not c.height_band_m[0] <= lo <= hi <= c.height_band_m[1]:
            return False
        # Unknown physical size is not inferred from a semantic label. Point
        # results are sensor-qualified occupancy claims; size guarantees need
        # measured dimensions. Reject explicit below-resolution objects.
        return o.dimensions_m is None or min(o.dimensions_m[:2]) >= c.min_size_m

    def assess(self, message: ObservationMessage, now_us: int) -> Assessment:
        if not isinstance(message, ObservationMessage):
            raise ValueError("assessment requires an admitted observation")
        if type(now_us) is not int or now_us < 0:
            raise ValueError("now_us must be nonnegative")
        # This verifies/authenticates but is not a replacement for Checker.
        message = self.store.get(self.store.add(message))
        cells, denominator = footprint(message, self.coverage_map.spec)
        if not self._active(message, now_us):
            return Assessment("unverified", ("UNVERIFIED_SINGLE_SOURCE",), cells, (), (),
                              "UNVERIFIED_SINGLE_SOURCE: inactive observation")
        self.coverage_map.add_observation(message, now_us)
        candidates = [m for m in self.store.recent_observations(
            now_us - self.coverage_map.ttl_us, now_us) if self._active(m, now_us)]
        def relevant(root, claim):
            return self._active(root, now_us) and self._eligible(root) and self.agrees(root, claim)
        witnesses = self.store.independent_witnesses(message, candidates, relevant)
        semantic_refs = set()
        for candidate in candidates:
            for ref in self.store.authenticated_roots(candidate):
                root = self.store.get(ref)
                if isinstance(root, ObservationMessage) and relevant(root, message):
                    anchor = self.store.anchors[root.envelope.source_id]
                    if anchor.physical_observer is not None:
                        semantic_refs.add(ref)

        free_cells = set()
        free_refs = set()
        geometric_refs = set()
        for cell in cells:
            for record in self.coverage_map.records_at(cell, now_us):
                coverage = record.message
                if not isinstance(coverage, CoverageMessage) or not self._range_applies(coverage, message):
                    continue
                origin = record.range_origin
                if (origin is None or not self._active(origin, now_us)
                        or not self._range_applies(origin, message)):
                    continue
                if record.state == "free" and self._eligible(origin):
                    free_cells.add(cell)
                    free_refs.add(record.evidence_ref)
                elif record.state == "occupied" and self._independent_range(origin, message):
                    geometric_refs.add(record.evidence_ref)
        contradiction = bool(cells) and len(free_cells) * 2 >= denominator
        local = message.envelope.source_id == self.local_source_id
        if local:
            # Assessing remote input never removes/downgrades an ego detection.
            contradiction = False
            free_refs.clear()
        independent_support = witnesses >= 2 or bool(geometric_refs)
        if contradiction and independent_support:
            state, code = "contested", "CONTESTED"
        elif contradiction:
            state, code = "contradicted", "CONTRADICTED_BY_FREE_SPACE"
        elif witnesses >= 2:
            state, code = "corroborated", "CORROBORATED"
        elif geometric_refs:
            state, code = "unverified", "POSITION_CORROBORATED_CLASS_UNVERIFIED"
        else:
            state, code = "unverified", "UNVERIFIED_SINGLE_SOURCE"
        return Assessment(state, (code,), cells, tuple(sorted(semantic_refs | geometric_refs)),
                          tuple(sorted(free_refs)) if contradiction else (),
                          f"{code}: witnesses={witnesses}; free_cells={len(free_cells)}/{denominator}; local={int(local)}")
