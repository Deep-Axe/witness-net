"""Synthetic 2-D LiDAR: ray-cast coverage and geometric object clusters.

Uses the shared `coopx.coverage` grid and coverage types (one schema).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, floor, hypot, pi, sin

from coopx.coverage import CoverageContent, GridSpec

S2_SCAN_DT_US = 100_000
S2_GRID = GridSpec(-10.0, -30.0, 1.0, 60, 60)


@dataclass(frozen=True, slots=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    kind: str = "obstacle"

    def __post_init__(self) -> None:
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("rect needs positive area")

    def intersects_cell(self, spec: GridSpec, cell: tuple[int, int]) -> bool:
        x0, y0, x1, y1 = spec.cell_bounds(cell)
        return self.xmin < x1 and self.xmax > x0 and self.ymin < y1 and self.ymax > y0


@dataclass(frozen=True, slots=True)
class MovingBody:
    """Axis-aligned box whose centre follows a constant-velocity track."""

    name: str
    x0: float
    y0: float
    vx: float
    vy: float
    length_m: float
    width_m: float
    kind: str = "object"

    def rect_at(self, t_s: float) -> Rect:
        cx = self.x0 + self.vx * t_s
        cy = self.y0 + self.vy * t_s
        hx, hy = self.length_m / 2.0, self.width_m / 2.0
        return Rect(cx - hx, cy - hy, cx + hx, cy + hy, self.kind)


@dataclass
class SynthWorld:
    """Static rectangles plus constant-velocity movers. Deterministic given t."""

    obstacles: tuple[Rect, ...]
    movers: tuple[MovingBody, ...] = ()

    def rects_at(self, t_s: float) -> tuple[Rect, ...]:
        return self.obstacles + tuple(m.rect_at(t_s) for m in self.movers)

    def occupied_cells(self, spec: GridSpec, t_s: float) -> frozenset[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for rect in self.rects_at(t_s):
            ix0 = floor((rect.xmin - spec.origin_x) / spec.cell_m)
            iy0 = floor((rect.ymin - spec.origin_y) / spec.cell_m)
            ix1 = floor((rect.xmax - spec.origin_x) / spec.cell_m) + 1
            iy1 = floor((rect.ymax - spec.origin_y) / spec.cell_m) + 1
            for ix in range(ix0, ix1):
                for iy in range(iy0, iy1):
                    cell = (ix, iy)
                    if spec.contains(cell) and rect.intersects_cell(spec, cell):
                        cells.add(cell)
        return frozenset(cells)


def cluster_occupied(cells: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    """4-connected components, sorted by (min_ix, min_iy) for determinism."""
    remaining = set(cells)
    clusters: list[frozenset[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        found = {start}
        while stack:
            ix, iy = stack.pop()
            for nx, ny in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    found.add((nx, ny))
                    stack.append((nx, ny))
        clusters.append(frozenset(found))
    clusters.sort(key=lambda c: min(c))
    return clusters


def cluster_geometry(cluster: frozenset[tuple[int, int]], spec: GridSpec) -> dict:
    xs, ys = zip(*(spec.cell_centre(cell) for cell in cluster))
    ix0, iy0 = min(c[0] for c in cluster), min(c[1] for c in cluster)
    ix1, iy1 = max(c[0] for c in cluster), max(c[1] for c in cluster)
    return {
        "xyz": (sum(xs) / len(xs), sum(ys) / len(ys), 0.5),
        "dimensions_m": (
            max(spec.cell_m, (ix1 - ix0 + 1) * spec.cell_m),
            max(spec.cell_m, (iy1 - iy0 + 1) * spec.cell_m),
            1.5,
        ),
        "cells": cluster,
    }


def default_s2_world() -> SynthWorld:
    """Parked truck, walking pedestrian, rolling car — site-frame metres."""
    truck = Rect(12.0, 8.0, 18.0, 11.0, "obstacle")
    pedestrian = MovingBody("pedestrian", 8.0, 1.0, 1.4, 0.0, 0.6, 0.6, "object")
    car = MovingBody("car", 12.0, 4.0, 6.0, 0.0, 4.0, 1.8, "object")
    return SynthWorld((truck,), (pedestrian, car))


class LidarObserver:
    """720-beam 2-D scan. Cells before the first return are free; the hit is occupied."""

    def __init__(self, spec: GridSpec = S2_GRID, *, pose: tuple[float, float] = (0.0, 0.0),
                 n_beams: int = 720, range_m: float = 40.0, height_band_m: tuple[float, float] = (0.2, 2.0),
                 min_size_m: float = 0.3, seed: int = 0, step_m: float | None = None,
                 dt_us: int = S2_SCAN_DT_US):
        if n_beams < 1 or range_m <= 0:
            raise ValueError("invalid lidar geometry")
        if type(dt_us) is not int or dt_us < 0:
            raise ValueError("dt_us must be a nonnegative int")
        self.spec = spec
        self.pose = pose
        self.n_beams = n_beams
        self.range_m = range_m
        self.height_band_m = height_band_m
        self.min_size_m = min_size_m
        self.step_m = spec.cell_m / 4.0 if step_m is None else step_m
        self.seed = seed
        self.dt_us = dt_us

    def scan(self, world: SynthWorld, *, t_us: int, window_us: tuple[int, int] | None = None) -> CoverageContent:
        t_s = t_us / 1_000_000
        blocked = world.occupied_cells(self.spec, t_s)
        free: set[tuple[int, int]] = set()
        occupied: set[tuple[int, int]] = set()
        observed: set[tuple[int, int]] = set()
        ox, oy = self.pose
        for i in range(self.n_beams):
            angle = 2.0 * pi * i / self.n_beams
            dx, dy = cos(angle), sin(angle)
            self._cast_beam(ox, oy, dx, dy, blocked, free, occupied, observed)
        # Coverage window ends at capture time (shared coverage invariant).
        window = window_us if window_us is not None else (max(0, t_us - self.dt_us), t_us)
        return CoverageContent(
            self.spec, window, "lidar", self.height_band_m, self.min_size_m,
            frozenset(observed), frozenset(free), frozenset(occupied))

    def objects(self, coverage: CoverageContent) -> list[dict]:
        clusters = cluster_occupied(coverage.occupied)
        return [cluster_geometry(cluster, coverage.spec) for cluster in clusters]

    def _cast_beam(self, ox: float, oy: float, dx: float, dy: float,
                   blocked: frozenset[tuple[int, int]],
                   free: set[tuple[int, int]], occupied: set[tuple[int, int]],
                   observed: set[tuple[int, int]]) -> None:
        dist = 0.0
        last: tuple[int, int] | None = None
        while dist <= self.range_m:
            x, y = ox + dx * dist, oy + dy * dist
            cell = self.spec.cell_at(x, y)
            if not self.spec.contains(cell):
                return
            if last == cell:
                dist += self.step_m
                continue
            last = cell
            observed.add(cell)
            if cell in blocked:
                occupied.add(cell)
                free.discard(cell)
                return
            if cell not in occupied:
                free.add(cell)
            dist += self.step_m

    def bearing_to(self, x: float, y: float) -> float:
        return atan2(y - self.pose[1], x - self.pose[0])

    def range_to(self, x: float, y: float) -> float:
        return hypot(x - self.pose[0], y - self.pose[1])
