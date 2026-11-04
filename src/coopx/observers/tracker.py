"""Per-camera world-coordinate tracking; no GT identities or future samples.

Association uses class, anchor, label and a conservative speed bound. Motion is
weighted least squares over the preceding two seconds, with measurement noise
and fit residuals both represented in velocity uncertainty. These are declared
operating bounds, not learned dataset constants; see PARAMETERS below.
"""

import math
from dataclasses import dataclass

import numpy as np

PARAMETERS = {
    "history_s": 2.0,  # four 2-Hz intervals; bounded constant-velocity fit
    "association_gap_s": 2.0,  # do not bridge more than four missed samples
    "measurement_std_floor_m": 0.1,  # LiDAR lift already floors XY at 0.2 m
    "min_velocity_std_mps": 0.1,  # avoid an overconfident noiseless fit
    # Rounded above p95 of 75 non-target tuning pairs (pedestrian 53 excluded).
    "max_center_step_per_second": 2.0,
    "max_appearance_distance": 0.7,
    "max_association_cost": 5.5,
    "speed_limits_mps": {1: 8.0, 2: 60.0, 3: 20.0, 0: 20.0},
}


@dataclass(frozen=True)
class Measurement:
    xyz: tuple
    std: tuple
    cls: int
    anchor: int
    label_id: int | None = None
    box_xyxy: tuple | None = None
    appearance: tuple | None = None

    def __post_init__(self):
        if (
            len(self.xyz) != 3
            or len(self.std) != 3
            or any(not math.isfinite(x) for x in (*self.xyz, *self.std))
            or any(x < 0 for x in self.std)
        ):
            raise ValueError(
                "measurement needs finite xyz and nonnegative standard deviation"
            )
        if self.box_xyxy is not None:
            box = self.box_xyxy
            if (
                len(box) != 4
                or not all(math.isfinite(v) for v in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError("invalid tracking box")


@dataclass(frozen=True)
class TrackEstimate:
    track_id: int
    velocity_mps: tuple | None
    velocity_std_mps: tuple | None
    samples: int


def _minimum_assignment(costs):
    """Rectangular Hungarian assignment; dummy columns allow unmatched rows.
    Deterministic, O(rows² * columns), with no scipy/runtime model dependency.
    """
    costs = np.asarray(costs, dtype=float)
    n, m = costs.shape
    if n > m or not np.isfinite(costs).all():
        raise ValueError("finite rectangular assignment required")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = {}
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


class CameraTracker:
    def __init__(self, *, first_id=200):
        if type(first_id) is not int or not 0 <= first_id < 2**32:
            raise ValueError("track ID must fit uint32")
        self._next = first_id
        self._tracks = {}

    @staticmethod
    def _fit(history):
        if len(history) < 2:
            return None, None
        t = np.array([h[0] for h in history], dtype=float) / 1e6
        t -= t[-1]
        p = np.array([h[2].xyz for h in history], dtype=float)
        noise = np.maximum(
            np.array([h[2].std for h in history], dtype=float),
            PARAMETERS["measurement_std_floor_m"],
        )
        w = 1 / noise**2
        means = (w * t[:, None]).sum(axis=0) / w.sum(axis=0)
        centered = t[:, None] - means
        xx = (w * centered**2).sum(axis=0)
        if np.any(xx <= 1e-12):
            return None, None
        pmean = (w * p).sum(axis=0) / w.sum(axis=0)
        v = (w * centered * (p - pmean)).sum(axis=0) / xx
        residual = p - pmean - centered * v
        # Known measurement variance plus lack-of-fit process variance.
        variance = (
            1 / xx + (w * residual**2).sum(axis=0) / max(1, len(history) - 2) / xx
        )
        std = np.maximum(np.sqrt(variance), PARAMETERS["min_velocity_std_mps"])
        return tuple(map(float, v)), tuple(map(float, std))

    def update(self, measurements, *, capture_us, available_us):
        """Batch assignment is one-to-one. Future captures and not-yet-available
        results cannot affect this estimate, including out-of-order completions.
        Returned estimates correspond to the input order.
        """
        if (
            type(capture_us) is not int
            or type(available_us) is not int
            or not 0 <= capture_us <= available_us < 2**64
        ):
            raise ValueError("invalid tracking times")
        measurements = list(measurements)
        eligible = {}
        association = {}
        for tid, history in self._tracks.items():
            known = [h for h in history if h[1] <= available_us and h[0] != capture_us]
            if not known:
                continue
            nearest = min(known, key=lambda h: (abs(capture_us - h[0]), h[0]))
            if abs(capture_us - nearest[0]) / 1e6 > PARAMETERS["association_gap_s"]:
                continue
            # Identity association can use measurements already available at
            # this decision. The signed velocity for an older capture uses
            # ONLY captures preceding that capture, never a newer position.
            eligible[tid] = [
                h
                for h in known
                if 0 < capture_us - h[0] <= PARAMETERS["history_s"] * 1e6
            ]
            association[tid] = nearest
        pairs = []
        for tid, (other_capture, _available, old) in association.items():
            dt = abs(capture_us - other_capture) / 1e6
            past = eligible[tid]
            velocity, _ = (
                self._fit(past) if other_capture < capture_us else (None, None)
            )
            predicted = tuple(
                x + dt * v for x, v in zip(old.xyz, velocity or (0.0, 0.0, 0.0))
            )
            for i, m in enumerate(measurements):
                if (m.cls, m.anchor, m.label_id) != (old.cls, old.anchor, old.label_id):
                    continue
                speed = PARAMETERS["speed_limits_mps"].get(int(m.cls), 20.0)
                noise = 2 * math.hypot(*[a + b for a, b in zip(m.std, old.std)])
                if math.dist(m.xyz, old.xyz) <= speed * dt + noise:
                    cost = math.dist(m.xyz, predicted) / max(
                        0.1, math.hypot(*[a + b for a, b in zip(m.std, old.std)])
                    )
                    if m.box_xyxy is not None and old.box_xyxy is not None:
                        a, b = m.box_xyxy, old.box_xyxy
                        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                            0, min(a[3], b[3]) - max(a[1], b[1])
                        )

                        def area(r):
                            return max(0, r[2] - r[0]) * max(0, r[3] - r[1])

                        union = area(a) + area(b) - intersection
                        iou = intersection / union if union else 0.0
                        center_distance = math.hypot(
                            (a[0] + a[2] - b[0] - b[2]) / 2,
                            (a[1] + a[3] - b[1] - b[3]) / 2,
                        )
                        diagonal = max(1.0, math.hypot(b[2] - b[0], b[3] - b[1]))
                        normalized_step = center_distance / diagonal
                        if (
                            normalized_step
                            > PARAMETERS["max_center_step_per_second"] * dt
                        ):
                            continue
                        cost += 2 * (1 - iou) + normalized_step
                    if m.appearance is not None and old.appearance is not None:
                        distance = math.sqrt(
                            sum(
                                (math.sqrt(a) - math.sqrt(b)) ** 2
                                for a, b in zip(m.appearance, old.appearance)
                            )
                            / 2
                        )
                        if distance > PARAMETERS["max_appearance_distance"]:
                            continue
                        cost += 2 * distance
                    if cost < PARAMETERS["max_association_cost"]:
                        pairs.append((cost, tid, i))
        # Dummies let a detection start a new track instead of being forced
        # onto an unrelated remaining person. Global assignment avoids greedy
        # swaps when two people compete for the same nearest prediction.
        ids = sorted(eligible)
        columns = {tid: index for index, tid in enumerate(ids)}
        cost_matrix = np.full(
            (len(measurements), len(ids) + len(measurements)),
            PARAMETERS["max_association_cost"],
        )
        if ids:
            cost_matrix[:, : len(ids)] = 1e6
        for cost, tid, i in pairs:
            cost_matrix[i, columns[tid]] = cost
        assigned = {
            i: ids[j]
            for i, j in _minimum_assignment(cost_matrix).items()
            if j < len(ids)
        }
        result = []
        for i, m in enumerate(measurements):
            tid = assigned.get(i)
            if tid is None:
                if self._next >= 2**32:
                    raise ValueError("track IDs exhausted")
                tid = self._next
                self._next += 1
            past = eligible.get(tid, [])
            history = [
                h for h in past if capture_us - h[0] <= PARAMETERS["history_s"] * 1e6
            ]
            history.append((capture_us, available_us, m))
            velocity, std = self._fit(history)
            speed = PARAMETERS["speed_limits_mps"].get(int(m.cls), 20.0)
            if velocity is not None and math.hypot(*velocity) > speed:
                # Association may be ambiguous; publish position without invented motion.
                velocity = std = None
            self._tracks.setdefault(tid, []).append((capture_us, available_us, m))
            self._tracks[tid].sort(key=lambda h: h[0])
            result.append(TrackEstimate(tid, velocity, std, len(history)))
        # Retain only the bounded history needed by later calls. A late result
        # can never rewrite an already published estimate.
        for tid in list(self._tracks):
            self._tracks[tid] = [
                h for h in self._tracks[tid] if capture_us - h[0] <= 4e6
            ]
            if not self._tracks[tid]:
                del self._tracks[tid]
        return result
