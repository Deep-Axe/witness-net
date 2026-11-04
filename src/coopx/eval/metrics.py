"""Phase C metrics on belief streams. Muse rows are S4b; this is the fast path.

A hazard is any belief whose state is not `contradicted`. Match radii: 1.5 m
pedestrian, 3 m vehicle. Lead time subtracts the measured fast-path latency.
"""

from __future__ import annotations

import itertools
from math import hypot

PED_MATCH_M = 1.5
VEHICLE_MATCH_M = 3.0
_VEHICLE = {"car", "vehicle", "bus", "van", "truck", "VEHICLE"}
_PED = {"pedestrian", "person", "PEDESTRIAN"}


def match_radius_m(label: str) -> float:
    name = str(label or "")
    if name in _VEHICLE or name.lower() in _VEHICLE:
        return VEHICLE_MATCH_M
    return PED_MATCH_M


def _xy(value):
    return (float(value[0]), float(value[1]))


def _dist(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _hazard(belief) -> bool:
    return belief.get("state") != "contradicted"


def _compatible(belief, target) -> bool:
    belief_cls = str(belief.get("cls") or "").lower()
    target_cls = str(target.get("label") or target.get("cls") or "").lower()
    ped = {"pedestrian", "person"}
    vehicle = {"vehicle", "car", "bus", "van", "truck"}
    if target_cls in ped:
        return belief_cls in ped
    if target_cls in vehicle:
        return belief_cls in vehicle
    return True


class TemporalHistory:
    """Decision-local GT index for scoring, never an observer/model input.
    Indexing by identity avoids scanning every GT object for every belief pair.
    """

    def __init__(self, rows):
        self.rows = tuple(rows)
        self.by_id = {}
        for t, objects in self.rows:
            for obj in objects:
                self.by_id.setdefault(obj["id"], []).append((t, _xy(obj["xy"])))
        for samples in self.by_id.values():
            samples.sort()

    def __iter__(self):
        return iter(self.rows)


def _recent_positions(target, *, latency_tolerance_s=0.0, now_us=None, gt_history=()):
    if not isinstance(
        latency_tolerance_s, (int, float)
    ) or not 0 <= latency_tolerance_s < float("inf"):
        raise ValueError("latency_tolerance_s must be finite and nonnegative")
    positions = [_xy(target["xy"])]
    if latency_tolerance_s:
        if now_us is None:
            raise ValueError("tolerant scoring requires decision time")
        lower = now_us - latency_tolerance_s * 1e6
        if isinstance(gt_history, TemporalHistory):
            positions.extend(
                xy
                for t, xy in gt_history.by_id.get(target["id"], ())
                if lower <= t <= now_us
            )
        else:
            positions.extend(
                _xy(g["xy"])
                for t, objects in gt_history
                if lower <= t <= now_us
                for g in objects
                if g["id"] == target["id"]
            )
    return positions


def _match_distance(belief, target, **kwargs):
    if not _compatible(belief, target):
        return float("inf")
    point = _xy(belief["xy"])
    best = min(_dist(point, p) for p in _recent_positions(target, **kwargs))
    tau = kwargs.get("latency_tolerance_s", 0.0)
    if tau:
        now = kwargs["now_us"]
        lower = now - tau * 1e6
        history = kwargs.get("gt_history", ())
        if isinstance(history, TemporalHistory):
            samples = [
                (t, xy) for t, xy in history.by_id.get(target["id"], ()) if t <= now
            ]
        else:
            samples = sorted(
                (t, _xy(g["xy"]))
                for t, objects in history
                if t <= now
                for g in objects
                if g["id"] == target["id"]
            )
        for (ta, a), (tb, b) in itertools.pairwise(samples):
            # Linear interpolation within consecutive 10-Hz annotations only.
            # Never invent a path across a missing-annotation interval.
            if not 0 < tb - ta <= 100_000 or tb < lower:
                continue
            lo = max(0.0, (lower - ta) / (tb - ta))
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            fraction = (
                lo
                if not length2
                else max(
                    lo,
                    min(
                        1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
                    ),
                )
            )
            best = min(best, _dist(point, (a[0] + fraction * dx, a[1] + fraction * dy)))
    return best


def _matches(
    belief, target, *, latency_tolerance_s=0.0, now_us=None, gt_history=()
) -> bool:
    return _match_distance(
        belief,
        target,
        latency_tolerance_s=latency_tolerance_s,
        now_us=now_us,
        gt_history=gt_history,
    ) <= match_radius_m(target.get("label") or target.get("cls") or "")


def occluded_scores(ticks: list[dict]) -> dict:
    """Recall over occluded targets; precision and false hazards over all hazards."""
    occluded_pairs = hits = 0
    hazards = matched = false = 0
    for tick in ticks:
        targets = list(tick.get("targets") or [])
        beliefs = [b for b in tick.get("beliefs") or [] if _hazard(b)]
        occluded = [t for t in targets if t.get("visibility") == "occluded"]
        for target in occluded:
            occluded_pairs += 1
            if any(_matches(b, target) for b in beliefs):
                hits += 1
        for belief in beliefs:
            hazards += 1
            if any(_matches(belief, target) for target in targets):
                matched += 1
            else:
                false += 1
    return {
        "occluded_recall": None if occluded_pairs == 0 else hits / occluded_pairs,
        "precision": None if hazards == 0 else matched / hazards,
        "false_hazards": false,
        "hazards": hazards,
        "occluded_targets": occluded_pairs,
    }


EVAL_RADIUS_M = 50.0  # scored region around the ego; V2X-Real annotates a finite range


def score_frame(
    beliefs: list[dict],
    gt_objects: list[dict],
    *,
    center,
    radius_m: float = EVAL_RADIUS_M,
    latency_tolerance_s: float = 0.0,
    now_us=None,
    gt_history=(),
) -> dict:
    """Score one decision against ALL labelled objects, one-to-one.

    Hazards outside the region are counted, not scored. Inside it, beliefs are
    associated greedily by distance to class-compatible GT within the match
    radius; each GT object takes one true positive, further beliefs on an
    already-matched object are duplicates (one object seen by several
    observers), and the rest are false hazards.
    """
    if latency_tolerance_s and not isinstance(gt_history, TemporalHistory):
        gt_history = TemporalHistory(gt_history)
    center = _xy(center)
    hazards = [b for b in beliefs if _hazard(b)]
    inside = [b for b in hazards if _dist(_xy(b["xy"]), center) <= radius_m]
    kwargs = {
        "latency_tolerance_s": latency_tolerance_s,
        "now_us": now_us,
        "gt_history": gt_history,
    }
    # Include identities which existed inside the scoring window and since left
    # the annotations. This applies the SAME tolerance to false hazards.
    identities = {g["id"]: g for g in gt_objects}
    if latency_tolerance_s:
        _recent_positions({"id": "validation", "xy": center}, **kwargs)
        for t, objects in gt_history:
            if now_us - latency_tolerance_s * 1e6 <= t <= now_us:
                for g in objects:
                    identities.setdefault(g["id"], g)
    gts = [
        g
        for g in identities.values()
        if min(_dist(p, center) for p in _recent_positions(g, **kwargs))
        <= radius_m + VEHICLE_MATCH_M
    ]
    pairs = sorted(
        (_match_distance(b, g, **kwargs), bi, gi)
        for bi, b in enumerate(inside)
        for gi, g in enumerate(gts)
        if _matches(b, g, **kwargs)
    )
    used_b, used_g = set(), set()
    for _d, bi, gi in pairs:
        if bi in used_b or gi in used_g:
            continue
        used_b.add(bi)
        used_g.add(gi)
    duplicates = sum(
        1
        for bi, b in enumerate(inside)
        if bi not in used_b and any(_matches(b, gts[gi], **kwargs) for gi in used_g)
    )
    true_positives = len(used_b)
    return {
        "hazards": len(inside),
        "true_positives": true_positives,
        "duplicates": duplicates,
        "false_hazards": len(inside) - true_positives - duplicates,
        "out_of_region": len(hazards) - len(inside),
        "gt_in_region": len(gts),
    }


def aggregate_frames(frames: list[dict]) -> dict:
    keys = ("hazards", "true_positives", "duplicates", "false_hazards", "out_of_region")
    total = {k: sum(f[k] for f in frames) for k in keys}
    scored = total["true_positives"] + total["false_hazards"]
    total["precision"] = None if scored == 0 else total["true_positives"] / scored
    total["decisions"] = len(frames)
    return total


def warning_lead_us(first_match_us, first_ego_visible_us, latency_chain_us: int):
    """Lead time of a fast-path warning before ego can see the target.

    The warning exists only after the latency chain (SAM + lift + transport +
    checker). None when either instant is missing.
    """
    if first_match_us is None or first_ego_visible_us is None:
        return None
    if type(latency_chain_us) is not int or latency_chain_us < 0:
        raise ValueError("latency_chain_us must be a nonnegative int")
    return int(first_ego_visible_us) - (int(first_match_us) + latency_chain_us)


def safety_flips(
    before: list[dict], after: list[dict], *, radius_m: float = PED_MATCH_M
) -> int:
    """Local beliefs present before and absent after. Witness must stay at 0."""
    flips = 0
    for belief in before:
        if not belief.get("local") or belief.get("state") == "contradicted":
            continue
        kept = False
        for other in after:
            if (
                other.get("local")
                and other.get("state") != "contradicted"
                and _dist(_xy(belief["xy"]), _xy(other["xy"])) <= radius_m
            ):
                kept = True
                break
        if not kept:
            flips += 1
    return flips


def attack_catch(injected_refs, beliefs: list[dict]) -> bool:
    """Caught when no non-contradicted belief cites an injected packet."""
    live = set()
    for belief in beliefs:
        if belief.get("state") == "contradicted":
            continue
        live.update(belief.get("refs") or [])
    return not (set(injected_refs) & live)


def false_distrust(beliefs: list[dict], targets: list[dict]) -> int:
    """GT matches that fusion marked contradicted."""
    count = 0
    for belief in beliefs:
        if belief.get("state") != "contradicted":
            continue
        if any(_matches(belief, target) for target in targets):
            count += 1
    return count


def unknown_region_rate(unknown_count: int, cells: int) -> float | None:
    if type(cells) is not int or cells <= 0:
        return None
    return unknown_count / cells


def bytes_per_detection(nbytes: int, useful: int) -> float | None:
    if useful <= 0:
        return None
    return nbytes / useful


def independence(message_count: int, witness_count: int) -> dict:
    """Message count versus EvidenceStore.independent_witnesses.

    Unanchored identities add messages and zero witnesses. Relays of one
    anchored observation count once.
    """
    return {"messages": message_count, "independent_witnesses": witness_count}


def independent_witness_count(anchors, messages) -> int:
    """Max independent witnesses of any observation in `messages`."""
    from coopx.envelope import EvidenceStore, ObservationMessage

    store = EvidenceStore(anchors)
    kept = []
    for message in messages:
        if not isinstance(message, ObservationMessage):
            continue
        ref = store.add(message)
        kept.append(store.get(ref))
    if not kept:
        return 0
    return max(store.independent_witnesses(message, kept) for message in kept)


def capture_identity(belief, gt_history):
    """Evaluation-only identity audit. Nearest capture-time annotation inside
    the unchanged class radius; ambiguity is reported, never used for tracking.
    No identity is inferred when the capture frame is not available.
    """
    capture = belief.get("capture_time_us")
    position = belief.get("captured_position")
    if capture is None or position is None:
        return None, False
    objects = next((objects for t, objects in gt_history if t == capture), ())
    original = dict(belief, xy=position[:2])
    # This audit asks which PERSON a person detector saw. The legacy scorer
    # permits matching other/unknown classes; do not mislabel a nearby trash
    # can as a "different pedestrian" in the new audit.
    if str(belief.get("cls", "")).lower() in ("pedestrian", "person"):
        objects = [
            g
            for g in objects
            if str(g.get("label") or g.get("cls") or "").lower()
            in ("pedestrian", "person")
        ]
    matches = sorted(
        (_dist(position[:2], g["xy"]), str(g["id"]))
        for g in objects
        if _matches(original, g)
    )
    return (matches[0][1] if matches else None), len(matches) > 1


def image_identity(box, projected):
    """Which projected GT person overlaps a SAM box? Audit only, fixed gates
    shared across clips: >=.25 IoU and .10 separation from the second choice.
    Ambiguous overlaps remain flagged rather than counted as proven identity.
    """

    def area(rect):
        return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])

    matches = []
    for obj in projected:
        other = obj["box_xyxy"]
        overlap = area(
            (
                max(box[0], other[0]),
                max(box[1], other[1]),
                min(box[2], other[2]),
                min(box[3], other[3]),
            )
        )
        union = area(box) + area(other) - overlap
        iou = overlap / union if union else 0.0
        if iou >= 0.25:
            matches.append((iou, str(obj["id"])))
    matches.sort(reverse=True)
    if not matches:
        return None, False
    ambiguous = len(matches) > 1 and matches[0][0] - matches[1][0] < 0.10
    return matches[0][1], ambiguous
