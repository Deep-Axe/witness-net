"""Phase C metrics on belief streams. Muse rows are S4b; this is the fast path.

A hazard is any belief whose state is not `contradicted`. Match radii: 1.5 m
pedestrian, 3 m vehicle. Lead time subtracts the measured fast-path latency.
"""

from __future__ import annotations

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


def _matches(belief, target) -> bool:
    return _compatible(belief, target) and _dist(
        _xy(belief["xy"]), _xy(target["xy"])) <= match_radius_m(
            target.get("label") or target.get("cls") or "")


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


def safety_flips(before: list[dict], after: list[dict], *, radius_m: float = PED_MATCH_M) -> int:
    """Local beliefs present before and absent after. Witness must stay at 0."""
    flips = 0
    for belief in before:
        if not belief.get("local") or belief.get("state") == "contradicted":
            continue
        kept = False
        for other in after:
            if (other.get("local") and other.get("state") != "contradicted"
                    and _dist(_xy(belief["xy"]), _xy(other["xy"])) <= radius_m):
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


def independence(message_count: int, root_count: int) -> dict:
    return {"messages": message_count, "independent_roots": root_count}
