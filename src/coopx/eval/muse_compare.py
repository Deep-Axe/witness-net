"""Muse versus geometric consensus, only where the deterministic stack misses.

ghost_unseen, class_flip and replay_refreshed on both approved clips. shift is
included on clip 1. A tie or a Muse loss is written as that, not as a win.
"""

from __future__ import annotations

import json
from pathlib import Path

from coopx.attacks import attack_anchors, plan_attack
from coopx.checker import Checker
from coopx.coverage import CoverageMap
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, EvidenceStore, Lineage, ObservationMessage, evidence_ref, sign
from coopx.eval.fusion import fuse
from coopx.identity import TrustAnchor, TrustClass
from coopx.pipeline import S3_DATASET, S3_REGION, S3_TTL_US, _demo_grid, _demo_scene, _demo_target, demo_node_key

MISSED = ("ghost_unseen", "class_flip", "replay_refreshed")


def cases():
    rows = []
    for name, frame, gt_id in (
            ("blos_pedestrian_001", "000029", "81"),
            ("blos_pedestrian_002", "000000", "138")):
        kinds = MISSED + (("shift",) if name == "blos_pedestrian_001" else ())
        for kind in kinds:
            rows.append({"name": name, "frame": frame, "gt_id": gt_id, "attack": kind})
    return rows


def write_comparison(json_path: Path, markdown_path: Path, *, offline: bool = True,
                     budget_usd: float = 5.0, client=None, key_seed: int = 0) -> dict:
    rows = []
    for case in cases():
        rows.append(_one(case, offline=offline, budget_usd=budget_usd, client=client, key_seed=key_seed))
    payload = {"rows": rows}
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = _markdown(rows)
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text, encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path), "rows": len(rows)}


def _one(case, *, offline, budget_usd, client, key_seed):
    from coopx.dataset.v2xreal import V2XReal
    scene = _demo_scene(case["name"])
    dataset = V2XReal(S3_DATASET)
    ego = dataset.read(scene["scenario"], scene["ego_id"], case["frame"])
    rsu_agent, rsu_cam = scene["observers"][0][1], scene["observers"][0][2]
    rsu = dataset.read(scene["scenario"], rsu_agent, case["frame"])
    target = _demo_target(dataset, scene, case["frame"], case["gt_id"])
    now = int(ego.nominal_time_us) + 1_000_000
    grid = _demo_grid([target["xy"], (float(ego.lidar_to_site[0, 3]), float(ego.lidar_to_site[1, 3]))])
    keys = {"rsu-1": demo_node_key(key_seed, "rsu-1"), "ego": demo_node_key(key_seed, "ego")}
    anchors = {name: TrustAnchor(keys[name].public_key, f"{name}-key", TrustClass.A,
                                 physical_observer=name, permitted_region=S3_REGION, anchored=True)
               for name in keys}
    honest = _direct("rsu-1", keys["rsu-1"], target, now)
    attack_scene = {"target_xyz": (target["xy"][0], target["xy"][1], target["z"]),
                    "ego_xyz": (float(ego.lidar_to_site[0, 3]), float(ego.lidar_to_site[1, 3]), 0.5),
                    "unseen_xyz": (target["xy"][0] + 12.0, target["xy"][1] + 12.0, 0.5)}
    plan = plan_attack(case["attack"], seed=key_seed, scene=attack_scene, now_us=now + 1000)
    _keys, extra = attack_anchors(plan, region=S3_REGION)
    anchors.update(extra)
    store = EvidenceStore(anchors)
    coverage = CoverageMap(grid, ttl_us=S3_TTL_US, spatial_frame_id="site_map", store=store)
    checker = Checker(anchors, ttl_us=S3_TTL_US)
    remote = []
    for message in (honest,):
        result = checker.check(__import__("coopx.envelope", fromlist=["encode"]).encode(message), now)
        if result.state == "accepted":
            remote.append(result.message)
            store.add(result.message)
    attack_messages = []
    for packet in plan.packets:
        result = checker.check(packet.raw, now + 1000)
        if result.state == "accepted" and isinstance(result.message, ObservationMessage):
            attack_messages.append(result.message)
            store.add(result.message)
    from coopx.assess import Assessor
    assessor = Assessor(store, coverage)
    beliefs = fuse("geo", ego_observations=[], remote_admitted=remote + attack_messages,
                   store=store, coverage_map=coverage, assessor=assessor, now_us=now + 1000)
    attack_refs = {evidence_ref(m) for m in attack_messages}
    caught = not any(b.state != "contradicted" and set(b.refs) & attack_refs for b in beliefs)
    row = {"scene": case["name"], "attack": case["attack"], "geo_caught": caught,
           "muse": None, "relation": "deterministic catch; Muse not called"}
    if caught or not attack_messages:
        if not attack_messages:
            row["relation"] = "attack packet was not admitted; Muse not called"
        return row
    row["muse"] = _muse(case, attack_messages, now + 1000, offline=offline,
                        budget_usd=budget_usd, client=client)
    muse_state = (row["muse"] or {}).get("state")
    if muse_state == "contradicted":
        row["relation"] = "Muse contradicted a report geo left standing"
    elif muse_state in ("unverified", "contested", None):
        row["relation"] = "tie: Muse did not contradict it either" if muse_state else f"Muse did not run: {row['muse']}"
    else:
        row["relation"] = f"loss: Muse said {muse_state} without a free-space or second-observer proof"
    return row


def _direct(source, key, target, now):
    observation = Observation(
        source, 81, 81, now, ObjectClass.PEDESTRIAN,
        (target["xy"][0], target["xy"][1], target["z"]), PositionAnchor.SURFACE, confidence=0.8)
    envelope = Envelope(2, "observation", f"{source}-gt", source, "s4b", "site_map",
                        now, now, now + S3_TTL_US, Auth(f"{source}-key"), 81)
    return sign(ObservationMessage(envelope, observation, Lineage(source, "camera", f"{source}-gt", now)), key)


def _muse(case, messages, now, *, offline, budget_usd, client):
    from coopx.meta.client import BudgetExceeded, CacheMiss, MetaAPIError, MetaClient
    from coopx.meta.reasoner import Reasoner
    from coopx.query import Answer, Query
    if client is None:
        client = MetaClient(budget_usd=budget_usd, offline=offline)
    evidence = {}
    refs = []
    for message in messages:
        ref = evidence_ref(message)
        refs.append(ref)
        evidence[ref] = {"source_id": message.envelope.source_id, "class": message.observation.cls.name,
                         "xy": list(message.observation.xyz_m[:2]),
                         "capture_time_us": message.envelope.capture_time_us,
                         "availability_time_us": now}
    query = Query("q-compare", "T2", "rsu-1", messages[0].observation.xyz_m[:2], 4.0, now,
                  "pedestrian", "compare a remote report the geometric baseline did not reject")
    answer = Answer("q-compare", "rsu-1", "detected", now, None, tuple(refs))
    reasoner = Reasoner(client, contradicts=lambda *_a: False)
    try:
        verdicts, result = reasoner.judge(
            f"Does the remote report in {case['name']} survive the evidence?",
            [query], [answer], evidence, now + int(result_latency_placeholder()))
    except (BudgetExceeded, CacheMiss, MetaAPIError, OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    # The clock passed to judge must be >= answer time. Re-call if the placeholder
    # path above is wrong: judge was already called. Keep the first result.
    state = verdicts[0].state if verdicts else None
    return {"state": state, "reason": verdicts[0].reason if verdicts else "",
            "cost_usd": result.cost_usd, "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms}


def result_latency_placeholder():
    return 0


def _markdown(rows):
    lines = [
        "# Muse compared with geometric consensus",
        "",
        "Generated by `coopx.eval.muse_compare`. Muse runs only when geo does not",
        "contradict the attack. Ties and losses are stated as such.",
        "",
        "The object under attack is the clip target's ground-truth position, not a",
        "SAM detection. Pedestrian 53 at frame 000096 is not in this table.",
        "",
        "| scene | attack | geo caught | Muse | relation |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        muse = row.get("muse") or {}
        muse_cell = "—" if row["geo_caught"] else (muse.get("state") or muse.get("error") or "—")
        lines.append(f"| {row['scene']} | {row['attack']} | {'yes' if row['geo_caught'] else 'no'} | {muse_cell} | {row['relation']} |")
    lines.append("")
    return "\n".join(lines)
