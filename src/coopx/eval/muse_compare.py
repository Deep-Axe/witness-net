"""Muse versus geometric consensus, only where the deterministic stack misses.

Real sliced LiDAR coverage, the real range callback, and a real plan, execute
and judge against the roadside observer. Ties and losses are reported as such.
"""

from __future__ import annotations

import json
from pathlib import Path

from coopx.attacks import attack_anchors, plan_attack
from coopx.assess import Assessor
from coopx.checker import Checker
from coopx.coverage import CoverageMap, CoverageMessage
from coopx.envelope import EvidenceStore, ObservationMessage, encode, evidence_ref
from coopx.eval.fusion import fuse
from coopx.identity import TrustAnchor, TrustClass
from coopx.pipeline import (DEMO_REGIONS, S3_DATASET, S3_REGION, S3_TTL_US,
                            _demo_grid, _demo_scene, _demo_target, _signed_coverage_window,
                            demo_node_key, range_contradicts_callback)

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


def judge_with_evidence(client, contradicts, question, queries, answers, evidence, now_us):
    """The same judge call the comparison uses. `contradicts` is the caller's callback."""
    from coopx.meta.reasoner import Reasoner
    reasoner = Reasoner(client, contradicts=contradicts)
    for query in queries:
        reasoner._plan_calls[query.query_id] = ("compare-plan", 0)
    return reasoner.judge(question, queries, answers, evidence, now_us)


def write_comparison(json_path: Path, markdown_path: Path, *, offline: bool = True,
                     budget_usd: float = 5.0, client=None, key_seed: int = 0) -> dict:
    rows = [_one(case, offline=offline, budget_usd=budget_usd, client=client, key_seed=key_seed)
            for case in cases()]
    payload = {"rows": rows}
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = _markdown(rows)
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(text, encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path), "rows": len(rows)}


def _one(case, *, offline, budget_usd, client, key_seed):
    from coopx.dataset.v2xreal import V2XReal
    from coopx.meta.client import BudgetExceeded, CacheMiss, MetaAPIError, MetaClient
    from coopx.meta.reasoner import Reasoner, execute
    from coopx.observers.camera import CameraObserver, jpeg_to_camera_frame, rsu_camera_capability
    from coopx.observers.camera_lift import lift_box
    from coopx.observers.lidar_real import lidar_coverage_sliced
    from coopx.query import Answer
    scene = _demo_scene(case["name"])
    dataset = V2XReal(S3_DATASET)
    ego = dataset.read(scene["scenario"], scene["ego_id"], case["frame"])
    rsu_agent, rsu_cam = scene["observers"][0][1], scene["observers"][0][2]
    rsu = dataset.read(scene["scenario"], rsu_agent, case["frame"])
    target = _demo_target(dataset, scene, case["frame"], case["gt_id"])
    capture = int(ego.nominal_time_us)
    region = DEMO_REGIONS[case["name"]]
    grid = _demo_grid([region["center"], target["xy"]])
    rsu_id = "rsu-1" if rsu_agent == "-1" else "rsu-2"
    keys = {rsu_id: demo_node_key(key_seed, rsu_id), "ego": demo_node_key(key_seed, "ego")}
    anchors = {name: TrustAnchor(keys[name].public_key, f"{name}-key", TrustClass.A,
                                 physical_observer=name, permitted_region=S3_REGION, anchored=True)
               for name in keys}
    store = EvidenceStore(anchors)
    coverage_map = CoverageMap(grid, ttl_us=S3_TTL_US, spatial_frame_id="site_map", store=store)
    checker = Checker(anchors, ttl_us=S3_TTL_US)
    try:
        slices = lidar_coverage_sliced(
            rsu, grid, capture_us=capture, window_us=(max(0, capture - 100_000), capture),
            max_range_m=80.0)
    except Exception as exc:
        return {"scene": case["name"], "attack": case["attack"], "geo_caught": False,
                "muse": {"error": f"lidar: {exc}"}, "relation": "LiDAR coverage failed"}
    for index, content in enumerate(slices):
        message = _signed_coverage_window(
            rsu_id, keys[rsu_id], content, sim_us=capture, ttl_us=S3_TTL_US,
            sensor_seq=1000 + index, message_id=f"{rsu_id}-cmp-{case['attack']}-{index}")
        result = checker.check(encode(message), capture)
        if result.state == "accepted":
            store.add(result.message)
            coverage_map.add(result.message, capture)
    if client is None:
        client = MetaClient(budget_usd=budget_usd, offline=offline)
    observer = CameraObserver(
        rsu_id, lambda: jpeg_to_camera_frame(rsu, rsu_cam), rsu_cam, client, lift_box,
        keys[rsu_id], session="s4b", ttl_us=S3_TTL_US)
    try:
        published = observer.observe()
    except (CacheMiss, MetaAPIError, BudgetExceeded, OSError) as exc:
        published = []
        sam_error = f"{type(exc).__name__}: {exc}"
    else:
        sam_error = None
    remote = []
    for message in published:
        result = checker.check(encode(message), message.envelope.generation_time_us)
        if result.state == "accepted" and isinstance(result.message, ObservationMessage):
            remote.append(result.message)
            store.add(result.message)
    attack_scene = {
        "target_xyz": (target["xy"][0], target["xy"][1], target["z"]),
        "ego_xyz": (float(ego.lidar_to_site[0, 3]), float(ego.lidar_to_site[1, 3]), 0.5),
        "unseen_xyz": (region["center"][0] + 8.0, region["center"][1] - 4.0, target["z"]),
    }
    when = capture + 100_000
    plan = plan_attack(case["attack"], seed=key_seed + 3, scene=attack_scene, now_us=when)
    _keys, extra = attack_anchors(plan, region=S3_REGION)
    anchors.update(extra)
    store.replace_anchors(anchors)
    checker = Checker(anchors, ttl_us=S3_TTL_US)
    attack_messages = []
    for packet in plan.packets:
        result = checker.check(packet.raw, when)
        if result.state == "accepted" and isinstance(result.message, ObservationMessage):
            attack_messages.append(result.message)
            store.add(result.message)
    assessor = Assessor(store, coverage_map)
    beliefs = fuse("geo", ego_observations=[], remote_admitted=remote + attack_messages,
                   store=store, coverage_map=coverage_map, assessor=assessor, now_us=when)
    attack_refs = {evidence_ref(m) for m in attack_messages}
    caught = bool(attack_messages) and not any(
        b.state != "contradicted" and set(b.refs) & attack_refs for b in beliefs)
    row = {"scene": case["name"], "attack": case["attack"], "geo_caught": caught,
           "muse": None, "relation": "deterministic catch; Muse not called"}
    if sam_error and not remote:
        row["relation"] = f"SAM unavailable ({sam_error}); geo used coverage and the attack only"
    if caught or not attack_messages:
        if not attack_messages:
            row["relation"] = "attack packet was not admitted; Muse not called"
        return row
    try:
        capability = rsu_camera_capability(rsu_id, rsu, rsu_cam)
    except Exception as exc:
        row["muse"] = {"error": f"capability: {exc}"}
        row["relation"] = "no footprint; Muse not called"
        return row
    clock = {"us": when + 1_000_000}
    reasoner = Reasoner(client, contradicts=range_contradicts_callback(store, coverage_map, clock))
    try:
        queries, plan_result = reasoner.plan(
            "Is the remote pedestrian report backed by this roadside camera?",
            [capability],
            {"time_us": capture, "asked_region": {"center": list(region["center"]),
                                                   "radius_m": region["radius_m"]},
             "physical_observers": {rsu_id: rsu_id}},
            clock["us"])
    except (BudgetExceeded, CacheMiss, MetaAPIError, OSError, ValueError) as exc:
        row["muse"] = {"error": f"{type(exc).__name__}: {exc}"}
        row["relation"] = f"Muse plan failed: {row['muse']['error']}"
        return row
    raw = execute(queries, [capability], {rsu_id: lambda query: observer.on_query(
        query, capability, plan_result.availability_time_us)[0]})
    answers = []
    for answer in raw:
        for message in observer.published:
            ref = evidence_ref(message)
            if store.get(ref) is None:
                checked = checker.check(encode(message), message.envelope.generation_time_us)
                if checked.state == "accepted" and isinstance(checked.message, ObservationMessage):
                    store.add(checked.message)
        kept = [ref for ref in answer.evidence_refs if store.get(ref) is not None]
        if answer.result == "detected" and not kept:
            answers.append(Answer(answer.query_id, answer.observer_id, "unavailable", None, None, ()))
        else:
            answers.append(Answer(answer.query_id, answer.observer_id, answer.result,
                                  answer.observed_time_us, answer.coverage_ref, tuple(kept)))
    evidence = {}
    for message in attack_messages:
        ref = evidence_ref(message)
        evidence[ref] = _summary(message, message.envelope.generation_time_us)
    for answer in answers:
        for ref in answer.evidence_refs:
            message = store.get(ref)
            if isinstance(message, ObservationMessage):
                evidence[ref] = _summary(message, message.envelope.generation_time_us)
    judge_time = max(plan_result.availability_time_us, clock["us"])
    clock["us"] = judge_time
    try:
        verdicts, result = reasoner.judge(
            "Is the remote pedestrian report backed by this roadside camera?",
            queries, answers, evidence, judge_time)
    except (BudgetExceeded, CacheMiss, MetaAPIError, OSError, ValueError) as exc:
        row["muse"] = {"error": f"{type(exc).__name__}: {exc}", "evidence_refs": sorted(evidence)}
        row["relation"] = f"Muse judge failed: {row['muse']['error']}"
        return row
    state = verdicts[0].state if verdicts else None
    row["muse"] = {"state": state, "reason": verdicts[0].reason if verdicts else "",
                   "cost_usd": result.cost_usd, "cache_hit": result.cache_hit,
                   "latency_ms": result.latency_ms,
                   "evidence_refs": sorted(evidence),
                   "answer_results": [a.result for a in answers]}
    if state == "contradicted" and not caught:
        row["relation"] = "Muse contradicted a report geo left standing"
    elif state in ("unverified", "contested", None):
        row["relation"] = "tie: Muse did not contradict it either" if state else "Muse returned no verdict"
    else:
        row["relation"] = f"loss: Muse said {state} without outrunning the evidence check"
    return row


def _summary(message, available_us):
    return {"source_id": message.envelope.source_id, "class": message.observation.cls.name,
            "xy": [message.observation.xyz_m[0], message.observation.xyz_m[1]],
            "capture_time_us": message.envelope.capture_time_us,
            "availability_time_us": available_us, "message_id": message.envelope.message_id}


def _markdown(rows):
    lines = [
        "# Muse compared with geometric consensus",
        "",
        "Generated by `coopx.eval.muse_compare` from sliced LiDAR coverage, geometric",
        "fusion, and a real plan, execute and judge. Muse runs only when geo does not",
        "contradict the attack. Ties and losses are stated as such.",
        "",
        "| scene | attack | geo caught | Muse | relation |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        muse = row.get("muse") or {}
        if row["geo_caught"]:
            muse_cell = "—"
        else:
            muse_cell = muse.get("state") or muse.get("error") or "—"
        lines.append(
            f"| {row['scene']} | {row['attack']} | "
            f"{'yes' if row['geo_caught'] else 'no'} | {muse_cell} | {row['relation']} |")
    lines.append("")
    return "\n".join(lines)
