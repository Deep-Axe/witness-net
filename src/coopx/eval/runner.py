"""Eval runner: one scene, one fusion, optional attack. Muse stays off.

Observations are signed local stand-ins at the scene's GT positions unless the
caller passes packets or a live client. Agent A's `fuse` is used when it
imports; ego/union (and geo/witness, labelled) have stubs until then.
"""

from __future__ import annotations

import json
from pathlib import Path

from coopx.attacks import apply_attack, attack_anchors, plan_attack, scene_grid
from coopx.checker import Checker
from coopx.coverage import CoverageMap, CoverageMessage
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, EvidenceStore, Lineage, ObservationMessage,
                            evidence_ref, sign)
from coopx.eval.metrics import (attack_catch, false_distrust, independence,
                                independent_witness_count, occluded_scores, safety_flips)
from coopx.identity import TrustAnchor, TrustClass
from coopx.pipeline import demo_node_key
from coopx.transport import PacketTransport

FUSION_MODES = ("ego", "union", "geo", "witness")
REGION = (-1000.0, -1000.0, -50.0, 1000.0, 1000.0, 80.0)
# Prior measured fast path on this dataset: SAM ~2.7 s + 100 ms transport.
PRIOR_LATENCY_US = 2_800_000

FIXTURE = {
    "name": "frame-000096",
    "scenario": "2023-04-04-14-28-53_45_0",
    "frames": ("000096",),
    "ego_id": "2",
    "split": "tune",
    "target_id": "53",
    "target_label": "pedestrian",
    "target_xyz": (21.181, -179.669, -0.608),
    "ego_xyz": (14.796, -171.72, 0.5),
    "visibility": "outside_view",
    "observers": ("rsu-2", "rsu-1"),
    "note": "Pedestrian 53 is outside the ego cameras on this frame, not labelled occluded.",
}


def load_eval_scenes(name: str = "all"):
    scenes = _scenes_from_a() or [dict(FIXTURE)]
    if name != "all":
        scenes = [s for s in scenes if s.get("name") == name]
        if not scenes:
            raise ValueError(f"unknown scene {name}")
    return scenes


def _scenes_from_a():
    try:
        from coopx.dataset.scenes import load_scenes
    except ImportError:
        return None
    loaded = load_scenes()
    out = []
    for scene in loaded:
        if isinstance(scene, dict):
            out.append(scene)
        else:
            out.append({
                "name": scene.name,
                "scenario": scene.scenario,
                "frames": scene.frames,
                "ego_id": scene.ego_id,
                "targets": scene.targets,
                "visibility": scene.visibility,
                "hidden_interval": scene.hidden_interval,
            })
    return out or None


def run_eval(*, scene: str = "frame-000096", fusion: str = "witness",
             attack: str | None = None, seed: int = 0, offline: bool = True,
             budget_usd: float = 2.0, out_dir: Path = Path("data/results"),
             client=None, perception: str = "sam") -> dict:
    from coopx.meta.client import BudgetExceeded
    if fusion not in FUSION_MODES and fusion != "all":
        raise ValueError(f"unknown fusion {fusion}")
    modes = FUSION_MODES if fusion == "all" else (fusion,)
    summaries = []
    partial = False
    stopped = None
    try:
        for mode in modes:
            summaries.append(_one(scene, mode, attack, seed, out_dir,
                                  perception=perception, offline=offline,
                                  budget_usd=budget_usd, client=client))
    except BudgetExceeded as exc:
        partial = True
        stopped = str(exc)
    result = {"runs": summaries, "partial": partial, "stopped": stopped,
              "offline": offline, "budget_usd": budget_usd, "client": client is not None}
    return result


def _one(scene_name, fusion, attack, seed, out_dir: Path, *,
         perception="sam", offline=True, budget_usd=2.0, client=None) -> dict:
    scene = dict(load_eval_scenes(scene_name)[0])
    now_us = 9_600_000
    error = None
    if perception == "gt-oracle":
        honest = _honest_messages(scene, _keys_for(("rsu-2", "rsu-1", "ego"), seed), now_us)
    elif perception == "sam":
        honest, error = _perceive_sam(scene, seed, now_us, offline=offline,
                                      budget_usd=budget_usd, client=client)
        ego_hits = [m for m in honest if m.envelope.source_id == "ego"]
        if ego_hits:
            scene["ego_xyz"] = ego_hits[0].observation.xyz_m
    else:
        raise ValueError(f"unknown perception {perception}")
    if honest:
        now_us = max(message.envelope.generation_time_us for message in honest)
    plan = None if attack is None else plan_attack(
        attack, seed=seed, scene=scene, now_us=now_us + 1_000)
    nodes = ["rsu-2", "rsu-1", "ego"]
    if plan is not None:
        nodes.extend(node.name for node in plan.nodes if node.name not in nodes)
    keys = {name: demo_node_key(seed, name) for name in nodes}
    anchors = {
        name: TrustAnchor(keys[name].public_key, f"{name}-key", TrustClass.A,
                          physical_observer=name, permitted_region=REGION, anchored=True)
        for name in ("rsu-2", "rsu-1", "ego") if name in keys
    }
    if plan is not None:
        _keys, extra = attack_anchors(plan, region=REGION)
        anchors.update(extra)
    checker = Checker(anchors, ttl_us=5_000_000)
    events = []
    admitted = []

    def on_delivery(delivery):
        result = checker.check(delivery.packet, delivery.available_sim_us)
        from coopx.pipeline import _check_event
        events.append(_check_event(delivery, result, delivery.available_sim_us))
        if result.state == "accepted" and result.message is not None:
            admitted.append(result.message)

    loss = 0.0 if plan is None else plan.loss
    transport = PacketTransport(
        tuple(nodes), delay_us=100_000, jitter_us=0, loss_probability=loss,
        seed=seed, event_callback=events.append, delivery_callback=on_delivery)
    sent_at = {}
    for message in honest:
        raw = _encode(message)
        source = message.envelope.source_id
        sim = message.envelope.generation_time_us
        if source in sent_at and sim <= sent_at[source]:
            sim = sent_at[source] + 1
        sent_at[source] = sim
        events.append({
            "schema": "coopx.events/1", "t_replay_us": sim, "type": "observation_published",
            "source_id": source, "dest_id": None,
            "message_id": message.envelope.message_id, "message_type": "observation",
            "evidence_ref": evidence_ref(message), "bytes": len(raw), "check_codes": [],
            "state": None, "detail": {
                "local": source == "ego",
                "xyz": list(message.observation.xyz_m),
                "cls": message.observation.cls.name,
            },
        })
        transport.send(source, raw, sim_us=sim,
                       message_id=message.envelope.message_id, message_type="observation")
    if plan is not None:
        apply_attack(transport, plan)
    transport.receive_until(now_us + 2_000_000)
    ego_obs = [m for m in admitted if isinstance(m, ObservationMessage) and m.envelope.source_id == "ego"]
    remote = [m for m in admitted if isinstance(m, ObservationMessage) and m.envelope.source_id != "ego"]
    coverages = [m for m in admitted if isinstance(m, CoverageMessage)]
    spec = scene_grid((scene["target_xyz"], scene.get("ego_xyz") or (0.0, 0.0)))
    store = EvidenceStore(anchors)
    for message in (*ego_obs, *remote, *coverages):
        store.add(message)
    coverage_map = CoverageMap(spec, ttl_us=5_000_000, spatial_frame_id="site_map", store=store)
    map_now = now_us + 2_000_000
    for message in coverages:
        coverage_map.add(message, map_now)
    before = [_belief(m, True, "confirmed") for m in ego_obs]
    beliefs, stub = _fuse(fusion, ego_obs, remote, coverages, anchors, store, coverage_map, map_now)
    targets = _targets(scene, perception)
    scores = occluded_scores([{"beliefs": beliefs, "targets": targets}])
    injected = [evidence_ref(_decode(p.raw)) for p in plan.packets] if plan else []
    flips = safety_flips(before, beliefs)
    # Ego-only never reads remote packets, so it does not "catch" an attack.
    # remote_clear is caught when the ego's own detections are still there.
    if fusion == "ego" or plan is None or plan.kind in ("loss_sweep", "disconnect"):
        caught = None
    elif plan.kind == "remote_clear":
        caught = bool(before) and flips == 0
    elif plan.kind == "majority_veto":
        caught = _protected(beliefs, scene["target_xyz"], scene.get("target_label", "pedestrian"))
    else:
        caught = attack_catch(injected, beliefs)
    attack_obs = []
    packet_count = 0
    if plan is not None:
        packet_count = len(plan.packets)
        attack_obs = [m for m in (_decode(p.raw) for p in plan.packets)
                      if isinstance(m, ObservationMessage)]
    witnesses = independent_witness_count(anchors, attack_obs or honest)
    metrics = {
        "scores": scores,
        "safety_flips": flips,
        "false_distrust": false_distrust(beliefs, targets),
        "attack_caught": caught,
        "independence": independence(len(honest) + packet_count, witnesses),
        "latency_chain_us": PRIOR_LATENCY_US,
        "lead_us": None,
        "visibility": scene.get("visibility"),
    }
    summary = {
        "record": "summary", "scene": scene["name"], "fusion": fusion,
        "attack": attack, "seed": seed, "fusion_stub": stub,
        "perception": perception, "partial": bool(error), "error": error,
        "metrics": metrics, "split": scene.get("split", "tune"),
    }
    path = _out_path(out_dir, scene["name"], fusion, attack, perception)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        handle.write(json.dumps({
            "record": "tick", "t_us": now_us, "beliefs": beliefs, "fusion": fusion,
        }, separators=(",", ":")) + "\n")
        handle.write(json.dumps(summary, separators=(",", ":")) + "\n")
    summary["path"] = str(path)
    return summary


def _honest_messages(scene, keys, now_us):
    target = tuple(scene["target_xyz"])
    messages = []
    for source in scene.get("observers") or ("rsu-2",):
        messages.append(_direct(source, keys[source], 200 + len(messages), target, now_us,
                                ObjectClass.PEDESTRIAN, f"{source}-gt"))
    messages.append(_direct("ego", keys["ego"], 1, tuple(scene["ego_xyz"]), now_us,
                            ObjectClass.VEHICLE, "ego-local"))
    return messages


def _direct(source, key, object_id, xyz, now_us, cls, message_id):
    observation = Observation(source, object_id, object_id, now_us, cls, xyz,
                              PositionAnchor.SURFACE, confidence=0.8)
    envelope = Envelope(
        2, "observation", message_id, source, "s4a", "site_map",
        now_us, now_us, now_us + 5_000_000, Auth(f"{source}-key"), object_id)
    lineage = Lineage(source, "camera", message_id, now_us, "direct")
    return sign(ObservationMessage(envelope, observation, lineage), key)


def _keys_for(names, seed):
    return {name: demo_node_key(seed, name) for name in names}


def _perceive_sam(scene, seed, now_us, *, offline, budget_usd, client):
    """SAM at 2 Hz on both RSUs and every ego camera. Cache hits cost nothing.

    A failure keeps the redacted exception text. It does not fall back to
    ground-truth positions; that comparison is the gt-oracle perception.
    """
    from coopx.dataset.v2xreal import V2XReal
    from coopx.meta.client import BudgetExceeded, CacheMiss, MetaAPIError, MetaClient
    from coopx.observers.camera import CameraObserver, jpeg_to_camera_frame
    from coopx.observers.camera_lift import lift_box
    from coopx.pipeline import S3_DATASET
    if client is None:
        client = MetaClient(budget_usd=budget_usd, offline=offline)
    dataset = V2XReal(S3_DATASET)
    scenario, frame_id = scene["scenario"], scene["frames"][0]
    agents = {
        "-2": dataset.read(scenario, "-2", frame_id),
        "-1": dataset.read(scenario, "-1", frame_id),
        "ego": dataset.read(scenario, scene.get("ego_id", "2"), frame_id),
    }
    jobs = [("rsu-2", agents["-2"], "cam2", 0), ("rsu-1", agents["-1"], "cam1", 0)]
    for index, cam in enumerate(name for name in ("cam1", "cam2", "cam3", "cam4")
                                if name in agents["ego"].cameras):
        jobs.append(("ego", agents["ego"], cam, index * 100))
    keys = _keys_for({job[0] for job in jobs}, seed)
    messages, errors = [], []
    for observer_id, agent, cam, origin in jobs:
        observer = CameraObserver(
            observer_id, lambda a=agent, c=cam: jpeg_to_camera_frame(a, c),
            cam, client, lift_box, keys[observer_id], session="s4a",
            ttl_us=5_000_000, period_us=500_000, seq_origin=origin)
        try:
            messages.extend(observer.observe())
        except BudgetExceeded as exc:
            errors.append(f"{observer_id} {cam}: BudgetExceeded: {exc}")
            break
        except (CacheMiss, MetaAPIError, OSError) as exc:
            errors.append(f"{observer_id} {cam}: {type(exc).__name__}: {exc}")
    return messages, ("; ".join(errors) or None)


def _protected(beliefs, xyz, label) -> bool:
    from coopx.eval.metrics import _matches
    target = {"xy": [xyz[0], xyz[1]], "label": label}
    return any(belief.get("state") != "contradicted" and _matches(belief, target)
               for belief in beliefs)


def _fuse(mode, ego_obs, remote, coverages, anchors, store, coverage_map, now_us):
    """Absence claims (contract 17 coverage_map).

    Until A's fuse lands, the stub agrees on this:
    - ego: ignore remote packets, including free-space claims.
    - union: any admitted lidar-free cell deletes every belief in that cell,
      the ego's included. That is the naive failure of safety asymmetry.
    - geo: delete a belief only when at least two evidence-eligible range
      observers claim that cell free. One remote_clear, or any number of
      unanchored majority_veto peers, does not.
    - witness: never delete a local belief. A free claim is not a deletion.
    """
    try:
        from coopx.eval.fusion import fuse
        beliefs = fuse(mode, ego_observations=ego_obs, remote_admitted=remote,
                       store=store, coverage_map=coverage_map, assessor=None, now_us=now_us)
        return [_public(b) for b in beliefs], False
    except ImportError:
        pass
    from coopx.coverage import evidence_eligible
    local = [_belief(m, True, "confirmed") for m in ego_obs]
    remote_beliefs = [_belief(m, False, "unverified") for m in remote]
    if mode == "ego":
        return local, True
    if mode == "geo":
        # Cheap geometric consensus: a remote object stays only with a second
        # source within 1.5 m. ROBOSAC/CAD-style; A's fuse replaces this.
        kept = []
        for belief in remote_beliefs:
            others = [b for b in remote_beliefs + local if b["refs"] != belief["refs"]]
            if any(_close(belief, other) for other in others):
                kept.append(dict(belief, state="confirmed"))
        remote_beliefs = kept
    beliefs = local + remote_beliefs
    if mode == "witness":
        return beliefs, True
    eligible = {name for name, anchor in anchors.items() if evidence_eligible(anchor)}
    def free_hits(belief):
        total = eligible_hits = 0
        for message in coverages:
            cell = message.content.spec.cell_at(belief["xy"][0], belief["xy"][1])
            if cell not in message.content.free:
                continue
            total += 1
            if message.envelope.source_id in eligible:
                eligible_hits += 1
        return total, eligible_hits
    kept = []
    for belief in beliefs:
        total, eligible_hits = free_hits(belief)
        drop = total > 0 if mode == "union" else eligible_hits >= 2
        if not drop:
            kept.append(belief)
    return kept, True


def _close(a, b) -> bool:
    return (a["xy"][0] - b["xy"][0]) ** 2 + (a["xy"][1] - b["xy"][1]) ** 2 <= 1.5 ** 2


def _belief(message, local, state) -> dict:
    return {
        "xy": [message.observation.xyz_m[0], message.observation.xyz_m[1]],
        "cls": message.observation.cls.name,
        "local": local, "state": state,
        "refs": [evidence_ref(message)],
    }


def _public(belief) -> dict:
    if isinstance(belief, dict):
        return belief
    return {
        "xy": list(belief.xy), "cls": belief.cls, "local": belief.local,
        "state": belief.state, "refs": list(belief.refs),
    }


def _targets(scene, perception: str) -> list[dict]:
    xyz = scene["target_xyz"]
    targets = [
        {"id": scene.get("target_id", "53"), "label": scene.get("target_label", "pedestrian"),
         "xy": [xyz[0], xyz[1]], "visibility": scene.get("visibility", "outside_view")},
    ]
    # The oracle publishes a vehicle at the ego pose so safety asymmetry has a
    # local detection to protect. That point is not a ground-truth car.
    if perception == "gt-oracle":
        ego = scene.get("ego_xyz") or (0.0, 0.0, 0.0)
        targets.append({"id": "ego-local", "label": "vehicle", "xy": [ego[0], ego[1]],
                        "visibility": "visible"})
    return targets


def _encode(message):
    from coopx.envelope import encode
    return encode(message)


def _decode(raw):
    from coopx.envelope import decode
    return decode(raw)


def _out_path(out_dir: Path, scene: str, fusion: str, attack: str | None,
              perception: str) -> Path:
    name = fusion if not attack else f"{fusion}-{attack}"
    return Path(out_dir) / perception / scene / f"{name}.jsonl"
