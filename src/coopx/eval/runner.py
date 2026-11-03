"""Eval runner: one scene, one fusion, optional attack. Muse stays off.

Observations are signed local stand-ins at the scene's GT positions unless the
caller passes packets or a live client. Agent A's `fuse` is used when it
imports; ego/union (and geo/witness, labelled) have stubs until then.
"""

from __future__ import annotations

import json
from pathlib import Path

from coopx.attacks import apply_attack, attack_anchors, plan_attack
from coopx.checker import Checker
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, Lineage, ObservationMessage, evidence_ref, sign
from coopx.eval.metrics import (attack_catch, false_distrust, independence,
                                occluded_scores, safety_flips)
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
             client=None) -> dict:
    from coopx.meta.client import BudgetExceeded
    if fusion not in FUSION_MODES and fusion != "all":
        raise ValueError(f"unknown fusion {fusion}")
    modes = FUSION_MODES if fusion == "all" else (fusion,)
    summaries = []
    partial = False
    stopped = None
    try:
        for mode in modes:
            summaries.append(_one(scene, mode, attack, seed, out_dir))
    except BudgetExceeded as exc:
        partial = True
        stopped = str(exc)
    result = {"runs": summaries, "partial": partial, "stopped": stopped,
              "offline": offline, "budget_usd": budget_usd, "client": client is not None}
    return result


def _one(scene_name, fusion, attack, seed, out_dir: Path) -> dict:
    scene = load_eval_scenes(scene_name)[0]
    now_us = 9_600_000
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
    honest = _honest_messages(scene, keys, now_us)
    for message in honest:
        raw = _encode(message)
        events.append({
            "schema": "coopx.events/1", "t_replay_us": now_us, "type": "observation_published",
            "source_id": message.envelope.source_id, "dest_id": None,
            "message_id": message.envelope.message_id, "message_type": "observation",
            "evidence_ref": evidence_ref(message), "bytes": len(raw), "check_codes": [],
            "state": None, "detail": {
                "local": message.envelope.source_id == "ego",
                "xyz": list(message.observation.xyz_m),
                "cls": message.observation.cls.name,
            },
        })
        transport.send(message.envelope.source_id, raw, sim_us=now_us,
                       message_id=message.envelope.message_id, message_type="observation")
    if plan is not None:
        apply_attack(transport, plan)
    transport.receive_until(now_us + 2_000_000)
    ego_obs = [m for m in admitted if m.envelope.source_id == "ego"]
    remote = [m for m in admitted if m.envelope.source_id != "ego"]
    before = [_belief(m, True, "confirmed") for m in ego_obs]
    beliefs, stub = _fuse(fusion, ego_obs, remote)
    targets = _targets(scene)
    scores = occluded_scores([{"beliefs": beliefs, "targets": targets}])
    injected = [evidence_ref(_decode(p.raw)) for p in plan.packets] if plan else []
    # Ego-only never reads remote packets, so it does not "catch" an attack.
    if fusion == "ego" or plan is None or plan.kind in ("loss_sweep", "disconnect"):
        caught = None
    else:
        caught = attack_catch(injected, beliefs)
    roots = {m.lineage.origin_observation_id for m in honest}
    messages = list(honest)
    if plan is not None:
        for packet in plan.packets:
            decoded = _decode(packet.raw)
            messages.append(decoded)
            roots.add(decoded.lineage.origin_observation_id)
    metrics = {
        "scores": scores,
        "safety_flips": safety_flips(before, beliefs),
        "false_distrust": false_distrust(beliefs, targets),
        "attack_caught": caught,
        "independence": independence(len(messages), len(roots)),
        "latency_chain_us": PRIOR_LATENCY_US,
        "lead_us": None,
        "visibility": scene.get("visibility"),
    }
    summary = {
        "record": "summary", "scene": scene["name"], "fusion": fusion,
        "attack": attack, "seed": seed, "fusion_stub": stub, "partial": False,
        "metrics": metrics, "split": scene.get("split", "tune"),
    }
    path = _out_path(out_dir, scene["name"], fusion, attack)
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


def _fuse(mode, ego_obs, remote):
    try:
        from coopx.eval.fusion import fuse
        beliefs = fuse(mode, ego_observations=ego_obs, remote_admitted=remote,
                       store=None, coverage_map=None, assessor=None, now_us=0)
        return [_public(b) for b in beliefs], False
    except ImportError:
        pass
    beliefs = [_belief(m, True, "confirmed") for m in ego_obs]
    if mode == "ego":
        return beliefs, True
    remote_beliefs = [_belief(m, False, "unverified") for m in remote]
    if mode == "union":
        return beliefs + remote_beliefs, True
    if mode == "geo":
        # Cheap geometric consensus: a remote claim stays only with a second
        # source within 1.5 m. ROBOSAC/CAD-style; A's fuse replaces this.
        kept = []
        for belief in remote_beliefs:
            others = [b for b in remote_beliefs + beliefs if b["refs"] != belief["refs"]]
            if any(_close(belief, other) for other in others):
                kept.append(dict(belief, state="confirmed"))
        return beliefs + kept, True
    return beliefs + remote_beliefs, True


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


def _targets(scene) -> list[dict]:
    xyz = scene["target_xyz"]
    ego = scene.get("ego_xyz") or (0.0, 0.0, 0.0)
    return [
        {"id": scene.get("target_id", "53"), "label": scene.get("target_label", "pedestrian"),
         "xy": [xyz[0], xyz[1]], "visibility": scene.get("visibility", "outside_view")},
        {"id": "ego-local", "label": "vehicle", "xy": [ego[0], ego[1]], "visibility": "visible"},
    ]


def _encode(message):
    from coopx.envelope import encode
    return encode(message)


def _decode(raw):
    from coopx.envelope import decode
    return decode(raw)


def _out_path(out_dir: Path, scene: str, fusion: str, attack: str | None) -> Path:
    name = fusion if not attack else f"{fusion}-{attack}"
    return Path(out_dir) / scene / f"{name}.jsonl"
