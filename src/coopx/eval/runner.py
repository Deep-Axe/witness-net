"""Causal replay of one mined scene, one fusion mode, optional attack. Muse off.

Per scene (contract 14): every camera endpoint in `scene.observers` plus every
ego camera runs SAM at 2 Hz; each roadside unit's own LiDAR publishes real
coverage (contract 15). Remote packets cross a lossy transport into the ego's
Checker; ego observations stay local. Results enter only at their real
availability times (cached calls replay their measured latency). Fusion
(contract 17) runs at every 10 Hz frame on what has arrived by then and is
scored against that frame's ground truth (all labelled objects, one-to-one).

GT is used for scoring and for placing synthetic attacks only, never as an
observer's perception, except in the explicitly labelled `gt-oracle` ablation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from coopx.attacks import apply_attack, attack_anchors, plan_attack
from coopx.checker import Checker
from coopx.coverage import CoverageContent, CoverageMap, CoverageMessage, GridSpec
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, EvidenceStore, Lineage, ObservationMessage,
                            encode, evidence_ref, sign)
from coopx.eval.fusion import FUSION_MODES, fuse
from coopx.eval.metrics import (EVAL_RADIUS_M, aggregate_frames, attack_catch, false_distrust,
                                independence, independent_witness_count, safety_flips,
                                score_frame, _matches)
from coopx.identity import TrustAnchor, TrustClass
from coopx.pipeline import demo_node_key
from coopx.transport import PacketTransport

REGION = (-1000.0, -1000.0, -50.0, 1000.0, 1000.0, 80.0)
TTL_US = 5_000_000
PERCEPTION_STRIDE = 5          # 10 Hz frames -> SAM at 2 Hz
LIDAR_PROCESSING_US = 100_000  # coverage build before publish (fast path, no model)
ORACLE_LATENCY_US = 100_000    # gt-oracle stand-ins: perception assumed instant + processing
TRANSPORT_DELAY_US = 100_000
LIDAR_RANGE_M = 60.0
# Observers run continuously: replay starts before the labelled clip so that
# detections captured earlier can arrive (after real latency) during it, and
# continues after it for precision. Occluded recall uses labelled frames only.
PRE_ROLL_US = 3_000_000
TAIL_US = 3_000_000

FIXTURE = dict(
    name="frame-000096", scenario="2023-04-04-14-28-53_45_0", frames=("000096",),
    ego_id="2", observers=(("rsu-2-cam2", "-2", "cam2"), ("rsu-1-cam1", "-1", "cam1")),
    target="53", visibility="outside_view")


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def load_eval_scenes(name: str = "all", *, path: str = "data/scenes/scenes.json"):
    """Mined clips (report split) plus the S3 frame (tune split, not occluded)."""
    from coopx.dataset.scenes import Scene, Target, Visibility, load_scenes
    scenes = []
    if Path(path).is_file():
        scenes = [(s, "report") for s in load_scenes(path)]
    f = FIXTURE
    fixture = Scene(f["name"], f["scenario"], f["frames"], f["ego_id"], f["observers"],
                    (Target(f["target"], "pedestrian"),),
                    {(f["target"], f["ego_id"], f["frames"][0]): Visibility(f["visibility"], {})},
                    {})
    scenes.append((fixture, "tune"))
    if name != "all":
        scenes = [item for item in scenes if item[0].name == name]
        if not scenes:
            raise ValueError(f"unknown scene {name}")
    return scenes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_eval(*, scene: str = "all", fusion: str = "witness", attack: str | None = None,
             seed: int = 0, offline: bool = True, budget_usd: float = 2.0,
             out_dir: Path = Path("data/results"), client=None, perception: str = "sam",
             dataset_root: Path | None = None) -> dict:
    from coopx.meta.client import BudgetExceeded, MetaClient
    if fusion not in FUSION_MODES and fusion != "all":
        raise ValueError(f"unknown fusion {fusion}")
    if perception not in ("sam", "gt-oracle"):
        raise ValueError(f"unknown perception {perception}")
    modes = FUSION_MODES if fusion == "all" else (fusion,)
    if perception == "sam" and client is None:
        client = MetaClient(budget_usd=budget_usd, offline=offline)
    summaries, partial, stopped = [], False, None
    try:
        for item, split in load_eval_scenes(scene):
            key = (item.name, seed, perception, str(dataset_root), id(client))
            replay = _REPLAYS.get(key)
            if replay is None:
                replay = _Replay(item, split, seed=seed, perception=perception, client=client,
                                 dataset_root=dataset_root)
                if not replay.errors:
                    _REPLAYS[key] = replay   # perception/coverage are pure given the inputs
            for mode in modes:
                summaries.append(replay.run(mode, attack, out_dir))
    except BudgetExceeded as exc:
        partial, stopped = True, str(exc)
    return {"runs": summaries, "partial": partial, "stopped": stopped,
            "offline": offline, "budget_usd": budget_usd}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

_REPLAYS: dict = {}


def _unit(agent_id: str) -> str:
    return f"unit{agent_id}"


class _Replay:
    """Perception and coverage are computed once per scene and shared by every mode."""

    def __init__(self, scene, split, *, seed, perception, client, dataset_root=None):
        from coopx.dataset.v2xreal import V2XReal
        from coopx.pipeline import S3_DATASET
        self.scene, self.split, self.seed, self.perception = scene, split, seed, perception
        self.client = client
        self.dataset = V2XReal(dataset_root or S3_DATASET)
        self.clip = tuple(scene.frames)
        self.agent_ids = tuple(dict.fromkeys([scene.ego_id] + [a for _e, a, _c in scene.observers]))
        aligned = list(self.dataset.aligned_frame_ids(scene.scenario, self.agent_ids))
        first, last = aligned.index(self.clip[0]), aligned.index(self.clip[-1])
        hz = getattr(self.dataset, "nominal_hz", 10)
        pre, post = int(PRE_ROLL_US * hz / 1e6), int(TAIL_US * hz / 1e6)
        self.frames = tuple(aligned[max(0, first - pre):last + post + 1])
        self.read = {}
        for frame in self.frames:
            for agent in self.agent_ids:
                self.read[(agent, frame)] = self.dataset.read(scene.scenario, agent, frame)
        self.time = {f: self.read[(scene.ego_id, f)].nominal_time_us for f in self.frames}
        ego0 = self.read[(scene.ego_id, self.frames[0])]
        self.endpoints = [(e, a, c) for e, a, c in scene.observers]
        self.endpoints += [(f"ego-{c}", scene.ego_id, c) for c in sorted(ego0.cameras)]
        self.lidar_units = [a for a in dict.fromkeys(a for _e, a, _c in scene.observers)
                            if a != scene.ego_id]
        self.grid = self._grid()
        self.errors = []
        self.model_latency_ms = []
        self._gt_cache = {}
        self.honest = self._perceive()          # list of (publish_us, message)
        self.coverage = self._coverage()        # list of (publish_us, message)

    # -- setup ---------------------------------------------------------------
    def _xy(self, agent, frame):
        m = self.read[(agent, frame)].lidar_to_site
        return float(m[0, 3]), float(m[1, 3])

    def _grid(self) -> GridSpec:
        pts = [self._xy(a, f) for a in self.agent_ids for f in (self.frames[0], self.frames[-1])]
        pad = LIDAR_RANGE_M
        x0 = math.floor(min(p[0] for p in pts) - pad)
        y0 = math.floor(min(p[1] for p in pts) - pad)
        nx = int(math.ceil(max(p[0] for p in pts) + pad) - x0)
        ny = int(math.ceil(max(p[1] for p in pts) + pad) - y0)
        cell = 1.0
        while nx * ny / (cell * cell) > 65_536:
            cell *= 2
        return GridSpec(float(x0), float(y0), cell, int(math.ceil(nx / cell)), int(math.ceil(ny / cell)))

    def _keys(self, names):
        return {n: demo_node_key(self.seed, n) for n in names}

    def anchors(self):
        out = {}
        names = [e for e, _a, _c in self.endpoints] + [f"{_unit(a)}-lidar" for a in self.lidar_units]
        keys = self._keys(names)
        units = {e: a for e, a, _c in self.endpoints}
        units.update({f"{_unit(a)}-lidar": a for a in self.lidar_units})
        for name in names:
            # Bind to the PHYSICAL unit: two cameras and a LiDAR on one unit are one witness.
            out[name] = TrustAnchor(keys[name].public_key, f"{name}-key", TrustClass.A,
                                    physical_observer=_unit(units[name]),
                                    permitted_region=REGION, anchored=True)
        return keys, out

    def perception_frames(self):
        return self.frames[::PERCEPTION_STRIDE]

    def _perceive(self):
        keys, _ = self.anchors()
        if self.perception == "gt-oracle":
            return self._oracle(keys)
        from coopx.meta.client import BudgetExceeded, CacheMiss, MetaAPIError
        from coopx.observers.camera import CameraObserver, jpeg_to_camera_frame
        from coopx.observers.camera_lift import lift_box
        out = []
        for index, (endpoint, agent, cam) in enumerate(self.endpoints):
            holder = {}
            observer = CameraObserver(
                endpoint, lambda h=holder: h.get("frame"), cam, self.client, lift_box,
                keys[endpoint], session="s4a", ttl_us=TTL_US, seq_origin=index * 10_000)
            for frame in self.perception_frames():
                agent_frame = self.read[(agent, frame)]
                if cam not in agent_frame.cameras:
                    continue
                holder["frame"] = jpeg_to_camera_frame(agent_frame, cam)
                try:
                    for message in observer.observe():
                        out.append((message.envelope.generation_time_us, message))
                except BudgetExceeded:
                    raise
                except (CacheMiss, MetaAPIError, OSError) as exc:
                    self.errors.append(f"{endpoint} {frame}: {type(exc).__name__}: {exc}")
                if observer.last_result is not None:
                    self.model_latency_ms.append(observer.last_result.latency_ms)
        return sorted(out, key=lambda item: (item[0], item[1].envelope.source_id))

    def _oracle(self, keys):
        """Labelled ablation: each endpoint reports scene targets it is labelled to see."""
        out, seq = [], {}
        for endpoint, agent, _cam in self.endpoints:
            for frame in self.perception_frames():
                for target in self.scene.targets:
                    vis = self.scene.visibility.get((target.gt_id, agent, frame))
                    box = self._gt_boxes(frame).get(target.gt_id)
                    if vis is None or vis.label != "visible" or box is None:
                        continue
                    capture = self.time[frame]
                    n = seq[endpoint] = seq.get(endpoint, -1) + 1
                    xyz = tuple(float(v) for v in box.center_site())
                    cls = ObjectClass.PEDESTRIAN if target.label == "pedestrian" else ObjectClass.VEHICLE
                    mid = f"{endpoint}-oracle-{n}"
                    obs = Observation(endpoint, 100 + int(target.gt_id), n, capture, cls, xyz,
                                      PositionAnchor.SURFACE, confidence=0.8)
                    env = Envelope(2, "observation", mid, endpoint, "s4a", "site_map", capture,
                                   capture + ORACLE_LATENCY_US,
                                   capture + ORACLE_LATENCY_US + TTL_US, Auth(f"{endpoint}-key"), n)
                    msg = sign(ObservationMessage(env, obs, Lineage(endpoint, "oracle", mid, capture, "direct")),
                               keys[endpoint])
                    out.append((capture + ORACLE_LATENCY_US, msg))
        return sorted(out, key=lambda item: (item[0], item[1].envelope.source_id))

    def _coverage(self):
        """Each roadside unit's own LiDAR: ground-relative free space, one signed
        message per ground-height slice (sloped sites; see lidar_coverage_sliced)."""
        from coopx.observers.lidar_real import lidar_coverage_sliced
        keys, _ = self.anchors()
        out = []
        for agent in self.lidar_units:
            source = f"{_unit(agent)}-lidar"
            seq = 0
            for frame in self.perception_frames():
                f = self.read[(agent, frame)]
                capture = self.time[frame]
                publish = capture + LIDAR_PROCESSING_US
                for index, content in enumerate(lidar_coverage_sliced(
                        f, self.grid, capture_us=capture,
                        window_us=(max(0, capture - 100_000), capture), max_range_m=LIDAR_RANGE_M)):
                    mid = f"{source}-{frame}-{index}"
                    env = Envelope(2, "coverage", mid, source, "s4a", "site_map", capture, publish,
                                   publish + TTL_US, Auth(f"{source}-key"), seq)
                    seq += 1
                    msg = sign(CoverageMessage(env, content, Lineage(source, "lidar", mid, capture, "direct")),
                               keys[source])
                    out.append((publish, msg))
        return out

    def height_band(self, frame, xy=None):
        """Absolute band honest slices use at xy (absence attacks must match it)."""
        from coopx.observers.lidar_real import ground_band
        agent = self.lidar_units[0] if self.lidar_units else self.scene.ego_id
        xy = xy if xy is not None else self._xy(agent, frame)
        return ground_band(self.read[(agent, frame)], xy) or (0.2, 1.0)

    # -- one fusion mode ------------------------------------------------------
    def _gt_boxes(self, frame):
        """Scoring GT = union of every agent's labels in the scenario (as the miner)."""
        if frame not in self._gt_cache:
            boxes = {}
            for agent in self.dataset.agents(self.scene.scenario):
                frame_data = self.read.get((agent, frame))
                if frame_data is None:
                    try:
                        frame_data = self.dataset.read(self.scene.scenario, agent, frame)
                    except (FileNotFoundError, KeyError, OSError, ValueError):
                        continue
                for gid, box in frame_data.boxes.items():
                    boxes.setdefault(gid, box)
            self._gt_cache[frame] = boxes
        return self._gt_cache[frame]

    def gt(self, frame):
        objects = {}
        for _ in (0,):
            for gid, box in self._gt_boxes(frame).items():
                if gid not in objects:
                    c = box.center_site()
                    objects[gid] = {"id": gid, "label": box.label, "xy": [float(c[0]), float(c[1])]}
        return list(objects.values())

    def _attack_scene(self, at_frame):
        target = self.scene.targets[0]
        box = None
        # The target's labelled position nearest the attack time (GT can drop out).
        order = sorted(self.frames, key=lambda f: abs(self.time[f] - self.time[at_frame]))
        for frame in order:
            box = self._gt_boxes(frame).get(target.gt_id)
            if box is not None:
                break
        txyz = tuple(float(v) for v in box.center_site()) if box is not None else (0.0, 0.0, 0.0)
        ego_xy = self._xy(self.scene.ego_id, at_frame)
        local = [m for _t, m in self.honest if m.envelope.source_id.startswith("ego-")]
        ego_xyz = tuple(local[0].observation.xyz_m) if local else (ego_xy[0] + 3.0, ego_xy[1], 0.5)
        free_xyz = unseen_xyz = None
        # All slices captured at the attack frame (one message per ground slice).
        cov = [m for _t, m in self.coverage if m.envelope.capture_time_us == self.time[at_frame]]
        if cov:
            from coopx.observers.lidar_real import ground_model
            unit = self.lidar_units[0]
            plane = ground_model(self.read[(unit, at_frame)])
            spec = cov[0].content.spec
            free = set().union(*(m.content.free for m in cov))
            observed = set().union(*(m.content.observed for m in cov))
            def placed(cell):
                x, y = spec.cell_centre(cell)
                g = 0.0 if plane is None else float(plane[0] * x + plane[1] * y + plane[2])
                return (x, y, g + 0.7)   # pedestrian-centre height, inside the slice band
            # Ghost in measured free space ~4 m from the target, not on it.
            if free:
                free_xyz = placed(min(free, key=lambda cell: abs(math.dist(spec.cell_centre(cell), txyz[:2]) - 4.0)))
            unobserved = [(ix, iy) for ix in range(spec.nx) for iy in range(spec.ny)
                          if (ix, iy) not in observed
                          and 3.0 <= math.dist(spec.cell_centre((ix, iy)), txyz[:2]) <= 8.0]
            if unobserved:
                unseen_xyz = placed(min(unobserved, key=lambda cell: math.dist(spec.cell_centre(cell), txyz[:2])))
        replay = next((m for _t, m in self.honest if not m.envelope.source_id.startswith("ego-")), None)
        return {
            "target_xyz": txyz, "ego_xyz": ego_xyz, "free_xyz": free_xyz, "unseen_xyz": unseen_xyz,
            "grid": self.grid,
            "height_band_m": self.height_band(at_frame, txyz[:2]),
            "disconnect_node": self.endpoints[0][0],
            "replay_source": replay.envelope.source_id if replay else self.endpoints[0][0],
            "replay_message_id": replay.envelope.message_id if replay else "replay-old",
        }, (encode(replay) if replay else None)

    def run(self, mode, attack, out_dir: Path) -> dict:
        keys, anchors = self.anchors()
        decision_frames = self.frames
        # Attack at the middle perception frame: honest coverage exists at that instant.
        pframes = self.perception_frames()
        at_frame = pframes[len(pframes) // 2]
        plan = None
        if attack is not None:
            scene_dict, captured = self._attack_scene(at_frame)
            plan = plan_attack(attack, seed=self.seed, scene=scene_dict, now_us=self.time[at_frame],
                               captured=captured)
            _k, extra = attack_anchors(plan, region=REGION)
            anchors.update(extra)
        checker = Checker(anchors, ttl_us=TTL_US)
        store = EvidenceStore(anchors)
        cmap = CoverageMap(self.grid, ttl_us=TTL_US, spatial_frame_id="site_map", store=store)
        nodes = ["ego"] + [e for e, _a, _c in self.endpoints if not e.startswith("ego-")]
        nodes += [f"{_unit(a)}-lidar" for a in self.lidar_units]
        if plan is not None:
            nodes += [n.name for n in plan.nodes if n.name not in nodes]
        events, admitted, delivered_bytes = [], [], [0]

        def on_delivery(delivery):
            if delivery.receiver != "ego":
                return
            result = checker.check(delivery.packet, delivery.available_sim_us)
            from coopx.pipeline import _check_event
            events.append(_check_event(delivery, result, delivery.available_sim_us))
            delivered_bytes[0] += len(delivery.packet)
            if result.state != "accepted" or result.message is None:
                return
            if isinstance(result.message, CoverageMessage):
                try:
                    cmap.add(result.message, delivery.available_sim_us)
                except ValueError as exc:
                    self.errors.append(f"coverage not mapped: {exc}")
            else:
                admitted.append(result.message)

        transport = PacketTransport(
            tuple(nodes), delay_us=TRANSPORT_DELAY_US, jitter_us=0,
            loss_probability=0.0 if plan is None else plan.loss, seed=self.seed,
            event_callback=events.append, delivery_callback=on_delivery)
        # One time-ordered outgoing queue: honest remote packets, attack packets
        # and the disconnect control event, released as replay time advances.
        local_timeline, outgoing = [], []
        for publish, message in self.honest + self.coverage:
            source = message.envelope.source_id
            if source.startswith("ego-"):
                local_timeline.append((publish, message))   # local: never crosses the link
            else:
                outgoing.append((publish, 1, source, encode(message), message.envelope.message_id,
                                 message.envelope.message_type))
        if plan is not None:
            for packet in plan.packets:
                outgoing.append((packet.sim_us, 1, packet.source, packet.raw, packet.message_id,
                                 packet.message_type))
            if plan.disconnect is not None:
                node, when = plan.disconnect
                outgoing.append((when, 0, node, None, None, "disconnect"))
        outgoing.sort(key=lambda item: (item[0], item[1], item[2]))
        local_timeline.sort(key=lambda item: item[0])
        cursor = [0]

        def release(until_us):
            while cursor[0] < len(outgoing) and outgoing[cursor[0]][0] <= until_us:
                when, _order, source, raw, mid, mtype = outgoing[cursor[0]]
                cursor[0] += 1
                transport.receive_until(when)
                if mtype == "disconnect":
                    transport.disconnect(source, sim_us=when)
                else:
                    transport.send(source, raw, sim_us=when, message_id=mid, message_type=mtype)
            transport.receive_until(until_us)

        frames_scored, flips, distrust, ticks = [], 0, 0, []
        memory = {}
        occluded_pairs = occluded_hits = 0
        first_warning = {}
        for frame in decision_frames:
            now = self.time[frame]
            release(now)
            local = [m for t, m in local_timeline if t <= now]
            beliefs = [_public(b) for b in fuse(mode, ego_observations=local, remote_admitted=admitted,
                                                  store=store, coverage_map=cmap, assessor=None, now_us=now,
                                                  memory=memory)]
            before = [_public(b) for b in fuse("ego", ego_observations=local, remote_admitted=[],
                                                 store=store, coverage_map=cmap, assessor=None, now_us=now)]
            flips += safety_flips(before, beliefs)
            gt = self.gt(frame)
            ego_xy = self._xy(self.scene.ego_id, frame)
            frames_scored.append(score_frame(beliefs, gt, center=ego_xy, radius_m=EVAL_RADIUS_M))
            distrust += false_distrust(beliefs, gt)
            hazards = [b for b in beliefs if b.get("state") != "contradicted"]
            for target in self.scene.targets:
                vis = self.scene.visibility.get((target.gt_id, self.scene.ego_id, frame))
                g = next((o for o in gt if o["id"] == target.gt_id), None)
                if vis is None or vis.label != "occluded" or g is None:
                    continue
                occluded_pairs += 1
                if any(_matches(b, g) for b in hazards):
                    occluded_hits += 1
                    first_warning.setdefault(target.gt_id, now)
            ticks.append({"record": "tick", "t_us": now, "frame": frame, "beliefs": beliefs})

        injected = [evidence_ref(_decode(p.raw)) for p in plan.packets] if plan else []
        last = ticks[-1]["beliefs"] if ticks else []
        if mode == "ego" or plan is None or plan.kind in ("loss_sweep", "disconnect"):
            caught = None
        elif plan.kind in ("remote_clear", "majority_veto"):
            caught = flips == 0
        else:
            caught = attack_catch(injected, last)
        attack_obs = [m for m in (_decode(p.raw) for p in plan.packets)
                      if isinstance(m, ObservationMessage)] if plan else []
        lead = {}
        for target in self.scene.targets:
            visible_frame = (self.scene.hidden_interval.get(target.gt_id) or (None, None))[1]
            warn = first_warning.get(target.gt_id)
            visible_us = self.time.get(visible_frame) if visible_frame in self.time else None
            # None = unknown/censored (no warning, or the ego never sees it in the clip).
            lead[target.gt_id] = None if warn is None or visible_us is None else visible_us - warn
        scores = aggregate_frames(frames_scored)
        scores["occluded_recall"] = None if occluded_pairs == 0 else occluded_hits / occluded_pairs
        scores["occluded_targets"] = occluded_pairs
        useful = len(first_warning)
        metrics = {
            "scores": scores,
            "safety_flips": flips,
            "false_distrust": distrust,
            "attack_caught": caught,
            "independence": independence(
                len([1 for _t, m in self.honest if not m.envelope.source_id.startswith("ego-")])
                + (len(plan.packets) if plan else 0),
                independent_witness_count(anchors, attack_obs or [m for _t, m in self.honest])),
            "lead_us": lead,
            "bytes_to_ego": delivered_bytes[0],
            "bytes_per_hidden_detection": None if useful == 0 else delivered_bytes[0] / useful,
            "model_latency_ms_mean": (sum(self.model_latency_ms) / len(self.model_latency_ms)
                                      if self.model_latency_ms else None),
            "eval_radius_m": EVAL_RADIUS_M,
        }
        summary = {
            "record": "summary", "scene": self.scene.name, "fusion": mode, "attack": attack,
            "seed": self.seed, "fusion_stub": False, "perception": self.perception,
            "partial": bool(self.errors), "error": "; ".join(self.errors[:5]) or None,
            "metrics": metrics, "split": self.split,
            "decisions": len(ticks), "perception_frames": list(self.perception_frames()),
        }
        path = _out_path(out_dir, self.scene.name, mode, attack, self.perception)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            for tick in ticks:
                handle.write(json.dumps(tick, separators=(",", ":")) + "\n")
            handle.write(json.dumps(summary, separators=(",", ":")) + "\n")
        summary["path"] = str(path)
        return summary


def _public(belief) -> dict:
    if isinstance(belief, dict):
        return belief
    return {"xy": list(belief.xy), "cls": belief.cls, "local": belief.local,
            "state": belief.state, "refs": list(belief.refs)}


def _decode(raw):
    from coopx.envelope import decode
    return decode(raw)


def _out_path(out_dir: Path, scene: str, fusion: str, attack: str | None, perception: str) -> Path:
    name = fusion if not attack else f"{fusion}-{attack}"
    return Path(out_dir) / perception / scene / f"{name}.jsonl"
