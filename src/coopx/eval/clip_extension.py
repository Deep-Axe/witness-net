"""Post-hoc clip-2 inspection from existing traces, calibration and raw LiDAR.

No inference client is constructed. This appends diagnostic metadata only;
scored scenes, existing rows, thresholds and temporal tolerances stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from coopx.dataset.scenes import MAX_EGO_DISTANCE_M, _box_blocks, _samples, visibility
from coopx.dataset.v2xreal import project_camera
from coopx.eval.report import write_summary
from coopx.eval.runner import _Replay, load_eval_scenes

SCENE = "blos_pedestrian_002"
TARGET = "138"
DISCLAIMER = (
    "Window extended after results were seen; not part of the scored "
    "clip. Single clip, single source. No lead time is claimed."
)


def target_report(replay, tick, target_box):
    """Audit identity independently of position radius and report the distance.
    Contradicted beliefs are not silently hidden from this inspection.
    """
    found = []
    for belief in tick["beliefs"]:
        identity, ambiguous = replay.audit_identity(belief, [])
        if identity == TARGET and not ambiguous:
            found.append(describe(belief, target_box, identity, ambiguous))
    return (
        min(found, key=lambda row: row["distance_m"])
        if found and target_box is not None
        else (found[0] if found else None)
    )


def describe(belief, target_box, identity=None, ambiguous=False):
    return {
        "distance_m": None
        if target_box is None
        else math.dist(belief["xy"], target_box.center_site()[:2]),
        "source": belief.get("source_id"),
        "state": belief["state"],
        "capture_us": belief.get("capture_time_us"),
        "identity": identity,
        "ambiguous": ambiguous,
        "local": belief.get("local", False),
        "observation_ref": belief.get("observation_ref"),
    }


def nearest_ego(replay, tick, target_box):
    if target_box is None:
        return None
    candidates = [
        b for b in tick["beliefs"] if b.get("local") and b.get("cls") == "PEDESTRIAN"
    ]
    if not candidates:
        return None
    nearest = min(
        candidates, key=lambda b: math.dist(b["xy"], target_box.center_site()[:2])
    )
    identity, ambiguous = replay.audit_identity(nearest, [])
    return describe(nearest, target_box, identity, ambiguous)


def draw_overlays(replay, *, out_dir):
    frame_id = "000030"
    target = replay._gt_boxes(frame_id).get(TARGET)
    if target is None:
        raise ValueError("frame 000030 has no target GT; never extrapolate an overlay")
    paths = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for agent, cam, role, color in (
        (replay.scene.ego_id, "cam4", "ego", "orange"),
        ("-1", "cam2", "rsu", "lime"),
    ):
        frame = replay.read.get((agent, frame_id)) or replay.dataset.read(
            replay.scene.scenario, agent, frame_id
        )
        uv, _, valid = project_camera(
            frame.to_camera(target.corners_site(), cam), frame.cameras[cam].intrinsic
        )
        if not valid.all():
            raise ValueError(
                "target crosses the camera near plane; overlay needs explicit clipping"
            )
        with Image.open(frame.cameras[cam].image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for a, b in (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ):
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=3)
            lo, hi = uv.min(0), uv.max(0)
            rect = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
            draw.rectangle(rect, outline=color, width=3)
            draw.rectangle((0, 0, image.width, 35), fill="black")
            draw.text(
                (10, 10),
                f"{SCENE} frame {frame_id} target {TARGET} | {role} {agent} {cam} | GT box, human review pending",
                fill="white",
            )
            stem = f"{SCENE}-posthoc-{frame_id}-{role}-{agent}-{cam}"
            full = out_dir / f"{stem}-full.png"
            image.save(full)
            pad = max(48.0, 0.5 * max(rect[2] - rect[0], rect[3] - rect[1]))
            crop_box = (
                max(0, int(rect[0] - pad)),
                max(0, int(rect[1] - pad)),
                min(image.width, math.ceil(rect[2] + pad)),
                min(image.height, math.ceil(rect[3] + pad)),
            )
            crop = out_dir / f"{stem}-crop.png"
            image.crop(crop_box).save(crop)
            paths.extend([str(full), str(crop)])
    return paths


def build_extension(study_path, *, overlays=Path("data/scenes/overlays/")):
    study_path = Path(study_path)
    study = json.loads(study_path.read_text())
    if not any(
        row["scene"] == SCENE
        and row.get("variant") == "SAM without tracking"
        and row.get("latency_tolerance_s") == 0
        for row in study.get("sweep", [])
    ):
        raise ValueError("study has no strict SAM-without-tracking control for clip 2")
    scene = load_eval_scenes(SCENE)[0][0]
    trace = (
        study_path.parent / "ablation" / "untracked" / "sam" / SCENE / "witness.jsonl"
    )
    raw = trace.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    summary = next(row for row in reversed(records) if row.get("record") == "summary")
    if (
        summary.get("tracking") is not False
        or summary["perception"] != "sam"
        or summary["fusion"] != "witness"
        or summary.get("latency_tolerance_s", 0) != 0
    ):
        raise ValueError(
            "inspection requires the existing strict SAM-without-tracking witness trace"
        )
    summary = {**summary, "path": str(trace)}
    replay = _Replay.for_scoring(scene, summary)
    agents = replay.dataset.agents(scene.scenario)
    ticks = {row["frame"]: row for row in records if row.get("record") == "tick"}
    rows = []
    first = None
    for i in range(34):
        frame_id = f"{i:06d}"
        for agent in agents:
            if (agent, frame_id) not in replay.read:
                replay.read[(agent, frame_id)] = replay.dataset.read(
                    scene.scenario, agent, frame_id
                )
        target = replay._gt_boxes(frame_id).get(TARGET)
        gt_sources = [a for a in agents if TARGET in replay.read[(a, frame_id)].boxes]
        row = {
            "frame": frame_id,
            "t_us": ticks[frame_id]["t_us"],
            "gt_sources": gt_sources,
            "ego_visibility": None,
            "rsu_visibility": None,
            "ego_range_m": None,
            "ego_nearest": nearest_ego(replay, ticks[frame_id], target),
            "target_report": target_report(replay, ticks[frame_id], target),
        }
        if target is not None:
            for agent, name in (
                (scene.ego_id, "ego_visibility"),
                ("-1", "rsu_visibility"),
            ):
                f = replay.read[(agent, frame_id)]
                v = visibility(f, f.boxes.get(TARGET, target))
                row[name] = {"label": v.label, "signals": v.signals}
            row["ego_range_m"] = math.dist(
                target.center_site()[:2],
                replay.read[(scene.ego_id, frame_id)].lidar_to_site[:2, 3],
            )
        if first is None and row["target_report"] is not None:
            first = {
                "frame": frame_id,
                "t_us": row["t_us"],
                **row["target_report"],
                "ego_nearest": row["ego_nearest"],
            }
        rows.append(row)
    # Inspect the full annotation sequence, so a temporary gap is not called
    # the end of GT and a later return cannot be suppressed.
    presence = []
    for frame_id in replay.dataset.aligned_frame_ids(scene.scenario, agents):
        if TARGET in replay._gt_boxes(frame_id):
            presence.append(frame_id)
    last = presence[-1] if presence else None
    all_frames = replay.dataset.aligned_frame_ids(scene.scenario, agents)
    first_missing = next(
        (f for f in all_frames if presence and f > presence[0] and f not in presence),
        None,
    )
    stored = [
        f
        for (tid, agent, f), v in scene.visibility.items()
        if tid == TARGET and agent == scene.ego_id
    ]
    mining_last = max(stored)
    boundary = [
        row
        for row in rows
        if row["frame"] in (mining_last, f"{int(mining_last) + 1:06d}")
    ]
    frame30 = replay.read[(scene.ego_id, "000030")]
    target30 = replay._gt_boxes("000030")[TARGET]
    origin = frame30.camera_to_site("cam4")[:3, 3]
    blockers = []
    for obstacle in frame30.boxes.values():
        fraction = float(_box_blocks(origin, _samples(target30), obstacle).mean())
        if fraction > 0:
            blockers.append(
                {
                    "gt_id": obstacle.object_id,
                    "label": obstacle.label,
                    "blocked_sample_fraction": fraction,
                }
            )
    result = {
        "schema": "coopx.clip2-posthoc/1",
        "disclaimer": DISCLAIMER,
        "scene": SCENE,
        "target": TARGET,
        "trace": str(trace),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "tracking": False,
        "frame_000030_gt_blockers": blockers,
        "rows": rows,
        "first_report": first,
        "gt_last_present": last,
        "gt_first_missing": first_missing,
        "gt_returns_after_first_missing": [
            f for f in presence if first_missing and f > first_missing
        ],
        "mined_window_end": scene.frames[-1],
        "mining_last_record": mining_last,
        "mining_range_m": MAX_EGO_DISTANCE_M,
        "mining_boundary": boundary,
        "context_frames_after_record": int(scene.frames[-1]) - int(mining_last),
        "overlays": draw_overlays(replay, out_dir=overlays),
    }
    if trace.read_bytes() != raw:
        raise RuntimeError("source trace changed during inspection")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("data/s4b-agent-a/release/study.json")
    )
    parser.add_argument("--summary", type=Path, default=Path("results/summary.md"))
    parser.add_argument("--overlays", type=Path, default=Path("data/scenes/overlays/"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/s4b-agent-a/clip2-extension/report.json"),
    )
    args = parser.parse_args()
    extension = build_extension(args.study, overlays=args.overlays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extension, indent=2) + "\n")
    study = json.loads(args.study.read_text())
    study["clip2_extension"] = extension
    args.study.write_text(json.dumps(study, indent=2) + "\n")
    write_summary(args.study.parent / "matrix", args.summary, study_path=args.study)
    print(
        json.dumps(
            {
                "first_report": extension["first_report"],
                "gt_last_present": extension["gt_last_present"],
                "gt_first_missing": extension["gt_first_missing"],
                "overlays": extension["overlays"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
