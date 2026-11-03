"""Paid-call-free diagnostics; no live model or simulator success is implied."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.radio_codec import decode, encode
from coopx.transport import ReferenceTransport


def _write_event(handle, kind: str, sim_us: int, **fields) -> None:
    handle.write(json.dumps({"version": 1, "kind": kind, "sim_us": sim_us, **fields},
                            allow_nan=False, separators=(",", ":")) + "\n")


def synthetic(output: Path) -> dict:
    import numpy as np
    from coopx.geometry import CameraIntrinsics, decode_depth_bgra, lift_mask_surface

    intrinsics = CameraIntrinsics(64, 48, 90)
    packed = round(10 / 1000 * (256**3 - 1))
    raw = np.empty((48, 64, 4), dtype=np.uint8)
    raw[:] = [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255, 255]
    mask = np.zeros((48, 64), dtype=bool)
    mask[20:29, 28:37] = True
    transform = np.eye(4)
    transform[:3, 3] = [3, 2, 5]
    estimate = lift_mask_surface(decode_depth_bgra(raw), mask, intrinsics, transform,
                                 depth_mode="axial")
    observation = Observation("rsu-1", 7, 1, 2_000_000, ObjectClass.PEDESTRIAN,
                              estimate.xyz_m, PositionAnchor.SURFACE,
                              confidence=0.9)
    # No calibrated position std is asserted from a synthetic surface spread.
    payload = encode(observation)
    transport = ReferenceTransport(("rsu-1", "ego"), delay_us=100_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        _write_event(handle, "run.started", 0, mode="synthetic", seed=0,
                     depth_mode="axial", observation_ttl_us=500_000,
                     note="Synthetic mask/depth; no CARLA, SAM or Muse calls")
        _write_event(handle, "observation.created", 2_000_000,
                     observation=asdict(observation), surface_support=estimate.supporting_pixels)
        for frame, send_us in ((20, 2_000_000), (25, 2_500_000)):
            transport.send("rsu-1", payload, frame=frame, sim_us=send_us)
            _write_event(handle, "packet.sent", send_us, source="rsu-1", frame=frame, payload=payload)
            for delivery in transport.receive_until(send_us + 100_000):
                received = decode(delivery.payload, source=delivery.source)
                _write_event(handle, "packet.received", delivery.available_sim_us,
                             **asdict(delivery), fresh=received.is_fresh(delivery.available_sim_us, 500_000))
    return {"mode": "synthetic", "events": str(output), "packet_bytes": len(payload),
            "sent_bytes": transport.sent_bytes, "received_bytes": transport.received_bytes,
            "lifted_surface_xyz_m": observation.xyz_m,
            "replayed_deliveries": replay_packets(output), "paid_calls": 0}


def replay_packets(path: Path) -> list[dict]:
    """Decode received packets at logged availability; preserve capture age."""
    deliveries = []
    last_time = -1
    ttl_us = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("version") != 1 or event["sim_us"] < last_time:
                raise ValueError("unsupported event version or backwards event clock")
            last_time = event["sim_us"]
            if event["kind"] == "run.started":
                ttl_us = event["observation_ttl_us"]
            if event["kind"] == "packet.received":
                if ttl_us is None or event["available_sim_us"] != event["sim_us"]:
                    raise ValueError("missing run metadata or invalid availability")
                observation = decode(event["payload"], source=event["source"])
                deliveries.append({"source": observation.source, "object_id": observation.object_id,
                                   "captured_sim_us": observation.captured_sim_us,
                                   "available_sim_us": event["sim_us"],
                                   "fresh": observation.is_fresh(event["sim_us"], ttl_us)})
    return deliveries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("synthetic", help="Exercise depth -> objects -> packets -> delivery/replay")
    smoke.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are preserved")
    replay = sub.add_parser("replay-packets", help="Decode received packets from a synthetic event log")
    replay.add_argument("events", type=Path)
    s1 = sub.add_parser("s1", help="S1: signed observations through packet transport to ego")
    s1.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are preserved")
    s1.add_argument("--seed", type=int, default=0)
    s1.add_argument("--loss", type=float, default=0.1)
    s1.add_argument("--delay-us", type=int, default=100_000)
    s1.add_argument("--jitter-us", type=int, default=50_000)
    s1.add_argument("--disconnect-at-us", type=int, default=None)
    s1.add_argument("--key-seed", type=int, default=None)
    s1.add_argument("--overwrite", action="store_true",
                    help="Replace an existing output file")
    s2 = sub.add_parser("s2", help="S2: LiDAR coverage vs camera ghosts")
    s2.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are preserved")
    s2.add_argument("--seed", type=int, default=0)
    s2.add_argument("--loss", type=float, default=0.0)
    s2.add_argument("--delay-us", type=int, default=100_000)
    s2.add_argument("--jitter-us", type=int, default=0)
    s2.add_argument("--key-seed", type=int, default=0)
    s2.add_argument("--overwrite", action="store_true",
                    help="Replace an existing output file")
    s3 = sub.add_parser("s3", help="S3: one dataset frame through SAM and the Reasoner")
    s3.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are preserved")
    s3.add_argument("--overwrite", action="store_true",
                    help="Replace an existing output file")
    s3.add_argument("--offline", action="store_true",
                    help="Use the disk cache only; refuse uncached Meta calls")
    s3.add_argument("--key-seed", type=int, default=0)
    s3.add_argument("--budget-usd", type=float, default=3.0)
    evaluate = sub.add_parser("eval", help="S4a: fusion x attack metrics, Muse off")
    evaluate.add_argument("--scene", default="frame-000096")
    evaluate.add_argument("--fusion", default="witness",
                          choices=("ego", "union", "geo", "witness", "all"))
    evaluate.add_argument("--attack", default=None)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--offline", action="store_true",
                          help="SAM cache only; a miss is reported, not billed")
    evaluate.add_argument("--perception", default="sam", choices=("sam", "gt-oracle"),
                          help="sam is the headline; gt-oracle is the labelled comparison")
    evaluate.add_argument("--budget-usd", type=float, default=2.0)
    evaluate.add_argument("--output-dir", type=Path, default=Path("data/results"))
    evaluate.add_argument("--summary", type=Path, default=Path("results/summary.md"))
    ledger = sub.add_parser("ledger", help="Settled spend vs outstanding reservations")
    ledger.add_argument("--show", action="store_true")
    ledger.add_argument("--release", default=None, help="Reservation id to release")
    ledger.add_argument("--release-uncertain", action="store_true",
                        help="Release every outstanding reservation")
    ledger.add_argument("--confirm", action="store_true",
                        help="Required to append a release row")
    ledger.add_argument("--reason", default="")
    ledger.add_argument("--ledger", type=Path, default=Path("data/cost-log.jsonl"))
    args = parser.parse_args()
    try:
        if args.command == "synthetic":
            result = synthetic(args.output)
        elif args.command == "replay-packets":
            result = replay_packets(args.events)
        elif args.command == "s2":
            from coopx.pipeline import run_s2
            result = run_s2(args.output, seed=args.seed, loss=args.loss,
                            delay_us=args.delay_us, jitter_us=args.jitter_us,
                            key_seed=args.key_seed, overwrite=args.overwrite)
        elif args.command == "s3":
            from coopx.pipeline import run_s3
            result = run_s3(args.output, overwrite=args.overwrite, offline=args.offline,
                            key_seed=args.key_seed, budget_usd=args.budget_usd)
        elif args.command == "eval":
            from coopx.eval.report import write_summary
            from coopx.eval.runner import run_eval
            result = run_eval(scene=args.scene, fusion=args.fusion, attack=args.attack,
                              seed=args.seed, offline=args.offline, budget_usd=args.budget_usd,
                              out_dir=args.output_dir, perception=args.perception)
            write_summary(args.output_dir, args.summary)
            result["summary"] = str(args.summary)
        elif args.command == "ledger":
            from coopx.meta.client import MetaClient
            client = MetaClient(cost_log=args.ledger, budget_usd=1.0, environ={}, offline=True)
            if args.release or args.release_uncertain:
                result = client.release_reservations(
                    rid=args.release, uncertain=args.release_uncertain,
                    confirm=args.confirm, reason=args.reason)
            else:
                result = client.ledger_status()
        else:
            from coopx.pipeline import run_s1
            result = run_s1(args.output, seed=args.seed, loss=args.loss,
                            delay_us=args.delay_us, jitter_us=args.jitter_us,
                            disconnect_at_us=args.disconnect_at_us,
                            key_seed=args.key_seed, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"coopx: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
