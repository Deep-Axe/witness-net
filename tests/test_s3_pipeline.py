"""S3 pipeline: offline, recorded SAM fixture strings. No network."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from coopx import events as event_schema
from coopx.dataset.v2xreal import V2XReal, camera_image_size, project_camera
from coopx.meta.client import MetaClient
from coopx.pipeline import pick_s3_frame, run_s3


class _FakePost:
    def __init__(self, sequence):
        self.sequence = list(sequence)

    def __call__(self, url, headers, body, timeout):
        item = self.sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

DATASET = Path("/home/deep/meta_hackathon/data/v2x-real/val")


def _sam_text_for_gt(pick) -> str:
    dataset = V2XReal(DATASET)
    rsu = dataset.read(pick.scenario, pick.rsu_id, pick.frame_id)
    box = rsu.boxes[pick.pedestrian_id]
    camera = rsu.cameras[pick.camera_name]
    width, height = camera_image_size(camera)
    uv, _depth, valid = project_camera(
        rsu.to_camera(box.corners_site(), pick.camera_name), camera.intrinsic)
    xs, ys = uv[valid, 0], uv[valid, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}|>"


def _payload(text):
    return 200, {
        "text": text,
        "usage": {
            "input_tokens": 0, "output_tokens": 0, "images_processed": 1,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


@unittest.skipUnless(DATASET.is_dir(), "V2X-Real val set is not present")
class S3PipelineTests(unittest.TestCase):
    def test_offline_run_validates_and_orders_s3_events(self):
        pick = pick_s3_frame(DATASET)
        text = _sam_text_for_gt(pick)
        with tempfile.TemporaryDirectory() as tmp:
            post = _FakePost([_payload(text), _payload(text)])
            client = MetaClient(
                cache_dir=Path(tmp) / "cache", cost_log=Path(tmp) / "cost.jsonl",
                post=post, environ={"MODEL_API_KEY": "fixture-key-not-secret"},
                clock=iter([0.0, 0.4, 1.0, 1.4]).__next__)
            path = Path(tmp) / "s3.jsonl"
            overlay = Path(tmp) / "overlay.png"
            summary = run_s3(path, client=client, overwrite=True, overlay=overlay,
                             dataset_root=DATASET, delay_us=50_000)
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                      if line]
            for event in events:
                event_schema.validate(event)
            counts = Counter(e["type"] for e in events)
            self.assertEqual(summary["events_by_type"], dict(counts))
            self.assertEqual(summary["model_call_count"], counts.get("model_call", 0))
            self.assertEqual(summary["frame_id"], pick.frame_id)
            self.assertEqual(summary["pedestrian_id"], pick.pedestrian_id)
            self.assertGreater(counts.get("observation_published", 0), 0)
            self.assertGreater(counts.get("query", 0), 0)
            self.assertGreater(counts.get("answer", 0), 0)
            self.assertGreater(counts.get("verdict", 0), 0)
            self.assertGreater(summary["accepted"], 0)

            published_refs = {}
            seen_query = {}
            spark_avail = []
            first_check = None
            for event in events:
                if event["type"] in ("observation_published", "coverage_published"):
                    published_refs[event["evidence_ref"]] = event["t_replay_us"]
                if event["type"] == "query":
                    seen_query[event["message_id"]] = event["t_replay_us"]
                if event["type"] == "answer":
                    self.assertGreaterEqual(event["t_replay_us"], seen_query[event["message_id"]])
                    for ref in event["detail"]["evidence_refs"]:
                        self.assertIn(ref, published_refs)
                        self.assertLessEqual(published_refs[ref], event["t_replay_us"])
                if event["type"] == "verdict":
                    for call_id in event["detail"]["model_call_ids"]:
                        call = next(e for e in events if e["type"] == "model_call"
                                    and e["message_id"] == call_id)
                        self.assertGreaterEqual(event["t_replay_us"],
                                                call["detail"]["availability_time_us"])
                    for ref in event["detail"]["evidence_refs"]:
                        self.assertIn(ref, published_refs)
                        self.assertLessEqual(published_refs[ref], event["t_replay_us"])
                if (event["type"] == "model_call"
                        and event["detail"]["product"] == "muse-spark-1.3"):
                    spark_avail.append(event["detail"]["availability_time_us"])
                if event["type"] == "check" and event["state"] == "accepted" and first_check is None:
                    first_check = event["t_replay_us"]
            if spark_avail and first_check is not None:
                self.assertLessEqual(first_check, min(spark_avail))
            blob = (Path(tmp) / "cost.jsonl").read_text(encoding="utf-8") if (
                Path(tmp) / "cost.jsonl").is_file() else ""
            self.assertNotIn("fixture-key-not-secret", blob)
            self.assertNotIn("fixture-key-not-secret", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
