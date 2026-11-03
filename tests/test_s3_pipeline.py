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


class _FakeMeta:
    """Answers SAM with a recorded box token and Spark plan/judge from the request data.

    The plan also proposes an unknown observer, and the judge cites an invented
    ref: both must be rejected by the Reasoner's candidate-set validation.
    """

    def __init__(self, sam_text):
        self.sam_text = sam_text
        self.calls = []

    @staticmethod
    def _data(payload):
        text = payload["input"][0]["content"][0]["text"]
        return json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        fmt = (payload.get("text") or {}).get("format", {}).get("name")
        self.calls.append(fmt or payload["model"])
        if payload["model"] == "sam-3.1":
            return _payload(self.sam_text)
        data = self._data(payload)
        if fmt == "coopx_plan_v1":
            region = data["context"]["asked_region"]
            out = {"queries": [
                {"tier": "T3", "observer_id": "rsu-2", "center_xy": region["center"],
                 "radius_m": region["radius_m"], "concept": "person", "reason": "covers it"},
                {"tier": "T1", "observer_id": "rsu-9", "center_xy": region["center"],
                 "radius_m": 4.0, "concept": None, "reason": "invented observer"}]}
        else:
            ref = sorted(data["evidence"])[0]
            out = {"verdicts": [{"subject": data["evidence"][ref]["message_id"],
                                 "state": "corroborated",
                                 "evidence_refs": [ref, "f" * 64], "reason": "seen by rsu-2"}]}
        return _payload(json.dumps(out))

DATASET = Path(__file__).resolve().parents[1] / "data" / "v2x-real" / "val"


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
            post = _FakeMeta(text)
            ticks = iter(range(10_000))
            client = MetaClient(
                cache_dir=Path(tmp) / "cache", cost_log=Path(tmp) / "cost.jsonl",
                post=post, environ={"MODEL_API_KEY": "fixture-key-not-secret"},
                clock=lambda: next(ticks) * 0.4)
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
            # One SAM call (the T3 rerun is a cache hit), one plan, one judge.
            self.assertEqual(post.calls, ["sam-3.1", "coopx_plan_v1", "coopx_judge_v1"])
            self.assertEqual(summary["queries"], [
                {"observer_id": "rsu-2", "tier": "T3", "concept": "person"}])
            self.assertEqual(summary["answers"], ["detected"])
            # One observer cannot corroborate: the model's upgrade is refused.
            self.assertEqual([v["state"] for v in summary["verdicts"]], ["unverified"])
            codes = {r["code"] for r in summary["reasoner_rejections"]}
            self.assertEqual(codes, {"UNKNOWN_OBSERVER", "UNKNOWN_EVIDENCE",
                                     "SINGLE_SOURCE_CORROBORATION"})
            verdict = next(e for e in events if e["type"] == "verdict")
            self.assertEqual(len(verdict["detail"]["evidence_refs"]), 1)
            self.assertEqual(len(verdict["detail"]["model_call_ids"]), 2)
            hits = [e for e in events if e["type"] == "model_call" and e["detail"]["cache_hit"]]
            self.assertTrue(hits)
            self.assertTrue(all(e["detail"]["cost_usd"] == 0 and e["detail"]["latency_ms"] > 0
                                for e in hits))

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
