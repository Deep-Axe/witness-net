"""Camera observer wiring and frustum capability. Tests make no network calls."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from coopx.dataset.v2xreal import (V2XReal, camera_footprint_xy, point_in_polygon)
from coopx.domain import ObjectClass, PositionAnchor
from coopx.envelope import evidence_ref
from coopx.events import validate
from coopx.observers.camera import CameraFrame, CameraObserver
from coopx.observers.camera_lift import Lifted, lift_box
from coopx.query import Capability, Query
from coopx.meta.client import MetaClient
from coopx.pipeline import demo_node_key


class _FakePost:
    def __init__(self, sequence):
        self.sequence = list(sequence)

    def __call__(self, url, headers, body, timeout):
        item = self.sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _ok(text="hello", images=0):
    return 200, {
        "text": text,
        "usage": {
            "input_tokens": 0, "output_tokens": 0, "images_processed": images,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }

DATASET = Path(__file__).resolve().parents[1] / "data" / "v2x-real" / "val"
SCENARIO = "2023-04-04-14-28-53_45_0"


def _tiny_png():
    image = Image.new("RGB", (32, 32), (20, 20, 20))
    import io
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class CameraObserverTests(unittest.TestCase):
    def test_observe_emits_signed_observation_at_sam_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            post = _FakePost([_ok("<|box;x1=8;y1=8;x2=24;y2=24;w=32;h=32|>", images=1)])
            client = MetaClient(
                cache_dir=Path(tmp) / "c", cost_log=Path(tmp) / "cost.jsonl",
                post=post, environ={"MODEL_API_KEY": "k"},
                clock=iter([0.0, 0.25, 1.0, 1.0]).__next__)
            lifted = Lifted((3.0, 4.0, 1.0), (0.6, 0.6, 1.6), (0.1, 0.1, 0.1),
                            8, PositionAnchor.SURFACE)

            def lift_fn(*_args, **_kwargs):
                return lifted

            class DummyFrame:
                cameras = {"cam1": None}

            def source():
                return CameraFrame(_tiny_png(), (32, 32), 1_000_000, DummyFrame(), "cam1")

            observer = CameraObserver(
                "rsu-2", source, "cam1", client, lift_fn, demo_node_key(1, "rsu-2"),
                ttl_us=5_000_000)
            messages = observer.observe()
            events = observer.drain_events()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].observation.cls, ObjectClass.PEDESTRIAN)
            self.assertEqual(messages[0].envelope.generation_time_us,
                             messages[0].envelope.capture_time_us + 250_000)
            self.assertEqual(len(events), 1)
            validate(events[0])
            self.assertEqual(events[0]["type"], "model_call")
            self.assertEqual(events[0]["t_replay_us"], events[0]["detail"]["availability_time_us"])
            query = Query("q1", "T3", "rsu-2", (3.0, 4.0), 5.0, 1_250_000, "person", "look")
            cap = Capability("rsu-2", ("camera_rgb",), ("T0", "T1", "T2", "T3"), "open",
                             ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
            answer, at_us = observer.on_query(query, cap, 1_250_000)
            # Same frame, same concept: reuse the fast-path detections, no new
            # SAM call and no re-signed duplicates.
            self.assertEqual(at_us, 1_250_000)
            self.assertEqual(observer.drain_events(), [])
            self.assertEqual(len(observer.published), 1)
            self.assertEqual(answer.result, "detected")
            self.assertIn(evidence_ref(messages[0]), answer.evidence_refs)

    def test_new_concepts_are_found_and_stay_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = "<0f>0<|box;x1=8;y1=8;x2=24;y2=24;w=32;h=32|>"
            post = _FakePost([_ok(box, images=1), _ok(box, images=1)])
            client = MetaClient(cache_dir=Path(tmp) / "c", cost_log=Path(tmp) / "cost.jsonl",
                                post=post, environ={"MODEL_API_KEY": "k"},
                                clock=iter([0.0, 0.1, 1.0, 1.1]).__next__)
            spots = iter([(3.0, 4.0, 0.5), (7.0, 7.0, 0.5)])

            def lift_fn(*_args, **_kwargs):
                return Lifted(next(spots), (0.6, 0.6, 1.0), (0.1, 0.1, 0.1), 8,
                              PositionAnchor.SURFACE)

            class DummyFrame:
                cameras = {"cam1": None}

            observer = CameraObserver(
                "rsu-2", lambda: CameraFrame(_tiny_png(), (32, 32), 1_000_000, DummyFrame(), "cam1"),
                "cam1", client, lift_fn, demo_node_key(1, "rsu-2"), ttl_us=5_000_000,
                concepts=())
            cap = Capability("rsu-2", ("camera_rgb",), ("T3",), "open",
                             ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
            trash, _ = observer.on_query(
                Query("q1", "T3", "rsu-2", (3.0, 4.0), 1.0, 1_000_000, "trash can", "x"),
                cap, 1_000_000)
            cone, _ = observer.on_query(
                Query("q2", "T3", "rsu-2", (3.0, 4.0), 1.0, 1_000_000, "traffic cone", "x"),
                cap, 1_000_000)
            self.assertEqual(trash.result, "detected")
            # The cone was found at (7, 7); the trash can at (3, 4) is not a cone.
            self.assertEqual(cone.result, "not_detected")
            labels = {m.label_id for m in observer.published}
            self.assertEqual(len(labels), 2)
            self.assertEqual(sorted(observer.labels.values()), ["traffic cone", "trash can"])

    def test_lift_needs_min_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "scene" / "1"
            directory.mkdir(parents=True)
            extrinsic = [[0, 0, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
            metadata = dict(lidar_pose=[0, 0, 0, 0, 0, 0], vehicles={}, cam1=dict(
                intrinsic=[[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                extrinsic=extrinsic, cords=[0] * 6))
            (directory / "000002.yaml").write_text(yaml.safe_dump(metadata))
            np.array([[11, -1, -2, 12]], dtype="<f4").tofile(directory / "000002.bin")
            Image.new("RGB", (100, 80)).save(directory / "000002_cam1.jpeg")
            frame = V2XReal(root).read("scene", "1", "000002")
            self.assertIsNone(lift_box((0, 0, 10, 10), frame, "cam1", min_points=5))


@unittest.skipUnless(DATASET.is_dir(), "V2X-Real val set is not present")
class CameraCapabilityTests(unittest.TestCase):
    def test_chosen_pedestrian_inside_rsu2_outside_rsu1(self):
        from coopx.pipeline import pick_s3_frame
        pick = pick_s3_frame(DATASET)
        dataset = V2XReal(DATASET)
        rsu = dataset.read(pick.scenario, pick.rsu_id, pick.frame_id)
        other = dataset.read(pick.scenario, pick.other_rsu_id, pick.frame_id)
        pedestrian = rsu.boxes[pick.pedestrian_id]
        xy = tuple(float(v) for v in pedestrian.center_site()[:2])
        poly2 = camera_footprint_xy(rsu, pick.camera_name)
        self.assertTrue(point_in_polygon(xy, poly2),
                        f"pedestrian {pick.pedestrian_id} not in rsu-2 {pick.camera_name}")
        outside = False
        used = None
        for name in sorted(other.cameras):
            if not point_in_polygon(xy, camera_footprint_xy(other, name)):
                outside = True
                used = name
                break
        self.assertTrue(outside, "no rsu-1 camera footprint excludes the pedestrian")
        self.assertIsNotNone(used)


if __name__ == "__main__":
    unittest.main()
