import math
import unittest
from dataclasses import replace

from test_coverage import Fixture

from coopx.coverage import CoverageMap
from coopx.envelope import decode, encode, sign, verify
from coopx.eval.fusion import MAX_PREDICTION_US, fuse
from coopx.observers.tracker import CameraTracker, Measurement


def measurement(x, y=0, cls=1, std=0.1):
    return Measurement((float(x), float(y), 1.0), (std, std, std), cls, 0)


class TrackerTests(unittest.TestCase):
    def test_stable_ids_velocity_and_uncertainty_from_past_samples(self):
        tracker = CameraTracker()
        for i in range(5):
            result = tracker.update(
                [measurement(i * 0.75)],
                capture_us=i * 500_000,
                available_us=i * 500_000 + 2_800_000,
            )[0]
            self.assertEqual(result.track_id, 200)
            if i:
                self.assertAlmostEqual(result.velocity_mps[0], 1.5)
                self.assertGreaterEqual(result.velocity_std_mps[0], 0.1)
            else:
                self.assertIsNone(result.velocity_mps)

    def test_assignment_is_one_to_one_and_batch_order_does_not_rename_tracks(self):
        tracker = CameraTracker()
        a, b = tracker.update(
            [measurement(0), measurement(10)], capture_us=0, available_us=1
        )
        later = tracker.update(
            [measurement(10.5), measurement(0.5)],
            capture_us=500_000,
            available_us=500_001,
        )
        self.assertEqual([t.track_id for t in later], [b.track_id, a.track_id])

    def test_class_label_and_anchor_cannot_borrow_motion(self):
        tracker = CameraTracker()
        first = tracker.update([measurement(0)], capture_us=0, available_us=1)[0]
        later = tracker.update(
            [measurement(0.1, cls=2)], capture_us=100_000, available_us=100_001
        )[0]
        self.assertNotEqual(first.track_id, later.track_id)
        self.assertIsNone(later.velocity_mps)

    def test_expired_track_is_not_reused(self):
        tracker = CameraTracker()
        first = tracker.update([measurement(0)], capture_us=0, available_us=1)[0]
        later = tracker.update(
            [measurement(0)], capture_us=2_000_001, available_us=2_000_002
        )[0]
        self.assertNotEqual(first.track_id, later.track_id)

    def test_future_capture_and_unavailable_results_cannot_influence_velocity(self):
        tracker = CameraTracker()
        tracker.update([measurement(0)], capture_us=0, available_us=4_000_000)
        early = tracker.update(
            [measurement(1)], capture_us=500_000, available_us=1_000_000
        )[0]
        self.assertIsNone(early.velocity_mps)
        tracker.update([measurement(90)], capture_us=5_000_000, available_us=6_000_000)
        # A late completion for an earlier capture cannot see the future sample.
        old = tracker.update(
            [measurement(1.5)], capture_us=1_000_000, available_us=6_000_000
        )[0]
        if old.velocity_mps is not None:
            self.assertLess(math.hypot(*old.velocity_mps), 8)

    def test_noisy_measurements_increase_uncertainty(self):
        def fit(std):
            tracker = CameraTracker()
            tracker.update([measurement(0, std=std)], capture_us=0, available_us=1)
            return tracker.update(
                [measurement(1, std=std)], capture_us=1_000_000, available_us=1_000_001
            )[0]

        self.assertGreater(fit(1).velocity_std_mps[0], fit(0.1).velocity_std_mps[0])


class MotionEnvelopeTests(unittest.TestCase):
    def test_extension_is_signed_roundtrips_and_preserves_legacy(self):
        f = Fixture()
        old = f.observation()
        raw = encode(old)
        self.assertEqual(encode(decode(raw)), raw)
        moving = sign(
            replace(
                old,
                observation=replace(old.observation, velocity_mps=(1.0, 0.0, 0.0)),
                velocity_std_mps=(0.2, 0.3, 0.4),
            ),
            f.keys["pole"],
        )
        received = decode(encode(moving))
        self.assertEqual(received.velocity_std_mps, (0.2, 0.3, 0.4))
        self.assertTrue(verify(received, f.anchors))
        self.assertEqual(len(encode(moving)), len(raw) + 24)
        self.assertFalse(
            verify(replace(received, velocity_std_mps=(0.1, 0.3, 0.4)), f.anchors)
        )
        tampered = bytearray(encode(moving))
        tampered[-65] ^= 1
        self.assertFalse(verify(decode(bytes(tampered)), f.anchors))

    def test_uncertainty_requires_velocity_and_finite_nonnegative_axes(self):
        f = Fixture()
        message = f.observation()
        with self.assertRaises(ValueError):
            replace(message, velocity_std_mps=(1.0, 1.0, 1.0))
        message = replace(
            message,
            observation=replace(message.observation, velocity_mps=(0.0, 0.0, 0.0)),
        )
        for std in (
            (-1.0, 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            (float("inf"), 0.0, 0.0),
            (1.0, 1.0),
        ):
            with self.assertRaises(ValueError):
                replace(message, velocity_std_mps=std)


class PredictionTests(unittest.TestCase):
    def setup(self):
        f = Fixture()
        grid = CoverageMap(f.spec, ttl_us=5_000_000, store=f.store)
        return f, grid

    def moving(self, f, **kwargs):
        m = f.observation(expiry=10_000_000, **kwargs)
        return sign(
            replace(
                m,
                observation=replace(m.observation, velocity_mps=(1.0, 0.0, 0.0)),
                velocity_std_mps=(0.2, 0.2, 0.2),
            ),
            f.keys[m.envelope.source_id],
        )

    def call(self, f, grid, m, now, *, memory=None, local=False, tracking=True):
        return fuse(
            "witness",
            ego_observations=[m] if local else [],
            remote_admitted=[] if local else [m],
            store=f.store,
            coverage_map=grid,
            assessor=None,
            now_us=now,
            memory=memory,
            tracking=tracking,
        )

    def test_prediction_uses_capture_time_and_grows_uncertainty_without_altering_signature(
        self,
    ):
        f, grid = self.setup()
        m = self.moving(f, std=(0.1, 0.1, 0.1))
        wire = encode(m)
        a = self.call(f, grid, m, 2_000_000)[0]
        b = self.call(f, grid, m, 3_000_000)[0]
        self.assertAlmostEqual(a.xy[0], 4.5, delta=0.01)
        self.assertEqual(a.xy[1], 1.5)
        self.assertAlmostEqual(b.xy[0], 5.5, delta=0.01)
        self.assertEqual(b.xy[1], 1.5)
        self.assertGreater(b.position_std_m[0], a.position_std_m[0])
        self.assertEqual(a.track_id, m.observation.object_id)
        self.assertEqual(encode(m), wire)

    def test_horizon_is_exclusive_and_local_prediction_is_retained(self):
        f, grid = self.setup()
        m = self.moving(f, source="ego")
        self.assertTrue(
            self.call(f, grid, m, 1_000_000 + MAX_PREDICTION_US - 1, local=True)
        )
        self.assertEqual(
            self.call(f, grid, m, 1_000_000 + MAX_PREDICTION_US, local=True), []
        )

    def test_delayed_moving_ghost_is_checked_at_capture_not_predicted_position(self):
        f, grid = self.setup()
        m = self.moving(f, source="attacker")
        old = f.coverage(free={(3, 1)}, expiry=10_000_000)
        grid.add(old, 1_000_000)
        # New scan supersedes old free space in the live map.
        grid.add(
            f.coverage(occupied={(3, 1)}, capture=3_000_000, expiry=10_000_000),
            3_000_000,
        )
        memory = {"coverage_history": [old]}
        belief = self.call(f, grid, m, 3_000_000, memory=memory)[0]
        self.assertAlmostEqual(belief.xy[0], 5.5, delta=0.01)
        self.assertEqual(belief.xy[1], 1.5)
        self.assertEqual(belief.state, "contradicted")

    def test_refutation_survives_new_frame_and_cannot_poison_local_track(self):
        f, grid = self.setup()
        m = self.moving(f, source="attacker")
        grid.add(f.coverage(free={(3, 1)}, expiry=10_000_000), 1_000_000)
        memory = {}
        self.assertEqual(
            self.call(f, grid, m, 1_000_000, memory=memory)[0].state, "contradicted"
        )
        update = self.moving(
            f, source="attacker", capture=2_000_000, xyz=(10.5, 1.5, 1.0)
        )
        self.assertEqual(
            self.call(f, grid, update, 2_000_000, memory=memory)[0].state,
            "contradicted",
        )
        local = self.moving(f, source="ego", capture=2_000_000, xyz=(10.5, 1.5, 1.0))
        self.assertEqual(
            self.call(f, grid, local, 2_000_000, memory=memory, local=True)[0].state,
            "confirmed",
        )

    def test_untracked_ablation_uses_unpredicted_position(self):
        f, grid = self.setup()
        m = self.moving(f)
        b = self.call(f, grid, m, 3_000_000, tracking=False)[0]
        self.assertEqual(b.xy, (3.5, 1.5))
        self.assertIsNone(b.track_id)


class CameraTrackingIntegrationTests(unittest.TestCase):
    def test_two_frames_sign_stable_identity_velocity_and_uncertainty(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from coopx.domain import PositionAnchor
        from coopx.meta.sam import Detection
        from coopx.observers.camera import CameraFrame, CameraObserver
        from coopx.observers.camera_lift import Lifted

        f = Fixture()
        holder = {}
        frame = lambda: holder["frame"]

        def lift(*args, **kwargs):
            return Lifted(
                (holder["x"], 1.0, 1.0),
                (0.5, 0.5, 1.7),
                (0.1, 0.1, 0.1),
                8,
                PositionAnchor.SURFACE,
            )

        observer = CameraObserver(
            "pole", frame, "cam1", None, lift, f.keys["pole"], ttl_us=5_000_000
        )
        with (
            patch("coopx.observers.camera.segment") as segment,
            patch("coopx.observers.camera.model_call_event", return_value={}),
        ):
            output = []
            for i in range(3):
                holder["x"] = 3.0 + i * 0.5
                holder["frame"] = CameraFrame(
                    __import__("test_camera_observer")._tiny_png(),
                    (32, 32),
                    i * 500_000,
                    None,
                    "cam1",
                )
                segment.return_value = (
                    [Detection((0.0, 0.0, 10.0, 10.0), None, None)],
                    SimpleNamespace(availability_time_us=i * 500_000 + 2_800_000),
                )
                output.extend(observer.observe())
        self.assertEqual(len({m.observation.object_id for m in output}), 1)
        self.assertIsNone(output[0].observation.velocity_mps)
        for m in output[1:]:
            received = decode(encode(m))
            self.assertTrue(verify(received, f.anchors))
            self.assertAlmostEqual(received.observation.velocity_mps[0], 1.0)
            self.assertIsNotNone(received.velocity_std_mps)
            self.assertEqual(
                m.lineage.origin_capture_time_us, m.envelope.capture_time_us
            )

    def test_box_overlap_helps_when_depth_uncertainty_is_large(self):
        tracker = CameraTracker()

        def m(x, box):
            return replace(measurement(x, std=10.0), box_xyxy=box)

        first = tracker.update(
            [m(0, (0, 0, 10, 20)), m(0.5, (20, 0, 30, 20))],
            capture_us=0,
            available_us=1,
        )
        # Depth order reverses; image boxes still distinguish these people.
        later = tracker.update(
            [m(0.4, (0, 0, 10, 20)), m(0.1, (20, 0, 30, 20))],
            capture_us=500_000,
            available_us=500_001,
        )
        self.assertEqual([r.track_id for r in later], [r.track_id for r in first])

    def test_bad_depth_sample_gets_less_weight_in_velocity_fit(self):
        tracker = CameraTracker()
        for i, (x, std) in enumerate(((0, 0.1), (0.5, 0.1), (4.0, 50.0), (1.5, 0.1))):
            result = tracker.update(
                [measurement(x, std=std)],
                capture_us=i * 500_000,
                available_us=i * 500_000 + 1,
            )[0]
        self.assertAlmostEqual(result.velocity_mps[0], 1.0, places=3)


class TrackerBoundsTests(unittest.TestCase):
    def test_association_and_signed_fit_share_the_two_second_history_bound(self):
        from unittest.mock import patch

        tracker = CameraTracker()
        for i in range(6):
            tracker.update(
                [measurement(i * 0.5)],
                capture_us=i * 500_000,
                available_us=i * 500_000 + 1,
            )
        original = CameraTracker._fit
        with patch.object(CameraTracker, "_fit", wraps=original) as fit:
            tracker.update(
                [measurement(3.0)], capture_us=3_000_000, available_us=3_000_001
            )
        self.assertTrue(fit.call_args_list)
        for call in fit.call_args_list:
            self.assertTrue(all(3_000_000 - h[0] <= 2_000_000 for h in call.args[0]))

    def test_invalid_samples_and_times_fail_before_changing_tracks(self):
        tracker = CameraTracker()
        for args in ((-1, 1), (2, 1), (0, 2**64)):
            with self.assertRaises(ValueError):
                tracker.update(
                    [measurement(0)], capture_us=args[0], available_us=args[1]
                )
        for std in ((-1.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)):
            with self.assertRaises(ValueError):
                replace(measurement(0), std=std)
        self.assertEqual(
            tracker.update([measurement(0)], capture_us=0, available_us=1)[0].track_id,
            200,
        )


class AvailabilityTests(unittest.TestCase):
    setup = PredictionTests.setup
    moving = PredictionTests.moving
    call = PredictionTests.call

    def test_capture_does_not_make_an_unfinished_model_result_available(self):
        f, grid = self.setup()
        m = self.moving(f)
        unfinished = sign(
            replace(m, envelope=replace(m.envelope, generation_time_us=3_000_000)),
            f.keys["pole"],
        )
        self.assertEqual(self.call(f, grid, unfinished, 2_000_000), [])
        self.assertTrue(self.call(f, grid, unfinished, 3_000_000))

    def test_unavailable_range_report_cannot_refute_an_observation(self):
        f, grid = self.setup()
        m = self.moving(f, source="attacker")
        coverage = f.coverage(free={(3, 1)}, expiry=10_000_000)
        unfinished = sign(
            replace(
                coverage,
                envelope=replace(coverage.envelope, generation_time_us=3_000_000),
            ),
            f.keys["lidar"],
        )
        memory = {"coverage_history": [unfinished]}
        result = self.call(f, grid, m, 2_000_000, memory=memory)[0]
        self.assertEqual(result.state, "unverified")
        result = self.call(f, grid, m, 3_000_000, memory=memory)[0]
        self.assertEqual(result.state, "contradicted")


class AssignmentQualityTests(unittest.TestCase):
    def test_assignment_is_global_and_can_leave_a_detection_unmatched(self):
        from coopx.observers.tracker import _minimum_assignment

        result = _minimum_assignment([[1.0, 2.0, 5.5, 5.5], [1.1, 100.0, 5.5, 5.5]])
        self.assertEqual(result, {0: 1, 1: 0})
        self.assertEqual(_minimum_assignment([[100.0, 5.5]])[0], 1)

    def test_unrelated_image_region_cannot_inherit_a_track(self):
        tracker = CameraTracker()
        old = replace(measurement(0, std=50.0), box_xyxy=(0.0, 0.0, 10.0, 20.0))
        new = replace(measurement(0.1, std=50.0), box_xyxy=(100.0, 0.0, 110.0, 20.0))
        first = tracker.update([old], capture_us=0, available_us=1)[0]
        second = tracker.update([new], capture_us=500_000, available_us=500_001)[0]
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertIsNone(second.velocity_mps)

    def test_different_appearance_does_not_inherit_motion(self):
        tracker = CameraTracker()
        old = replace(measurement(0), appearance=(1.0, 0.0))
        new = replace(measurement(0.1), appearance=(0.0, 1.0))
        first = tracker.update([old], capture_us=0, available_us=1)[0]
        second = tracker.update([new], capture_us=500_000, available_us=500_001)[0]
        self.assertNotEqual(first.track_id, second.track_id)

    def test_slow_older_completion_keeps_the_known_track_without_future_velocity(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from test_camera_observer import _tiny_png

        from coopx.domain import PositionAnchor
        from coopx.meta.sam import Detection
        from coopx.observers.camera import CameraFrame, CameraObserver
        from coopx.observers.camera_lift import Lifted

        f = Fixture()
        observer = CameraObserver(
            "pole",
            lambda: None,
            "cam1",
            None,
            lambda *_args, **_kw: Lifted(
                (3.0, 1.0, 1.0),
                (0.5, 0.5, 1.7),
                (0.1, 0.1, 0.1),
                8,
                PositionAnchor.SURFACE,
            ),
            f.keys["pole"],
        )
        detection = Detection((0.0, 0.0, 10.0, 10.0), None, None)
        with patch("coopx.observers.camera.model_call_event", return_value={}):
            new, _ = observer.publish_result(
                CameraFrame(_tiny_png(), (32, 32), 500_000, None, "cam1"),
                "person",
                [detection],
                SimpleNamespace(availability_time_us=1_000_000),
            )
            old, _ = observer.publish_result(
                CameraFrame(_tiny_png(), (32, 32), 0, None, "cam1"),
                "person",
                [detection],
                SimpleNamespace(availability_time_us=2_000_000),
            )
        self.assertEqual(len(new), 1)
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0].observation.object_id, new[0].observation.object_id)
        self.assertIsNone(old[0].observation.velocity_mps)
        self.assertEqual(len(observer.published), 2)


class ReplayCompletionOrderTests(unittest.TestCase):
    def test_results_are_tracked_at_availability_not_capture_iteration_order(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from test_camera_observer import _tiny_png

        from coopx.domain import PositionAnchor
        from coopx.eval.runner import _Replay
        from coopx.meta.sam import Detection
        from coopx.observers.camera import CameraFrame
        from coopx.observers.camera_lift import Lifted

        f = Fixture()
        r = _Replay.__new__(_Replay)
        r.perception = "sam"
        r.tracking = True
        r.client = object()
        r.endpoints = [("pole", "-1", "cam1")]
        r.anchors = lambda: (f.keys, f.anchors)
        r.perception_frames = lambda: ("0", "1", "2")
        r.read = {
            ("-1", str(i)): SimpleNamespace(
                cameras={"cam1": None}, nominal_time_us=i * 500_000
            )
            for i in range(3)
        }
        r.errors = []
        r.model_latency_ms = []
        r.provenance = {}

        def camera_frame(frame, cam):
            return CameraFrame(_tiny_png(), (32, 32), frame.nominal_time_us, frame, cam)

        def lift(box, frame, cam, **kwargs):
            return Lifted(
                (3.0 + frame.nominal_time_us / 1e6, 1.0, 1.0),
                (0.5, 0.5, 1.7),
                (0.1, 0.1, 0.1),
                8,
                PositionAnchor.SURFACE,
            )

        responses = [
            (
                [Detection((0.0, 0.0, 10.0, 10.0), None, None)],
                SimpleNamespace(availability_time_us=t, latency_ms=1000.0),
            )
            for t in (3_000_000, 1_000_000, 2_000_000)
        ]
        with (
            patch(
                "coopx.observers.camera.jpeg_to_camera_frame", side_effect=camera_frame
            ),
            patch("coopx.observers.camera_lift.lift_box", side_effect=lift),
            patch("coopx.meta.sam.segment", side_effect=responses),
            patch("coopx.observers.camera.model_call_event", return_value={}),
        ):
            messages = r._perceive()
        self.assertEqual(
            [m.envelope.capture_time_us for _, m in messages], [500_000, 1_000_000, 0]
        )
        self.assertEqual(len({m.observation.object_id for _, m in messages}), 1)
        self.assertAlmostEqual(messages[1][1].observation.velocity_mps[0], 1.0)
        self.assertIsNone(messages[-1][1].observation.velocity_mps)
        self.assertEqual(len(r.model_latency_ms), 3)


class LatePositiveTests(unittest.TestCase):
    def test_newer_empty_result_is_not_evidence_against_a_late_positive(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from test_camera_observer import _tiny_png

        from coopx.domain import PositionAnchor
        from coopx.meta.sam import Detection
        from coopx.observers.camera import CameraFrame, CameraObserver
        from coopx.observers.camera_lift import Lifted

        f = Fixture()
        observer = CameraObserver(
            "pole",
            lambda: None,
            "cam1",
            None,
            lambda *_args, **_kw: Lifted(
                (3.0, 1.0, 1.0),
                (0.5, 0.5, 1.7),
                (0.1, 0.1, 0.1),
                8,
                PositionAnchor.SURFACE,
            ),
            f.keys["pole"],
        )
        with patch("coopx.observers.camera.model_call_event", return_value={}):
            observer.publish_result(
                CameraFrame(_tiny_png(), (32, 32), 500_000, None, "cam1"),
                "person",
                [],
                SimpleNamespace(availability_time_us=1_000_000),
            )
            old, _ = observer.publish_result(
                CameraFrame(_tiny_png(), (32, 32), 0, None, "cam1"),
                "person",
                [Detection((0.0, 0.0, 10.0, 10.0), None, None)],
                SimpleNamespace(availability_time_us=2_000_000),
            )
        self.assertEqual(len(old), 1)
        self.assertIsNone(old[0].observation.velocity_mps)


class OutOfSequenceAndRefutationTests(unittest.TestCase):
    def test_known_newer_capture_cannot_influence_old_signed_velocity(self):
        tracker = CameraTracker()
        first = tracker.update([measurement(0.0)], capture_us=0, available_us=1)[0]
        newer = tracker.update(
            [measurement(1.0)], capture_us=1_000_000, available_us=1_100_000
        )[0]
        old = tracker.update(
            [measurement(0.45)], capture_us=750_000, available_us=2_000_000
        )[0]
        self.assertEqual(old.track_id, first.track_id)
        self.assertAlmostEqual(old.velocity_mps[0], 0.6)
        self.assertAlmostEqual(newer.velocity_mps[0], 1.0)

    def test_renaming_sensor_cannot_reset_same_source_track_refutation(self):
        helper = PredictionTests()
        f, grid = helper.setup()
        memory = {}
        ghost = helper.moving(f, source="attacker")
        grid.add(f.coverage(free={(3, 1)}, expiry=10_000_000), 1_000_000)
        self.assertEqual(
            helper.call(f, grid, ghost, 1_000_000, memory=memory)[0].state,
            "contradicted",
        )
        update = helper.moving(
            f, source="attacker", capture=2_000_000, xyz=(10.5, 1.5, 1.0)
        )
        update = sign(
            replace(update, lineage=replace(update.lineage, origin_sensor="new-name")),
            f.keys["attacker"],
        )
        after = helper.call(f, grid, update, 2_000_000, memory=memory)[0]
        self.assertEqual(after.state, "contradicted")
        from coopx.envelope import evidence_ref

        self.assertIn(evidence_ref(ghost), after.refs)
