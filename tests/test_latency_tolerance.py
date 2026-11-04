import unittest

from coopx.eval.metrics import _matches, capture_identity, score_frame


def target(x, identity="p"):
    return {"id": identity, "xy": [x, 0.0], "label": "Pedestrian"}


def belief(x):
    return {"xy": [x, 0.0], "cls": "PEDESTRIAN", "state": "unverified"}


class LatencyToleranceTests(unittest.TestCase):
    def setUp(self):
        self.now = 4_000_000
        self.history = [(i * 1_000_000, [target(i * 2.0)]) for i in range(5)]
        self.gt = [target(8.0)]

    def score(self, beliefs, tau=0, history=None):
        return score_frame(
            beliefs,
            self.gt,
            center=(0.0, 0.0),
            latency_tolerance_s=tau,
            now_us=self.now,
            gt_history=self.history if history is None else history,
        )

    def test_zero_is_strict_default_and_fixed_radius_does_not_expand(self):
        self.assertEqual(self.score([belief(2.0)])["false_hazards"], 1)
        self.assertEqual(self.score([belief(2.0)], 2)["false_hazards"], 1)
        self.assertEqual(self.score([belief(2.0)], 3)["true_positives"], 1)
        self.assertFalse(_matches(belief(2.5), target(4.1)))

    def test_tolerance_applies_to_duplicates_and_false_hazards_not_only_recall(self):
        strict = self.score([belief(2.0), belief(2.1), belief(-20)])
        relaxed = self.score([belief(2.0), belief(2.1), belief(-20)], 3)
        self.assertEqual(
            (strict["true_positives"], strict["duplicates"], strict["false_hazards"]),
            (0, 0, 3),
        )
        self.assertEqual(
            (
                relaxed["true_positives"],
                relaxed["duplicates"],
                relaxed["false_hazards"],
            ),
            (1, 1, 1),
        )

    def test_no_future_or_other_identity_can_supply_target_history(self):
        history = [(5_000_000, [target(2.0)]), (2_000_000, [target(2.0, "neighbour")])]
        self.assertFalse(
            _matches(
                belief(2.0),
                self.gt[0],
                latency_tolerance_s=4,
                now_us=self.now,
                gt_history=history,
            )
        )

    def test_temporal_boundary_is_inclusive(self):
        self.assertTrue(
            _matches(
                belief(0.0),
                self.gt[0],
                latency_tolerance_s=4,
                now_us=self.now,
                gt_history=self.history,
            )
        )
        self.assertFalse(
            _matches(
                belief(0.0),
                self.gt[0],
                latency_tolerance_s=3.999,
                now_us=self.now,
                gt_history=self.history,
            )
        )

    def test_recently_departed_object_receives_same_tolerance(self):
        gt = [target(8.0)]
        history = [(2_000_000, [target(20.0, "departed")])]
        score = score_frame(
            [belief(20.0)],
            gt,
            center=(0, 0),
            latency_tolerance_s=3,
            now_us=self.now,
            gt_history=history,
        )
        self.assertEqual(score["true_positives"], 1)

    def test_invalid_tolerance_is_rejected(self):
        for tau in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.score([belief(2)], tau)

    def test_identity_audit_reports_neighbour_and_ambiguity(self):
        b = dict(belief(8), capture_time_us=0, captured_position=[0.1, 0, 1])
        identity, ambiguous = capture_identity(
            b, [(0, [target(0, "other"), target(1, "p")])]
        )
        self.assertEqual(identity, "other")
        self.assertTrue(ambiguous)
        self.assertEqual(capture_identity(b, []), (None, False))


class SweepReplayTests(unittest.TestCase):
    def replay(self):
        from types import SimpleNamespace

        from coopx.eval.runner import _Replay

        replay = _Replay.__new__(_Replay)
        replay.scene = SimpleNamespace(
            name="test-clip",
            ego_id="ego",
            targets=[SimpleNamespace(gt_id="p")],
            visibility={("p", "ego", "4"): SimpleNamespace(label="occluded")},
            hidden_interval={"p": ("4", None)},
        )
        replay.time = {"1": 1_000_000, "4": 4_000_000}
        replay.perception = "sam"
        replay.gt = lambda frame: [target(2.0 if frame == "1" else 8.0)]
        replay._xy = lambda agent, frame: (0.0, 0.0)
        return replay

    def test_rescoring_changes_only_scores_preserves_beliefs_and_censored_lead(self):
        import json
        import tempfile
        from pathlib import Path

        replay = self.replay()
        b = dict(
            belief(2.0), capture_time_us=1_000_000, captured_position=[2.0, 0.0, 1.0]
        )
        ticks = [
            {"record": "tick", "frame": "1", "t_us": 1_000_000, "beliefs": []},
            {"record": "tick", "frame": "4", "t_us": 4_000_000, "beliefs": [b]},
        ]
        original = replay.score_ticks(ticks)
        self.assertEqual(original["scores"]["occluded_recall"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "original.jsonl"
            src.write_text("\n".join(json.dumps(t) for t in ticks) + "\n")
            summary = {
                "scene": "test-clip",
                "fusion": "witness",
                "path": str(src),
                "metrics": {**original, "bytes_to_ego": 100},
                "latency_tolerance_s": 0,
            }
            new = replay.rescore(
                summary, latency_tolerance_s=3, out_dir=Path(tmp) / "rescore"
            )
            self.assertEqual(new["metrics"]["scores"]["occluded_recall"], 1.0)
            self.assertEqual(new["metrics"]["audited_occluded_recall"], 1.0)
            self.assertIsNone(new["metrics"]["lead_us"]["p"])
            self.assertEqual(new["metrics"]["position_error_at_decision_m"], 6.0)
            rows = [
                json.loads(line) for line in Path(new["path"]).read_text().splitlines()
            ]
            self.assertEqual(rows[:-1], ticks)
            self.assertEqual(summary["latency_tolerance_s"], 0)

    def test_attack_default_stays_strict_and_sweep_values_are_fixed(self):
        import inspect

        from coopx.eval.runner import LATENCY_TOLERANCES_S, run_eval

        self.assertEqual(
            inspect.signature(run_eval).parameters["latency_tolerance_s"].default, 0.0
        )
        self.assertEqual(LATENCY_TOLERANCES_S, (0, 1, 2, 3, 4))

    def test_report_clearly_labels_tolerant_scores(self):
        from coopx.eval.report import render_summary

        report = render_summary([], study={"sweep": []})
        self.assertIn(
            "τ > 0 scores detection of a recent position, not the current position.",
            report,
        )
        self.assertIn("Attack rows below always use τ = 0", report)
        self.assertIn("wrong pedestrian / hits", report)


class ContinuousWindowTests(unittest.TestCase):
    def test_between_frame_positions_are_interpolated_only_within_tolerance(self):
        history = [(0, [target(-2.0)]), (100_000, [target(2.0)])]
        self.assertTrue(
            _matches(
                belief(0.0),
                target(2.0),
                latency_tolerance_s=0.1,
                now_us=100_000,
                gt_history=history,
            )
        )
        self.assertFalse(
            _matches(
                belief(0.0),
                target(2.0),
                latency_tolerance_s=0.001,
                now_us=100_000,
                gt_history=history,
            )
        )
        self.assertFalse(_matches(belief(0.0), target(2.0)))

    def test_missing_annotations_do_not_invent_a_path(self):
        history = [(0, [target(-2.0)]), (1_000_000, [target(2.0)])]
        self.assertFalse(
            _matches(
                belief(0.0),
                target(2.0),
                latency_tolerance_s=1,
                now_us=1_000_000,
                gt_history=history,
            )
        )


class IdentityClassTests(unittest.TestCase):
    def test_nearby_non_person_is_not_counted_as_another_pedestrian(self):
        b = dict(belief(0.0), capture_time_us=0, captured_position=[0.0, 0.0, 1.0])
        objects = [
            {"id": "bin", "label": "TrashCan", "xy": [0.0, 0.0]},
            target(0.4, "person"),
        ]
        self.assertEqual(capture_identity(b, [(0, objects)]), ("person", False))


class SummaryIsolationTests(unittest.TestCase):
    def test_tolerant_run_cannot_overwrite_default_strict_result_path(self):
        from pathlib import Path

        from coopx.eval.runner import _out_path

        strict = _out_path(Path("data/results"), "scene", "witness", None, "sam")
        tolerant = _out_path(
            Path("data/results"), "scene", "witness", None, "sam", latency_tolerance_s=3
        )
        self.assertNotEqual(strict, tolerant)
        self.assertEqual(strict, Path("data/results/sam/scene/witness.jsonl"))
        self.assertEqual(tolerant, Path("data/results/tau-3/sam/scene/witness.jsonl"))

    def test_decision_error_p90_uses_nearest_rank(self):
        from types import SimpleNamespace

        replay = SweepReplayTests().replay()
        replay.scene.visibility = {
            ("p", "ego", str(i)): SimpleNamespace(label="occluded") for i in range(10)
        }
        replay.gt = lambda frame: [target(float(frame))]
        ticks = [
            {
                "frame": str(i),
                "t_us": i * 100_000,
                "beliefs": [
                    dict(
                        belief(0.0),
                        capture_time_us=0,
                        captured_position=[0.0, 0.0, 1.0],
                    )
                ],
            }
            for i in range(10)
        ]
        scores = replay.score_ticks(ticks)
        self.assertEqual(scores["position_error_at_decision_m"], 4.5)
        self.assertEqual(scores["position_error_at_decision_p90_m"], 8.0)


class ImageIdentityTests(unittest.TestCase):
    def test_audit_uses_source_image_and_flags_overlapping_neighbours(self):
        from coopx.eval.metrics import image_identity

        boxes = [
            {"id": "target", "box_xyxy": (0, 0, 10, 20)},
            {"id": "neighbour", "box_xyxy": (20, 0, 30, 20)},
        ]
        self.assertEqual(image_identity((20, 0, 30, 20), boxes), ("neighbour", False))
        self.assertEqual(image_identity((100, 0, 110, 20), boxes), (None, False))
        boxes.append({"id": "overlapping", "box_xyxy": (0.1, 0, 10.1, 20)})
        self.assertTrue(image_identity((0, 0, 10, 20), boxes)[1])


class AuditedRecallTests(unittest.TestCase):
    def test_a_neighbour_match_does_not_become_a_verified_target_match(self):
        replay = SweepReplayTests().replay()
        replay.audit_identity = lambda belief, history: ("neighbour", False)
        ticks = [
            {"frame": "1", "t_us": 1_000_000, "beliefs": []},
            {"frame": "4", "t_us": 4_000_000, "beliefs": [belief(2.0)]},
        ]
        scores = replay.score_ticks(ticks, latency_tolerance_s=3)
        self.assertEqual(scores["scores"]["occluded_recall"], 1.0)
        self.assertEqual(scores["wrong_pedestrian_matches"], 1)
        self.assertEqual(scores["audited_occluded_recall"], 0.0)
        self.assertIsNone(scores["audited_lead_us"]["p"])


class TemporalIndexTests(unittest.TestCase):
    def test_index_keeps_identical_distances_with_missing_and_future_frames(self):
        import random

        from coopx.eval.metrics import TemporalHistory, _match_distance

        rng = random.Random(11)
        history = [
            (i * 100_000, [target(rng.uniform(-5, 5), str(j)) for j in range(8)])
            for i in range(10)
            if i != 5
        ]
        indexed = TemporalHistory(history)
        for tau in (0, 0.15, 0.4, 1.0):
            for obj in history[4][1]:
                b = belief(rng.uniform(-5, 5))
                raw = _match_distance(
                    b, obj, latency_tolerance_s=tau, now_us=700_000, gt_history=history
                )
                fast = _match_distance(
                    b, obj, latency_tolerance_s=tau, now_us=700_000, gt_history=indexed
                )
                self.assertEqual(raw, fast)
