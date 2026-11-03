"""Metrics on hand-built beliefs, and one offline fixture run. No network."""
from pathlib import Path
import tempfile
import unittest

from coopx.eval.metrics import (attack_catch, bytes_per_detection, false_distrust,
                                independence, occluded_scores, safety_flips,
                                unknown_region_rate, warning_lead_us)
from coopx.eval.report import render_summary
from coopx.eval.runner import run_eval


class MetricTests(unittest.TestCase):
    def test_recall_precision_and_false_hazards(self):
        ticks = [{"beliefs": [
            {"xy": [0.2, 0.0], "cls": "PEDESTRIAN", "state": "confirmed", "local": False, "refs": ["a"]},
            {"xy": [10.0, 10.0], "cls": "PEDESTRIAN", "state": "unverified", "local": False, "refs": ["b"]},
        ], "targets": [
            {"id": "p", "label": "pedestrian", "xy": [0.0, 0.0], "visibility": "occluded"},
        ]}]
        scores = occluded_scores(ticks)
        self.assertEqual(scores["occluded_recall"], 1.0)
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["false_hazards"], 1)

    def test_lead_time_subtracts_the_latency_chain(self):
        self.assertEqual(warning_lead_us(0, 2_000_000, 500_000), 1_500_000)
        self.assertIsNone(warning_lead_us(None, 2_000_000, 500_000))

    def test_witness_keeps_local_beliefs(self):
        local = [{"xy": [1.0, 1.0], "cls": "PEDESTRIAN", "state": "confirmed", "local": True, "refs": ["e"]}]
        self.assertEqual(safety_flips(local, local), 0)
        self.assertEqual(safety_flips(local, []), 1)

    def test_catch_distrust_loss_and_roots(self):
        beliefs = [{"xy": [0.0, 0.0], "cls": "PEDESTRIAN", "state": "contradicted",
                    "local": False, "refs": ["ghost"]}]
        targets = [{"label": "pedestrian", "xy": [0.0, 0.0]}]
        self.assertTrue(attack_catch(["ghost"], beliefs))
        self.assertEqual(false_distrust(beliefs, targets), 1)
        self.assertEqual(unknown_region_rate(25, 100), 0.25)
        self.assertEqual(bytes_per_detection(400, 2), 200)
        self.assertEqual(independence(6, 1), {"messages": 6, "independent_witnesses": 1})

    def test_sam_failure_keeps_the_error_text(self):
        from coopx.meta.client import MetaAPIError

        class Boom:
            def respond(self, **kwargs):
                raise MetaAPIError(
                    "Meta API network error: [Errno -2] Name or service not known",
                    status=None)

        if not DATASET.is_dir():
            self.skipTest("V2X-Real val set is not present")
        with tempfile.TemporaryDirectory() as tmp:
            run = run_eval(scene="frame-000096", fusion="ego", seed=1, offline=True,
                           out_dir=Path(tmp), perception="sam", client=Boom())["runs"][0]
            self.assertIn("Name or service not known", run["error"])
            self.assertTrue(run["partial"])


DATASET = Path(__file__).resolve().parents[1] / "data" / "v2x-real" / "val"
SCENES = Path(__file__).resolve().parents[1] / "data" / "scenes" / "scenes.json"


@unittest.skipUnless(DATASET.is_dir() and SCENES.is_file(), "dataset or mined scenes absent")
class MinedClipReplayTests(unittest.TestCase):
    """Oracle perception on mined clip 1: deterministic, no network."""

    def run_mode(self, fusion, attack=None):
        with tempfile.TemporaryDirectory() as tmp:
            return run_eval(scene="blos_pedestrian_001", fusion=fusion, attack=attack, seed=1,
                            offline=True, out_dir=Path(tmp), perception="gt-oracle")["runs"][0]

    def test_remote_help_reaches_the_occluded_target_ego_alone_does_not(self):
        ego, witness = self.run_mode("ego"), self.run_mode("witness")
        self.assertEqual(ego["metrics"]["scores"]["occluded_recall"], 0.0)
        self.assertGreater(witness["metrics"]["scores"]["occluded_recall"], 0.0)
        self.assertEqual(witness["split"], "report")

    def test_remote_clear_never_removes_a_local_detection(self):
        for fusion in ("union", "witness"):
            run = self.run_mode(fusion, "remote_clear")
            self.assertEqual(run["metrics"]["safety_flips"], 0, fusion)

    def test_sybil_adds_no_witness_and_a_relay_counts_once(self):
        sybil = self.run_mode("witness", "sybil")
        relay = self.run_mode("witness", "relay_dup")
        self.assertEqual(sybil["metrics"]["independence"]["independent_witnesses"], 0)
        self.assertEqual(relay["metrics"]["independence"]["independent_witnesses"], 1)

    def test_summary_renders_from_runs(self):
        text = render_summary([self.run_mode("witness")])
        self.assertIn("blos_pedestrian_001", text)
        self.assertIn("| report |", text)

if __name__ == "__main__":
    unittest.main()


class ScoreFrameTests(unittest.TestCase):
    def test_all_gt_one_to_one_duplicates_and_region(self):
        from coopx.eval.metrics import aggregate_frames, score_frame
        gt = [{"id": "a", "label": "Pedestrian", "xy": [0.0, 0.0]},
              {"id": "b", "label": "Pedestrian", "xy": [10.0, 0.0]}]
        beliefs = [{"xy": [0.2, 0.0], "cls": "PEDESTRIAN", "state": "unverified"},
                   {"xy": [0.0, 0.5], "cls": "PEDESTRIAN", "state": "confirmed"},
                   {"xy": [-0.4, 0.1], "cls": "PEDESTRIAN", "state": "unverified"},
                   {"xy": [10.3, 0.0], "cls": "PEDESTRIAN", "state": "unverified"},
                   {"xy": [200.0, 0.0], "cls": "PEDESTRIAN", "state": "unverified"},
                   {"xy": [5.0, 5.0], "cls": "PEDESTRIAN", "state": "contradicted"}]
        f = score_frame(beliefs, gt, center=(0.0, 0.0), radius_m=50.0)
        self.assertEqual((f["true_positives"], f["duplicates"], f["false_hazards"], f["out_of_region"]),
                         (2, 2, 0, 1))
        self.assertEqual(aggregate_frames([f])["precision"], 1.0)
        f2 = score_frame([{"xy": [30.0, 0.0], "cls": "PEDESTRIAN", "state": "unverified"}], gt,
                         center=(0.0, 0.0))
        self.assertEqual(aggregate_frames([f, f2])["precision"], 2 / 3)
