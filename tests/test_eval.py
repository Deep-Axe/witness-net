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
        self.assertEqual(independence(6, 1), {"messages": 6, "independent_roots": 1})

    def test_offline_fixture_witness_does_not_drop_the_ego(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_eval(scene="frame-000096", fusion="witness", attack="remote_clear",
                              seed=1, offline=True, out_dir=Path(tmp))
            run = result["runs"][0]
            self.assertFalse(result["partial"])
            self.assertTrue(run["fusion_stub"])
            self.assertEqual(run["metrics"]["safety_flips"], 0)
            self.assertIsNone(run["metrics"]["scores"]["occluded_recall"])
            text = render_summary([run])
            self.assertIn("frame-000096", text)
            self.assertIn("n/a", text)


if __name__ == "__main__":
    unittest.main()
