"""Offline checks for the S4b demo claims. No network."""
import unittest

from coopx.coverage import CoverageContent, CoverageMap, GridSpec
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import Auth, Envelope, EvidenceStore, Lineage, ObservationMessage, sign
from coopx.identity import TrustAnchor, TrustClass
from coopx.pipeline import (S3_REGION, beat_badge, coverage_change_caption, demo_node_key,
                            witness_count, _signed_coverage_window)
from coopx.query import Answer, Query


class BadgeTests(unittest.TestCase):
    def test_badge_matches_the_beat(self):
        self.assertEqual(beat_badge("causal"), "Causal replay")
        self.assertEqual(beat_badge("snapshot"), "Paused snapshot Q&A")
        self.assertEqual(beat_badge("single"), "Single frame")


class CoverageExpiryTests(unittest.TestCase):
    def test_disconnect_timeout_removes_free_cells(self):
        spec = GridSpec(0.0, 0.0, 1.0, 4, 4)
        content = CoverageContent(
            spec, (0, 0), "lidar", (0.2, 2.0), 0.3,
            frozenset({(1, 1)}), frozenset({(1, 1)}), frozenset())
        key = demo_node_key(0, "rsu-1")
        anchors = {"rsu-1": TrustAnchor(
            key.public_key, "rsu-1-key", TrustClass.A, physical_observer="rsu-1",
            permitted_region=S3_REGION, anchored=True)}
        store = EvidenceStore(anchors)
        coverage = CoverageMap(spec, ttl_us=1_000_000, spatial_frame_id="site_map", store=store)
        message = _signed_coverage_window(
            "rsu-1", key, content, sim_us=0, ttl_us=1_000_000, sensor_seq=1, message_id="cov")
        store.add(message)
        coverage.add(message, 0)
        before = len(coverage.snapshot(0)["free"])
        after = len(coverage.snapshot(1_000_001)["free"])
        self.assertEqual(before, 1)
        self.assertEqual(after, 0)
        self.assertIn("unknown", coverage_change_caption(before, after, "rsu-1"))


class GhostWitnessTests(unittest.TestCase):
    def test_unanchored_attacker_scores_zero(self):
        key = demo_node_key(1, "ghost")
        anchors = {"ghost": TrustAnchor(
            key.public_key, "ghost-key", TrustClass.B, physical_observer=None,
            permitted_region=S3_REGION, anchored=False)}
        store = EvidenceStore(anchors)
        observation = Observation(
            "ghost", 1, 1, 100_000, ObjectClass.PEDESTRIAN, (1.0, 2.0, 0.5),
            PositionAnchor.SURFACE, confidence=0.4)
        envelope = Envelope(
            2, "observation", "ghost-1", "ghost", "s4b", "site_map",
            100_000, 100_000, 6_000_000, Auth("ghost-key"), 1)
        message = sign(ObservationMessage(
            envelope, observation, Lineage("ghost", "attack", "ghost-1", 100_000)), key)
        store.add(message)
        self.assertEqual(witness_count(store, [message]), 0)


class MuseEvidenceTests(unittest.TestCase):
    def test_contradicted_stands_only_when_the_callback_is_true(self):
        from types import SimpleNamespace
        from coopx.eval.muse_compare import judge_with_evidence
        ref_a, ref_b = "ab" * 32, "cd" * 32
        queries = [Query("q1", "T1", "cam", (0.0, 0.0), 2.0, 1_000_000, "person", "look"),
                   Query("q2", "T1", "lidar", (0.0, 0.0), 2.0, 1_000_000, None, "range")]
        answers = [Answer("q1", "cam", "detected", 1_000_000, None, (ref_a,)),
                   Answer("q2", "lidar", "observed_free", 1_000_000, ref_b, ())]
        evidence = {ref_a: {"source_id": "cam", "xy": [0.0, 0.0], "availability_time_us": 1_000_000},
                    ref_b: {"source_id": "lidar", "availability_time_us": 1_000_000}}

        class Client:
            def respond(self, **kwargs):
                return SimpleNamespace(
                    json={"verdicts": [{"subject": "q1", "state": "contradicted",
                                        "evidence_refs": [ref_a, ref_b], "reason": "free"}]},
                    text="", call_id="j", availability_time_us=kwargs["query_time_us"] + 1)

        denied, _ = judge_with_evidence(
            Client(), lambda *_args: False, "question", queries, answers, evidence, 2_000_000)
        allowed, _ = judge_with_evidence(
            Client(), lambda *_args: True, "question", queries, answers, evidence, 2_000_000)
        self.assertEqual(denied[0].state, "unverified")
        self.assertEqual(allowed[0].state, "contradicted")
        self.assertIn(ref_a, evidence)
        self.assertIn(ref_b, evidence)


if __name__ == "__main__":
    unittest.main()
