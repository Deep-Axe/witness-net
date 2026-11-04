"""The post-hoc inspection is append-only and cannot silently change scores."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

from coopx.eval.clip_extension import DISCLAIMER, nearest_ego, target_report
from coopx.eval.report import render_summary


def belief(
    identity, xy=(0.1, 0.0), *, local=False, state="unverified", ambiguous=False
):
    return {
        "identity": identity,
        "xy": xy,
        "local": local,
        "state": state,
        "cls": "PEDESTRIAN",
        "source_id": "ego-cam4" if local else "rsu-1-cam2",
        "capture_time_us": 0,
        "ambiguous": ambiguous,
    }


def extension_fixture():
    def row(frame, distance):
        return {
            "frame": frame,
            "t_us": int(frame) * 100_000,
            "gt_sources": ["2"],
            "ego_visibility": {"label": "occluded"},
            "rsu_visibility": {"label": "visible"},
            "ego_range_m": distance,
            "ego_nearest": None,
            "target_report": None,
        }

    rows = [row("000018", 39.7), row("000019", 40.6), row("000030", 49.6)]
    first = {
        "frame": "000030",
        "t_us": 3_000_000,
        "distance_m": 0.12,
        "source": "rsu-1-cam2",
        "state": "unverified",
        "capture_us": 0,
        "ego_nearest": None,
    }
    return {
        "disclaimer": DISCLAIMER,
        "mining_boundary": rows[:2],
        "mining_range_m": 40.0,
        "mining_last_record": "000018",
        "context_frames_after_record": 3,
        "mined_window_end": "000021",
        "gt_last_present": "000033",
        "gt_first_missing": "000034",
        "gt_returns_after_first_missing": [],
        "first_report": first,
        "rows": rows,
        "overlays": ["data/scenes/overlays/example.png"],
        "trace_sha256": "0" * 64,
    }


class ClipExtensionTests(unittest.TestCase):
    def setUp(self):
        self.replay = SimpleNamespace(
            audit_identity=lambda b, _history: (b["identity"], b["ambiguous"])
        )
        self.box = SimpleNamespace(center_site=lambda: (0.0, 0.0, 1.0))

    def test_identity_not_distance_selects_target_and_state_is_not_hidden(self):
        near_wrong = belief("other", (0.01, 0.0))
        target = belief("138", (0.5, 0.0), state="contradicted")
        found = target_report(self.replay, {"beliefs": [near_wrong, target]}, self.box)
        self.assertEqual(found["distance_m"], 0.5)
        self.assertEqual(found["state"], "contradicted")
        ambiguous = belief("138", ambiguous=True)
        self.assertIsNone(
            target_report(self.replay, {"beliefs": [ambiguous]}, self.box)
        )

    def test_nearest_ego_is_local_and_reports_another_identity(self):
        rows = [belief("138", (0.01, 0.0)), belief("other", (2.0, 0.0), local=True)]
        nearest = nearest_ego(self.replay, {"beliefs": rows}, self.box)
        self.assertEqual(nearest["distance_m"], 2.0)
        self.assertEqual(nearest["identity"], "other")
        self.assertIsNone(nearest_ego(self.replay, {"beliefs": rows}, None))

    def test_no_gt_cannot_invent_a_distance(self):
        found = target_report(self.replay, {"beliefs": [belief("138")]}, None)
        self.assertIsNone(found["distance_m"])

    def test_append_preserves_every_existing_row_and_input(self):
        runs = [
            {
                "scene": "blos_pedestrian_002",
                "split": "report",
                "perception": "sam",
                "fusion": "witness",
                "metrics": {"scores": {"occluded_recall": 0.0}},
            }
        ]
        study = {"sweep": [], "matrix": []}
        before = render_summary(runs, study=study)
        extended = {**study, "clip2_extension": extension_fixture()}
        unchanged = deepcopy(extended)
        after = render_summary(runs, study=extended)
        self.assertEqual(extended, unchanged)
        old_rows = [line for line in before.splitlines() if line.startswith("|")]
        new_rows = [line for line in after.splitlines() if line.startswith("|")]
        self.assertEqual(new_rows[: len(old_rows)], old_rows)
        self.assertIn(DISCLAIMER, after)
        self.assertIn("experimental, not a headline", after)
        self.assertIn("| SAM without tracking | 000030 / 3.000", after)
        self.assertIn("cutoff was not the end of GT", after)
        self.assertIn("000034", after)
