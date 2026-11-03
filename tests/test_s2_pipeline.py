"""S2 pipeline: coverage / assessment / map events, ghost admission, determinism.

The local validator mirrors the viewer's checks; every event must also pass
`coopx.events.validate` (the single schema source).
"""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from coopx import events as event_schema
from coopx.checker import Checker
from coopx.envelope import decode, encode, evidence_ref
from coopx.events import ASSESSMENT_CODES, CHECK_CODES, FIELDS, SCHEMA
from coopx.observers.lidar_synth import S2_GRID, LidarObserver, default_s2_world
from coopx.pipeline import (
    S2_GHOST_FREE_XYZ,
    S2_GHOST_SILENT_XYZ,
    S2_NODES,
    S2_REFLECTION_XYZ,
    S2_TTL_US,
    _anchors_and_keys,
    _signed_coverage,
    run_s2,
)
from coopx.transport import EVENT_KEYS, EVENT_TYPES

S2_TYPES = set(EVENT_TYPES)
ASSESSMENT_STATES = {"corroborated", "contradicted", "unverified", "contested"}
CHECK_STATES = {"accepted", "rejected"}


def _is_cell(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and type(value[0]) is int and type(value[1]) is int)


def validate_log_event(event: dict) -> None:
    """Full S1+S2 field/type/consistency check. Count-friendly: raises ValueError."""
    if type(event) is not dict or set(event) != set(EVENT_KEYS) or set(event) != set(FIELDS):
        raise ValueError("keys")
    if event["schema"] != SCHEMA:
        raise ValueError("schema")
    if type(event["t_replay_us"]) is not int or event["t_replay_us"] < 0:
        raise ValueError("t_replay_us")
    if event["type"] not in S2_TYPES:
        raise ValueError("type")
    if type(event["source_id"]) is not str or not event["source_id"]:
        raise ValueError("source_id")
    for name in ("dest_id", "message_id", "message_type"):
        if event[name] is not None and (type(event[name]) is not str or not event[name]):
            raise ValueError(name)
    ref = event["evidence_ref"]
    if ref is not None:
        if type(ref) is not str or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            raise ValueError("evidence_ref")
    if event["bytes"] is not None and (type(event["bytes"]) is not int or event["bytes"] < 0):
        raise ValueError("bytes")
    codes = event["check_codes"]
    if type(codes) is not list or any(type(c) is not str for c in codes):
        raise ValueError("check_codes")
    if type(event["detail"]) is not dict:
        raise ValueError("detail")
    kind = event["type"]
    if kind == "check":
        if event["state"] not in CHECK_STATES or not codes:
            raise ValueError("check state")
        if any(c not in CHECK_CODES for c in codes):
            raise ValueError("check code")
        if event["state"] == "accepted" and codes != ["OK"]:
            raise ValueError("accepted")
        if event["state"] == "rejected" and "OK" in codes:
            raise ValueError("rejected OK")
    elif kind == "assessment":
        if event["state"] not in ASSESSMENT_STATES:
            raise ValueError("assessment state")
        if not codes or any(c not in ASSESSMENT_CODES for c in codes):
            raise ValueError("assessment codes")
        detail = event["detail"]
        if not isinstance(detail.get("cells"), list) or any(not _is_cell(c) for c in detail["cells"]):
            raise ValueError("assessment cells")
        for key in ("supporting", "contradicting"):
            if not isinstance(detail.get(key), list) or any(type(x) is not str for x in detail[key]):
                raise ValueError(key)
        if type(detail.get("reason")) is not str:
            raise ValueError("reason")
    elif kind == "map_snapshot":
        if event["state"] is not None or codes:
            raise ValueError("map_snapshot codes")
        detail = event["detail"]
        for key in ("occupied", "free", "contested"):
            if not isinstance(detail.get(key), list) or any(not _is_cell(c) for c in detail[key]):
                raise ValueError(key)
        if type(detail.get("unknown_count")) is not int or detail["unknown_count"] < 0:
            raise ValueError("unknown_count")
    elif kind == "coverage_published":
        if event["state"] is not None or codes:
            raise ValueError("coverage codes")
        if event["message_type"] != "coverage":
            raise ValueError("coverage message_type")
        detail = event["detail"]
        grid = detail.get("grid") or {}
        if type(grid.get("cell_m")) not in (int, float) or type(grid.get("nx")) is not int:
            raise ValueError("grid")
        if not isinstance(detail.get("free"), list) or any(not _is_cell(c) for c in detail["free"]):
            raise ValueError("free")
        if not isinstance(detail.get("occupied"), list) or any(not _is_cell(c) for c in detail["occupied"]):
            raise ValueError("occupied")
        if type(detail.get("observed_count")) is not int:
            raise ValueError("observed_count")
    else:
        if event["state"] is not None or codes:
            raise ValueError("non-check state/codes")


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class S2PipelineTests(unittest.TestCase):
    def _run(self, path: Path, **kwargs):
        defaults = dict(seed=0, loss=0.0, delay_us=50_000, jitter_us=0, key_seed=1)
        defaults.update(kwargs)
        return run_s2(path, **defaults)

    def test_log_has_s2_event_types_and_passes_full_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            summary = self._run(path)
            events = load_events(path)
            for event in events:
                validate_log_event(event)
                event_schema.validate(event)
            counts = Counter(e["type"] for e in events)
            self.assertGreater(counts["coverage_published"], 0)
            self.assertGreater(counts["assessment"], 0)
            self.assertGreater(counts["map_snapshot"], 0)
            self.assertEqual(summary["events_by_type"], dict(counts))
            times = [e["t_replay_us"] for e in events]
            self.assertEqual(times, sorted(times))

    def test_malformed_check_codes_are_rejected_by_validator(self):
        with self.assertRaises(ValueError):
            validate_log_event({
                "schema": SCHEMA, "t_replay_us": 0, "type": "check",
                "source_id": "a", "dest_id": "ego", "message_id": "m",
                "message_type": "observation", "evidence_ref": "0" * 64,
                "bytes": 1, "check_codes": 1, "state": "accepted", "detail": {},
            })

    def test_ghost_packets_are_admitted_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            summary = self._run(path)
            events = load_events(path)
            ghost_checks = [e for e in events if e["type"] == "check"
                            and e["source_id"] in ("ghost1", "ghost2")]
            self.assertTrue(ghost_checks)
            self.assertTrue(all(e["state"] == "accepted" for e in ghost_checks))
            self.assertTrue(all(e["check_codes"] == ["OK"] for e in ghost_checks))
            self.assertEqual(summary["rejected"], 0)
            decode_errors = [e for e in events if e["type"] == "check"
                             and "DECODE_ERROR" in e["check_codes"]]
            self.assertEqual(decode_errors, [])

    def test_coverage_is_never_logged_as_decode_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            self._run(path)
            events = load_events(path)
            coverage_checks = [e for e in events if e["type"] == "check"
                               and e["message_type"] == "coverage"]
            self.assertTrue(coverage_checks)
            self.assertTrue(all(e["check_codes"] == ["OK"] for e in coverage_checks))

    def test_ghost_in_free_space_and_silence_cells(self):
        lidar = LidarObserver(S2_GRID, pose=(0.0, 0.0), seed=0)
        coverage = lidar.scan(default_s2_world(), t_us=2_000_000)
        free_cell = S2_GRID.cell_at(*S2_GHOST_FREE_XYZ[:2])
        silent_cell = S2_GRID.cell_at(*S2_GHOST_SILENT_XYZ[:2])
        self.assertIn(free_cell, coverage.free)
        self.assertNotIn(silent_cell, coverage.observed)

    def test_key_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            self._run(a, key_seed=3)
            self._run(b, key_seed=3)
            self.assertEqual(load_events(a), load_events(b))

    def test_free_space_ghost_contradicted_silent_ghost_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            self._run(path)
            events = load_events(path)
            coverage_refs = {e["evidence_ref"] for e in events
                             if e["type"] == "coverage_published" and e["source_id"] == "lidar1"}
            by_source = {}
            for e in events:
                if e["type"] == "assessment":
                    by_source.setdefault(e["source_id"], []).append(e)
            ghost1 = by_source["ghost1"]
            self.assertTrue(ghost1)
            for e in ghost1:
                self.assertEqual((e["state"], e["check_codes"]),
                                 ("contradicted", ["CONTRADICTED_BY_FREE_SPACE"]))
                self.assertTrue(e["detail"]["contradicting"])
                self.assertLessEqual(set(e["detail"]["contradicting"]), coverage_refs)
            # Silence is not free space: behind the truck no range sensor looks.
            for e in by_source["ghost2"]:
                self.assertEqual((e["state"], e["check_codes"]),
                                 ("unverified", ["UNVERIFIED_SINGLE_SOURCE"]))
                self.assertEqual(e["detail"]["contradicting"], [])
            # Geometric LiDAR support corroborates position only, never class.
            pole_codes = Counter(e["check_codes"][0] for e in by_source["pole1"])
            self.assertGreater(pole_codes["POSITION_CORROBORATED_CLASS_UNVERIFIED"], 0)
            self.assertEqual(pole_codes["CORROBORATED"], 0)

    def test_contested_cell_links_lidar_free_and_ghost_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            self._run(path)
            events = load_events(path)
            published = {e["evidence_ref"]: e for e in events
                         if e["type"] in ("coverage_published", "observation_published")}
            ghost_cell = list(S2_GRID.cell_at(*S2_GHOST_FREE_XYZ[:2]))
            snaps = [e for e in events if e["type"] == "map_snapshot"]
            contested = [s for s in snaps if ghost_cell in s["detail"]["contested"]]
            self.assertTrue(contested)
            self.assertTrue(all(s["t_replay_us"] > 2_000_000 for s in contested))
            for snap in contested:
                entry = next(c for c in snap["detail"]["contested_claims"]
                             if c["cell"] == ghost_cell)
                kinds = {(published[ref]["source_id"], kind) for ref, kind in entry["claims"]}
                self.assertIn(("lidar1", "free"), kinds)
                self.assertIn(("ghost1", "occupied"), kinds)
            silent_cell = list(S2_GRID.cell_at(*S2_GHOST_SILENT_XYZ[:2]))
            self.assertFalse(any(silent_cell in s["detail"]["contested"] for s in snaps))

    def test_coverage_is_signed_and_forgery_is_rejected(self):
        keys, anchors = _anchors_and_keys(S2_NODES, key_seed=1)
        checker = Checker(anchors, ttl_us=S2_TTL_US)
        scan = LidarObserver(S2_GRID, seed=0).scan(default_s2_world(), t_us=100_000)
        honest = _signed_coverage("lidar1", keys["lidar1"], scan, sim_us=100_000,
                                  sensor_seq=0, message_id="c0")
        raw = encode(honest)
        self.assertEqual(raw[:4], b"CX2C")
        self.assertEqual(evidence_ref(decode(raw)), evidence_ref(honest))
        self.assertEqual(checker.check(raw, 150_000).codes, ("OK",))
        # A ghost cannot publish free space in the LiDAR's name.
        forged = encode(_signed_coverage("lidar1", keys["ghost1"], scan, sim_us=100_000,
                                         sensor_seq=1, message_id="c1"))
        self.assertEqual(checker.check(forged, 150_000).codes, ("BAD_SIGNATURE",))

    def test_reflection_cameras_are_contested_by_lidar_free_space(self):
        lidar = LidarObserver(S2_GRID, pose=(0.0, 0.0), seed=0)
        coverage = lidar.scan(default_s2_world(), t_us=3_000_000)
        cell = S2_GRID.cell_at(*S2_REFLECTION_XYZ[:2])
        self.assertIn(cell, coverage.free)
        self.assertGreater(
            ((S2_REFLECTION_XYZ[0] - S2_GHOST_FREE_XYZ[0]) ** 2
             + (S2_REFLECTION_XYZ[1] - S2_GHOST_FREE_XYZ[1]) ** 2) ** 0.5,
            1.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.jsonl"
            self._run(path)
            events = load_events(path)
            for event in events:
                event_schema.validate(event)
            coverage_refs = {e["evidence_ref"] for e in events
                             if e["type"] == "coverage_published" and e["source_id"] == "lidar1"}
            published = {e["evidence_ref"]: e for e in events
                         if e["type"] in ("coverage_published", "observation_published")}
            by_source = {}
            for e in events:
                if e["type"] == "assessment":
                    by_source.setdefault(e["source_id"], []).append(e)
            for source in ("pole2", "pole3"):
                self.assertTrue(by_source[source])
                for e in by_source[source]:
                    self.assertEqual((e["state"], e["check_codes"]),
                                     ("contested", ["CONTESTED"]))
                    supporting = {published[ref]["source_id"] for ref in e["detail"]["supporting"]}
                    self.assertEqual(supporting, {"pole2", "pole3"})
                    self.assertTrue(e["detail"]["contradicting"])
                    self.assertLessEqual(set(e["detail"]["contradicting"]), coverage_refs)
            reflection_cell = list(cell)
            snaps = [e for e in events if e["type"] == "map_snapshot"
                     and reflection_cell in e["detail"]["contested"]]
            self.assertTrue(snaps)
            entry = next(c for c in snaps[-1]["detail"]["contested_claims"]
                         if c["cell"] == reflection_cell)
            kinds = {(published[ref]["source_id"], kind) for ref, kind in entry["claims"]}
            self.assertIn(("lidar1", "free"), kinds)
            self.assertIn(("pole2", "occupied"), kinds)
            self.assertIn(("pole3", "occupied"), kinds)
            self.assertIn("pole2", S2_NODES)
            self.assertIn("pole3", S2_NODES)

if __name__ == "__main__":
    unittest.main()
