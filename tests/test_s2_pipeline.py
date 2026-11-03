"""S2 pipeline: coverage / assessment / map events, ghost admission, determinism.

S2 event types are not yet in coopx.events on this branch (Agent A adds them).
The validator below is the local full check the viewer also implements.
"""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from coopx.events import CHECK_CODES, FIELDS, SCHEMA
from coopx.observers.lidar_synth import S2_GRID
from coopx.pipeline import (
    ASSESSMENT_CODES,
    COVERAGE_PREFIX,
    S2_EVENT_TYPES,
    S2_GHOST_FREE_XYZ,
    S2_GHOST_SILENT_XYZ,
    StubCoverageChecker,
    decode_coverage_stub,
    encode_coverage_stub,
    run_s2,
)
from coopx.observers.lidar_synth import CoverageContent, LidarObserver, default_s2_world
from coopx.transport import EVENT_KEYS, EVENT_TYPES

S2_TYPES = set(EVENT_TYPES) | set(S2_EVENT_TYPES)
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
        free_cell = S2_GRID.cell_of(*S2_GHOST_FREE_XYZ[:2])
        silent_cell = S2_GRID.cell_of(*S2_GHOST_SILENT_XYZ[:2])
        self.assertIn(free_cell, coverage.free)
        self.assertNotIn(silent_cell, coverage.observed)

    def test_key_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            self._run(a, key_seed=3)
            self._run(b, key_seed=3)
            self.assertEqual(load_events(a), load_events(b))

    def test_stub_coverage_checker_admits_standin_only(self):
        lidar = LidarObserver(S2_GRID, seed=0)
        coverage = lidar.scan(default_s2_world(), t_us=0)
        raw = encode_coverage_stub(
            coverage, source_id="lidar1", message_id="c0",
            capture_time_us=0, valid_until_us=1_000_000)
        self.assertTrue(raw.startswith(COVERAGE_PREFIX))
        checker = StubCoverageChecker()
        self.assertEqual(checker.check(raw, 0).codes, ("OK",))
        self.assertEqual(checker.check(b"not-cov", 0).codes, ("DECODE_ERROR",))
        decoded = decode_coverage_stub(raw)
        self.assertIsInstance(decoded.content, CoverageContent)
        self.assertEqual(decoded.source_id, "lidar1")


if __name__ == "__main__":
    unittest.main()
