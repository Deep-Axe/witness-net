from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from coopx.pipeline import PassThroughChecker, run_s1
from coopx.transport import EVENT_KEYS, EVENT_SCHEMA, EVENT_TYPES


def validate_event(event: dict) -> None:
    if set(event) != set(EVENT_KEYS):
        raise ValueError(f"keys {sorted(event)} != {sorted(EVENT_KEYS)}")
    if event["schema"] != EVENT_SCHEMA:
        raise ValueError("bad schema")
    if type(event["t_replay_us"]) is not int or event["t_replay_us"] < 0:
        raise ValueError("t_replay_us")
    if event["type"] not in EVENT_TYPES:
        raise ValueError("type")
    if not isinstance(event["source_id"], str) or not event["source_id"]:
        raise ValueError("source_id")
    if event["dest_id"] is not None and not isinstance(event["dest_id"], str):
        raise ValueError("dest_id")
    if event["message_id"] is not None and not isinstance(event["message_id"], str):
        raise ValueError("message_id")
    if event["message_type"] is not None and not isinstance(event["message_type"], str):
        raise ValueError("message_type")
    ref = event["evidence_ref"]
    if ref is not None:
        if not isinstance(ref, str) or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            raise ValueError("evidence_ref")
    if event["bytes"] is not None and (type(event["bytes"]) is not int or event["bytes"] < 0):
        raise ValueError("bytes")
    codes = event["check_codes"]
    if not isinstance(codes, list) or any(not isinstance(c, str) for c in codes):
        raise ValueError("check_codes")
    if event["type"] == "check":
        if event["state"] not in ("accepted", "rejected") or not codes:
            raise ValueError("check state/codes")
        if event["state"] == "accepted" and codes != ["OK"]:
            raise ValueError("accepted must be OK")
        if event["state"] == "rejected" and codes == ["OK"]:
            raise ValueError("rejected OK")
    else:
        if event["state"] is not None or codes:
            raise ValueError("non-check state/codes")
    if not isinstance(event["detail"], dict):
        raise ValueError("detail")


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            events.append(json.loads(line))
    return events


class PipelineTests(unittest.TestCase):
    def test_run_s1_writes_contract_jsonl_and_checks_follow_ego_deliveries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            summary = run_s1(path, seed=0, loss=0.0, delay_us=100_000, jitter_us=0)
            events = load_events(path)
            for event in events:
                validate_event(event)
            counts = Counter(event["type"] for event in events)
            self.assertEqual(summary["events_by_type"], dict(counts))
            self.assertEqual(summary["accepted"] + summary["rejected"], counts["check"])
            self.assertEqual(summary["accepted"], counts["check"])
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(summary["checker"], "Checker")
            self.assertEqual(summary["sent_bytes"], sum(
                e["bytes"] for e in events if e["type"] == "packet_sent"))
            self.assertEqual(summary["received_bytes"], sum(
                e["bytes"] for e in events if e["type"] == "packet_delivered"))
            self.assertEqual(summary["drops"], {"loss": 0, "disconnected": 0, "budget": 0})

            for i, event in enumerate(events):
                if event["type"] == "packet_delivered" and event["dest_id"] == "ego":
                    self.assertLess(i + 1, len(events))
                    check = events[i + 1]
                    self.assertEqual(check["type"], "check")
                    self.assertEqual(check["evidence_ref"], event["evidence_ref"])
                    self.assertEqual(check["state"], "accepted")
                    self.assertEqual(check["check_codes"], ["OK"])

            self.assertGreaterEqual(counts["observation_published"], 100)
            self.assertGreaterEqual(counts["packet_sent"], 100)
            self.assertGreaterEqual(counts["packet_delivered"], 100)
            self.assertGreaterEqual(counts["check"], 100)

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(OSError):
                run_s1(path, seed=0, loss=0.0, delay_us=0, jitter_us=0)

    def test_disconnect_is_logged_and_summary_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            summary = run_s1(path, seed=1, loss=0.0, delay_us=50_000, jitter_us=0,
                             disconnect_at_us=3_000_000)
            events = load_events(path)
            for event in events:
                validate_event(event)
            self.assertEqual(summary["events_by_type"], dict(Counter(e["type"] for e in events)))
            disconnects = [e for e in events if e["type"] == "observer_disconnected"]
            self.assertEqual(len(disconnects), 1)
            self.assertEqual(disconnects[0]["source_id"], "pole1")
            self.assertGreater(summary["drops"]["disconnected"], 0)
            self.assertEqual(summary["drops"]["disconnected"],
                             sum(1 for e in events if e["type"] == "packet_dropped"
                                 and e["detail"].get("reason") == "disconnected"))

    def test_passthrough_checker_rejects_garbage(self):
        checker = PassThroughChecker()
        result = checker.check(b"not-a-packet", 0)
        self.assertEqual(result.state, "rejected")
        self.assertEqual(result.codes, ("DECODE_ERROR",))
        self.assertIsNone(result.message)


if __name__ == "__main__":
    unittest.main()
