"""S1 end to end on merged code: transport + real Checker + event schema."""
import json
from pathlib import Path
import tempfile
import unittest

from coopx import events
from coopx.checker import Checker
from coopx.domain import ObjectClass
from coopx.envelope import encode, evidence_ref, decode
from coopx.pipeline import S1_TTL_US, _anchors_and_keys, _signed_observation, _handle_deliveries, run_s1
from coopx.transport import PacketTransport


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class S1IntegrationTests(unittest.TestCase):
    def test_every_event_passes_the_schema_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            run_s1(path, seed=3, loss=0.2, jitter_us=150_000, disconnect_at_us=3_000_000)
            for event in load(path):
                events.validate(event)

    def test_honest_jittered_lossy_run_is_never_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            # Jitter larger than the publish period forces reordering.
            summary = run_s1(path, seed=1, loss=0.1, delay_us=100_000, jitter_us=250_000)
            checks = [e for e in load(path) if e["type"] == "check"]
            self.assertEqual(summary["checker"], "Checker")
            self.assertGreater(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 0, [e["check_codes"] for e in checks
                                                       if e["state"] != "accepted"])
            seqs = [e["detail"]["sensor_seq"] for e in checks if e["source_id"] == "pole1"]
            self.assertTrue(any(b < a for a, b in zip(seqs, seqs[1:])),
                            "test is vacuous unless some packets arrived out of order")

    def test_disconnected_observer_stops_reaching_ego(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s1.jsonl"
            run_s1(path, seed=0, jitter_us=0, disconnect_at_us=3_000_000)
            log = load(path)
            cut = next(e["t_replay_us"] for e in log if e["type"] == "observer_disconnected")
            late_pole_checks = [e for e in log if e["type"] == "check"
                                and e["source_id"] == "pole1" and e["t_replay_us"] > cut + 200_000]
            self.assertEqual(late_pole_checks, [])
            self.assertTrue(any(e["type"] == "check" and e["source_id"] == "car1"
                                and e["t_replay_us"] > cut + 200_000 for e in log))

    def test_replayed_and_forged_packets_rejected_after_transport(self):
        keys, anchors = _anchors_and_keys()
        checker = Checker(anchors, ttl_us=S1_TTL_US)
        log = []
        transport = PacketTransport(("pole1", "car1", "ego"), delay_us=100_000,
                                    jitter_us=80_000, seed=7, event_callback=log.append)
        packets = []
        for frame in range(10):
            t = frame * 100_000
            raw = encode(_signed_observation(
                "pole1", keys["pole1"], object_id=1, frame=frame, sim_us=t,
                cls=ObjectClass.PEDESTRIAN, xyz=(8.0 + 0.14 * frame, 1.0, 0.5),
                velocity=(1.4, 0.0, 0.0)))
            packets.append(raw)
            transport.send("pole1", raw, sim_us=t, message_type="observation")
        accepted, rejected = _handle_deliveries(transport.receive_until(1_100_000), checker, log)
        self.assertEqual((accepted, rejected), (10, 0))
        latest = checker.latest("pole1", 1)
        # Car signs a packet claiming to be the pole: signature fails.
        forged = encode(_signed_observation(
            "pole1", keys["car1"], object_id=1, frame=11, sim_us=1_300_000,
            cls=ObjectClass.PEDESTRIAN, xyz=(9.5, 1.0, 0.5), velocity=(1.4, 0.0, 0.0)))
        for raw, send_us, expected in ((packets[5], 1_100_000, "DUPLICATE"),
                                        (forged, 1_300_000, "BAD_SIGNATURE"),
                                        (packets[0], 5_000_000, "STALE")):
            envelope = decode(raw).envelope
            transport.send("pole1", raw, sim_us=send_us, message_id=envelope.message_id,
                           message_type="observation")
            before = len(log)
            accepted, rejected = _handle_deliveries(transport.receive_until(send_us + 200_000), checker, log)
            self.assertEqual((accepted, rejected), (0, 1))
            checks = [event for event in log[before:] if event["type"] == "check"]
            self.assertEqual(len(checks), 1)
            self.assertIn(expected, checks[0]["check_codes"])
            self.assertEqual(checks[0]["state"], "rejected")
            self.assertEqual(checks[0]["evidence_ref"], evidence_ref(decode(raw)))
            self.assertEqual(checker.latest("pole1", 1), latest)
        for event in log:
            events.validate(event)
        self.assertEqual(checker.latest("pole1", 1).envelope.sensor_seq, 9)


if __name__ == "__main__":
    unittest.main()
