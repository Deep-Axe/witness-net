import unittest

from coopx.events import EVENT_TYPES as CANONICAL_TYPES, FIELDS, SCHEMA
from coopx.radio_codec import encode
from coopx.domain import ObjectClass, Observation
from coopx.transport import EVENT_TYPES, PacketTransport, ReferenceTransport, EVENT_KEYS, EVENT_SCHEMA


def _ascii_payload():
    return encode(Observation("rsu-1", 7, 1, 1_000_000, ObjectClass.PEDESTRIAN, (1.0, 2.0, 0.5)))


class PacketTransportTests(unittest.TestCase):
    def test_reference_transport_still_counts_one_send(self):
        payload = _ascii_payload()
        bus = ReferenceTransport(("rsu-1", "helper", "ego"), delay_us=100)
        bus.send("rsu-1", payload, frame=1, sim_us=0)
        self.assertEqual(bus.sent_bytes, 84)
        self.assertEqual(len(bus.receive_until(100)), 2)

    def test_bytes_sent_once_per_transmission(self):
        packet = b"\x00" * 40
        bus = PacketTransport(("a", "b", "c"), delay_us=0, jitter_us=0, seed=0)
        bus.send("a", packet, sim_us=0)
        self.assertEqual(bus.sent_bytes, 40)
        got = bus.receive_until(0)
        self.assertEqual(len(got), 2)
        self.assertEqual(bus.received_bytes, 80)
        self.assertEqual({d.receiver for d in got}, {"b", "c"})

    def test_seed_is_deterministic(self):
        packet = b"\x01" * 8

        def run(seed):
            bus = PacketTransport(("a", "b"), delay_us=10, jitter_us=50,
                                  loss_probability=0.3, seed=seed)
            for t in range(40):
                bus.send("a", packet, sim_us=t * 100)
            deliveries = bus.receive_until(10_000)
            return [(d.available_sim_us, d.receiver) for d in deliveries], dict(bus.drops)

        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(4))

    def test_jitter_produces_out_of_order_delivery(self):
        found = False
        for seed in range(300):
            bus = PacketTransport(("a", "b"), delay_us=0, jitter_us=1_000, seed=seed)
            bus.send("a", b"\x00" * 4, sim_us=0, message_id="first")
            bus.send("a", b"\x01" * 4, sim_us=1, message_id="second")
            got = bus.receive_until(2_000)
            if len(got) == 2 and got[0].packet[0] == 1:
                found = True
                break
        self.assertTrue(found, "no seed reordered two sends under jitter")

    def test_loss_rate_matches_configuration(self):
        n = 500
        bus = PacketTransport(("a", "b"), delay_us=0, loss_probability=0.25, seed=11)
        for i in range(n):
            bus.send("a", b"\x02" * 8, sim_us=i)
        bus.receive_until(n)
        rate = bus.drops["loss"] / n
        self.assertAlmostEqual(rate, 0.25, delta=0.08)
        self.assertEqual(bus.drops["loss"] + len([e for e in bus.events if e["type"] == "packet_delivered"]), n)

    def test_disconnect_drops_inflight_and_later_reconnect_resumes(self):
        packet = b"\x03" * 16
        bus = PacketTransport(("a", "b", "c"), delay_us=1_000, jitter_us=0, seed=0)
        bus.send("a", packet, sim_us=0, message_id="p0")
        dropped = bus.disconnect("b", sim_us=500)
        self.assertTrue(any(e["type"] == "observer_disconnected" for e in dropped))
        self.assertEqual(bus.drops["disconnected"], 1)
        arrived = bus.receive_until(2_000)
        self.assertEqual([d.receiver for d in arrived], ["c"])
        bus.send("a", packet, sim_us=3_000, message_id="p1")
        later = bus.receive_until(5_000)
        self.assertEqual([d.receiver for d in later], ["c"])
        self.assertGreaterEqual(bus.drops["disconnected"], 2)
        bus.reconnect("b", sim_us=6_000)
        bus.send("a", packet, sim_us=7_000, message_id="p2")
        resumed = bus.receive_until(9_000)
        self.assertEqual(sorted(d.receiver for d in resumed), ["b", "c"])

    def test_disconnected_source_neither_sends_nor_receives(self):
        bus = PacketTransport(("a", "b"), delay_us=0, jitter_us=0, seed=0)
        bus.disconnect("a", sim_us=0)
        bus.send("a", b"\x04" * 8, sim_us=0)
        self.assertEqual(bus.sent_bytes, 0)
        self.assertEqual(bus.drops["disconnected"], 1)
        self.assertEqual(bus.receive_until(0), [])
        bus.send("b", b"\x05" * 8, sim_us=1)
        self.assertEqual(bus.receive_until(1), [])
        self.assertEqual(bus.drops["disconnected"], 2)

    def test_budget_drops_are_counted(self):
        packet = b"\x06" * 100
        bus = PacketTransport(("a", "b"), delay_us=0, jitter_us=0, budget_bytes_per_s=100, seed=0)
        bus.send("a", packet, sim_us=0)
        bus.send("a", packet, sim_us=1)
        self.assertEqual(bus.drops["budget"], 1)
        self.assertEqual(len(bus.receive_until(1)), 1)
        bus.send("a", packet, sim_us=1_000_000)
        self.assertEqual(len(bus.receive_until(1_000_000)), 1)
        self.assertEqual(bus.drops["budget"], 1)

    def test_events_match_shared_contract(self):
        bus = PacketTransport(("a", "b"), delay_us=0, jitter_us=0, loss_probability=1, seed=0)
        bus.send("a", b"\x07" * 8, sim_us=0, message_id="m", message_type="observation")
        bus.receive_until(0)
        self.assertTrue(bus.events)
        for event in bus.events:
            self.assertEqual(set(event), set(EVENT_KEYS))
            self.assertEqual(event["schema"], EVENT_SCHEMA)

    def test_schema_is_imported_from_events(self):
        self.assertIs(EVENT_TYPES, CANONICAL_TYPES)
        self.assertEqual(tuple(EVENT_KEYS), FIELDS)
        self.assertEqual(EVENT_SCHEMA, SCHEMA)

    def test_due_delivery_survives_later_disconnect(self):
        """Peer review: send at 0 delay 100; disconnect at 500; arrival at 100 is due."""
        bus = PacketTransport(("a", "b"), delay_us=100, jitter_us=0, seed=0)
        bus.send("a", b"\x08" * 8, sim_us=0, message_id="due")
        dropped = bus.disconnect("b", sim_us=500)
        self.assertEqual(bus.drops["disconnected"], 0)
        self.assertFalse(any(e["type"] == "packet_dropped" for e in dropped))
        got = bus.receive_until(500)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].available_sim_us, 100)
        self.assertEqual(got[0].receiver, "b")

    def test_future_inflight_is_dropped_at_disconnect(self):
        bus = PacketTransport(("a", "b"), delay_us=1_000, jitter_us=0, seed=0)
        bus.send("a", b"\x09" * 8, sim_us=0, message_id="late")
        bus.disconnect("b", sim_us=500)
        self.assertEqual(bus.drops["disconnected"], 1)
        self.assertEqual(bus.receive_until(2_000), [])

    def test_nonmonotonic_control_is_rejected(self):
        bus = PacketTransport(("a", "b"), delay_us=0, jitter_us=0, seed=0)
        bus.disconnect("b", sim_us=500)
        with self.assertRaises(ValueError):
            bus.reconnect("b", sim_us=400)

    def test_each_send_has_unique_tx_id(self):
        packet = b"\x0a" * 8
        bus = PacketTransport(("a", "b"), delay_us=50, jitter_us=0, seed=0)
        bus.send("a", packet, sim_us=0)
        bus.send("a", packet, sim_us=10)
        sent = [e for e in bus.events if e["type"] == "packet_sent"]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["evidence_ref"], sent[1]["evidence_ref"])
        tx = [e["detail"]["tx_id"] for e in sent]
        self.assertEqual(len(set(tx)), 2)
        got = bus.receive_until(100)
        self.assertEqual(len(got), 2)
        self.assertEqual({d.tx_id for d in got}, set(tx))
        delivered = [e for e in bus.events if e["type"] == "packet_delivered"]
        self.assertEqual({e["detail"]["tx_id"] for e in delivered}, set(tx))

    def test_delivery_callback_pairs_each_delivery(self):
        seen = []
        bus = PacketTransport(
            ("a", "b", "c"), delay_us=0, jitter_us=0, seed=0,
            delivery_callback=seen.append)
        bus.send("a", b"\x0b" * 4, sim_us=0)
        got = bus.receive_until(0)
        self.assertEqual(len(got), 2)
        self.assertEqual([d.receiver for d in seen], [d.receiver for d in got])
        self.assertEqual([d.tx_id for d in seen], [d.tx_id for d in got])


if __name__ == "__main__":
    unittest.main()
