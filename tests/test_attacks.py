"""Attack plans are deterministic and signed by the attacker. No network."""
import unittest

from coopx.attacks import ATTACK_KINDS, apply_attack, attack_anchors, plan_attack
from coopx.checker import Checker
from coopx.envelope import decode, verify
from coopx.transport import PacketTransport

SCENE = {"target_xyz": (21.0, -180.0, 0.5), "ego_xyz": (14.0, -171.0, 0.5),
         "disconnect_node": "rsu-2"}
REGION = (-1000.0, -1000.0, -50.0, 1000.0, 1000.0, 80.0)
NOW = 10_000_000


class AttackTests(unittest.TestCase):
    def test_every_kind_is_deterministic_and_signed(self):
        for kind in ATTACK_KINDS:
            with self.subTest(kind=kind):
                first = plan_attack(kind, seed=3, scene=SCENE, now_us=NOW)
                second = plan_attack(kind, seed=3, scene=SCENE, now_us=NOW)
                self.assertEqual([p.raw for p in first.packets], [p.raw for p in second.packets])
                self.assertEqual(first.loss, second.loss)
                self.assertEqual(first.disconnect, second.disconnect)
                _keys, anchors = attack_anchors(first, region=REGION)
                for packet in first.packets:
                    message = decode(packet.raw)
                    self.assertTrue(verify(message, anchors), kind)

    def test_replay_old_is_stale_and_refreshed_is_not_a_forgery(self):
        old = plan_attack("replay_old", seed=1, scene=SCENE, now_us=NOW)
        checker = Checker(attack_anchors(old, region=REGION)[1], ttl_us=5_000_000)
        result = checker.check(old.packets[0].raw, NOW + 100_000)
        self.assertIn("STALE", result.codes)
        fresh = plan_attack("replay_refreshed", seed=1, scene=SCENE, now_us=NOW)
        self.assertNotEqual(old.packets[0].raw, fresh.packets[0].raw)
        self.assertEqual(decode(fresh.packets[0].raw).envelope.capture_time_us, NOW)

    def test_each_attack_is_visible_on_the_transport(self):
        for kind in ATTACK_KINDS:
            with self.subTest(kind=kind):
                plan = plan_attack(kind, seed=5, scene=SCENE, now_us=NOW)
                nodes = ["ego", "rsu-2"] + [node.name for node in plan.nodes]
                events = []
                transport = PacketTransport(
                    tuple(dict.fromkeys(nodes)), delay_us=0, loss_probability=plan.loss,
                    seed=5, event_callback=events.append)
                if kind == "loss_sweep":
                    from coopx.attacks import _one
                    nodes_extra, packets = _one("ghost", (1.0, 2.0, 0.5), __import__(
                        "coopx.domain", fromlist=["ObjectClass"]).ObjectClass.PEDESTRIAN, NOW, 5)
                    transport = PacketTransport(
                        ("ego", "ghost"), delay_us=0, loss_probability=plan.loss, seed=5,
                        event_callback=events.append)
                    transport.send("ghost", packets[0].raw, sim_us=NOW, message_id="g")
                else:
                    apply_attack(transport, plan)
                transport.receive_until(NOW + 2_000_000)
                kinds = {event["type"] for event in events}
                self.assertTrue(
                    kinds & {"packet_sent", "packet_dropped", "observer_disconnected"}, kind)


if __name__ == "__main__":
    unittest.main()
