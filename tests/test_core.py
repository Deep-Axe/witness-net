import base64
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from coopx.cli import replay_packets, synthetic
from coopx.control import FrameBarrier
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.radio_codec import RECORD, decode, encode, validate_payload
from coopx.transport import ReferenceTransport


def observation(**fields):
    original = Observation("rsu-1", 7, 3, 1_000_000, ObjectClass.PEDESTRIAN,
                           (12.25, -3.5, 1.0))
    return replace(original, **fields)


class CodecTests(unittest.TestCase):
    def test_full_record_roundtrip_and_quantization(self):
        original = observation(anchor=PositionAnchor.CENTRE,
                               velocity_mps=(0, -1.3, 0.2), dimensions_m=(0.6, 0.6, 1.7),
                               yaw_deg=-90.125, confidence=0.913, position_std_m=(0.2, 0.4, 0.1))
        payload = encode(original)
        self.assertEqual(len(payload), 84)
        received = decode(payload, source="rsu-1")
        self.assertEqual(received.key, original.key)
        self.assertEqual(received.sequence, original.sequence)
        self.assertEqual(received.captured_sim_us, original.captured_sim_us)
        self.assertEqual(received.anchor, PositionAnchor.CENTRE)
        for name, tolerance in (("xyz_m", 1e-5), ("velocity_mps", 1e-5),
                                ("dimensions_m", 0.005), ("position_std_m", 0.005)):
            for value, expected in zip(getattr(received, name), getattr(original, name)):
                self.assertAlmostEqual(value, expected, delta=tolerance)
        self.assertAlmostEqual(received.yaw_deg, original.yaw_deg, delta=0.0051)
        self.assertAlmostEqual(received.confidence, original.confidence, delta=0.5 / 255)

    def test_null_and_zero_are_distinct(self):
        absent = decode(encode(observation()), source="rsu-1")
        self.assertIsNone(absent.velocity_mps)
        self.assertIsNone(absent.confidence)
        zero = decode(encode(observation(velocity_mps=(0, 0, 0), confidence=0)), source="rsu-1")
        self.assertEqual(zero.velocity_mps, (0, 0, 0))
        self.assertEqual(zero.confidence, 0)

    def test_source_namespace_is_from_receiver_mapping(self):
        payload = encode(observation())
        self.assertNotEqual(decode(payload, source="rsu-1").key,
                            decode(payload, source="helper-1").key)

    def test_reject_unrepresentable_values(self):
        for changes in ({"xyz_m": (math.nan, 1, 2)}, {"confidence": math.inf},
                        {"object_id": -1}, {"sequence": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observation(**changes)
        for changes in ({"xyz_m": (1e100, 1, 2)}, {"dimensions_m": (1000, 1, 2)},
                        {"dimensions_m": (0.001, 1, 2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                encode(observation(**changes))

    def test_reject_malformed_radio_input(self):
        for payload in ("x" * 100, "abc\x00", "é", "CX1:?!", "CX2:abcd", "CX1:AAAA"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode(payload, source="rsu-1")
        values = list(RECORD.unpack(base64.b64decode(encode(observation())[4:])))
        values[2] |= 1 << 15
        malformed = "CX1:" + base64.b64encode(RECORD.pack(*values)).decode()
        with self.assertRaises(ValueError):
            decode(malformed, source="rsu-1")
        self.assertEqual(validate_payload(encode(observation())), 84)

    def test_freshness_uses_capture_time(self):
        received = decode(encode(observation()), source="rsu-1")
        self.assertTrue(received.is_fresh(1_400_000, 500_000))
        self.assertFalse(received.is_fresh(1_600_000, 500_000))
        self.assertFalse(received.is_fresh(900_000, 500_000))


class ControlTests(unittest.TestCase):
    def test_barrier_blocks_until_every_control_worker_acknowledges(self):
        barrier = FrameBarrier(frozenset({"scenario", "ego-control"}))
        barrier.begin(40)
        barrier.acknowledge("scenario", 40)
        with self.assertRaises(RuntimeError):
            barrier.begin(41)
        barrier.acknowledge("ego-control", 40)
        self.assertTrue(barrier.ready)
        barrier.begin(41)

    def test_duplicate_wrong_frame_and_unknown_ack_fail(self):
        barrier = FrameBarrier(frozenset({"scenario"}))
        barrier.begin(10)
        for worker, frame in (("scenario", 9), ("scenario", 11), ("other", 10)):
            with self.assertRaises(ValueError):
                barrier.acknowledge(worker, frame)
        barrier.acknowledge("scenario", 10)
        with self.assertRaises(ValueError):
            barrier.acknowledge("scenario", 10)
        with self.assertRaises(ValueError):
            barrier.begin(10)


class TransportTests(unittest.TestCase):
    def test_delay_and_broadcast_accounting(self):
        bus = ReferenceTransport(("rsu-1", "helper", "ego"), delay_us=100)
        bus.send("rsu-1", encode(observation()), frame=10, sim_us=1000)
        self.assertEqual(bus.receive_until(1099), [])
        self.assertEqual(bus.sent_bytes, 84)
        ready = bus.receive_until(1100)
        self.assertEqual({r.receiver for r in ready}, {"helper", "ego"})
        self.assertEqual(bus.received_bytes, 168)

    def test_loss_does_not_reduce_sender_bytes(self):
        bus = ReferenceTransport(("rsu-1", "ego"), loss_probability=1)
        bus.send("rsu-1", encode(observation()), frame=10, sim_us=1_000_000)
        self.assertEqual(bus.receive_until(2_000_000), [])
        self.assertEqual(bus.sent_bytes, 84)
        self.assertEqual(bus.received_bytes, 0)
        self.assertEqual(bus.dropped_deliveries, 1)

    def test_no_second_send_per_tick_or_backwards_clock(self):
        bus = ReferenceTransport(("rsu-1", "ego"))
        payload = encode(observation())
        bus.send("rsu-1", payload, frame=10, sim_us=1_000_000)
        with self.assertRaises(ValueError):
            bus.send("rsu-1", payload, frame=10, sim_us=1_000_000)
        bus.receive_until(1_100_000)
        with self.assertRaises(ValueError):
            bus.send("rsu-1", payload, frame=11, sim_us=1_000_000)
        with self.assertRaises(ValueError):
            bus.receive_until(1_000_000)

    def test_seeded_loss_reproducibility(self):
        def run():
            bus = ReferenceTransport(("rsu-1", "ego"), loss_probability=0.5, seed=41)
            for frame in range(20):
                bus.send("rsu-1", encode(observation()), frame=frame, sim_us=frame * 1000)
            return bus.receive_until(1_000_000), bus.dropped_deliveries
        self.assertEqual(run(), run())


class GeometryTests(unittest.TestCase):
    def test_depth_channel_order_and_wide_arithmetic(self):
        import numpy as np
        from coopx.geometry import decode_depth_bgra
        pixels = np.array([[[1, 2, 3, 255], [255, 255, 255, 255]]], dtype=np.uint8)
        values = decode_depth_bgra(pixels)
        self.assertAlmostEqual(values[0, 0], (3 + 2 * 256 + 65536) * 1000 / (256**3 - 1))
        self.assertEqual(values[0, 1], 1000)

    def test_off_axis_axial_range_and_world_transform(self):
        import numpy as np
        from coopx.geometry import CameraIntrinsics, backproject
        camera = CameraIntrinsics(100, 100, 90)
        axial = backproject([[100, 50]], [10], camera, np.eye(4), depth_mode="axial")
        radial = backproject([[100, 50]], [10], camera, np.eye(4), depth_mode="range")
        np.testing.assert_allclose(axial, [[10, 10, 0]], atol=1e-12)
        np.testing.assert_allclose(radial, [[10 / math.sqrt(2), 10 / math.sqrt(2), 0]])
        transform = np.array([[0, -1, 0, 5], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        centre = backproject([[50, 50]], [10], camera, transform, depth_mode="axial")
        np.testing.assert_allclose(centre, [[5, 12, 3]])
        with self.assertRaises(ValueError):
            backproject([[50, 50]], [10], camera, np.diag([2, 2, 2, 1]), depth_mode="axial")

    def test_mask_lift_excludes_background_and_requires_support(self):
        import numpy as np
        from coopx.geometry import CameraIntrinsics, lift_mask_surface
        camera = CameraIntrinsics(20, 20, 90)
        depth = np.full((20, 20), 1000.0)
        depth[6:15, 6:15] = 10
        mask = np.zeros((20, 20), dtype=bool)
        mask[6:15, 6:15] = True
        result = lift_mask_surface(depth, mask, camera, np.eye(4), depth_mode="axial")
        np.testing.assert_allclose(result.xyz_m, [10, 0, 0], atol=1e-12)
        self.assertEqual(result.supporting_pixels, 49)
        with self.assertRaises(ValueError):
            lift_mask_surface(depth, np.zeros_like(mask), camera, np.eye(4), depth_mode="axial")


class PipelineTests(unittest.TestCase):
    def test_synthetic_replay_preserves_capture_age_and_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            result = synthetic(path)
            self.assertEqual(result["paid_calls"], 0)
            self.assertEqual(result["sent_bytes"], 168)
            self.assertEqual(result["received_bytes"], 168)
            replay = replay_packets(path)
            self.assertEqual([r["fresh"] for r in replay], [True, False])
            self.assertEqual([r["captured_sim_us"] for r in replay], [2_000_000] * 2)
            with self.assertRaises(FileExistsError):
                synthetic(path)


if __name__ == "__main__":
    unittest.main()
