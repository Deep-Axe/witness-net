from dataclasses import replace
import unittest

from coopx.domain import ObjectClass, Observation
from coopx.envelope import (Auth, Envelope, Lineage, ObservationMessage, Geometry,
                           EvidenceStore, sign, encode, decode, verify,
                           evidence_ref, byte_accounting)
from coopx.identity import NodeKey, TrustAnchor, TrustClass

class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.keys = {s: NodeKey.generate() for s in ("pole", "car", "relay", "other")}
        self.anchors = {
            s: TrustAnchor(k.public_key, s+"-key", TrustClass.A,
                           physical_observer=s,
                           permitted_region=(-100., -100., -5., 100., 100., 10.))
            for s, k in self.keys.items()}
        self.store = EvidenceStore(self.anchors)

    def original(self, source="pole", mid="o1", sensor="camera", seq=1, parents=(), **kwargs):
        o = Observation(source, 7, seq, 1000+seq, ObjectClass.PEDESTRIAN, (10., 2., 1.))
        e = Envelope(2, "observation", mid, source, "session", "site_map",
                     o.captured_sim_us, 2000, 5000, Auth(source+"-key"), seq)
        l = Lineage(source, sensor, mid, o.captured_sim_us, parents=parents)
        return decode(encode(sign(ObservationMessage(e, o, l, **kwargs), self.keys[source])))

    def relay(self, original, mid="relay1", **lineage_changes):
        e = replace(original.envelope, message_id=mid, source_id="relay",
                    generation_time_us=2100, auth=Auth("relay-key"))
        l = replace(original.lineage, derivation="relayed",
                    root_ref=evidence_ref(original), **lineage_changes)
        return sign(replace(original, envelope=e, lineage=l,
                            observation=replace(original.observation, source="relay")),
                    self.keys["relay"])

    def test_roundtrip_signature_and_full_accounting(self):
        message = self.original(label_id=0)
        self.assertEqual(decode(encode(message)), message)
        self.assertTrue(verify(message, self.anchors))
        info = byte_accounting(message)
        self.assertEqual(info["signature"], 64)
        self.assertEqual(info["object_body"], 59)
        self.assertEqual(info["total"], len(encode(message)))
        self.assertEqual(sum(info[k] for k in ("signature", "key_ref", "object_body", "envelope_and_lineage")), info["total"])
        self.assertEqual(message.label_id, 0)
        self.assertIsNone(self.original().label_id)

    def test_tampered_signature_body_and_metadata_fail(self):
        message = self.original()
        raw = encode(message)
        tampered = raw[:-1] + bytes([raw[-1] ^ 1])
        self.assertFalse(verify(decode(tampered), self.anchors))
        mutated = replace(message, observation=replace(message.observation, xyz_m=(11.,2.,1.)))
        self.assertFalse(verify(mutated, self.anchors))
        mutated = replace(message, envelope=replace(message.envelope, spatial_frame_id="wrong_map"))
        self.assertFalse(verify(mutated, self.anchors))
        with self.assertRaises(ValueError):
            self.store.add(mutated)

    def test_unknown_key_wrong_reference_and_unsigned_rejected(self):
        message = self.original()
        self.assertFalse(verify(message, {}))
        wrong = replace(message, envelope=replace(message.envelope, auth=Auth("other-key", message.envelope.auth.signature)))
        self.assertFalse(verify(wrong, self.anchors))
        with self.assertRaises(ValueError):
            encode(replace(message, envelope=replace(message.envelope, auth=Auth("pole-key"))))

    def test_original_and_five_relays_are_one_witness(self):
        original = self.original()
        self.store.add(original)
        relays = [self.relay(original, mid=f"r{i}") for i in range(5)]
        for r in relays:
            self.store.add(r)
            self.assertEqual(self.store.authenticated_roots(r), {evidence_ref(original)})
            self.assertEqual(decode(encode(r)).lineage.root_ref, evidence_ref(original))
        self.assertEqual(self.store.independent_witnesses([original]+relays), 1)

    def test_missing_original_fabricated_origin_or_modified_relay_no_credit(self):
        original = self.original()
        relay = self.relay(original)
        self.store.add(relay)
        self.assertEqual(self.store.independent_witnesses([relay]), 0)
        self.store.add(original)
        fake = self.relay(original, origin_observer="other")
        self.assertEqual(self.store.authenticated_roots(fake), set())
        moved = sign(replace(relay, observation=replace(relay.observation, xyz_m=(40.,2.,1.))), self.keys["relay"])
        self.assertEqual(self.store.authenticated_roots(moved), set())
        wrong_frame = sign(replace(relay, envelope=replace(relay.envelope, spatial_frame_id="elsewhere")), self.keys["relay"])
        self.assertEqual(self.store.authenticated_roots(wrong_frame), set())

    def test_ten_frames_one_sensor_one_witness(self):
        frames = [self.original(mid=f"f{i}", seq=i) for i in range(10)]
        for message in frames:
            self.store.add(message)
        self.assertEqual(self.store.independent_witnesses(frames), 1)

    def test_lidar_lift_and_same_lidar_one_witness(self):
        lidar = self.original(mid="lidar1", sensor="lidar")
        ref = self.store.add(lidar)
        camera = self.original(mid="camera1", parents=(ref,))
        self.store.add(camera)
        self.assertEqual(self.store.independent_witnesses([camera, lidar]), 1)

    def test_independent_observers_and_shared_causes(self):
        pole, car = self.original(), self.original("car")
        for message in (pole, car):
            self.store.add(message)
        self.assertEqual(self.store.independent_witnesses([pole, car]), 2)
        anchors = {s: replace(a, common_causes=("upstream-camera",)) for s, a in self.anchors.items()}
        shared = EvidenceStore(anchors)
        shared.add(pole)
        shared.add(car)
        self.assertEqual(shared.independent_witnesses([pole, car]), 1)

    def test_unanchored_peer_and_class_d_have_zero_weight(self):
        pole = self.original()
        anchors = {"pole": replace(self.anchors["pole"], physical_observer=None, trust_class=TrustClass.B)}
        store = EvidenceStore(anchors)
        store.add(pole)
        self.assertEqual(store.independent_witnesses([pole]), 0)
        anchors["pole"] = replace(self.anchors["pole"], trust_class=TrustClass.D)
        store = EvidenceStore(anchors)
        store.add(pole)
        self.assertEqual(store.independent_witnesses([pole]), 0)

    def test_bad_wire_truncation_trailing_flags_and_canonical_null(self):
        raw = encode(self.original())
        for n in range(len(raw)):
            with self.subTest(truncated=n), self.assertRaises(ValueError):
                decode(raw[:n])
        for bad in (raw+b"x", raw[:6]+bytes([128])+raw[7:], b"BAD!"+raw[4:]):
            with self.assertRaises(ValueError):
                decode(bad)
        # First unused velocity slot in the CX1 body must stay canonical zero.
        index = len(raw)-64-59+32
        bad = raw[:index]+bytes([1])+raw[index+1:]
        with self.assertRaises(ValueError):
            decode(bad)

    def test_permissions_and_metadata_validation(self):
        anchor = self.anchors["pole"]
        self.assertTrue(anchor.permits((0,0,0)))
        self.assertFalse(anchor.permits((101,0,0)))
        self.assertFalse(anchor.permits((float("nan"),0,0)))
        message = self.original()
        with self.assertRaises(ValueError):
            replace(message.envelope, valid_until_us=0)
        with self.assertRaises(ValueError):
            replace(message, geometry=Geometry.ORIENTED_BOX)
        with self.assertRaises(ValueError):
            replace(message, lineage=replace(message.lineage, origin_observer="other"))

    def test_fusion_preserves_independent_roots_and_dependency_collapses(self):
        pole, car = self.original(), self.original("car")
        pole_ref, car_ref = self.store.add(pole), self.store.add(car)
        fused = self.relay(pole)
        fused = sign(replace(fused, lineage=replace(
            fused.lineage, derivation="fused", parents=(car_ref,))), self.keys["relay"])
        self.store.add(fused)
        self.assertEqual(self.store.independent_witnesses([fused]), 2)
        # A measurement derived using another observer's evidence is correlated
        # with that upstream observer, even though its signer is different.
        derived = self.original("car", mid="derived", parents=(pole_ref,))
        self.store.add(derived)
        self.assertEqual(self.store.independent_witnesses([derived, pole]), 1)

    def test_relay_extra_parents_cannot_manufacture_witnesses(self):
        pole, car = self.original(), self.original("car")
        self.store.add(pole)
        ref = self.store.add(car)
        relay = self.relay(pole, parents=(ref,))
        self.store.add(relay)
        self.assertEqual(self.store.independent_witnesses([relay]), 1)

    def test_multiple_trusted_identities_one_physical_observer(self):
        anchors = {s: replace(a, physical_observer="one-body") for s, a in self.anchors.items()}
        store = EvidenceStore(anchors)
        messages = [self.original(s) for s in self.keys]
        for message in messages:
            store.add(message)
        self.assertEqual(store.independent_witnesses(messages), 1)

    def test_missing_parent_fails_closed(self):
        message = self.original(parents=("0"*64,))
        self.store.add(message)
        self.assertEqual(self.store.authenticated_roots(message), set())

if __name__ == "__main__":
    unittest.main()
