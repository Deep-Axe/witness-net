from dataclasses import replace
import unittest
from unittest.mock import patch

from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, Lineage, ObservationMessage, Geometry,
                           EvidenceStore, sign, encode, decode, verify,
                           evidence_ref, byte_accounting, default_agrees)
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

    def alter_observation(self, message, **changes):
        observation = replace(message.observation, **changes)
        capture = observation.captured_sim_us
        envelope = replace(message.envelope, capture_time_us=capture,
                           generation_time_us=max(capture, message.envelope.generation_time_us),
                           valid_until_us=max(capture, message.envelope.generation_time_us) + 1_000_000)
        lineage = message.lineage
        if lineage.derivation == "direct":
            lineage = replace(lineage, origin_capture_time_us=capture)
        return sign(replace(message, observation=observation, envelope=envelope,
                            lineage=lineage), self.keys[envelope.source_id])

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
        self.assertEqual(self.store.independent_witnesses(original, [original]+relays), 1)

    def test_missing_original_fabricated_origin_or_modified_relay_no_credit(self):
        original = self.original()
        relay = self.relay(original)
        self.store.add(relay)
        self.assertEqual(self.store.independent_witnesses(original, [relay]), 0)
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
        self.assertEqual(self.store.independent_witnesses(frames[-1], frames), 1)

    def test_lidar_lift_and_same_lidar_one_witness(self):
        lidar = self.original(mid="lidar1", sensor="lidar")
        ref = self.store.add(lidar)
        camera = self.original(mid="camera1", parents=(ref,))
        self.store.add(camera)
        self.assertEqual(self.store.independent_witnesses(camera, [camera, lidar]), 1)

    def test_independent_observers_and_shared_causes(self):
        pole, car = self.original(), self.original("car")
        for message in (pole, car):
            self.store.add(message)
        self.assertEqual(self.store.independent_witnesses(pole, [pole, car]), 2)
        anchors = {s: replace(a, common_causes=("upstream-camera",)) for s, a in self.anchors.items()}
        shared = EvidenceStore(anchors)
        shared.add(pole)
        shared.add(car)
        self.assertEqual(shared.independent_witnesses(pole, [pole, car]), 1)

    def test_unanchored_peer_and_class_d_have_zero_weight(self):
        pole = self.original()
        anchors = {"pole": replace(self.anchors["pole"], physical_observer=None, trust_class=TrustClass.B)}
        store = EvidenceStore(anchors)
        store.add(pole)
        self.assertEqual(store.independent_witnesses(pole, [pole]), 0)
        anchors["pole"] = replace(self.anchors["pole"], trust_class=TrustClass.D)
        store = EvidenceStore(anchors)
        store.add(pole)
        self.assertEqual(store.independent_witnesses(pole, [pole]), 0)

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
        self.assertEqual(self.store.independent_witnesses(fused, [fused]), 2)
        # A measurement derived using another observer's evidence is correlated
        # with that upstream observer, even though its signer is different.
        derived = self.original("car", mid="derived", parents=(pole_ref,))
        self.store.add(derived)
        self.assertEqual(self.store.independent_witnesses(derived, [derived, pole]), 1)

    def test_relay_extra_parents_cannot_manufacture_witnesses(self):
        pole, car = self.original(), self.original("car")
        self.store.add(pole)
        ref = self.store.add(car)
        relay = self.relay(pole, parents=(ref,))
        self.store.add(relay)
        self.assertEqual(self.store.independent_witnesses(pole, [relay]), 1)

    def test_multiple_trusted_identities_one_physical_observer(self):
        anchors = {s: replace(a, physical_observer="one-body") for s, a in self.anchors.items()}
        store = EvidenceStore(anchors)
        messages = [self.original(s) for s in self.keys]
        for message in messages:
            store.add(message)
        self.assertEqual(store.independent_witnesses(messages[0], messages), 1)

    def test_missing_parent_fails_closed(self):
        message = self.original(parents=("0"*64,))
        self.store.add(message)
        self.assertEqual(self.store.authenticated_roots(message), set())

    def test_signature_and_roots_are_cached_under_immutable_anchor_snapshot(self):
        original = self.original()
        actual = NodeKey.verify
        with patch.object(NodeKey, 'verify', wraps=actual) as verify_call:
            ref = self.store.add(original)
            for _ in range(10):
                self.store.add(encode(original))
                self.assertEqual(self.store.authenticated_roots(original), {ref})
            self.assertEqual(verify_call.call_count, 1)
        with self.assertRaises(TypeError):
            self.store.anchors['pole'] = self.anchors['car']
        # Changing the caller's input mapping has no effect on the snapshot.
        self.anchors.clear()
        self.assertEqual(self.store.authenticated_roots(original), {ref})
        self.store.replace_anchors({})
        self.assertEqual(self.store.authenticated_roots(original), set())

    def test_pruned_original_fails_closed_then_resolves_when_restored(self):
        original = self.original()
        ref = self.store.add(original)
        relay = self.relay(original)
        self.store.add(relay)
        self.assertEqual(self.store.authenticated_roots(relay), {ref})
        self.store.discard(ref)
        self.assertEqual(self.store.authenticated_roots(relay), set())
        self.store.add(original)
        self.assertEqual(self.store.authenticated_roots(relay), {ref})

    def test_recent_observation_index_is_inclusive_and_discard_safe(self):
        messages = [self.original(mid=f'index-{i}', seq=i) for i in range(20)]
        for message in reversed(messages):
            self.store.add(message)
        recent = self.store.recent_observations(1007, 1009)
        self.assertEqual([m.envelope.capture_time_us for m in recent], [1007, 1008, 1009])
        self.store.discard(evidence_ref(recent[1]))
        self.assertEqual(len(self.store.recent_observations(1007, 1009)), 2)
        self.assertEqual(self.store.recent_observations(5000, 6000), ())

    def test_nested_anchor_inputs_are_frozen_not_aliased_into_caches(self):
        region=[-100.,-100.,-5.,100.,100.,10.]
        causes=['calibration-1']
        anchor=replace(self.anchors['pole'], permitted_region=region, common_causes=causes)
        store=EvidenceStore({'pole':anchor})
        region[3]=10000.
        causes.append('changed')
        self.assertEqual(store.anchors['pole'].permitted_region[3],100.)
        self.assertEqual(store.anchors['pole'].common_causes,('calibration-1',))

    def test_decoded_wire_cache_is_invalidated_by_dataclass_replacement(self):
        original = self.original()
        self.assertIsNotNone(original._wire)
        modified = replace(original, observation=replace(original.observation, xyz_m=(50., 2., 1.)))
        self.assertIsNone(modified._wire)
        self.assertFalse(verify(modified, self.anchors))

    def test_lineage_cannot_launder_unrelated_content_into_witness_credit(self):
        pole = self.original()
        self.store.add(pole)
        tracked = self.relay(pole)
        tracked = self.alter_observation(replace(tracked, lineage=replace(
            tracked.lineage, derivation="tracked")), xyz_m=(40., -20., 1.))
        self.store.add(tracked)
        ghost = self.alter_observation(self.original("relay", mid="ghost"),
                                       xyz_m=(40., -20., 1.))
        self.store.add(ghost)
        self.assertEqual(self.store.independent_witnesses(tracked, [tracked]), 0)
        self.assertEqual(self.store.independent_witnesses(tracked, [tracked, ghost]), 1)
        self.assertEqual(self.store.independent_witnesses(pole, [pole]), 1)
        self.assertEqual(self.store.independent_witnesses(pole, [pole, self.original("car")]), 2)

    def test_legitimate_fusion_can_differ_from_its_supporting_roots(self):
        pole = self.original()
        car = self.alter_observation(self.original("car"), xyz_m=(11., 2., 1.))
        self.store.add(pole)
        car_ref = self.store.add(car)
        fused = self.relay(pole)
        fused = replace(fused, lineage=replace(fused.lineage, derivation="fused", parents=(car_ref,)))
        fused = self.alter_observation(fused, xyz_m=(10.5, 2., 1.))
        self.assertEqual(self.store.independent_witnesses(fused, [fused]), 2)

    def test_default_agreement_time_class_distance_and_velocity_gates(self):
        root = self.original()
        on_edge = self.alter_observation(root, xyz_m=(11.5, 2., 1.), captured_sim_us=301001)
        self.assertTrue(default_agrees(root, on_edge))
        self.assertFalse(default_agrees(root, self.alter_observation(on_edge, captured_sim_us=301002)))
        self.assertFalse(default_agrees(root, self.alter_observation(root, xyz_m=(11.501, 2., 1.))))
        self.assertFalse(default_agrees(root, self.alter_observation(root, xyz_m=(10., 2., 2.501))))
        self.assertFalse(default_agrees(root, self.alter_observation(root, cls=ObjectClass.CYCLIST)))
        car = self.alter_observation(root, cls=ObjectClass.VEHICLE)
        self.assertTrue(default_agrees(car, self.alter_observation(car, xyz_m=(13., 2., 1.))))
        self.assertFalse(default_agrees(car, self.alter_observation(car, xyz_m=(13.001, 2., 1.))))
        moving = self.alter_observation(root, velocity_mps=(10., 0., 0.))
        later = self.alter_observation(root, captured_sim_us=201001, xyz_m=(13.4, 2., 1.))
        self.assertTrue(default_agrees(moving, later))
        self.assertFalse(default_agrees(root, later))
        # An invented velocity on the derived claim cannot widen a genuine root.
        self.assertFalse(default_agrees(root, self.alter_observation(later, velocity_mps=(1e6, 0., 0.))))

    def test_default_agreement_rejects_incompatible_frame_anchor_and_label(self):
        root = self.original(label_id=7)
        self.assertFalse(default_agrees(root, replace(root, label_id=8)))
        self.assertFalse(default_agrees(root, replace(root, label_id=None)))
        self.assertFalse(default_agrees(root, replace(root, envelope=replace(
            root.envelope, spatial_frame_id="different_map"))))
        self.assertFalse(default_agrees(root, self.alter_observation(root, anchor=PositionAnchor.CENTRE)))
        unknown = self.alter_observation(root, cls=ObjectClass.UNKNOWN)
        self.assertFalse(default_agrees(unknown, root))

    def test_custom_agreement_receives_original_content(self):
        root = self.original()
        self.store.add(root)
        derived = self.relay(root)
        derived = self.alter_observation(replace(derived, lineage=replace(
            derived.lineage, derivation="tracked")), xyz_m=(40., -20., 1.))
        seen = []
        def agrees(original, claim):
            seen.append((original.observation.xyz_m, claim.observation.xyz_m))
            return original.observation.xyz_m == claim.observation.xyz_m
        self.assertEqual(self.store.independent_witnesses(derived, [derived], agrees), 0)
        self.assertEqual(seen, [(root.observation.xyz_m, derived.observation.xyz_m)])

    def test_configured_class_b_sybils_need_receiver_established_anchoring(self):
        messages = []
        anchors = {}
        for index in range(5):
            source = f"sybil-{index}"
            self.keys[source] = NodeKey.generate()
            anchors[source] = TrustAnchor(self.keys[source].public_key, source+"-key", TrustClass.B,
                                           physical_observer=f"body-{index}")
            messages.append(self.original(source))
        store = EvidenceStore(anchors)
        for message in messages:
            store.add(message)
        self.assertEqual(store.independent_witnesses(messages[0], messages), 0)
        anchored = EvidenceStore({source: replace(anchor, anchored=True) for source, anchor in anchors.items()})
        for message in messages:
            anchored.add(message)
        self.assertEqual(anchored.independent_witnesses(messages[0], messages), 5)
        same_body = EvidenceStore({source: replace(anchor, anchored=True, physical_observer="one-body")
                                   for source, anchor in anchors.items()})
        self.assertEqual(same_body.independent_witnesses(messages[0], messages), 1)
        class_d = EvidenceStore({source: replace(anchor, anchored=True, trust_class=TrustClass.D)
                                 for source, anchor in anchors.items()})
        self.assertEqual(class_d.independent_witnesses(messages[0], messages), 0)
        class_c = EvidenceStore({source: replace(anchor, trust_class=TrustClass.C)
                                 for source, anchor in anchors.items()})
        self.assertEqual(class_c.independent_witnesses(messages[0], messages), 5)
        with self.assertRaises(ValueError):
            replace(next(iter(anchors.values())), anchored="yes")

if __name__ == "__main__":
    unittest.main()
