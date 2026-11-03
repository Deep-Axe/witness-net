from dataclasses import replace
import random
import unittest

from coopx.checker import Checker
from coopx.coverage import (CoverageContent, CoverageMap, CoverageMessage, GridSpec,
                           MAX_GRID_CELLS)
from coopx.domain import ObjectClass, Observation, PositionAnchor
from coopx.envelope import (Auth, Envelope, EvidenceStore, Geometry, Lineage, ObservationMessage,
                           byte_accounting, decode, encode, evidence_ref, sign, verify)
from coopx.identity import NodeKey, TrustAnchor, TrustClass


class Fixture:
    def __init__(self, spec=None):
        self.spec = spec or GridSpec(0., 0., 1., 20, 6)
        self.keys = {s: NodeKey.generate() for s in ('pole', 'lidar', 'car', 'ego', 'attacker', 'other')}
        self.anchors = {s: TrustAnchor(key.public_key, s+'-key',
            TrustClass.B if s == 'attacker' else TrustClass.A, physical_observer=s,
            permitted_region=(-100., -100., -10., 100., 100., 10.)) for s, key in self.keys.items()}
        self.store = EvidenceStore(self.anchors)
        self.sequences = {s: 0 for s in self.keys}

    def envelope(self, source, kind, capture, expiry, seq):
        if seq is None:
            seq = self.sequences[source]
            self.sequences[source] += 1
        e = Envelope(2, kind, f'{source}-{kind}-{seq}', source, 'run', 'site_map',
                     capture, capture, capture + 1_000_000 if expiry is None else expiry,
                     Auth(source+'-key'), seq)
        return e, Lineage(source, 'camera' if kind == 'observation' else 'lidar', e.message_id, capture)

    def coverage(self, *, source='lidar', free=(), occupied=(), observed=None,
                 modality='lidar', capture=1_000_000, expiry=None, seq=None,
                 height=(0.2, 2.0), min_size=.3):
        free, occupied = frozenset(free), frozenset(occupied)
        c = CoverageContent(self.spec, (capture, capture), modality, height, min_size,
                            free | occupied if observed is None else frozenset(observed), free, occupied)
        e, l = self.envelope(source, 'coverage', capture, expiry, seq)
        return sign(CoverageMessage(e, c, l), self.keys[source])

    def observation(self, *, source='pole', xyz=(3.5, 1.5, 1.), cls=ObjectClass.PEDESTRIAN,
                    capture=1_000_000, expiry=None, seq=None, object_id=7, std=None,
                    dimensions=None, yaw=None, anchor=PositionAnchor.SURFACE, parents=()):
        e, l = self.envelope(source, 'observation', capture, expiry, seq)
        l = replace(l, parents=parents)
        o = Observation(source, object_id, e.sensor_seq, capture, cls, xyz, anchor,
                        dimensions_m=dimensions, yaw_deg=yaw, position_std_m=std)
        geometry = Geometry.ORIENTED_BOX if dimensions is not None and yaw is not None else Geometry.POINT_STD
        return sign(ObservationMessage(e, o, l, geometry=geometry), self.keys[source])


class CoverageCodecTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()

    def test_signed_bitmap_roundtrip_store_and_accounting(self):
        original = self.f.coverage(free={(1, 2), (3, 4)}, occupied={(10, 1)})
        raw = encode(original)
        received = decode(raw)
        self.assertEqual(received, original)
        self.assertTrue(verify(received, self.f.anchors))
        self.assertEqual(self.f.store.add(raw), evidence_ref(original))
        self.assertEqual(self.f.store.authenticated_roots(original), {evidence_ref(original)})
        info = byte_accounting(received)
        self.assertEqual(info['signature'], 64)
        self.assertEqual(info['total'], len(raw))
        self.assertEqual(info['coverage_body'], 69 + 3 * ((20*6+7)//8))
        self.assertEqual(sum(info[k] for k in ('signature', 'key_ref', 'coverage_body', 'envelope_and_lineage')), len(raw))
        self.assertEqual(self.f.store.messages(), (received,))

    def test_tampered_signature_and_content_are_not_authenticated(self):
        original = self.f.coverage(free={(1, 2)})
        raw = encode(original)
        tampered = raw[:-1] + bytes([raw[-1] ^ 1])
        self.assertFalse(verify(decode(tampered), self.f.anchors))
        with self.assertRaises(ValueError):
            self.f.store.add(tampered)
        changed = replace(original, content=replace(original.content, min_size_m=.5))
        self.assertFalse(verify(changed, self.f.anchors))

    def test_invalid_grid_and_content_invariants(self):
        for args in ((0, 0, 0, 1, 1), (0, 0, 1, True, 1), (0, 0, 1, 0, 1),
                     (0, 0, 1, 512, 512), (float('nan'), 0, 1, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                GridSpec(*args)
        c = self.f.coverage(free={(1, 1)}).content
        for fields in ({'observed': frozenset()}, {'occupied': frozenset({(1, 1)})},
                       {'free': frozenset({(-1, 1)})}, {'free': {(1, 1)}},
                       {'observed': frozenset({(True, 1)})}, {'window_us': (2, 1)},
                       {'window_us': (-1, 0)}, {'height_band_m': (2, 1)},
                       {'min_size_m': 0}, {'min_size_m': float('inf')}, {'modality': 'radar'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(c, **fields)
        with self.assertRaises(ValueError):
            self.f.coverage(free={(1, 1)}, modality='camera_rgb')
        original = self.f.coverage()
        with self.assertRaises(ValueError):
            replace(original, content=replace(original.content, window_us=(0, 1)))
        with self.assertRaises(ValueError):
            replace(original, lineage=replace(original.lineage, origin_observer='other'))

    def test_grid_half_open_boundaries(self):
        spec = GridSpec(-10, -30, 1, 60, 60)
        self.assertEqual(spec.cell_at(-10, -30), (0, 0))
        self.assertEqual(spec.cell_at(-10.01, -30.01), (-1, -1))
        self.assertFalse(spec.contains(spec.cell_at(50, 30)))
        self.assertTrue(spec.contains((59, 59)))

    def test_hard_size_bound_and_noncanonical_padding_and_truncation(self):
        f = Fixture(GridSpec(0, 0, 1, 256, 256))
        original = f.coverage(free={(0, 0), (255, 255)})
        raw = encode(original)
        self.assertEqual(f.spec.nx*f.spec.ny, MAX_GRID_CELLS)
        self.assertLess(len(raw), 65_536)
        self.assertEqual(decode(raw), original)
        with self.assertRaises(ValueError):
            decode(raw + bytes(65_537-len(raw)))
        small = Fixture(GridSpec(0, 0, 1, 3, 1)).coverage(free={(0, 0)})
        raw = encode(small)
        for n in (0, 4, 63, 100, len(raw)-1):
            with self.subTest(length=n), self.assertRaises(ValueError):
                decode(raw[:n])
        # Bit 7 is outside the 3-cell observed bitmap: ignored bits are not canonical.
        bad = bytearray(raw)
        bad[-64-3] |= 128
        with self.assertRaises(ValueError):
            decode(bytes(bad))

    def test_coverage_relay_preserves_authenticated_root_but_not_semantic_witness(self):
        original = self.f.coverage(occupied={(3, 1)})
        ref = self.f.store.add(original)
        envelope = replace(original.envelope, source_id='car', message_id='relay', auth=Auth('car-key'))
        lineage = replace(original.lineage, derivation='relayed', root_ref=ref)
        relay = sign(replace(original, envelope=envelope, lineage=lineage), self.f.keys['car'])
        self.f.store.add(relay)
        self.assertEqual(self.f.store.authenticated_roots(relay), {ref})
        subject = self.f.observation()
        self.assertEqual(self.f.store.independent_witnesses(subject, [original, relay]), 0)

    def test_checker_admits_coverage_and_keeps_observation_latest_separate(self):
        checker = Checker(self.f.anchors, roi=(0, 0, 0, 20, 6, 2))
        observation = self.f.observation(source='lidar')
        self.assertEqual(checker.check(encode(observation), 1_000_000).codes, ('OK',))
        message = self.f.coverage(free={(1, 1)})
        self.assertEqual(checker.check(encode(message), 1_000_000).codes, ('OK',))
        self.assertIsInstance(checker.check(encode(message), 1_000_000).message, CoverageMessage)
        self.assertEqual(checker.latest('lidar', 7).observation, observation.observation)

    def test_checker_auth_time_replay_and_full_grid_bounds(self):
        m = self.f.coverage(free={(1, 1)})
        raw = encode(m)
        forged = raw[:-1] + bytes([raw[-1] ^ 1])
        self.assertEqual(Checker(self.f.anchors).check(forged, 1_000_000).codes, ('BAD_SIGNATURE',))
        self.assertEqual(Checker(self.f.anchors).check(raw, 2_000_001).codes, ('STALE', 'EXPIRED'))
        checker = Checker(self.f.anchors)
        self.assertEqual(checker.check(raw, 1_000_000).codes, ('OK',))
        self.assertEqual(checker.check(raw, 1_000_000).codes, ('DUPLICATE',))
        self.assertEqual(Checker(self.f.anchors, roi=(0, 0, 0, 19, 6, 2)).check(raw, 1_000_000).codes,
                         ('OUT_OF_BOUNDS',))
        anchors = dict(self.f.anchors)
        anchors['lidar'] = replace(anchors['lidar'], permitted_region=(0, 0, 0, 19, 6, 2))
        self.assertEqual(Checker(anchors).check(raw, 1_000_000).codes, ('OUTSIDE_PERMITTED_REGION',))

    def test_checker_window_handles_coverage_and_observation_jitter_together(self):
        checker = Checker(self.f.anchors)
        messages = [self.f.coverage(seq=10), self.f.observation(source='lidar', seq=12),
                    self.f.coverage(seq=11), self.f.observation(source='lidar', seq=13)]
        for message in messages:
            self.assertEqual(checker.check(encode(message), 1_000_000).codes, ('OK',))
        self.assertEqual(checker.latest('lidar', 7).envelope.sensor_seq, 12)

    def test_one_thousand_mutated_coverage_packets_fail_without_exceptions(self):
        checker = Checker(self.f.anchors)
        raw = encode(self.f.coverage(free={(1, 1), (2, 1)}, occupied={(3, 1)}))
        rng = random.Random(2)
        for index in range(1000):
            packet = bytearray(raw)
            offset = rng.randrange(len(packet))
            packet[offset] ^= rng.randrange(1, 256)
            with self.subTest(index=index):
                result = checker.check(bytes(packet), 1_000_000)
                self.assertEqual(result.state, 'rejected')
                self.assertNotIn('OK', result.codes)


class CoverageMapTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.map = CoverageMap(self.f.spec, store=self.f.store)

    def test_wall_free_before_return_unknown_behind_and_camera_absence(self):
        message = self.f.coverage(free={(x, 1) for x in range(10)}, occupied={(10, 1)})
        self.map.add(message, 1_000_000)
        snapshot = self.map.snapshot(1_000_000)
        self.assertIn([9, 1], snapshot['free'])
        self.assertIn([10, 1], snapshot['occupied'])
        self.assertEqual(self.map.claims_at((11, 1), 1_000_000), [])
        camera = self.f.coverage(source='pole', modality='camera_rgb', observed={(11, 1)})
        self.map.add(camera, 1_000_000)
        self.assertEqual(self.map.claims_at((11, 1), 1_000_000), [])
        self.assertEqual(self.map.snapshot(1_000_000)['unknown_count'], 120-11)

    def test_contested_preserves_both_sources_and_counts_as_occupied(self):
        occupied = self.f.coverage(source='pole', modality='camera_rgb', occupied={(3, 1)})
        free = self.f.coverage(free={(3, 1)})
        for message in (occupied, free):
            self.map.add(message, 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['contested'], [[3, 1]])
        self.assertEqual(set(self.map.claims_at((3, 1), 1_000_000)),
                         {(evidence_ref(occupied), 'occupied'), (evidence_ref(free), 'free')})
        self.assertTrue(self.map.is_occupied((3, 1), 1_000_000))

    def test_expiry_and_ttl_revert_to_unknown_at_inclusive_boundary(self):
        message = self.f.coverage(free={(3, 1)}, expiry=1_100_000)
        self.map.add(message, 1_000_000)
        self.assertEqual(self.map.snapshot(1_100_000)['free'], [[3, 1]])
        self.assertEqual(self.map.snapshot(1_100_001)['unknown_count'], 120)
        other = CoverageMap(self.f.spec, ttl_us=100_000)
        other.add(self.f.coverage(), 1_000_000)
        with self.assertRaises(ValueError):
            other.add(self.f.coverage(free={(3, 1)}), 1_100_001)

    def test_old_coverage_cannot_overwrite_newer_and_unknown_clears_same_sensor_free(self):
        newer = self.f.coverage(occupied={(3, 1)}, capture=1_100_000)
        self.map.add(newer, 1_100_000)
        self.map.add(self.f.coverage(free={(3, 1)}, capture=1_000_000), 1_150_000)
        self.assertEqual(self.map.claims_at((3, 1), 1_150_000), [(evidence_ref(newer), 'occupied')])
        self.map.add(self.f.coverage(observed={(3, 1)}, capture=1_200_000), 1_200_000)
        self.assertEqual(self.map.claims_at((3, 1), 1_200_000), [])

    def test_incompatible_time_is_occupied_without_false_contested(self):
        self.map.add(self.f.coverage(source='pole', modality='camera_rgb', occupied={(3, 1)}), 1_000_000)
        self.map.add(self.f.coverage(free={(3, 1)}, capture=1_300_001), 1_300_001)
        snapshot = self.map.snapshot(1_300_001)
        self.assertEqual(snapshot['contested'], [])
        self.assertEqual(snapshot['occupied'], [[3, 1]])

    def test_object_motion_clears_previous_footprint_and_old_arrival_cannot_move_it_back(self):
        self.map.add_observation(self.f.observation(), 1_000_000)
        newer = self.f.observation(xyz=(5.5, 1.5, 1), capture=1_100_000)
        self.map.add_observation(newer, 1_100_000)
        self.map.add_observation(self.f.observation(capture=1_050_000), 1_100_000)
        self.assertEqual(self.map.claims_at((3, 1), 1_100_000), [])
        self.assertEqual(self.map.snapshot(1_100_000)['occupied'], [[5, 1]])

    def test_unanchored_peer_and_class_d_cannot_turn_map_free(self):
        self.map.add(self.f.coverage(source='attacker', free={(3, 1)}), 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['free'], [])
        self.f.store.anchors['other'] = replace(self.f.anchors['other'], trust_class=TrustClass.D)
        self.map.add(self.f.coverage(source='other', free={(4, 1)}), 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['unknown_count'], 120)

    def test_relay_inherits_origin_trust_and_expiry_not_just_relay_signature(self):
        untrusted = self.f.coverage(source='attacker', free={(3, 1)})
        ref = self.f.store.add(untrusted)
        relay = sign(replace(untrusted, envelope=replace(untrusted.envelope,
            source_id='pole', message_id='relay1', auth=Auth('pole-key')),
            lineage=replace(untrusted.lineage, derivation='relayed', root_ref=ref)), self.f.keys['pole'])
        self.map.add(relay, 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['free'], [])
        genuine = self.f.coverage(free={(4, 1)}, expiry=1_100_000)
        ref = self.f.store.add(genuine)
        relay = sign(replace(genuine, envelope=replace(genuine.envelope,
            source_id='car', message_id='relay2', valid_until_us=2_000_000, auth=Auth('car-key')),
            lineage=replace(genuine.lineage, derivation='relayed', root_ref=ref)), self.f.keys['car'])
        self.map.add(relay, 1_000_000)
        self.assertIn([4, 1], self.map.snapshot(1_100_000)['free'])
        self.assertEqual(self.map.snapshot(1_100_001)['free'], [])

    def test_unresolved_or_laundered_coverage_does_not_establish_free_space(self):
        original = self.f.coverage(free={(3, 1)})
        unresolved = sign(replace(original, lineage=replace(original.lineage,
            derivation='relayed', root_ref='0'*64)), self.f.keys['lidar'])
        self.map.add(unresolved, 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['free'], [])
        real = self.f.coverage(source='pole', free={(10, 1)})
        ref = self.f.store.add(real)
        laundered = sign(replace(original, lineage=replace(original.lineage,
            derivation='tracked', root_ref=ref, origin_observer='pole',
            origin_observation_id=real.envelope.message_id)), self.f.keys['lidar'])
        self.map.add(laundered, 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['free'], [])

    def test_invalid_or_forged_map_inputs_rejected(self):
        message = self.f.coverage(free={(3, 1)})
        with self.assertRaises(ValueError):
            self.map.add(replace(message, content=replace(message.content, min_size_m=.7)), 1_000_000)
        with self.assertRaises(ValueError):
            self.map.add(message, 999_999)
        self.map.add(message, 1_000_000)
        with self.assertRaises(ValueError):
            self.map.snapshot(999_999)
        with self.assertRaises(ValueError):
            self.map.claims_at((-1, 0), 1_000_000)

    def test_disjoint_height_or_below_resolution_size_is_not_false_contested(self):
        self.map.add(self.f.coverage(free={(3, 1)}), 1_000_000)
        self.map.add_observation(self.f.observation(xyz=(3.5, 1.5, 3.)), 1_000_000)
        self.assertEqual(self.map.snapshot(1_000_000)['contested'], [])
        self.map.add_observation(self.f.observation(xyz=(3.5, 1.5, 1.),
            dimensions=(.1, .1, .1), capture=1_100_000), 1_100_000)
        self.assertEqual(self.map.snapshot(1_100_000)['contested'], [])
