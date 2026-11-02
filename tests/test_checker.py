from dataclasses import asdict
import json
import random
import struct
import unittest

from coopx.checker import Checker, CheckResult
from coopx.domain import ObjectClass, Observation
from coopx.envelope import Auth, Envelope, Lineage, ObservationMessage, encode, sign
from coopx.events import CHECK_CODES, EVENT_TYPES, Event, SCHEMA, to_json_line, validate
from coopx.identity import NodeKey, TrustAnchor, TrustClass


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.keys = {source: NodeKey.generate() for source in ('pole', 'car', 'foreign')}
        self.anchors = {source: TrustAnchor(self.keys[source].public_key, source+'-key', TrustClass.A,
                                           physical_observer=source,
                                           permitted_region=(-50, -50, -5, 50, 50, 10))
                        for source in ('pole', 'car')}
        self.checker = Checker(self.anchors)

    def packet(self, *, source='pole', seq=1, mid='m1', capture=1_000_000,
               generation=None, expiry=None, xyz=(1., 2., 1.), object_id=7,
               key_ref=None, has_sensor_seq=True, session='run'):
        generation = capture if generation is None else generation
        expiry = generation+2_000_000 if expiry is None else expiry
        observation = Observation(source, object_id, seq, capture, ObjectClass.PEDESTRIAN, xyz)
        envelope = Envelope(2, 'observation', mid, source, session, 'site_map', capture,
                            generation, expiry, Auth(key_ref or source+'-key'),
                            seq if has_sensor_seq else None)
        lineage = Lineage(source, 'camera', mid, capture)
        return encode(sign(ObservationMessage(envelope, observation, lineage), self.keys[source]))

    def test_valid_packet_and_decodable_rejected_message(self):
        raw = self.packet()
        result = self.checker.check(raw, 1_000_000)
        self.assertEqual((result.codes, result.state), (('OK',), 'accepted'))
        self.assertEqual(result.message.envelope.message_id, 'm1')
        duplicate = self.checker.check(raw, 1_000_000)
        self.assertEqual(duplicate.codes, ('DUPLICATE',))
        self.assertIsNotNone(duplicate.message)

    def test_decode_and_auth_failures_stop_before_time_and_bounds(self):
        raw = self.packet(capture=0, expiry=1, xyz=(100., 0., 0.))
        result = self.checker.check(raw[:-1]+bytes([raw[-1] ^ 1]), 2_000_000)
        self.assertEqual(result.codes, ('BAD_SIGNATURE',))
        self.assertEqual(result.state, 'rejected')
        self.assertEqual(self.checker.check(b'broken', 1_000_000).codes, ('DECODE_ERROR',))
        self.assertIsNone(self.checker.check(b'broken', 1_000_000).message)
        self.assertEqual(self.checker.check(self.packet(source='foreign'), 1_000_000).codes,
                         ('UNKNOWN_SOURCE',))
        self.assertEqual(self.checker.check(self.packet(key_ref='different-key'), 1_000_000).codes,
                         ('KEY_MISMATCH',))
        for raw in (None, 'CX2', bytearray(b'CX2'), b''):
            with self.subTest(raw=raw):
                self.assertEqual(self.checker.check(raw, 0).codes, ('DECODE_ERROR',))

    def test_honest_jitter_in_window_is_accepted(self):
        for seq in (10, 12, 11, 13):
            with self.subTest(seq=seq):
                result = self.checker.check(self.packet(seq=seq, mid=f'm{seq}'), 1_000_000)
                self.assertEqual(result.codes, ('OK',))

    def test_window_boundary_replays_and_duplicate_sequence(self):
        checker = Checker(self.anchors, replay_window=4)
        for seq in (10, 13, 11):
            self.assertEqual(checker.check(self.packet(seq=seq, mid=f'm{seq}'), 1_000_000).codes, ('OK',))
        # [highest-window+1, highest] is inclusive. 10 was already consumed.
        self.assertEqual(checker.check(self.packet(seq=10, mid='new-id'), 1_000_000).codes, ('DUPLICATE',))
        self.assertEqual(checker.check(self.packet(seq=9, mid='old'), 1_000_000).codes,
                         ('OUTSIDE_REPLAY_WINDOW',))
        self.assertEqual(checker.check(self.packet(seq=20, mid='jump'), 1_000_000).codes, ('OK',))
        self.assertEqual(checker.check(self.packet(seq=10, mid='m10'), 1_000_000).codes,
                         ('DUPLICATE', 'OUTSIDE_REPLAY_WINDOW'))
        self.assertEqual(checker.check(self.packet(seq=0, mid='resigned-old'), 1_000_000).codes,
                         ('OUTSIDE_REPLAY_WINDOW',))

    def test_rejected_packets_do_not_poison_ids_window_or_latest(self):
        rejected = self.checker.check(self.packet(seq=1000, mid='reused', capture=2_000_000), 1_000_000)
        self.assertEqual(rejected.codes, ('FUTURE_TIME',))
        self.assertIsNone(self.checker.latest('pole', 7))
        result = self.checker.check(self.packet(seq=1, mid='reused'), 1_000_000)
        self.assertEqual(result.codes, ('OK',))
        raw = self.packet(seq=2000, mid='bad-signature')
        self.checker.check(raw[:-1]+bytes([raw[-1]^1]), 1_000_000)
        self.assertEqual(self.checker.check(self.packet(seq=2, mid='second'), 1_000_000).codes, ('OK',))

    def test_sources_are_isolated_and_session_change_does_not_reset_replays(self):
        self.assertEqual(self.checker.check(self.packet(seq=100), 1_000_000).codes, ('OK',))
        self.assertEqual(self.checker.check(self.packet(source='car'), 1_000_000).codes, ('OK',))
        self.assertEqual(self.checker.check(self.packet(seq=1, mid='restart', session='new'), 1_000_000).codes,
                         ('OUTSIDE_REPLAY_WINDOW',))

    def test_missing_sensor_sequence_uses_ids_and_timing_without_advancing_window(self):
        self.assertEqual(self.checker.check(self.packet(seq=1000, has_sensor_seq=False), 1_000_000).codes,
                         ('OK',))
        self.assertEqual(self.checker.check(self.packet(seq=1, mid='sequenced'), 1_000_000).codes, ('OK',))
        self.assertEqual(self.checker.check(self.packet(seq=2000, has_sensor_seq=False), 1_000_000).codes,
                         ('DUPLICATE',))

    def test_future_capture_and_generation_tolerance_is_inclusive(self):
        self.assertEqual(self.checker.check(self.packet(capture=2_000_000), 1_000_000).codes, ('FUTURE_TIME',))
        self.assertEqual(self.checker.check(self.packet(capture=1_200_001), 1_000_000).codes, ('FUTURE_TIME',))
        self.assertEqual(self.checker.check(self.packet(capture=1_200_000), 1_000_000).codes, ('OK',))
        checker = Checker(self.anchors)
        self.assertEqual(checker.check(self.packet(generation=1_200_001), 1_000_000).codes, ('FUTURE_TIME',))

    def test_ttl_and_expiry_boundaries_and_combined_failures(self):
        raw = self.packet(capture=0, expiry=1_000_000)
        self.assertEqual(self.checker.check(raw, 1_000_000).codes, ('OK',))
        self.assertEqual(Checker(self.anchors).check(raw, 1_000_001).codes, ('STALE', 'EXPIRED'))
        self.assertEqual(Checker(self.anchors).check(self.packet(capture=0), 1_000_001).codes, ('STALE',))
        self.assertEqual(Checker(self.anchors).check(self.packet(capture=900_000, expiry=950_000),
                                                  1_000_000).codes, ('EXPIRED',))

    def test_bounds_and_all_applicable_non_auth_codes(self):
        checker = Checker(self.anchors, replay_window=4, roi=(-10, -10, -2, 10, 10, 5))
        self.assertEqual(checker.check(self.packet(), 1_000_000).codes, ('OK',))
        self.assertEqual(checker.check(self.packet(seq=100, mid='high'), 1_000_000).codes, ('OK',))
        self.assertEqual(checker.check(self.packet(seq=101, mid='roi', xyz=(11., 0., 0.)), 1_000_000).codes,
                         ('OUT_OF_BOUNDS',))
        self.assertEqual(checker.check(self.packet(seq=102, mid='region', xyz=(60., 0., 0.)), 1_000_000).codes,
                         ('OUT_OF_BOUNDS', 'OUTSIDE_PERMITTED_REGION'))
        result = checker.check(self.packet(capture=0, expiry=1, xyz=(100., 0., 0.)), 2_000_000)
        self.assertEqual(result.codes, ('STALE', 'EXPIRED', 'DUPLICATE', 'OUTSIDE_REPLAY_WINDOW',
                                        'OUT_OF_BOUNDS', 'OUTSIDE_PERMITTED_REGION'))
        self.assertEqual(result.state, 'rejected')
        self.assertEqual(Checker(self.anchors, roi=(-10, -10, -2, 10, 10, 5)).check(
            self.packet(xyz=(10., -10., 5.)), 1_000_000).codes, ('OK',))
        self.assertEqual(Checker(self.anchors).check(self.packet(xyz=(50., -50., 10.)),
                                                  1_000_000).codes, ('OK',))
        self.assertEqual(Checker(self.anchors).check(self.packet(xyz=(51., 0., 0.)),
                                                  1_000_000).codes, ('OUTSIDE_PERMITTED_REGION',))
        # NaN/Inf are rejected by CX1 decoding before they reach ROI checks.
        raw = bytearray(self.packet())
        struct.pack_into('<f', raw, len(raw)-64-59+20, float('inf'))
        self.assertEqual(checker.check(bytes(raw), 1_000_000).codes, ('DECODE_ERROR',))

    def test_latest_keeps_newest_capture_not_latest_arrival_or_sequence(self):
        self.assertIsNone(self.checker.latest('pole', 7))
        newest = self.checker.check(self.packet(seq=12, mid='new', capture=1_000_000), 1_000_000).message
        self.assertEqual(self.checker.check(self.packet(seq=11, mid='late', capture=900_000),
                                          1_000_000).codes, ('OK',))
        self.assertEqual(self.checker.latest('pole', 7), newest)
        self.assertEqual(self.checker.check(self.packet(seq=13, mid='higher-seq', capture=950_000),
                                          1_000_000).codes, ('OK',))
        self.assertEqual(self.checker.latest('pole', 7), newest)
        self.checker.check(self.packet(seq=14, mid='tie', xyz=(2., 3., 1.)), 1_000_000)
        self.assertEqual(self.checker.latest('pole', 7), newest)
        self.checker.check(self.packet(seq=15, mid='other-object', object_id=8), 1_000_000)
        self.assertEqual(self.checker.latest('pole', 8).observation.object_id, 8)

    def test_invalid_configuration_or_call_clock_is_an_api_error(self):
        for values in ({'ttl_us': -1}, {'ttl_us': True}, {'future_tolerance_us': -1},
                       {'replay_window': 0}, {'replay_window': 1.2},
                       {'roi': (0, 0, 0, -1, 1, 1)}, {'roi': (0, 0, 0)},
                       {'roi': (0, 0, 0, float('nan'), 1, 1)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Checker(self.anchors, **values)
        for now in (-1, True, 1.1):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.checker.check(self.packet(), now)

    def test_one_thousand_random_tampered_old_and_foreign_packets_never_raise(self):
        rng = random.Random(716)
        raw = self.packet()
        checker = Checker(self.anchors)
        self.assertEqual(checker.check(self.packet(seq=1000), 1_000_000).state, 'accepted')
        for index in range(1000):
            category = index % 4
            if category == 0:
                damaged = bytearray(raw)
                offset = rng.randrange(len(damaged))
                damaged[offset] ^= rng.randrange(1, 256)
                candidate = bytes(damaged)
            elif category == 1:
                candidate = self.packet(seq=rng.randrange(100), mid=f'old-{index}', capture=0)
            elif category == 2:
                candidate = self.packet(source='foreign', seq=index, mid=f'foreign-{index}')
            else:
                candidate = rng.randbytes(rng.randrange(1024))
            with self.subTest(index=index):
                result = checker.check(candidate, 1_000_000)
                self.assertIsInstance(result, CheckResult)
                self.assertEqual(result.state, 'rejected')
                self.assertNotIn('OK', result.codes)
                self.assertTrue(set(result.codes) <= set(CHECK_CODES))


class EventTests(unittest.TestCase):
    def event(self, **changes):
        value = dict(schema=SCHEMA, t_replay_us=0, type='check', source_id='pole', dest_id='car',
                     message_id='m1', message_type='observation', evidence_ref='ab'*32,
                     bytes=300, check_codes=['OK'], state='accepted', detail={})
        value.update(changes)
        return value

    def test_roundtrip_dataclass_and_module_serializer(self):
        value = self.event(detail={'xyz': [1, 2, 3], 'nested': {'flag': True, 'empty': None},
                                   'text': 'line one\nline two'})
        event = Event(**value)
        self.assertEqual(asdict(event), value)
        for line in (event.to_json_line(), to_json_line(event), to_json_line(value)):
            self.assertTrue(line.endswith('\n'))
            self.assertEqual(len(line.splitlines()), 1)
            decoded = json.loads(line)
            self.assertEqual(decoded, value)
            self.assertIsNone(validate(decoded))

    def test_missing_extra_and_non_dict_rejected(self):
        for key in self.event():
            value = self.event()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate(value)
        for value in (None, [], {**self.event(), 'extra': None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(value)

    def test_every_field_wrong_type_or_value_rejected(self):
        cases = {
            'schema': [None, 1, 'coopx.events/2'],
            't_replay_us': [True, -1, 0.5, '0'],
            'type': [[], 5, 'unknown'],
            'source_id': [None, 1, '', '  '],
            'dest_id': [1, [], ''], 'message_id': [False, {}, ''],
            'message_type': [0, [], ''],
            'evidence_ref': [1, 'a'*63, 'A'*64, 'g'*64, 'a'*65],
            'bytes': [True, -1, 3.5, '300'],
            'check_codes': [('OK',), None, [1], ['UNKNOWN_CODE'], ['OK', 'OK']],
            'state': [1, [], None, 'rejected', 'unknown'], 'detail': [None, [], 'text'],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate(self.event(**{field: value}))

    def test_every_event_type_and_check_code_consistency(self):
        for event_type in EVENT_TYPES - {'check'}:
            valid = self.event(type=event_type, check_codes=[], state=None)
            validate(valid)
            for change in ({'check_codes': ['OK']}, {'state': 'accepted'}, {'state': 'rejected'}):
                with self.subTest(event_type=event_type, change=change), self.assertRaises(ValueError):
                    validate({**valid, **change})
        for code in CHECK_CODES:
            validate(self.event(check_codes=[code], state='accepted' if code == 'OK' else 'rejected'))
        validate(self.event(check_codes=['STALE', 'EXPIRED'], state='rejected'))
        for changes in ({'check_codes': []}, {'check_codes': ['OK', 'STALE'], 'state': 'rejected'},
                        {'check_codes': ['STALE'], 'state': 'accepted'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(self.event(**changes))

    def test_null_optional_fields_and_zero_bytes_are_allowed(self):
        validate(self.event(dest_id=None, message_id=None, message_type=None, evidence_ref=None, bytes=None))
        validate(self.event(bytes=0))

    def test_detail_requires_finite_json_values_and_no_cycles(self):
        values = [float('nan'), float('inf'), float('-inf'), {1: 'value'}, {'bad': {1, 2}},
                  {'bad': (1, 2)}, {'bad': object()}]
        cycle = {}
        cycle['loop'] = cycle
        values.append(cycle)
        deep = {}
        current = deep
        for _ in range(1100):
            current['child'] = {}
            current = current['child']
        values.append(deep)
        for index, value in enumerate(values):
            detail = value if type(value) is dict else {'value': value}
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate(self.event(detail=detail))
        with self.assertRaises(ValueError):
            Event(**self.event(detail=cycle))

    def test_serialization_revalidates_nested_mutations(self):
        event = Event(**self.event())
        event.check_codes.append('STALE')
        with self.assertRaises(ValueError):
            event.to_json_line()
        with self.assertRaises(ValueError):
            to_json_line(self.event(detail={'bad': float('nan')}))


if __name__ == '__main__':
    unittest.main()
