from copy import deepcopy
from dataclasses import replace
import json
import unittest

from coopx.assess import Assessor, footprint
from coopx.checker import Checker
from coopx.coverage import CoverageMap, CoverageMessage
from coopx.domain import ObjectClass, PositionAnchor
from coopx.envelope import default_agrees, encode, evidence_ref, sign
from coopx.events import Event, validate
from coopx.identity import TrustClass
from coopx.transport import PacketTransport
from test_coverage import Fixture


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.map = CoverageMap(self.f.spec)
        self.assessor = Assessor(self.f.store, self.map)

    def test_ghost_in_range_free_space_is_contradicted_but_map_stays_contested(self):
        free = self.f.coverage(free={(3, 1)})
        self.map.add(free, 1_000_000)
        ghost = self.f.observation(source='attacker')
        result = self.assessor.assess(ghost, 1_000_000)
        self.assertEqual((result.state, result.codes), ('contradicted', ('CONTRADICTED_BY_FREE_SPACE',)))
        self.assertEqual(result.contradicting, (evidence_ref(free),))
        self.assertEqual(self.map.snapshot(1_000_000)['contested'], [[3, 1]])
        self.assertEqual(set(self.map.claims_at((3, 1), 1_000_000)),
                         {(evidence_ref(free), 'free'), (evidence_ref(ghost), 'occupied')})

    def test_silence_or_rgb_absence_is_not_free_space(self):
        ghost = self.f.observation(source='attacker')
        self.assertEqual(self.assessor.assess(ghost, 1_000_000).codes, ('UNVERIFIED_SINGLE_SOURCE',))
        camera = self.f.coverage(source='pole', modality='camera_rgb', observed={(3, 1)})
        self.map.add(camera, 1_000_000)
        result = self.assessor.assess(ghost, 1_000_000)
        self.assertEqual((result.state, result.contradicting), ('unverified', ()))

    def test_range_occupied_confirms_position_but_not_the_pedestrian_class(self):
        occupied = self.f.coverage(occupied={(3, 1)})
        self.map.add(occupied, 1_000_000)
        unknown = self.f.observation(source='lidar', cls=ObjectClass.UNKNOWN)
        self.f.store.add(unknown)
        subject = self.f.observation()
        self.assertFalse(default_agrees(unknown, subject))
        result = self.assessor.assess(subject, 1_000_000)
        self.assertEqual(result.state, 'unverified')
        self.assertEqual(result.codes, ('POSITION_CORROBORATED_CLASS_UNVERIFIED',))
        self.assertIn(evidence_ref(occupied), result.supporting)

    def test_independent_agreeing_observers_semantically_corroborate(self):
        subject = self.f.observation()
        agreeing = self.f.observation(source='car', xyz=(3.8, 1.5, 1.))
        self.f.store.add(agreeing)
        result = self.assessor.assess(subject, 1_000_000)
        self.assertEqual((result.state, result.codes), ('corroborated', ('CORROBORATED',)))
        self.assertEqual(set(result.supporting), {evidence_ref(subject), evidence_ref(agreeing)})

    def test_contradiction_plus_independent_support_is_contested(self):
        self.map.add(self.f.coverage(free={(3, 1)}), 1_000_000)
        self.f.store.add(self.f.observation(source='car'))
        result = self.assessor.assess(self.f.observation(), 1_000_000)
        self.assertEqual((result.state, result.codes), ('contested', ('CONTESTED',)))
        self.assertTrue(result.supporting)
        self.assertTrue(result.contradicting)

    def test_half_footprint_threshold_and_unknown_outside_map_denominator(self):
        # +/- 1m about x=4 covers x=3,4; one of two cells is enough.
        subject = self.f.observation(xyz=(4., 1.5, 1.), std=(1., 0., 0.))
        self.map.add(self.f.coverage(free={(3, 1)}), 1_000_000)
        self.assertEqual(self.assessor.assess(subject, 1_000_000).state, 'contradicted')
        # Three footprint cells: only one free cell is insufficient.
        larger = self.f.observation(xyz=(4.5, 1.5, 1.), std=(1.5, 0., 0.), object_id=8)
        self.assertEqual(self.assessor.assess(larger, 1_000_000).state, 'unverified')
        outside = self.f.observation(xyz=(.5, 1.5, 1.), std=(2., 0., 0.), object_id=9)
        self.map.add(self.f.coverage(source='other', free={(0, 1)}), 1_000_000)
        self.assertEqual(self.assessor.assess(outside, 1_000_000).state, 'unverified')

    def test_time_height_and_measured_size_scope_free_space(self):
        self.map.add(self.f.coverage(free={(3, 1)}, capture=700_000), 1_000_000)
        self.assertEqual(self.assessor.assess(self.f.observation(), 1_000_000).state, 'contradicted')
        older_map = CoverageMap(self.f.spec)
        other = Assessor(self.f.store, older_map)
        older_map.add(self.f.coverage(free={(3, 1)}, capture=699_999), 1_000_000)
        self.assertEqual(other.assess(self.f.observation(), 1_000_000).state, 'unverified')
        high = self.f.observation(xyz=(3.5, 1.5, 3.), object_id=8)
        self.assertEqual(self.assessor.assess(high, 1_000_000).state, 'unverified')
        tiny = self.f.observation(dimensions=(.1, .1, .1), object_id=9)
        self.assertEqual(self.assessor.assess(tiny, 1_000_000).state, 'unverified')

    def test_expired_evidence_cannot_corroborate_or_contradict(self):
        self.map.add(self.f.coverage(free={(3, 1)}, expiry=1_000_000), 1_000_000)
        self.f.store.add(self.f.observation(source='car', expiry=1_000_000))
        result = self.assessor.assess(self.f.observation(capture=1_100_000), 1_100_000)
        self.assertEqual((result.state, result.contradicting), ('unverified', ()))

    def test_laundered_and_sybil_claims_never_become_corroborated(self):
        real = self.f.observation(xyz=(10., 5., 1.))
        root_ref = self.f.store.add(real)
        ghost = self.f.observation(source='attacker')
        tracked = replace(ghost, lineage=replace(ghost.lineage, derivation='tracked',
            root_ref=root_ref, origin_observer=real.envelope.source_id,
            origin_observation_id=real.envelope.message_id, origin_sensor=real.lineage.origin_sensor,
            origin_capture_time_us=real.envelope.capture_time_us))
        tracked = sign(tracked, self.f.keys['attacker'])
        self.f.store.add(ghost)
        self.assertEqual(self.assessor.assess(tracked, 1_000_000).state, 'unverified')
        for source in ('car', 'other'):
            self.f.store.anchors[source] = replace(self.f.anchors[source], trust_class=TrustClass.B)
            self.f.store.add(self.f.observation(source=source))
        self.assertEqual(self.assessor.assess(ghost, 1_000_000).state, 'unverified')

    def test_unanchored_and_same_body_range_cannot_supply_independent_position(self):
        subject = self.f.observation()
        self.map.add(self.f.coverage(source='attacker', occupied={(3, 1)}), 1_000_000)
        self.assertEqual(self.assessor.assess(subject, 1_000_000).codes, ('UNVERIFIED_SINGLE_SOURCE',))
        self.f.store.anchors['lidar'] = replace(self.f.anchors['lidar'], physical_observer='pole')
        self.map.add(self.f.coverage(occupied={(3, 1)}), 1_000_000)
        self.assertEqual(self.assessor.assess(subject, 1_000_000).codes, ('UNVERIFIED_SINGLE_SOURCE',))

    def test_lidar_used_for_lift_cannot_independently_confirm_that_geometry(self):
        occupied = self.f.coverage(occupied={(3, 1)})
        self.map.add(occupied, 1_000_000)
        subject = self.f.observation(parents=(evidence_ref(occupied),))
        self.assertEqual(self.assessor.assess(subject, 1_000_000).codes, ('UNVERIFIED_SINGLE_SOURCE',))

    def test_safety_asymmetry_remote_free_space_does_not_downgrade_ego(self):
        local = self.f.observation(source='ego')
        original_bytes = encode(local)
        checker = Checker(self.f.anchors)
        self.assertEqual(checker.check(original_bytes, 1_000_000).state, 'accepted')
        latest = checker.latest('ego', 7)
        self.f.store.add(local)
        self.map.add(self.f.coverage(free={(3, 1)}), 1_000_000)
        assessment = self.assessor.assess(local, 1_000_000)
        self.assertEqual(assessment.codes, ('UNVERIFIED_SINGLE_SOURCE',))
        self.assertEqual(assessment.contradicting, ())
        self.assertEqual(encode(self.f.store.get(evidence_ref(local))), original_bytes)
        self.assertTrue(self.map.is_occupied((3, 1), 1_000_000))
        # Remote claims do not mutate or remove that local evidence either.
        self.assessor.assess(self.f.observation(source='attacker'), 1_000_000)
        self.assertIsNotNone(self.f.store.get(evidence_ref(local)))
        self.assertEqual(checker.latest('ego', 7), latest)

    def test_oriented_box_footprint_and_point_boundary(self):
        box = self.f.observation(xyz=(5., 3., 1.), dimensions=(4., 2., 1.),
                                 yaw=90., anchor=PositionAnchor.CENTRE)
        cells, count = footprint(box, self.f.spec)
        self.assertEqual(set(cells), {(x, y) for x in (4, 5) for y in (1, 2, 3, 4)})
        self.assertEqual(count, 8)
        point = self.f.observation(xyz=(0., 0., 1.))
        self.assertEqual(footprint(point, self.f.spec), (((0, 0),), 1))

    def test_generated_reason_never_echoes_remote_identifier_text(self):
        subject = self.f.observation()
        original_id = subject.envelope.message_id
        malicious_id = '<img onerror=remote-text>'
        subject = sign(replace(subject, envelope=replace(subject.envelope, message_id=malicious_id),
                                lineage=replace(subject.lineage, origin_observation_id=malicious_id)), self.f.keys['pole'])
        reason = self.assessor.assess(subject, 1_000_000).reason
        self.assertNotIn(malicious_id, reason)
        self.assertNotIn(original_id, reason)

    def test_signed_coverage_and_ghost_through_transport_to_assessment_and_map(self):
        checker = Checker(self.f.anchors)
        bus = PacketTransport(('lidar', 'attacker', 'ego'), delay_us=10_000)
        coverage = self.f.coverage(free={(3, 1)})
        ghost = self.f.observation(source='attacker')
        for message in (coverage, ghost):
            bus.send(message.envelope.source_id, encode(message), sim_us=1_000_000,
                     message_id=message.envelope.message_id, message_type=message.envelope.message_type)
        assessments = []
        for delivery in bus.receive_until(1_010_000):
            if delivery.receiver != 'ego':
                continue
            result = checker.check(delivery.packet, delivery.available_sim_us)
            self.assertEqual(result.codes, ('OK',))  # Signed ghost is admitted, not rejected.
            if isinstance(result.message, CoverageMessage):
                self.map.add(result.message, delivery.available_sim_us)
            else:
                assessments.append(self.assessor.assess(result.message, delivery.available_sim_us))
        self.assertEqual(len(assessments), 1)
        assessment = assessments[0]
        self.assertEqual(assessment.state, 'contradicted')
        self.assertEqual(assessment.contradicting, (evidence_ref(coverage),))
        self.assertEqual(self.map.snapshot(1_010_000)['contested'], [[3, 1]])
        self.assertEqual(set(self.map.claims_at((3, 1), 1_010_000)),
                         {(evidence_ref(coverage), 'free'), (evidence_ref(ghost), 'occupied')})
        event = Event(schema='coopx.events/1', t_replay_us=1_010_000, type='assessment',
                      source_id='ego', dest_id=None, message_id=ghost.envelope.message_id,
                      message_type='observation', evidence_ref=evidence_ref(ghost), bytes=None,
                      check_codes=list(assessment.codes), state=assessment.state,
                      detail={'cells': [list(cell) for cell in assessment.cells],
                              'supporting': list(assessment.supporting),
                              'contradicting': list(assessment.contradicting), 'reason': assessment.reason})
        validate(json.loads(event.to_json_line()))


class S2EventTests(unittest.TestCase):
    def event(self, kind, detail, **changes):
        d = dict(schema='coopx.events/1', t_replay_us=1_000_000, type=kind, source_id='ego',
                 dest_id=None, message_id=None, message_type=None, evidence_ref=None, bytes=None,
                 check_codes=[], state=None, detail=detail)
        d.update(changes)
        return d

    def coverage(self):
        return self.event('coverage_published', dict(
            grid={'origin': [-10., -30.], 'cell_m': 1., 'nx': 60, 'ny': 60}, modality='lidar',
            window_us=[1_000_000, 1_000_000], free=[[1, 1]], occupied=[[2, 1]], observed_count=2),
            source_id='lidar', message_id='c1', message_type='coverage', evidence_ref='ab'*32, bytes=1600)

    def assessment(self, state='contradicted', code='CONTRADICTED_BY_FREE_SPACE'):
        return self.event('assessment', dict(cells=[[1, 1]], supporting=['cd'*32],
                         contradicting=['ab'*32], reason='generated: free_cells=1'),
                         message_id='o1', message_type='observation', evidence_ref='ef'*32,
                         state=state, check_codes=[code])

    def snapshot(self):
        return self.event('map_snapshot', dict(occupied=[[1, 1]], free=[[2, 1]],
                                              contested=[[3, 1]], unknown_count=3597))

    def test_all_new_event_types_and_assessment_states_roundtrip(self):
        events = [self.coverage(), self.snapshot(), self.assessment(),
                  self.assessment('corroborated', 'CORROBORATED'),
                  self.assessment('contested', 'CONTESTED'),
                  self.assessment('unverified', 'UNVERIFIED_SINGLE_SOURCE'),
                  self.assessment('unverified', 'POSITION_CORROBORATED_CLASS_UNVERIFIED')]
        for event in events:
            with self.subTest(kind=event['type'], state=event['state']):
                validate(event)
                self.assertEqual(json.loads(Event(**event).to_json_line()), event)

    def test_wrong_cells_refs_counts_and_state_code_pairs_rejected(self):
        cases = []
        for cells in ([[True, 1]], [[60, 1]], [[-1, 1]], [[1, 1], [1, 1]], [(1, 1)], [[1.0, 1]]):
            event = self.assessment()
            event['detail']['cells'] = cells
            cases.append(event)
        for fields in ({'state': 'unverified'}, {'check_codes': ['OK']}, {'check_codes': []},
                       {'check_codes': ['CONTESTED', 'CORROBORATED']}, {'evidence_ref': None}):
            cases.append({**self.assessment(), **fields})
        for name, value in (('supporting', ['invalid']), ('contradicting', []), ('reason', 3)):
            event = self.assessment()
            event['detail'][name] = value
            cases.append(event)
        for update in ({'modality': 'camera_rgb'}, {'observed_count': 1},
                       {'window_us': [2, 1]}, {'occupied': [[1, 1]]}, {'grid': {}}):
            event = self.coverage()
            event['detail'].update(update)
            cases.append(event)
        for update in ({'unknown_count': 0}, {'free': [[1, 1]]}, {'contested': [[2, 1]]}):
            event = self.snapshot()
            event['detail'].update(update)
            cases.append(event)
        for i, event in enumerate(cases):
            with self.subTest(index=i), self.assertRaises(ValueError):
                validate(event)

    def test_missing_detail_fields_rejected_for_each_s2_type(self):
        for event in (self.coverage(), self.snapshot(), self.assessment()):
            for field in event['detail']:
                value = deepcopy(event)
                del value['detail'][field]
                with self.subTest(kind=event['type'], field=field), self.assertRaises(ValueError):
                    validate(value)
