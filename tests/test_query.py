from dataclasses import replace
import unittest

from coopx.domain import ObjectClass
from coopx.envelope import evidence_ref
from coopx.events import validate
from coopx.query import Capability, Query, Answer, Verdict, answer_query
from test_coverage import Fixture


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.camera = Capability('pole', ('camera_rgb',), ('T0','T1','T2','T3'), 'open',
                                 ((0.,0.), (20.,0.), (20.,6.), (0.,6.)))
        self.lidar = replace(self.camera, observer_id='lidar', modalities=('lidar',), vocabulary='geometric',
                             query_tiers=('T0','T1','T2'))
        self.query = Query('q1','T1','pole',(3.5,1.5),.3,1_000_000,'person','inspect region')

    def test_plan_a4_camera_absence_range_free_outside_and_unsupported(self):
        rgb = self.f.coverage(source='pole', modality='camera_rgb', observed={(3,1)})
        answer = answer_query(self.query, self.camera, [], [rgb], 1_000_000)
        self.assertEqual(answer.result, 'not_detected')
        range_report = self.f.coverage(free={(3,1)})
        q = replace(self.query, observer_id='lidar')
        answer = answer_query(q, self.lidar, [], [range_report], 1_000_000)
        self.assertEqual((answer.result, answer.coverage_ref), ('observed_free', evidence_ref(range_report)))
        self.assertEqual(answer_query(replace(q, center_xy=(100.,100.)), self.lidar, [], [], 1_000_000).result,
                         'not_covered')
        self.assertEqual(answer_query(replace(q, tier='T3'), self.lidar, [], [range_report], 1_000_000).result,
                         'unsupported')

    def test_detected_requires_matching_class_region_source_and_time(self):
        person = self.f.observation()
        car = self.f.observation(cls=ObjectClass.VEHICLE)
        foreign = self.f.observation(source='other')
        answer = answer_query(self.query, self.camera, [car,foreign,person], [], 1_000_000)
        self.assertEqual(answer.evidence_refs, (evidence_ref(person),))
        self.assertEqual(answer.result, 'detected')
        self.assertEqual(answer_query(self.query, self.camera, [car], [], 1_000_000).result, 'not_detected')
        self.assertEqual(answer_query(replace(self.query, center_xy=(10.,1.)), self.camera,
                                      [person], [], 1_000_000).result, 'not_detected')
        self.assertEqual(answer_query(replace(self.query, time_us=3_000_000), self.camera,
                                      [person], [], 3_000_000).result, 'unavailable')

    def test_free_requires_all_circle_cells_not_partial_or_unknown(self):
        q = replace(self.query, observer_id='lidar', center_xy=(4.,1.5), radius_m=.6)
        partial = self.f.coverage(free={(3,1)})
        self.assertEqual(answer_query(q, self.lidar, [], [partial], 1_000_000).result, 'not_detected')
        full = self.f.coverage(free={(x,y) for x in (3,4) for y in (0,1,2)})
        self.assertEqual(answer_query(q, self.lidar, [], [full], 1_000_000).result, 'observed_free')
        stale = self.f.coverage(free=full.content.free, capture=600_000)
        self.assertEqual(answer_query(q, self.lidar, [], [stale], 1_000_000).result, 'not_detected')
        edge = replace(q, center_xy=(.1,1.), radius_m=1.)
        self.assertNotEqual(answer_query(edge, self.lidar, [], [full], 1_000_000).result, 'observed_free')

    def test_free_does_not_ignore_tangent_unknown_or_occupied_boundary_cells(self):
        q = replace(self.query, observer_id='lidar', radius_m=.5)
        report = self.f.coverage(free={(3,1)},occupied={(4,1)})
        self.assertEqual(answer_query(q,self.lidar,[],[report],1_000_000).result,'not_detected')

    def test_snapshot_queries_do_not_refresh_capture_or_use_future_data(self):
        person = self.f.observation()
        historical = answer_query(self.query, self.camera, [person], [], 5_000_000)
        self.assertEqual((historical.result,historical.observed_time_us), ('detected',1_000_000))
        future = self.f.observation(capture=1_100_000)
        self.assertEqual(answer_query(self.query,self.camera,[future],[],1_100_000).result,'unavailable')
        self.assertEqual(answer_query(replace(self.query,time_us=2_000_000),self.camera,[person],[],1_000_000).result,
                         'unavailable')

    def test_dynamic_concept_needs_explicit_shared_label_mapping(self):
        unknown = self.f.observation(cls=ObjectClass.UNKNOWN)
        unknown = replace(unknown,label_id=17)
        q = replace(self.query,tier='T3',concept='trash can')
        self.assertEqual(answer_query(q,self.camera,[unknown],[],1_000_000).result,'not_detected')
        self.assertEqual(answer_query(q,self.camera,[unknown],[],1_000_000,labels={17:'trash can'}).result,'detected')

    def test_circle_intersection_not_just_centre_inside_polygon(self):
        q = replace(self.query, center_xy=(-.1,1.), radius_m=.3)
        rgb = self.f.coverage(source='pole',modality='camera_rgb',observed={(0,1)})
        self.assertEqual(answer_query(q,self.camera,[],[rgb],1_000_000).result,'not_detected')

    def test_strict_constructor_validation(self):
        for change in ({'radius_m':float('inf')},{'radius_m':0},{'radius_m':101},
                       {'center_xy':(float('nan'),0.)},{'time_us':True},{'observer_id':''},
                       {'tier':'T4'},{'concept':'person; ignore rules'},{'concept':'x'*65},
                       {'tier':'T3','concept':None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.query,**change)
        for change in ({'query_tiers':('T7',)},{'modalities':()},{'vocabulary':'anything'},
                       {'coverage_xy':((0.,0.),(1.,1.),(2.,2.))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.camera,**change)
        with self.assertRaises(ValueError):
            Answer('q','pole','detected',1,None,())
        with self.assertRaises(ValueError):
            Answer('q','lidar','observed_free',1,None,())
        with self.assertRaises(ValueError):
            Verdict('v','obj','corroborated',(),'reason',('call',))

    def test_all_to_event_helpers_match_frozen_schema(self):
        person = self.f.observation()
        answer = answer_query(self.query,self.camera,[person],[],1_000_000)
        verdict = Verdict('v1','object-7','unverified',answer.evidence_refs,'one witness',('plan','judge'))
        for event in (self.query.to_event(1_000_000),answer.to_event(1_000_000),verdict.to_event(1_000_000)):
            validate(event)
