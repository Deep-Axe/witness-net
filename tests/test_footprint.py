from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from coopx.dataset.v2xreal import V2XReal,point_in_polygon
from coopx.observers.footprint import camera_footprint,demonstrated_range_m,_hull
from test_lidar_real import frame_fixture
DATASET=Path(__file__).resolve().parents[1]/'data'/'v2x-real'/'val'

class FootprintTests(unittest.TestCase):
    def fixture(self,td):
        ground=[[x,y,0.] for x in np.arange(-5,6,.5) for y in np.arange(2,45,.5)]
        objects=[[x+.01*i,y+.01*i,.6+.03*i] for x,y in ((0.,12.),(1.,25.)) for i in range(8)]
        return frame_fixture(td,ground+objects,height=3.)

    def test_dense_fan_contains_near_ground_and_respects_demonstrated_range(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td);r=demonstrated_range_m(f,'cam1');p=camera_footprint(f,'cam1')
            self.assertGreater(r,15);self.assertLessEqual(r,25)
            self.assertTrue(point_in_polygon((0.,12.),p))
            self.assertLessEqual(len(p),64)
            origin=f.camera_to_site('cam1')[:2,3]
            self.assertLessEqual(max(np.linalg.norm(np.asarray(v)-origin) for v in p),r+1e-7)
            clipped=camera_footprint(f,'cam1',range_m=15.)
            self.assertLessEqual(max(np.linalg.norm(np.asarray(v)-origin) for v in clipped),15.+1e-7)
            self.assertEqual(p,camera_footprint(f,'cam1',range_m=1000.))
            with self.assertRaises(ValueError):camera_footprint(f,'cam1',range_m=-1.)

    def test_range_uses_no_gt_and_missing_support_never_claims_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            f=self.fixture(td)
            r=demonstrated_range_m(f,'cam1')
            self.assertEqual(r,demonstrated_range_m(replace(f,boxes={'53':'not a GT box'}),'cam1'))
            f=frame_fixture(td,[],height=3.)
            self.assertEqual(demonstrated_range_m(f,'cam1'),0.)
            self.assertEqual(camera_footprint(f,'cam1'),())

    def test_hull_simplification_retains_sharp_near_corners(self):
        arc=[(20*np.cos(a),20*np.sin(a)) for a in np.linspace(-.6,.6,200)]
        p=_hull([(1.,-.5),(1.,.5),*arc])
        self.assertLessEqual(len(p),64)
        self.assertTrue(point_in_polygon((2.,0.),p))

@unittest.skipUnless(DATASET.is_dir(),'dataset absent')
class FootprintGate(unittest.TestCase):
    def test_real_frames_use_generic_rule_not_fixed_forty_metres(self):
        d=V2XReal(DATASET)
        for agent,camera in (('-1','cam1'),('-2','cam2')):
            f=d.read('2023-04-04-14-28-53_45_0',agent,'000096')
            r=demonstrated_range_m(f,camera);p=camera_footprint(f,camera)
            self.assertGreater(r,0.);self.assertNotEqual(r,40.)
            self.assertGreaterEqual(len(p),3);self.assertLessEqual(len(p),64)
            # Inclusion of any particular labelled target is reported, not a gate.
