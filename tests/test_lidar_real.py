from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from coopx.dataset.v2xreal import AgentFrame,Camera,V2XReal
from coopx.coverage import GridSpec,CoverageMap
from coopx.envelope import evidence_ref
from coopx.observers.lidar_real import lidar_coverage,lidar_objects,range_contradicts,ray_cells
from test_coverage import Fixture

DATASET=Path(__file__).resolve().parents[1]/'data'/'v2x-real'/'val'

def frame_fixture(directory,points,*,height=1.):
    directory=Path(directory);path=directory/'points.bin'
    points=np.asarray(points,dtype=float).reshape(-1,3)
    local=points-[0,0,height]
    np.column_stack([local,np.ones(len(points))]).astype('<f4').tofile(path)
    pose=np.eye(4);pose[2,3]=height
    extrinsic=np.eye(4);extrinsic[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    image=directory/'image.png';Image.new('RGB',(100,100)).save(image)
    camera=Camera('cam1',image,np.array([[100.,0,50],[0,100,50],[0,0,1]]),extrinsic,())
    return AgentFrame('synthetic','-1','000000',path,pose,{'cam1':camera},{},{},0)

class RealLidarTests(unittest.TestCase):
    def test_wall_free_before_unknown_behind_and_no_invented_sectors(self):
        with tempfile.TemporaryDirectory() as td:
            f=frame_fixture(td,[[10,0,1]])
            spec=GridSpec(0.,-5.,1.,20,10)
            c=lidar_coverage(f,spec,capture_us=0,window_us=(0,0),max_range_m=20)
            self.assertIn((5,5),c.free);self.assertIn((10,5),c.occupied)
            self.assertNotIn((11,5),c.observed);self.assertNotIn((5,6),c.observed)
            clipped=lidar_coverage(f,spec,capture_us=0,window_us=(0,0),max_range_m=6)
            self.assertNotIn((10,5),clipped.observed);self.assertFalse(clipped.occupied)
            self.assertLessEqual(max(x for x,y in clipped.free),5)

    def test_ray_height_sparsity_zero_padding_and_bad_windows(self):
        with tempfile.TemporaryDirectory() as td:
            f=frame_fixture(td,[[10,0,1]],height=4.)
            spec=GridSpec(0.,-2.,1.,20,4)
            c=lidar_coverage(f,spec,capture_us=0,window_us=(0,0),max_range_m=20)
            self.assertNotIn((2,2),c.free) # beam is still above the stated slice
            self.assertIn((9,2),c.free)
            with self.assertRaises(ValueError):lidar_coverage(f,spec,capture_us=2,window_us=(0,1),max_range_m=20)
            f=frame_fixture(td,[],height=1.)
            self.assertFalse(lidar_coverage(f,spec,capture_us=0,window_us=(0,0),max_range_m=20).observed)

    def test_grid_clipping_and_diagonal_boundary_touches(self):
        spec=GridSpec(0.,0.,1.,4,4)
        self.assertEqual(ray_cells((-1000,.5),(1000,.5),spec),((0,0),(1,0),(2,0),(3,0)))
        self.assertEqual(ray_cells((0,0),(4,4),spec),((0,0),(1,1),(2,2),(3,3)))
        self.assertEqual(ray_cells((-1,-1),(-2,-2),spec),())

    def test_unknown_clusters_above_own_ground(self):
        with tempfile.TemporaryDirectory() as td:
            ground=[[x,y,0.] for x in np.arange(-5,6,.5) for y in np.arange(2,20,.5)]
            obj=[[.1+i*.03,10.,.8+i*.03] for i in range(8)]
            f=frame_fixture(td,ground+obj,height=3.)
            objects=lidar_objects(f)
            self.assertEqual(len(objects),1);self.assertEqual(objects[0]['cls'],'UNKNOWN')
            self.assertEqual(objects[0]['n_points'],8)
            f=frame_fixture(td,obj,height=3.)
            self.assertEqual(lidar_objects(f),[])

    def test_specific_range_proof_shares_assessor_policy(self):
        from coopx.assess import Assessor
        f=Fixture();map=CoverageMap(f.spec,store=f.store);assessor=Assessor(f.store,map)
        cov=f.coverage(free={(3,1)});map.add(cov,1_000_000)
        detection=f.observation(source='attacker');f.store.add(detection)
        d,c=evidence_ref(detection),evidence_ref(cov)
        self.assertTrue(range_contradicts(f.store,map,d,c,1_000_000))
        self.assertEqual(assessor.assess(detection,1_000_000).state,'contradicted')
        self.assertFalse(range_contradicts(f.store,map,d,'ff'*32,1_000_000))
        old=f.observation(source='attacker',capture=0);f.store.add(old)
        self.assertFalse(range_contradicts(f.store,map,evidence_ref(old),c,1_000_000))
        high=f.observation(source='attacker',xyz=(3.5,1.5,5.),object_id=8);f.store.add(high)
        self.assertFalse(range_contradicts(f.store,map,evidence_ref(high),c,1_000_000))

@unittest.skipUnless(DATASET.is_dir(),'dataset absent')
class RealLidarGate(unittest.TestCase):
    def test_target_not_free_and_packet_is_serializable(self):
        from coopx.pipeline import _signed_coverage,demo_node_key
        from coopx.envelope import encode
        frame=V2XReal(DATASET).read('2023-04-04-14-28-53_45_0','-2','000096')
        spec=GridSpec(-10.,-240.,1.,120,120)
        c=lidar_coverage(frame,spec,capture_us=frame.nominal_time_us,window_us=(frame.nominal_time_us,frame.nominal_time_us),max_range_m=60.)
        target=frame.boxes['53'].center_site()
        self.assertNotIn(spec.cell_at(*map(float,target[:2])),c.free)
        self.assertTrue(c.free);self.assertTrue(c.occupied)
        m=_signed_coverage('lidar1',demo_node_key(0,'lidar1'),c,sim_us=frame.nominal_time_us,sensor_seq=1,message_id='coverage')
        self.assertTrue(encode(m).startswith(b'CX2C'))
        # Real range evidence, no model: signed ghost is refuted by the same
        # deterministic proof used by the supervisor callback.
        from coopx.pipeline import _anchors_and_keys,_signed_observation
        from coopx.envelope import EvidenceStore
        from coopx.domain import ObjectClass
        from coopx.assess import Assessor
        keys,anchors=_anchors_and_keys(('lidar1','attacker','ego'),key_seed=0,
                                      region=(-1000.,-1000.,-20.,1000.,1000.,20.))
        store=EvidenceStore(anchors);coverage_map=CoverageMap(spec,store=store)
        coverage_map.add(m,frame.nominal_time_us)
        x,y=spec.cell_centre(sorted(c.free)[0])
        ghost=_signed_observation('attacker',keys['attacker'],object_id=1,frame=1,
                                  sim_us=frame.nominal_time_us,cls=ObjectClass.PEDESTRIAN,
                                  xyz=(x,y,1.),velocity=(0.,0.,0.),dimensions=(.6,.6,1.5))
        store.add(ghost)
        self.assertTrue(range_contradicts(store,coverage_map,evidence_ref(ghost),evidence_ref(m),frame.nominal_time_us))
        self.assertEqual(Assessor(store,coverage_map).assess(ghost,frame.nominal_time_us).state,'contradicted')



class BatchedRayTests(unittest.TestCase):
    def test_batch_matches_individual_exact_rays(self):
        from coopx.observers.lidar_real import _ray_cells_many
        rng=np.random.default_rng(42);spec=GridSpec(-3.,-2.,.5,20,20)
        starts=rng.uniform(-10,15,(250,2));ends=rng.uniform(-10,15,(250,2))
        expected={c for a,b in zip(starts,ends) for c in ray_cells(a,b,spec)}
        self.assertEqual(_ray_cells_many(starts,ends,spec),expected)
        for a,b in zip(starts[:25],ends[:25]):
            self.assertEqual(_ray_cells_many(a[None],b[None],spec),set(ray_cells(a,b,spec)))
