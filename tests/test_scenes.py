from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from coopx.dataset.v2xreal import GroundTruthBox,V2XReal
from coopx.dataset.scenes import visibility,Scene,Target,Visibility,load_scenes,_scene_json
from test_lidar_real import frame_fixture
DATASET=Path(__file__).resolve().parents[1]/'data'/'v2x-real'/'val'

def box(ident,center,extent=(.3,.3,.8),label='Pedestrian'):
    return GroundTruthBox(ident,label,tuple(center),(0.,0.,0.),tuple(extent),(0.,0.,0.))

class SceneTests(unittest.TestCase):
    def test_box_behind_box_occluded_but_outside_fov_is_not_occlusion(self):
        with tempfile.TemporaryDirectory() as td:
            target=box('target',(0.,12.,1.));blocker=box('blocker',(0.,6.,1.),(2.,1.,2.),'Car')
            f=frame_fixture(td,[],height=1.);f=replace(f,boxes={'target':target,'blocker':blocker})
            v=visibility(f,target);self.assertEqual(v.label,'occluded');self.assertTrue(v.signals['obstruction'])
            behind=box('outside',(0.,-12.,1.))
            self.assertEqual(visibility(f,behind).label,'outside_view')

    def test_few_points_without_obstruction_are_insufficient_not_hidden(self):
        with tempfile.TemporaryDirectory() as td:
            target=box('target',(0.,12.,1.))
            f=frame_fixture(td,[[0,12,1]],height=1.);f=replace(f,boxes={'target':target})
            v=visibility(f,target);self.assertEqual(v.label,'insufficient_evidence')
            self.assertFalse(v.signals['obstruction'])
            f=frame_fixture(td,[[0,12,1+i*.05] for i in range(6)],height=1.);f=replace(f,boxes={'target':target})
            self.assertEqual(visibility(f,target).label,'visible')

    def test_scene_round_trip_preserves_tuple_keyed_visibility(self):
        with tempfile.TemporaryDirectory() as td:
            scene=Scene('test','scenario',('000000','000001'),'1',(('rsu-1','-1','cam1'),),
                        (Target('t','pedestrian'),),{('t','1','000000'):Visibility('occluded',{'points':0,'obstruction':True,'in_camera':True})},
                        {'t':('000000',None)})
            path=Path(td)/'scenes.json';path.write_text(json.dumps({'scenes':[_scene_json(scene)]}))
            self.assertEqual(load_scenes(path),[scene])

@unittest.skipUnless(DATASET.is_dir(),'dataset absent')
class VisibilityGate(unittest.TestCase):
    def test_partial_ego_projection_and_conflicting_signals_are_not_a_blos_clip(self):
        f=V2XReal(DATASET).read('2023-04-04-14-28-53_45_0','2','000096')
        v=visibility(f,'53')
        self.assertEqual(v.label,'insufficient_evidence')
        self.assertTrue(v.signals['in_camera'])
        self.assertGreaterEqual(v.signals['points'],5)

class MiningGate(unittest.TestCase):
    def test_one_second_hidden_interval_and_later_visibility_are_mined(self):
        import yaml
        from coopx.dataset.scenes import mine_scenes
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'dataset'
            for agent,x in (('1',0.),('-1',5.)):
                directory=root/'scene'/agent;directory.mkdir(parents=True)
                for number in range(13):
                    fid=f'{number:06d}'
                    target=dict(obj_type='Pedestrian',location=[0.,12.,1.],center=[0.,0.,0.],extent=[.3,.3,.8],angle=[0.,0.,0.])
                    blocker=dict(obj_type='Car',location=[0. if number<=10 else 8.,6.,1.],center=[0.,0.,0.],extent=[2.,1.,2.],angle=[0.,0.,0.])
                    metadata=dict(lidar_pose=[x,0.,1.,0.,0.,0.],vehicles=({'b':blocker} if agent=='1' and number<=10 else {'t':target,'b':blocker}),
                        cam1=dict(intrinsic=[[100,0,50],[0,100,50],[0,0,1]],extrinsic=[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],cords=[]))
                    (directory/f'{fid}.yaml').write_text(yaml.safe_dump(metadata))
                    world=np.array([[0.,12.,1.+i*.03] for i in range(6)]) if agent=='-1' or number>=11 else np.empty((0,3))
                    local=world-[x,0,1.]
                    np.column_stack([local,np.ones(len(local))]).astype('<f4').tofile(directory/f'{fid}.bin')
                    from PIL import Image
                    Image.new('RGB',(100,100)).save(directory/f'{fid}_cam1.jpeg')
            output=Path(td)/'scenes'/'scenes.json'
            scenes=mine_scenes(root,out=output)
            self.assertEqual(len(scenes),1)
            self.assertEqual(scenes[0].hidden_interval,{'t':('000000','000011')})
            self.assertEqual(load_scenes(output),scenes)
            self.assertTrue(list((output.parent/'overlays').glob('*.png')))


class MultipleCameraGate(unittest.TestCase):
    def test_one_clear_camera_prevents_an_agent_level_occlusion_label(self):
        with tempfile.TemporaryDirectory() as td:
            target=box('target',(0.,12.,1.));blocker=box('blocker',(0.,6.,1.),(2.,1.,2.),'Car')
            f=frame_fixture(td,[],height=1.)
            cam=f.cameras['cam1'];extrinsic=cam.camera_to_lidar.copy();extrinsic[0,3]=5.
            f=replace(f,boxes={'target':target,'blocker':blocker},cameras={'cam1':cam,'cam2':replace(cam,name='cam2',camera_to_lidar=extrinsic)})
            v=visibility(f,target)
            self.assertEqual(v.label,'insufficient_evidence')
            self.assertFalse(v.signals['obstruction'])
            self.assertEqual(v.signals['camera'],'cam2')


class MissingCameraGate(unittest.TestCase):
    def test_missing_images_never_establish_outside_view_or_hidden(self):
        with tempfile.TemporaryDirectory() as td:
            f=frame_fixture(td,[],height=1.)
            target=box('target',(0.,12.,1.))
            f.cameras['cam1'].image_path.unlink()
            self.assertEqual(visibility(f,target).label,'insufficient_evidence')


class EdgeOfImageGate(unittest.TestCase):
    def test_box_overlap_without_in_image_sight_lines_is_not_occlusion(self):
        with tempfile.TemporaryDirectory() as td:
            # Box grazes the right edge, but its interior sight-line samples
            # are outside; an outside-image blocker cannot certify occlusion.
            target=box('target',(6.8,12.,1.),(.7,.3,.8))
            blocker=box('blocker',(3.4,6.,1.),(1.,1.,2.),'Car')
            f=frame_fixture(td,[],height=1.);f=replace(f,boxes={'target':target,'blocker':blocker})
            v=visibility(f,target)
            self.assertIn(v.label,('outside_view','insufficient_evidence'))
