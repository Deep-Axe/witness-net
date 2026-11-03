from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from coopx.dataset.v2xreal import (AgentFrame, Camera, V2XReal, pose_to_site,
                                 project_camera, transform_points)
from coopx.domain import PositionAnchor
from coopx.observers.camera_lift import lift_box

DATASET = Path('/home/deep/meta_hackathon/data/v2x-real/val')
DATASET_METRICS = {}


class LiftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.frame = AgentFrame('scene', '-2', '000000', directory/'points.bin', np.eye(4),
            {'cam1': Camera('cam1', directory/'unused.jpeg', np.array([[100,0,50],[0,100,50],[0,0,1]]),
                            np.eye(4), (999,)*6)}, {}, {}, 0)

    def test_nearest_supported_cluster_beats_more_populated_background(self):
        near = np.array([[x,y,10.+.01*i] for i,(x,y) in enumerate(
            [(-.2,-.2),(-.1,.1),(0,0),(.1,-.1),(.2,.2),(.1,.2)])])
        wall = np.array([[x,y,30.] for x in np.linspace(-1,1,15) for y in np.linspace(-1,1,15)])
        # A nearer unsupported outlier must not replace the supported person.
        points = np.vstack([[[0,0,3]], near, wall])
        result = lift_box((40,40,60,60), self.frame, 'cam1', points_site=points)
        self.assertIsNotNone(result)
        self.assertEqual(result.anchor, PositionAnchor.SURFACE)
        self.assertEqual(result.n_points, len(near))
        self.assertLess(abs(result.xyz_site[2]-10), .1)
        self.assertGreaterEqual(result.std_m[0], .2)

    def test_erosion_sparse_support_and_behind_camera(self):
        points = np.array([[0,0,10], [.1,0,10], [0,.1,10], [0,0,-10], [.9,0,10]])
        self.assertIsNone(lift_box((40,40,60,60), self.frame, 'cam1', points_site=points))
        result = lift_box((40,40,60,60), self.frame, 'cam1', points_site=points, min_points=3)
        self.assertEqual(result.n_points, 3)
        self.assertIsNone(lift_box((100,100,120,120), self.frame, 'cam1', points_site=points))

    def test_own_lidar_uses_site_transform_and_ignores_cords(self):
        points = np.array([[.1*i, .05*i, 10, 1] for i in range(6)], dtype='<f4')
        points.tofile(self.frame.lidar_path)
        frame = replace(self.frame, lidar_to_site=pose_to_site([20,30,2,0,90,0]))
        result = lift_box((35,35,65,65), frame, 'cam1')
        expected = transform_points(points[:,:3], frame.lidar_to_site)
        np.testing.assert_allclose(result.xyz_site, np.median(expected, axis=0), atol=1e-6)

    def test_invalid_inputs_rejected_and_nonfinite_cached_points_ignored(self):
        for box in ((0,0,0,1), (0,0,float('nan'),1), (0,0,1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                lift_box(box, self.frame, 'cam1', points_site=np.zeros((5,3)))
        for settings in ({'shrink':.5}, {'min_points':True}, {'min_points':0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                lift_box((0,0,10,10), self.frame, 'cam1', points_site=np.zeros((5,3)), **settings)
        self.assertIsNone(lift_box((0,0,10,10), self.frame, 'cam1', points_site=np.array([[np.nan,0,1]])))


@unittest.skipUnless((DATASET/'2023-04-04-14-28-53_45_0').is_dir(), 'local V2X-Real dataset absent')
class DatasetLiftGate(unittest.TestCase):
    def test_gt_projected_boxes_on_three_frames_without_sam(self):
        global DATASET_METRICS
        dataset = V2XReal(DATASET)
        errors = {'Pedestrian': [], 'Car': []}
        surface_errors = {key: [] for key in errors}
        footprint_errors = {key: [] for key in errors}
        ambiguous = {key: 0 for key in errors}
        eligible = {key: 0 for key in errors}
        supported_frames = {key: set() for key in errors}
        none_count = {key: 0 for key in errors}
        for frame_id in ('000000', '000049', '000097'):
            frame = dataset.read('2023-04-04-14-28-53_45_0', '-2', frame_id)
            points = transform_points(frame.lidar()[:,:3], frame.lidar_to_site)
            # GT is used only to select/evaluate fake detections, never by lift_box.
            for box in frame.boxes.values():
                if box.label not in errors:
                    continue
                local = transform_points(points, np.linalg.inv(pose_to_site((*box.location, *box.angle))))
                if ((np.abs(local-box.center) <= np.asarray(box.extent)).all(axis=1)).sum() < 20:
                    continue
                for name in frame.cameras:
                    uv, _, valid = project_camera(frame.to_camera(box.corners_site(), name), frame.cameras[name].intrinsic)
                    if not valid.all() or not ((uv >= 0).all() and (uv[:,0] < 1920).all() and (uv[:,1] < 1080).all()):
                        continue
                    eligible[box.label] += 1
                    lifted = lift_box((*uv.min(axis=0), *uv.max(axis=0)), frame, name, points_site=points)
                    if lifted is None:
                        none_count[box.label] += 1
                        continue
                    error = np.linalg.norm(np.asarray(lifted.xyz_site)[:2]-box.center_site()[:2])
                    errors[box.label].append(float(error))
                    matrix = pose_to_site((*box.location, *box.angle))
                    inverse = np.linalg.inv(matrix)
                    sensor_local = transform_points(frame.camera_to_site(name)[:3,3][None], inverse)[0]-box.center
                    nearest_local = np.clip(sensor_local, -np.asarray(box.extent), np.asarray(box.extent))
                    nearest = transform_points((nearest_local+np.asarray(box.center))[None],matrix)[0]
                    surface_errors[box.label].append(float(np.linalg.norm(np.asarray(lifted.xyz_site)[:2]-nearest[:2])))
                    relative = transform_points(np.asarray(lifted.xyz_site)[None],inverse)[0]-box.center
                    footprint_errors[box.label].append(float(np.linalg.norm(np.maximum(np.abs(relative[:2])-box.extent[:2],0))))
                    ambiguous[box.label] += int(lifted.association_ambiguous)
                    supported_frames[box.label].add(frame_id)
        for label in errors:
            self.assertEqual(len(supported_frames[label]), 3, f'need {label} lifts on all 3 frames')
        DATASET_METRICS = {label: {'eligible_boxes': eligible[label], 'lifted_boxes': len(values),
            'insufficient_support': none_count[label], 'frames': sorted(supported_frames[label]),
            'median_xy_error_m': float(np.median(values)), 'p90_xy_error_m': float(np.quantile(values,.9)),
            'sensor_nearest_surface_xy_median_m': float(np.median(surface_errors[label])),
            'sensor_nearest_surface_xy_p90_m': float(np.quantile(surface_errors[label],.9)),
            'footprint_xy_median_m': float(np.median(footprint_errors[label])),
            'footprint_xy_p90_m': float(np.quantile(footprint_errors[label],.9)),
            'association_ambiguous': ambiguous[label]}
            for label, values in errors.items()}
        print('GT_BOX_LIFT_METRICS '+json.dumps(DATASET_METRICS,sort_keys=True))


class MaskAndGroundLiftTests(unittest.TestCase):
    def setUp(self):
        LiftTests.setUp(self)
        from PIL import Image
        Image.new('RGB',(100,100)).save(self.frame.cameras['cam1'].image_path)

    def test_l_shaped_mask_excludes_an_unrelated_foreground_wall(self):
        # Camera depth is site Z for this synthetic calibration. The L covers
        # left vertical and bottom horizontal bars, leaving the wall in its hole.
        mask=np.zeros((100,100),dtype=bool)
        mask[35:66,35:44]=True
        mask[57:66,35:66]=True
        pixels=np.array([(39,40),(39,45),(39,50),(40,59),(48,60),(55,61)])
        person=np.column_stack(((pixels-50)*.2,np.full(6,20.)))
        wall_pixels=np.array([(50,45),(51,46),(52,47),(53,48),(54,49),(55,50)])
        wall=np.column_stack(((wall_pixels-50)*.1,np.full(6,10.)))
        pts=np.vstack([wall,person])
        box=lift_box((30,30,70,70),self.frame,'cam1',points_site=pts)
        masked=lift_box((30,30,70,70),self.frame,'cam1',points_site=pts,mask=mask)
        self.assertAlmostEqual(box.xyz_site[2],10.)
        self.assertTrue(box.association_ambiguous)
        self.assertGreaterEqual(box.std_m[2],10.)
        self.assertAlmostEqual(masked.xyz_site[2],20.)
        self.assertFalse(masked.association_ambiguous)
        self.assertEqual(masked.method,'lidar')
        self.assertIsNone(lift_box((30,30,70,70),self.frame,'cam1',points_site=pts,
                                  mask=np.zeros((100,100),dtype=bool)))

    def test_invalid_mask_size_and_type(self):
        for mask in (np.zeros((20,20),dtype=bool),np.zeros((100,100)),[[True]],'mask'):
            with self.subTest(mask_type=type(mask)),self.assertRaises(ValueError):
                lift_box((30,30,70,70),self.frame,'cam1',points_site=np.zeros((0,3)),mask=mask)

    def ground_scene(self, *, slope=0.):
        rotation=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
        extrinsic=np.eye(4);extrinsic[:3,:3]=rotation;extrinsic[:3,3]=[0,0,3]
        cam=replace(self.frame.cameras['cam1'],camera_to_lidar=extrinsic)
        frame=replace(self.frame,cameras={'cam1':cam})
        ground=np.array([(x,y,slope*y) for x in np.arange(-5.5,6,.5) for y in np.arange(2,31,.5)])
        return frame,ground

    def test_ground_contact_and_uncertainty_grows_with_range(self):
        from coopx.observers.camera_lift import ground_ray_lift
        frame,ground=self.ground_scene()
        near=ground_ray_lift((40,45,60,80),frame,'cam1',points_site=ground)
        far=ground_ray_lift((40,45,60,65),frame,'cam1',points_site=ground)
        self.assertIsNotNone(near)
        self.assertIsNotNone(far)
        np.testing.assert_allclose(near.xyz_site,(0,10,0),atol=1e-8)
        self.assertEqual((near.method,near.anchor,near.n_points),('ground_ray',PositionAnchor.GROUND,0))
        self.assertGreaterEqual(near.ground_support,30)
        self.assertGreater(far.std_m[0],near.std_m[0])
        self.assertGreaterEqual(near.std_m[2],.15)

    def test_fallback_is_opt_in_only_when_object_support_is_insufficient(self):
        frame,ground=self.ground_scene()
        mask=np.zeros((100,100),dtype=bool);mask[49:52,49:52]=True
        self.assertIsNone(lift_box((40,45,60,80),frame,'cam1',points_site=ground,mask=mask))
        result=lift_box((40,45,60,80),frame,'cam1',points_site=ground,mask=mask,ground_fallback=True)
        self.assertEqual(result.method,'ground_ray')
        points=np.array([[.01*i,10.,3.] for i in range(6)])
        # These object points project to (50,50), and must win over ground fallback.
        result=lift_box((40,45,60,80),frame,'cam1',points_site=np.vstack([ground,points]),
                        mask=mask,ground_fallback=True)
        self.assertEqual(result.method,'lidar')
        self.assertEqual(result.n_points,6)
        self.assertIsNone(lift_box((40,45,60,80),frame,'cam1',points_site=ground,
                                  mask=np.zeros((100,100),dtype=bool),ground_fallback=True))

    def test_ground_rejects_clipping_horizon_missing_plane_and_unsupported_hit(self):
        from coopx.observers.camera_lift import ground_ray_lift
        frame,ground=self.ground_scene()
        for box,points in (((-1,40,60,80),ground),((40,40,60,100),ground),
                           ((40,30,60,40),ground),((40,40,60,51),ground),
                           ((40,40,60,80),ground[:20]),
                           ((40,40,60,80),ground[ground[:,1]>25])):
            with self.subTest(box=box,n=len(points)):
                self.assertIsNone(ground_ray_lift(box,frame,'cam1',points_site=points))
        frame,steep=self.ground_scene(slope=.5)
        self.assertIsNone(ground_ray_lift((40,40,60,80),frame,'cam1',points_site=steep))


@unittest.skipUnless(DATASET.is_dir(),'local V2X-Real dataset absent')
class ThinSupportGate(unittest.TestCase):
    def test_rsu1_thin_support_never_invokes_fallback_when_adequate(self):
        from unittest.mock import patch
        from coopx.observers.camera_lift import ground_ray_lift
        dataset=V2XReal(DATASET)
        for fid in ('000096','000097'):
            frame=dataset.read('2023-04-04-14-28-53_45_0','-1',fid)
            gt=frame.boxes['53'];name='cam1'
            uv,_,_=project_camera(frame.to_camera(gt.corners_site(),name),frame.cameras[name].intrinsic)
            box=(*uv.min(0),*uv.max(0))
            with patch('coopx.observers.camera_lift.ground_ray_lift',side_effect=AssertionError('fallback with adequate points')):
                measured=lift_box(box,frame,name,ground_fallback=True)
            self.assertEqual(measured.method,'lidar')
            self.assertGreaterEqual(measured.n_points,5)
            self.assertLess(np.linalg.norm(np.asarray(measured.xyz_site[:2])-gt.center_site()[:2]),.3)
            conditional=ground_ray_lift(box,frame,name)
            self.assertIsNotNone(conditional)
            self.assertEqual(conditional.anchor,PositionAnchor.GROUND)
            self.assertGreaterEqual(min(conditional.std_m[:2]),1.)
            self.assertLess(np.linalg.norm(np.asarray(conditional.xyz_site[:2])-gt.center_site()[:2]),.5)
