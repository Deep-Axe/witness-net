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
                    supported_frames[box.label].add(frame_id)
        for label in errors:
            self.assertEqual(len(supported_frames[label]), 3, f'need {label} lifts on all 3 frames')
        DATASET_METRICS = {label: {'eligible_boxes': eligible[label], 'lifted_boxes': len(values),
            'insufficient_support': none_count[label], 'frames': sorted(supported_frames[label]),
            'median_xy_error_m': float(np.median(values)), 'p90_xy_error_m': float(np.quantile(values,.9))}
            for label, values in errors.items()}
        print('GT_BOX_LIFT_METRICS '+json.dumps(DATASET_METRICS,sort_keys=True))
