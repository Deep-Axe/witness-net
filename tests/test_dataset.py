"""Geometry boundary tests: no private dataset assets required."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from coopx.dataset.v2xreal import (GroundTruthBox, V2XReal, point_in_polygon,
                                 pose_to_site, project_camera, transform_points)


class DatasetGeometryTests(unittest.TestCase):
    def test_release_angle_order_and_signs(self):
        np.testing.assert_allclose(transform_points([[1, 0, 0]],
            pose_to_site([10, 20, 30, 0, 90, 0])), [[10, 21, 30]], atol=1e-12)
        np.testing.assert_allclose(transform_points([[1, 0, 0]],
            pose_to_site([0, 0, 0, 0, 0, 90])), [[0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(transform_points([[0, 1, 0]],
            pose_to_site([0, 0, 0, 90, 0, 0])), [[0, 0, -1]], atol=1e-12)

    def test_box_half_extents_and_rotated_local_center(self):
        box = GroundTruthBox('7', 'Car', (10, 20, 0), (1, 0, 0), (2, 1, .5), (0, 90, 0))
        np.testing.assert_allclose(box.center_site(), [10, 21, 0], atol=1e-12)
        np.testing.assert_allclose(np.ptp(box.corners_site(), axis=0), [2, 4, 1], atol=1e-12)
        self.assertTrue(box.contains_site([10, 21, 0]))
        self.assertFalse(box.contains_site([10, 25, 0]))

    def test_projection_filters_near_and_behind_without_mirroring(self):
        k = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        uv, depth, valid = project_camera([[1, 2, 10], [1, 2, -10], [0, 0, 0]], k)
        np.testing.assert_allclose(uv[0], [60, 60])
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertTrue(np.isnan(uv[1:]).all())
        self.assertEqual(depth.tolist(), [10, -10, 0])

    def test_transform_inverse_and_invalid_rigid_transform(self):
        points = np.array([[1, 2, 3], [-10, 0, .5]])
        transform = pose_to_site([10, -20, 1, 3, 57, -2])
        np.testing.assert_allclose(transform_points(transform_points(points, transform),
                                                   np.linalg.inv(transform)), points, atol=1e-12)
        with self.assertRaises(ValueError):
            transform_points(points, np.diag([1, -1, 1, 1]))

    def test_point_in_polygon_counts_boundary(self):
        square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
        self.assertTrue(point_in_polygon((1.0, 1.0), square))
        self.assertTrue(point_in_polygon((0.0, 1.0), square))
        self.assertFalse(point_in_polygon((3.0, 1.0), square))


class DatasetReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Camera optical (right,down,forward) -> LiDAR (forward,left,up).
        extrinsic = [[0, 0, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        for agent, location in [('1', [0, 0, 0, 0, 0, 0]), ('-1', [10, 0, 0, 0, 0, 0])]:
            directory = self.root / 'scene' / agent
            directory.mkdir(parents=True)
            metadata = dict(lidar_pose=location, vehicles={}, cam1=dict(
                intrinsic=[[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                extrinsic=extrinsic, cords=[999]*6))
            (directory / '000002.yaml').write_text(yaml.safe_dump(metadata))
            np.array([[11, -1, -2, 12]], dtype='<f4').tofile(directory / '000002.bin')
            Image.new('RGB', (100, 80)).save(directory / '000002_cam1.jpeg')
        self.dataset = V2XReal(self.root)

    def test_reader_preserves_intensity_and_nominal_clock_label(self):
        frame = self.dataset.read('scene', '1', '000002')
        self.assertEqual(frame.lidar().tolist(), [[11, -1, -2, 12]])
        self.assertEqual(frame.nominal_time_us, 200_000)
        self.assertIn('capture time unavailable', frame.timestamp_basis)
        self.assertEqual(frame.cameras['cam1'].image().shape, (80, 100, 3))
        self.assertEqual(self.dataset.aligned_frame_ids('scene', ['1', '-1']), ('000002',))

    def test_camera_direction_and_cross_agent_chain_ignore_cords(self):
        vehicle = self.dataset.read('scene', '1', '000002')
        roadside = self.dataset.read('scene', '-1', '000002')
        # site [11,-1,-2] -> camera [1,2,10], via inverse extrinsic.
        own = vehicle.to_camera([[11, -1, -2]], 'cam1', source_to_site=vehicle.lidar_to_site)
        cross = vehicle.to_camera([[1, -1, -2]], 'cam1', source_to_site=roadside.lidar_to_site)
        np.testing.assert_allclose(own, [[1, 2, 10]])
        np.testing.assert_allclose(cross, own)

    def test_malformed_lidar_layout_rejected(self):
        frame = self.dataset.read('scene', '1', '000002')
        frame.lidar_path.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            frame.lidar()


if __name__ == '__main__':
    unittest.main()
