import unittest

from coopx.coverage import CoverageContent, GridSpec
from coopx.observers.lidar_synth import (
    LidarObserver,
    MovingBody,
    Rect,
    S2_GRID,
    SynthWorld,
    cluster_occupied,
)


def wall_world() -> SynthWorld:
    # Thin wall at x = 10 m, spanning y = [-5, 5].
    return SynthWorld((Rect(10.0, -5.0, 11.0, 5.0, "obstacle"),))


class LidarSynthTests(unittest.TestCase):
    def test_wall_at_10m_is_free_before_and_unobserved_behind(self):
        spec = S2_GRID
        lidar = LidarObserver(spec, pose=(0.0, 0.0), n_beams=720, range_m=40.0, seed=0)
        coverage = lidar.scan(wall_world(), t_us=0)
        self.assertTrue(coverage.free <= coverage.observed)
        self.assertTrue(coverage.occupied <= coverage.observed)
        self.assertFalse(coverage.free & coverage.occupied)
        self.assertEqual(coverage.modality, "lidar")

        before = spec.cell_at(9.0, 0.0)
        wall = spec.cell_at(10.2, 0.0)
        behind = spec.cell_at(14.0, 0.0)
        self.assertIn(before, coverage.free)
        self.assertNotIn(before, coverage.occupied)
        self.assertIn(wall, coverage.occupied)
        self.assertNotIn(behind, coverage.observed)
        self.assertNotIn(behind, coverage.free)
        self.assertNotIn(behind, coverage.occupied)

    def test_occluded_pedestrian_behind_truck_is_not_observed(self):
        truck = Rect(12.0, 8.0, 18.0, 11.0, "obstacle")
        hidden = MovingBody("hidden", 24.0, 15.2, 0.0, 0.0, 0.6, 0.6, "object")
        world = SynthWorld((truck,), (hidden,))
        lidar = LidarObserver(S2_GRID, pose=(0.0, 0.0), n_beams=720, range_m=40.0, seed=0)
        coverage = lidar.scan(world, t_us=0)
        hidden_cell = S2_GRID.cell_at(24.0, 15.2)
        self.assertNotIn(hidden_cell, coverage.observed)
        self.assertNotIn(hidden_cell, coverage.occupied)
        truck_cell = S2_GRID.cell_at(12.2, 8.2)
        self.assertIn(truck_cell, coverage.occupied)

    def test_clusters_match_visible_objects(self):
        truck = Rect(12.0, 8.0, 18.0, 11.0, "obstacle")
        pedestrian = MovingBody("ped", 8.0, 1.0, 0.0, 0.0, 0.6, 0.6, "object")
        world = SynthWorld((truck,), (pedestrian,))
        lidar = LidarObserver(S2_GRID, pose=(0.0, 0.0), n_beams=720, range_m=40.0, seed=0)
        coverage = lidar.scan(world, t_us=0)
        objects = lidar.objects(coverage)
        self.assertGreaterEqual(len(objects), 2)
        centres = [obj["xyz"][:2] for obj in objects]
        self.assertTrue(any(abs(x - 8.0) < 2.0 and abs(y - 1.0) < 2.0 for x, y in centres))
        self.assertTrue(any(abs(x - 15.0) < 4.0 and abs(y - 9.5) < 3.0 for x, y in centres))
        hidden_centres = [c for c in centres if abs(c[0] - 22.0) < 1.5]
        self.assertEqual(hidden_centres, [])

    def test_contract2_invariants_hold_for_default_world(self):
        from coopx.observers.lidar_synth import default_s2_world
        lidar = LidarObserver(S2_GRID, pose=(0.0, 0.0), seed=3)
        for t_us in (0, 1_000_000, 2_500_000):
            coverage = lidar.scan(default_s2_world(), t_us=t_us)
            self.assertTrue(coverage.free <= coverage.observed)
            self.assertTrue(coverage.occupied <= coverage.observed)
            self.assertFalse(coverage.free & coverage.occupied)
            self.assertEqual(coverage.modality, "lidar")
            self.assertGreater(len(coverage.observed), 0)

    def test_camera_rgb_cannot_claim_free(self):
        spec = GridSpec(-10.0, -30.0, 1.0, 60, 60)
        with self.assertRaises(ValueError):
            CoverageContent(
                spec, (0, 1), "camera_rgb", (0.2, 2.0), 0.3,
                frozenset({(1, 1)}), frozenset({(1, 1)}), frozenset())

    def test_cluster_occupied_is_deterministic(self):
        cells = frozenset({(2, 2), (3, 2), (10, 10), (2, 3)})
        a = cluster_occupied(cells)
        b = cluster_occupied(cells)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)

    def test_scan_is_deterministic_for_a_seed(self):
        world = wall_world()
        a = LidarObserver(S2_GRID, seed=4).scan(world, t_us=0)
        b = LidarObserver(S2_GRID, seed=4).scan(world, t_us=0)
        self.assertEqual(a.free, b.free)
        self.assertEqual(a.occupied, b.occupied)

    def test_scan_window_ends_at_capture_time(self):
        lidar = LidarObserver(S2_GRID, seed=0)
        mid = lidar.scan(wall_world(), t_us=250_000)
        self.assertEqual(mid.window_us, (150_000, 250_000))
        self.assertIsInstance(mid, CoverageContent)
        self.assertIsInstance(mid.spec, GridSpec)
        first = lidar.scan(wall_world(), t_us=0)
        self.assertEqual(first.window_us, (0, 0))


if __name__ == "__main__":
    unittest.main()
