import unittest
import numpy as np
from spatial_mapping.clearance import path_voxel_clearance, point_to_path, acceptance_entry_point


class ClearanceTest(unittest.TestCase):
    def distance(self, points, voxels=((0, 0, 0),), resolution=1.):
        return path_voxel_clearance(points, voxels, resolution)['distance_m']

    def test_distance_to_face_not_voxel_center(self):
        self.assertAlmostEqual(self.distance([(-.2, .5, .5)]), .2)

    def test_parallel_segment_nearest_interior(self):
        self.assertAlmostEqual(self.distance([(-2., 2., .5), (3., 2., .5)]), 1.)

    def test_corner_nearest_interior_between_face_crossings(self):
        self.assertAlmostEqual(self.distance([(-2., 0., .5), (0., -2., .5)]), np.sqrt(2.))

    def test_intersection_and_tangent_are_zero(self):
        self.assertEqual(self.distance([(-2., .5, .5), (2., .5, .5)]), 0.)
        self.assertEqual(self.distance([(-1., -1., 0.), (0., 0., 0.)]), 0.)

    def test_point_and_zero_length_segment_agree(self):
        p = (-2., -3., -4.)
        self.assertEqual(self.distance([p]), self.distance([p, p]))

    def test_negative_voxels_scale_and_reverse(self):
        points = np.array([[-4., -1., .5], [0., -1., .5]])
        cells = [(-3, -3, 0)]
        self.assertAlmostEqual(self.distance(points, cells), 1.)
        self.assertAlmostEqual(self.distance(points[::-1] * .75, cells, .75), .75)

    def test_subdivision_preserves_clearance(self):
        a, b = np.array([-2., 0., .5]), np.array([0., -2., .5])
        self.assertAlmostEqual(self.distance([a, b]), self.distance([a, a * .37 + b * .63, b]))

    def test_multiple_voxels_and_segments_report_minimum(self):
        d = path_voxel_clearance([(-4., .5, .5), (-2., .5, .5), (-1., .5, .5)],
                                 [(10, 0, 0), (0, 0, 0)], 1.)
        self.assertAlmostEqual(d['distance_m'], 1.)
        self.assertEqual(d['segment_index'], 1)
        self.assertEqual(d['nearest_inflated_voxel'], [0, 0, 0])

    def test_random_against_independent_dense_upper_bound(self):
        rng = np.random.default_rng(21)
        for _ in range(40):
            a, b = rng.uniform(-4., 4., size=(2, 3))
            cells = rng.integers(-3, 3, size=(5, 3))
            exact = self.distance([a, b], cells)
            samples = a + np.linspace(0., 1., 2001)[:, None] * (b - a)
            diff = samples[:, None, :] - np.clip(samples[:, None, :], cells, cells + 1)
            sampled = float(np.sqrt(np.sum(diff**2, axis=2)).min())
            self.assertLessEqual(exact, sampled + 1e-9)
            self.assertLessEqual(sampled - exact, np.linalg.norm(b - a) / 2000 + 1e-9)

    def test_empty_obstacle_set_is_not_a_safety_certificate(self):
        self.assertIsNone(self.distance([(0., 0., 0.)], []))

    def test_invalid_inputs(self):
        for points, cells, res in (([], [], 1.), ([(float('nan'), 0, 0)], [], 1.),
                                   ([(0, 0, 0)], [(0.5, 0, 0)], 1.),
                                   ([(0, 0, 0)], [], 0.)):
            with self.assertRaises(ValueError):
                path_voxel_clearance(points, cells, res)

    def test_acceptance_entry_is_counterfactual_not_waypoint_center(self):
        self.assertEqual(acceptance_entry_point((0, 0, 0), (2, 0, 0), .8), [1.2, 0., 0.])
        self.assertEqual(acceptance_entry_point((0, 0, 0), (.4, 0, 0), .8), [0., 0., 0.])

    def test_real_excursion_is_inside_preexisting_inflation_off_safe_segment(self):
        points = [[-49.939876556396484, 69.5733871459961, -1.5915603637695312],
                  [-50.625, 66.375, -1.5832649230957032]]
        pose = [-50.01469039916992, 67.36858367919922, -1.5697059631347656]
        cells = [(-67, 89, -3)]
        self.assertGreater(self.distance(points, cells, .75), 0.)
        self.assertEqual(self.distance([pose], cells, .75), 0.)
        self.assertAlmostEqual(point_to_path(pose, points)['distance_m'], .38900174056203973)


if __name__ == '__main__':
    unittest.main()
