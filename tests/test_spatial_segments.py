"""Regressao de voxels atravessados entre amostras, sem ROS/Gazebo."""
import unittest
import numpy as np
from itertools import product
from unittest.mock import patch
from spatial_mapping.navigation import SpatialNavigator, SpatialNavigationConfig
from spatial_mapping.traversal import segment_voxels


class SegmentSafetyTest(unittest.TestCase):
    def missed_cell_scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=.75, drone_clearance_radius_m=1.25,
            drone_vertical_clearance_m=.4,
        ))
        start = np.array([-29.32791519165039, 51.774959564208984, -1.5300025939941406])
        end = np.array([-30.637563132923542, 50.36243686707646, -1.565423583984375])
        missed = (-41, 68, -3)
        obstacle = nav.grid.voxel_to_world((-42, 69, -3))
        nav.grid.integrate_points(obstacle + [3., 0., 0.], [obstacle])
        for voxel in [*nav.grid._ray_voxels(start, end), missed]:
            nav.grid.mark_ego_voxel_free(nav.grid.voxel_to_world(voxel))
        return nav, start, end, missed

    def test_plan51_segment_cannot_skip_already_inflated_voxel(self):
        nav, start, end, missed = self.missed_cell_scene()
        self.assertNotIn(missed, nav.grid._ray_voxels(start, end))
        self.assertIn(missed, nav._inflated_obstacles())
        diagnostic = nav.path_safety_diagnostics(start, [end])
        self.assertFalse(diagnostic["safe"])
        self.assertEqual(diagnostic["first_unsafe_voxel"], list(missed))
        self.assertEqual(diagnostic["failure_reason"], "inflated")

    def test_reference_veto_and_reentry_use_complete_segment(self):
        nav, start, end, missed = self.missed_cell_scene()
        self.assertFalse(nav.path_avoids_obstacles_allowing_initial_escape(start, [end]))
        self.assertEqual(nav.first_obstacle_reentry_voxel(start, [end]), missed)

    def test_shortcut_cannot_cut_unknown_corner(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.))
        free = {(x, y, 0) for x in range(3) for y in range(3)} - {(1, 0, 0)}
        path = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0)]
        shortcut = nav._shortcut_path(path, free, set())
        self.assertGreater(len(shortcut), 2)
        for a, b in zip(shortcut, shortcut[1:]):
            self.assertTrue(set(nav.grid.segment_voxels(
                nav.grid.voxel_to_world(a), nav.grid.voxel_to_world(b)
            )) <= free)

    def test_integrating_depth_does_not_use_or_change_the_safety_traversal(self):
        nav = SpatialNavigator()
        with patch.object(nav.grid, "segment_voxels", side_effect=AssertionError("integration changed")):
            nav.grid.integrate_points([.1, .1, .1], [[3.1, .1, .1]])
        before = dict(nav.grid._log_odds)
        nav.grid.segment_voxels([.1, .1, .1], [3.1, .1, .1])
        self.assertEqual(before, nav.grid._log_odds)


class ClosedVoxelTraversalTest(unittest.TestCase):
    def test_diagonal_covers_all_corner_contacts_in_three_dimensions(self):
        self.assertEqual(set(segment_voxels([.5]*3, [1.5]*3, 1.)),
                         set(product((0, 1), repeat=3)))

    def test_line_on_grid_face_checks_both_sides(self):
        self.assertEqual(set(segment_voxels([0, .2, .5], [0, 2.2, .5], 1.)),
                         {(x, y, 0) for x in (-1, 0) for y in range(3)})

    def test_line_on_grid_edge_checks_four_columns(self):
        self.assertEqual(set(segment_voxels([0, 0, .2], [0, 0, 1.2], 1.)),
                         set(product((-1, 0), (-1, 0), (0, 1))))

    def test_zero_length_corner_and_interior(self):
        self.assertEqual(set(segment_voxels([0]*3, [0]*3, 1.)),
                         set(product((-1, 0), repeat=3)))
        self.assertEqual(segment_voxels([.5]*3, [.5]*3, 1.), [(0, 0, 0)])

    def test_endpoint_on_boundary_includes_cell_beyond_endpoint(self):
        self.assertEqual(set(segment_voxels([.5]*3, [1., .5, .5], 1.)),
                         {(0, 0, 0), (1, 0, 0)})

    @staticmethod
    def box_intersects(a, b, voxel):
        # Independent slab/box oracle, not a sampled line or grid-plane walk.
        enter, leave = 0., 1.
        for axis in range(3):
            delta = b[axis] - a[axis]
            low, high = voxel[axis], voxel[axis] + 1
            if delta == 0:
                if not low <= a[axis] <= high:
                    return False
            else:
                t0, t1 = sorted(((low - a[axis]) / delta, (high - a[axis]) / delta))
                enter, leave = max(enter, t0), min(leave, t1)
                if enter > leave + 1e-12:
                    return False
        return True

    def test_random_negative_and_positive_segments_against_box_oracle(self):
        rng = np.random.default_rng(51)
        for _ in range(100):
            a, b = rng.integers(-6, 7, size=(2, 3)) / 4.
            candidates = product(*(range(int(np.floor(min(x, y))) - 1,
                                        int(np.floor(max(x, y))) + 2)
                                   for x, y in zip(a, b)))
            expected = {v for v in candidates if self.box_intersects(a, b, v)}
            got = segment_voxels(a * .75, b * .75, .75)
            self.assertEqual(set(got), expected, (a, b))
            self.assertEqual(len(got), len(set(got)))
            self.assertEqual(set(got), set(segment_voxels(b * .75, a * .75, .75)))
            middle = .37 * a + .63 * b
            split = set(segment_voxels(a * .75, middle * .75, .75))
            split.update(segment_voxels(middle * .75, b * .75, .75))
            self.assertEqual(set(got), split)

    def test_invalid_input_is_rejected_without_unbounded_loop(self):
        for a, b, resolution in (([0]*3, [1]*3, 0), ([0]*3, [1]*3, np.inf),
                                 ([np.nan]*3, [1]*3, 1), ([0]*2, [1]*3, 1)):
            with self.assertRaises(ValueError):
                segment_voxels(a, b, resolution)


if __name__ == "__main__":
    unittest.main()
