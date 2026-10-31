import unittest
import numpy as np

from spatial_mapping.depth_intervals import DepthIntervalProfile, clip_vertical_segments
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.geometry import CameraIntrinsics
from spatial_mapping.occupancy import OccupancyGrid3D, OccupancyGridConfig


def profile():
    return dict(schema='depth_interval_v1', predicted_bin_edges_m=[.5,5,30],
                residual_bounds_m=[[-2,2],[-3,6]])


class DepthIntervalTest(unittest.TestCase):
    def test_empirical_bounds_include_nominal_and_clip_calibrated_domain(self):
        lower, upper = DepthIntervalProfile(profile()).bounds([.5,4,5,30])
        np.testing.assert_allclose(lower,[.5,2,2,27])
        np.testing.assert_allclose(upper,[2.5,6,11,30])
        with self.assertRaises(ValueError):
            DepthIntervalProfile(profile()).bounds([float('nan')])
        with self.assertRaises(ValueError):
            DepthIntervalProfile(profile()).bounds([31])

    def test_rejects_nonfinite_unsorted_or_displaced_only_profiles(self):
        for change in [dict(predicted_bin_edges_m=[.5,30,5]),
                       dict(residual_bounds_m=[[1,2],[-3,6]]),
                       dict(residual_bounds_m=[[-1,float('inf')],[-3,6]])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                DepthIntervalProfile(dict(profile(),**change))

    def test_free_carving_stops_before_nearest_possible_obstacle(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.))
        grid.integrate_rays([.5,.5,.5],[[4.5,.5,.5]],[True],
            occupied_segments=[[[2.5,.5,.5],[6.5,.5,.5]]],free_endpoints=[[2.5,.5,.5]])
        self.assertIn((6,0,0),grid.occupied_voxels())
        self.assertTrue({(x,0,0) for x in range(2,7)} <= grid.occupied_voxels())
        self.assertTrue(grid.free_voxels().isdisjoint(grid.occupied_voxels()))
        self.assertNotIn((2,0,0),grid.free_voxels())

    def test_interval_crossing_body_band_is_kept_even_when_point_is_outside(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.,depth_stride=1,
            obstacle_vertical_band_m=.2,depth_interval_profile=profile()))
        nav.integrate_depth([[3.]],CameraIntrinsics(1,1,0,0),np.eye(4),obstacle_reference_ned_z=4.)
        self.assertIn((0,0,4),nav.grid.occupied_voxels())
        self.assertNotIn((0,0,2),nav.grid.occupied_voxels())

    def test_horizontal_segments_outside_band_are_rejected(self):
        mask, segments = clip_vertical_segments(np.array([[0.,0.,2.],[0.,0.,0.]]),
            np.array([[3.,0.,2.],[3.,0.,0.]]),0.,.2)
        np.testing.assert_array_equal(mask,[False,True])
        self.assertEqual(segments.shape,(2,2,3))

    def test_does_not_silently_ignore_legacy_error_margins(self):
        with self.assertRaises(ValueError):
            SpatialNavigator(SpatialNavigationConfig(depth_interval_profile=profile(),
                                                     depth_free_space_margin_m=3.))
