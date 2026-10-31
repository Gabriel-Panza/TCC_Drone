import unittest
import numpy as np

from spatial_mapping.interval_fusion import IntervalHypotheses
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.occupancy import OccupancyGrid3D, OccupancyGridConfig
from spatial_mapping.snapshot import capture_planning_snapshot, restore_planning_snapshot


class IntervalFusionTests(unittest.TestCase):
    voxel = (4,0,0)

    def test_wide_interval_repetitions_never_confirm_the_entire_volume(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.,interval_hypothesis_fusion=True))
        for x in [.5,.5,1.5,2.5]:
            grid.integrate_rays([x,.5,.5],[[4.5,.5,.5]],[True],
                occupied_segments=[[[3.5,.5,.5],[6.5,.5,.5]]],free_endpoints=[[3.5,.5,.5]])
        self.assertFalse(grid.occupied_voxels())
        self.assertIn(self.voxel,grid.interval_hypotheses.possible)
        self.assertNotIn(self.voxel,grid.free_voxels())
        self.assertEqual(grid.state(self.voxel),'unknown')

    def test_repeated_same_pose_and_time_do_not_disprove(self):
        layer = IntervalHypotheses(1.,2)
        layer.observe({self.voxel},set(),[0,0,0],set())
        for _ in range(100):
            layer.observe(set(),{self.voxel},[0,0,0],set())
        self.assertIn(self.voxel,layer.possible)
        layer.observe(set(),{self.voxel},[1,0,0],set())
        for _ in range(10):
            layer.observe(set(),{self.voxel},[1,0,0],set())
        self.assertIn(self.voxel,layer.possible)

    def test_separated_free_evidence_shrinks_hypothesis_without_freeing_unseen_space(self):
        layer = IntervalHypotheses(1.,2)
        other = (5,0,0)
        layer.observe({self.voxel,other},set(),[0,0,0],set())
        layer.observe({self.voxel,other},{self.voxel},[1,0,0],set())
        layer.observe({self.voxel,other},{self.voxel},[2,0,0],set())
        self.assertNotIn(self.voxel,layer.possible)
        self.assertIn(other,layer.possible)
        self.assertEqual(layer.cleared_by_multiview,1)
        layer.observe({self.voxel},set(),[3,0,0],set())
        self.assertNotIn(self.voxel,layer.possible)

    def test_atomic_evidence_overrides_prior_disproof_and_never_gets_cleared(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.,interval_hypothesis_fusion=True,
                                                occupied_observations_required=1))
        grid.interval_hypotheses.disproved.add(self.voxel)
        grid.integrate_points([.5,.5,.5],[[4.5,.5,.5]])
        for x in [.5,1.5,2.5]:
            grid.integrate_rays([x,.5,.5],[[8.5,.5,.5]],[False],
                occupied_segments=[[[8.5,.5,.5],[9.5,.5,.5]]],free_endpoints=[[8.5,.5,.5]])
        self.assertIn(self.voxel,grid.occupied_voxels())
        self.assertNotIn(self.voxel,grid.free_voxels())

    def test_narrow_confirmation_requires_distinct_recent_views(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.,interval_hypothesis_fusion=True,
                                                occupied_observations_required=2))
        def narrow(x):
            grid.integrate_rays([x,.5,.5],[[4.5,.5,.5]],[True],
                occupied_segments=[[[4.4,.5,.5],[4.6,.5,.5]]],free_endpoints=[[4.4,.5,.5]],
                narrow_interval_mask=[True])
        narrow(.5); narrow(.5)
        self.assertNotIn(self.voxel,grid.occupied_voxels())
        narrow(1.5)
        self.assertIn(self.voxel,grid.occupied_voxels())

    def test_snapshot_preserves_partially_collected_free_views(self):
        profile = dict(schema='depth_interval_v1',predicted_bin_edges_m=[.5,30.],residual_bounds_m=[[-1.,2.]])
        nav = SpatialNavigator(SpatialNavigationConfig(depth_interval_profile=profile,
            interval_hypothesis_fusion=True,interval_fusion_baseline_m=1.,interval_fusion_free_views=2))
        nav.grid.interval_hypotheses.observe({self.voxel},set(),[0,0,0],set())
        nav.grid.interval_hypotheses.observe(set(),{self.voxel},[1,0,0],set())
        data = capture_planning_snapshot(nav)
        restored = restore_planning_snapshot(data)
        self.assertEqual(restored.grid.interval_hypotheses.free_views,nav.grid.interval_hypotheses.free_views)
        restored.grid.interval_hypotheses.observe(set(),{self.voxel},[2,0,0],set())
        self.assertNotIn(self.voxel,restored.grid.interval_hypotheses.possible)
        data.pop('interval_birth')
        with self.assertRaises(ValueError):
            restore_planning_snapshot(data)

    def test_fusion_requires_profile_and_hypotheses_are_inflated(self):
        with self.assertRaises(ValueError):
            SpatialNavigator(SpatialNavigationConfig(interval_hypothesis_fusion=True))
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.))
        grid.interval_hypotheses.observe({self.voxel},set(),[0,0,0],set())
        self.assertIn((3,0,0),grid.inflated_occupied_voxels(1.,.2,include_hypotheses=True))
        self.assertFalse(grid.inflated_occupied_voxels(1.,.2))

    def test_stale_narrow_views_do_not_count_as_recent_confirmation(self):
        layer = IntervalHypotheses(1.,2)
        layer.record_narrow(self.voxel,[0,0,0],1,6)
        layer.record_narrow(self.voxel,[1,0,0],10,6)
        layer.record_narrow(self.voxel,[1,0,0],11,6)
        self.assertEqual(len(layer.narrow_views[self.voxel]),1)
        layer.record_narrow(self.voxel,[2,0,0],12,6)
        self.assertEqual(len(layer.narrow_views[self.voxel]),2)

    def test_removing_hypothesis_alone_does_not_invent_free_space(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(interval_hypothesis_fusion=True,
            interval_fusion_baseline_m=1.,interval_fusion_free_views=2))
        grid.interval_hypotheses.observe({self.voxel},set(),[0,0,0],set())
        grid.interval_hypotheses.observe(set(),{self.voxel},[1,0,0],set())
        grid.interval_hypotheses.observe(set(),{self.voxel},[2,0,0],set())
        self.assertNotIn(self.voxel,grid.interval_hypotheses.possible)
        self.assertNotIn(self.voxel,grid.free_voxels())

    def test_ray_only_negative_evidence_cannot_disprove_voxel(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.,interval_hypothesis_fusion=True))
        grid.interval_hypotheses.observe({self.voxel},set(),[-2,0,0],set())
        for x in [.5,1.5,2.5]:
            grid.integrate_rays([x,.5,.5],[[8.5,.5,.5]],[False],
                occupied_segments=[[[8.5,.5,.5],[9.5,.5,.5]]],free_endpoints=[[8.5,.5,.5]])
        self.assertIn(self.voxel,grid.interval_hypotheses.possible)
        self.assertNotIn(self.voxel,grid.free_voxels())

    def test_free_validator_cannot_invent_unobserved_voxels(self):
        grid = OccupancyGrid3D(OccupancyGridConfig(resolution_m=1.,interval_hypothesis_fusion=True))
        with self.assertRaises(ValueError):
            grid.integrate_rays([.5,.5,.5],[[4.5,.5,.5]],[False],
                occupied_segments=[[[4.5,.5,.5],[5.5,.5,.5]]],free_endpoints=[[4.5,.5,.5]],
                interval_free_validator=lambda voxels:{(100,100,100)})
