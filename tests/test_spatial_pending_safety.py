from dataclasses import replace
import unittest
import numpy as np

from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.snapshot import capture_planning_snapshot, restore_planning_snapshot


class PendingSafetyTests(unittest.TestCase):
    def navigator(self):
        return SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.,
            drone_clearance_radius_m=1., drone_vertical_clearance_m=.2,
            occupied_observations_required=2, occupied_evidence_window_frames=2,
            protect_pending_obstacles=True))

    def test_pending_inflates_but_remains_unconfirmed(self):
        nav = self.navigator()
        nav.grid.integrate_points([.5,.5,.5], [[3.5,.5,.5]])
        self.assertFalse(nav.grid.occupied_voxels())
        self.assertIn((3,0,0), nav.grid.pending_occupied_voxels())
        self.assertIn((2,0,0), nav._inflated_obstacles())
        self.assertFalse(nav.position_is_safe([2.5,.5,.5]))
        restored = restore_planning_snapshot(capture_planning_snapshot(nav))
        self.assertEqual(restored._inflated_obstacles(), nav._inflated_obstacles())

    def test_static_safety_latch_survives_silence_and_conflicting_free_rays(self):
        nav = self.navigator()
        nav.grid.integrate_points([.5,.5,.5], [[3.5,.5,.5]])
        for _ in range(5):
            nav.grid.integrate_rays([.5,.5,.5], np.empty((0,3)), np.empty(0, dtype=bool))
        self.assertIn((3,0,0), nav.grid.pending_occupied_voxels())
        self.assertIn((2,0,0), nav._inflated_obstacles())
        for _ in range(3):
            nav.grid.integrate_rays([.5,.5,.5], [[5.5,.5,.5]], [False])
        self.assertIn((3,0,0), nav.grid.pending_occupied_voxels())
        self.assertNotIn((3,0,0), nav.grid.free_voxels())
        nav.grid.mark_ego_voxel_free([3.5,.5,.5])
        self.assertNotIn((3,0,0), nav.grid.pending_occupied_voxels())

    def test_legacy_pending_still_expires(self):
        nav = SpatialNavigator(replace(self.navigator().config, protect_pending_obstacles=False))
        nav.grid.integrate_points([.5,.5,.5], [[3.5,.5,.5]])
        nav.grid._integration_frame += 3
        nav.grid._expire_old_occupied_evidence()
        self.assertFalse(nav.grid.pending_occupied_voxels())

    def test_initial_escape_never_excuses_unknown_or_occupied_prefix(self):
        nav = self.navigator()
        for x in range(5):
            nav.grid._log_odds[(x,0,0)] = -2.
        nav._inflated_obstacles = lambda: {(0,0,0),(1,0,0)}
        self.assertTrue(nav.initial_escape_path_safety_diagnostics([.5,.5,.5], [[4.5,.5,.5]])['safe'])
        nav.grid._log_odds.pop((1,0,0))
        result = nav.initial_escape_path_safety_diagnostics([.5,.5,.5], [[4.5,.5,.5]])
        self.assertEqual(result['failure_reason'], 'unknown')
        nav.grid._confirmed_occupied.add((1,0,0))
        result = nav.initial_escape_path_safety_diagnostics([.5,.5,.5], [[4.5,.5,.5]])
        self.assertEqual(result['failure_reason'], 'occupied')
