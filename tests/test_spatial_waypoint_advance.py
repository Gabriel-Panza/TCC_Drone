"""Veto online de troca e metricas de segmento, sem ROS/PX4."""
import unittest
import numpy as np
from spatial_mapping.execution import (
    goal_settle_decision,
    plan_publication_decision,
    segment_tracking_diagnostics,
    waypoint_advance_decision,
)
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator


class SegmentTrackingTest(unittest.TestCase):
    def test_cross_track_projection_and_speed(self):
        result = segment_tracking_diagnostics(
            [1., .4, 0.], [0., 0., 0.], [2., 0., 0.], [3., 4., 0.])
        self.assertAlmostEqual(result['cross_track_m'], .4)
        self.assertAlmostEqual(result['along_track_m'], 1.)
        self.assertAlmostEqual(result['projection_fraction'], .5)
        self.assertAlmostEqual(result['speed_m_s'], 5.)
        self.assertFalse(result['overshot_segment'])

    def test_overshoot_is_not_hidden_by_clipped_projection(self):
        result = segment_tracking_diagnostics([3., .4, 0.], [0.,0.,0.], [2.,0.,0.])
        self.assertTrue(result['overshot_segment'])
        self.assertAlmostEqual(result['unclipped_projection_fraction'], 1.5)
        self.assertAlmostEqual(result['cross_track_m'], np.hypot(1., .4))

    def test_degenerate_segment_and_invalid_values(self):
        result = segment_tracking_diagnostics([1.,0.,0.], [0.,0.,0.], [0.,0.,0.])
        self.assertEqual(result['segment_length_m'], 0.)
        self.assertEqual(result['cross_track_m'], 1.)
        with self.assertRaises(ValueError):
            segment_tracking_diagnostics([np.nan,0,0], [0,0,0], [1,0,0])
        with self.assertRaises(ValueError):
            segment_tracking_diagnostics([0,0,0], [0,0,0], [1,0,0], [1,2])


class WaypointAdvanceTest(unittest.TestCase):
    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.))
        for voxel in [(0,0,0), (0,1,0), (1,1,0), (2,1,0)]:
            nav.grid.mark_ego_voxel_free(nav.grid.voxel_to_world(voxel))
        return nav

    def test_unsafe_early_switch_keeps_current_safe_target(self):
        result = waypoint_advance_decision(
            self.scene(), [.5,.7,.5],
            [[.5,1.5,.5], [2.5,1.5,.5]], 0, 0.8)
        self.assertFalse(result['advance'])
        self.assertEqual(result['action'], 'keep_target')
        self.assertEqual(result['reason'], 'unsafe_early_switch_keep_target')
        self.assertEqual(result['outgoing_path_safety']['failure_reason'], 'unknown')
        self.assertTrue(result['retained_path_safety']['safe'])

    def test_switch_advances_from_actual_safe_pose(self):
        result = waypoint_advance_decision(
            self.scene(), [.5,1.3,.5],
            [[.5,1.5,.5], [2.5,1.5,.5]], 0, .8)
        self.assertTrue(result['advance'])
        self.assertEqual(result['reason'], 'outgoing_path_safe')

    def test_both_paths_unsafe_requires_hold_and_replan(self):
        nav = self.scene()
        nav.grid._log_odds.pop((0,1,0))
        result = waypoint_advance_decision(
            nav, [.5,.7,.5], [[.5,1.5,.5], [2.5,1.5,.5]], 0, .8)
        self.assertFalse(result['advance'])
        self.assertEqual(result['action'], 'hold_and_replan')
        self.assertFalse(result['retained_path_safety']['safe'])


    def test_recovery_can_advance_while_escaping_initial_inflation(self):
        class RecoveryNavigator:
            def path_safety_diagnostics(self, current, path):
                return {'safe': False, 'failure_reason': 'inflated'}
            def path_avoids_obstacles_allowing_initial_escape(self, current, path):
                return True
        result = waypoint_advance_decision(
            RecoveryNavigator(), [0., 0., 0.],
            [[.5, 0., 0.], [2., 0., 0.]], 0, .8,
            allow_initial_escape=True)
        self.assertTrue(result['advance'])
        self.assertEqual(result['outgoing_path_safety']['policy'],
                         'initial_escape_no_reentry')

    def test_recovery_still_vetoes_obstacle_reentry(self):
        class RecoveryNavigator:
            def path_avoids_obstacles_allowing_initial_escape(self, current, path):
                return False
        result = waypoint_advance_decision(
            RecoveryNavigator(), [0., 0., 0.],
            [[.5, 0., 0.], [2., 0., 0.]], 0, .8,
            allow_initial_escape=True)
        self.assertFalse(result['advance'])
        self.assertEqual(result['action'], 'hold_and_replan')
        self.assertEqual(result['outgoing_path_safety']['failure_reason'],
                         'obstacle_reentry')

    def test_outside_acceptance_never_advances(self):
        result = waypoint_advance_decision(
            self.scene(), [.5,.5,.5], [[.5,1.5,.5], [2.5,1.5,.5]], 0, .8)
        self.assertFalse(result['advance'])
        self.assertEqual(result['reason'], 'outside_acceptance')

    def test_terminal_pending_and_final_recovery_target(self):
        pending = waypoint_advance_decision(
            self.scene(), [.5,1.3,.5], [[.5,1.5,.5]], 0, .8,
            terminal_arrival_pending=True)
        self.assertFalse(pending['advance'])
        self.assertEqual(pending['reason'], 'terminal_arrival_pending')
        final = waypoint_advance_decision(
            self.scene(), [.5,1.3,.5], [[.5,1.5,.5]], 0, .8)
        self.assertTrue(final['advance'])
        self.assertEqual(final['reason'], 'final_target_accepted')

    def test_empty_bad_index_and_invalid_input_fail_closed(self):
        self.assertEqual(
            waypoint_advance_decision(self.scene(), [0,0,0], [], 0, 1.)['action'],
            'no_target')
        with self.assertRaises(ValueError):
            waypoint_advance_decision(self.scene(), [0,0,0], [[1,0,0]], 0, -1)
        with self.assertRaises(ValueError):
            waypoint_advance_decision(self.scene(), [np.nan,0,0], [[1,0,0]], 0, 1.)

class AsyncPlanningPolicyTest(unittest.TestCase):
    def test_goal_settle_is_bounded_and_speed_aware(self):
        self.assertTrue(goal_settle_decision(.5, 0., 1., 3., .6)['waiting'])
        self.assertTrue(goal_settle_decision(1.2, 2., 1., 3., .6)['waiting'])
        self.assertFalse(goal_settle_decision(1.2, .4, 1., 3., .6)['waiting'])
        deadline = goal_settle_decision(3., 2., 1., 3., .6)
        self.assertFalse(deadline['waiting'])
        self.assertTrue(deadline['deadline_reached'])

    def test_current_generation_can_publish(self):
        result = plan_publication_decision(7, 7, False, True)
        self.assertTrue(result['publish'])
        self.assertFalse(result['stale'])
        self.assertEqual(result['reason'], 'current')

    def test_recovery_invalidates_inflight_plan_and_preserves_path(self):
        result = plan_publication_decision(7, 8, True, True)
        self.assertFalse(result['publish'])
        self.assertTrue(result['stale'])
        self.assertTrue(result['preserve_active_path'])
        self.assertEqual(result['reason'], 'recovery_active')

    def test_goal_transition_discards_old_result_without_path(self):
        result = plan_publication_decision(7, 8, False, False)
        self.assertFalse(result['publish'])
        self.assertTrue(result['stale'])
        self.assertFalse(result['preserve_active_path'])
        self.assertEqual(result['reason'], 'generation_changed')



if __name__ == '__main__':
    unittest.main()
