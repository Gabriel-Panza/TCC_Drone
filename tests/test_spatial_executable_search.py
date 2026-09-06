"""Busca por caminho executavel, sem aceitar arclength como progresso."""
import unittest
from unittest.mock import Mock
import numpy as np

from spatial_mapping.execution_reserve import (
    ExecutionReserveNavigator, executable_candidate_diagnostics,
)
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator, SpatialPlan


class ExecutableSearchTest(unittest.TestCase):
    def scene(self):
        source = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., frontier_standoff_m=2.5,
            lock_path_altitude_to_goal=True, goal_acceptance_radius_m=1.5,
        ))
        for x in range(4):
            for y in range(5):
                source.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        return ExecutionReserveNavigator(source, .4), [.5, .5, .5], [10.5, .5, .5]

    def test_short_primary_replaced_by_safe_executable_alternative(self):
        nav, current, goal = self.scene()
        original = nav.plan(current, goal)
        first = executable_candidate_diagnostics(nav, current, original,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertFalse(first['accepted'])
        result = nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=1.5)
        self.assertTrue(result.success, result.reason)
        search = result.diagnostics['candidate_search']
        self.assertGreater(search['selected_candidate_index'], 0)
        self.assertLessEqual(len(search['attempts']), 16)
        self.assertFalse(search['attempts'][0]['validation']['accepted'])
        check = result.diagnostics['executable_candidate_validation']
        self.assertTrue(check['accepted'])
        self.assertGreaterEqual(check['executable_path_length_m'], 1.5)
        self.assertGreaterEqual(check['endpoint_goal_distance_reduction_m'], .5)
        self.assertAlmostEqual(result.frontier_standoff_applied_m, 2.5)

    def test_budget_exhausted_keeps_empty_failure_not_rejected_primary(self):
        nav, current, goal = self.scene()
        result = nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=1.5,
                                     max_candidates=1)
        self.assertFalse(result.success)
        self.assertEqual(result.waypoints_ned_m, [])
        self.assertEqual(len(result.diagnostics['candidate_search']['attempts']), 1)

    def test_long_loop_without_endpoint_progress_is_rejected(self):
        nav, current, goal = self.scene()
        plan = SpatialPlan(True, 'local_subgoal', tuple(goal),
                           waypoints_ned_m=[(.5,3.5,.5), (.7,3.5,.5), (.7,.5,.5)])
        check = executable_candidate_diagnostics(nav, current, plan,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertGreater(check['executable_path_length_m'], 6.)
        self.assertAlmostEqual(check['endpoint_goal_distance_reduction_m'], .2)
        self.assertEqual(check['reason'], 'insufficient_endpoint_progress_after_standoff')
        self.assertFalse(check['accepted'])

    def test_post_standoff_cut_is_checked_not_raw_length(self):
        nav, current, goal = self.scene()
        plan = SpatialPlan(True, 'local_subgoal', tuple(goal), raw_path_length_m=100.,
                           waypoints_ned_m=[(1.7,.5,.5)])
        check = executable_candidate_diagnostics(nav, current, plan,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertFalse(check['accepted'])
        self.assertEqual(check['reason'], 'insufficient_executable_length_after_standoff')

    def test_terminal_arrival_is_not_rejected_only_for_being_short(self):
        nav, current, _ = self.scene()
        plan = SpatialPlan(True, 'goal_observed', (1.5,.5,.5),
                           waypoints_ned_m=[(1.5,.5,.5)])
        check = executable_candidate_diagnostics(nav, current, plan,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertTrue(check['accepted'])

    def test_unsafe_pruning_restores_strict_entry_before_validation(self):
        source = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.))
        for voxel in [(0,0,0), (0,1,0), (1,1,0), (2,1,0)]:
            source.grid.mark_ego_voxel_free(source.grid.voxel_to_world(voxel))
        nav = ExecutionReserveNavigator(source, .4)
        waypoints = [(.5,1.5,.5), (2.5,1.5,.5)]
        plan = SpatialPlan(True, 'local_subgoal', (4.5,1.5,.5), waypoints_ned_m=waypoints)
        check = executable_candidate_diagnostics(nav, [.5,.7,.5], plan,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertTrue(check['accepted'])
        self.assertTrue(check['strict_entry_waypoint_required'])
        self.assertEqual(plan.waypoints_ned_m, waypoints)

    def test_unknown_segment_never_accepted(self):
        nav, current, goal = self.scene()
        nav.grid._log_odds.pop((1,0,0))
        plan = SpatialPlan(True, 'local_subgoal', tuple(goal),
                           waypoints_ned_m=[(3.5,.5,.5)])
        check = executable_candidate_diagnostics(nav, current, plan,
                                               acceptance_m=.8, minimum_m=1.5)
        self.assertEqual(check['reason'], 'unsafe_executable_path')

    def test_external_recovery_or_reference_veto_is_preserved(self):
        nav, current, goal = self.scene()
        veto = Mock(return_value={'accepted': False, 'reason': 'reference_veto'})
        result = nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=1.5,
                                     candidate_validator=veto, max_candidates=4)
        self.assertFalse(result.success)
        veto.assert_called()
        reasons = [a['validation']['reason'] for a in result.diagnostics['candidate_search']['attempts']]
        self.assertIn('reference_veto', reasons)
        self.assertLessEqual(len(reasons), 4)

    def test_unsafe_start_never_attempts_alternative(self):
        nav, current, goal = self.scene()
        nav.grid._log_odds[(0,0,0)] = 3.
        nav.grid._confirmed_occupied.add((0,0,0))
        # Ego update may clear its cell; a neighboring confirmed obstacle still
        # inflates the start and cannot be ignored by the search wrapper.
        nav.grid._log_odds[(1,0,0)] = 3.
        nav.grid._confirmed_occupied.add((1,0,0))
        veto = Mock()
        result = nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=1.5,
                                     candidate_validator=veto)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, 'execution_reserve_unavailable_at_start')
        veto.assert_not_called()

    def test_invalid_inputs_and_unguarded_front_exploration_rejected(self):
        nav, current, goal = self.scene()
        for value in [-1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=value)
        with self.assertRaises(ValueError):
            nav.plan_executable(current, goal, acceptance_m=.8, minimum_m=1.5,
                                recovery_frontiers=True)


if __name__ == '__main__':
    unittest.main()
