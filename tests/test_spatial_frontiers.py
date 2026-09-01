"""Fronteiras sao destinos observados; desconhecido serve apenas como pista."""
import json
import unittest
from unittest.mock import Mock
import numpy as np

from spatial_mapping.execution import RecoveryCandidatePolicy, RecoveryProgressGuard
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator, SpatialPlan


class ObservationFrontierTest(unittest.TestCase):
    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., drone_clearance_radius_m=.1,
            lock_path_altitude_to_goal=True, frontier_standoff_m=2.5,
            local_plan_radius_m=15., goal_acceptance_radius_m=1.5))
        free = {(x, y, 0) for x in range(-8, 3) for y in range(-3, 4)}
        for v in free:
            nav.grid.mark_ego_voxel_free(nav.grid.voxel_to_world(v))
        return nav, np.array([.5, .5, .5]), np.array([10.5, .5, .5]), free

    def test_frontier_recovery_can_consider_behind_without_changing_normal_rank(self):
        nav, current, goal, free = self.scene()
        old = nav._rank_local_subgoals(current, goal, free)
        new = nav._rank_observation_frontiers(current, goal, free)
        self.assertTrue(any(v[0] < 0 for v in new))
        self.assertTrue(all(v[0] > 0 for v in old))
        self.assertNotIn((0, 0, 0), new)  # Interior, not an observation frontier.
        self.assertEqual(new, nav._rank_observation_frontiers(current, goal, free))

    def test_only_reachable_free_voxels_with_unknown_faces_are_candidates(self):
        nav, current, goal, free = self.scene()
        subset = {v for v in free if v[0] < -3}
        ranked = nav._rank_observation_frontiers(current, goal, subset)
        self.assertTrue(ranked)
        self.assertTrue(set(ranked) <= subset)
        for v in ranked:
            self.assertTrue(nav._frontier_unknown_faces(v, set()))

    def test_no_unknown_face_is_fabricated_from_inflation_or_occupied(self):
        nav, _, _, _ = self.scene()
        v = (-8, 0, 0)
        self.assertEqual(nav._frontier_unknown_faces(v, set()), [(-9, 0, 0)])
        self.assertEqual(nav._frontier_unknown_faces(v, {(-9, 0, 0)}), [])
        nav.grid._log_odds[(-9, 0, 0)] = 3.5
        self.assertEqual(nav._frontier_unknown_faces(v, set()), [])

    def test_exploration_still_respects_radius_and_no_goal_overshoot(self):
        nav, current, goal, free = self.scene()
        goal = np.array([1.5, .5, .5])
        for v in nav._rank_observation_frontiers(current, goal, free):
            p = nav.grid.voxel_to_world(v)
            self.assertLessEqual(p[0], goal[0])
            self.assertLessEqual(np.linalg.norm(p - current), nav.config.local_plan_radius_m)

    def test_unknown_is_not_added_to_executable_path(self):
        nav, current, goal, _ = self.scene()
        before = nav.grid.export_arrays()
        diag = nav.observation_frontier_diagnostics(current, [(-5.5, .5, .5)], (-7.5, .5, .5))
        self.assertTrue(diag['observable'])
        self.assertTrue(diag['unknown_neighbor_voxels'])
        after = nav.grid.export_arrays()
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(diag['unknown_space_traversed'])

    def test_occluded_frontier_after_standoff_is_rejected(self):
        nav, current, _, _ = self.scene()
        nav.grid._log_odds[(-6, 0, 0)] = 3.5
        diag = nav.observation_frontier_diagnostics(current, [(-4.5, .5, .5)], (-7.5, .5, .5))
        self.assertFalse(diag['observable'])
        self.assertEqual(diag['reason'], 'frontier_occluded_after_standoff')

    def test_new_policy_rejects_lateral_endpoint_without_frontier(self):
        nav = Mock()
        nav.path_is_safe.return_value = True
        nav.observation_frontier_diagnostics.return_value = {
            'observable': False, 'reason': 'not_observation_frontier'}
        guard = RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        policy = RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5, explore_frontiers=True)
        plan = SpatialPlan(True, 'local_subgoal', (10., 0., 0.),
                           waypoints_ned_m=[(-2., 3., 0.)])
        result = policy.evaluate(nav, np.array([-2., 0., 0.]), plan)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'not_observation_frontier')

    def test_frontier_cannot_bypass_reference_veto(self):
        nav = Mock()
        nav.path_is_safe.return_value = True
        nav.observation_frontier_diagnostics.return_value = {'observable': True}
        reference = Mock()
        reference.path_avoids_obstacles_allowing_initial_escape.return_value = False
        guard = RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        policy = RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5,
                                         reference_required=True, explore_frontiers=True)
        plan = SpatialPlan(True, 'local_subgoal', (10., 0., 0.),
                           waypoints_ned_m=[(-2., 3., 0.)])
        self.assertEqual(policy.evaluate(nav, np.array([-2., 0., 0.]), plan, reference)['reason'],
                         'reference_veto')

    def test_policy_roundtrip_and_legacy_default(self):
        guard = RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        policy = RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5, explore_frontiers=True)
        data = json.loads(json.dumps(policy.as_dict()))
        self.assertTrue(RecoveryCandidatePolicy.from_dict(data).explore_frontiers)
        del data['explore_frontiers']
        self.assertFalse(RecoveryCandidatePolicy.from_dict(data).explore_frontiers)

    def test_exploration_requires_a_validator_and_obeys_attempt_budget(self):
        nav, current, goal, _ = self.scene()
        with self.assertRaises(ValueError):
            nav.plan(current, goal, recovery_frontiers=True)
        plan = nav.plan(current, goal, recovery_frontiers=True, max_candidates=4,
                        candidate_validator=lambda p: {'accepted': False, 'reason': 'veto'})
        self.assertFalse(plan.success)
        self.assertEqual(plan.waypoints_ned_m, [])
        self.assertLessEqual(len(plan.diagnostics['candidate_search']['attempts']), 4)

    def test_frontiers_are_tried_even_when_no_forward_subgoal_exists(self):
        nav, _, goal, _ = self.scene()
        nav.grid._log_odds = {v: value for v, value in nav.grid._log_odds.items() if v[0] < 0}
        current = np.array([-.5, .5, .5])
        guard = RecoveryProgressGuard(0, tuple(goal), (3., .5, .5), (3., .5, .5))
        policy = RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5, explore_frontiers=True)
        self.assertFalse(nav.plan(current, goal).success)
        plan = nav.plan(current, goal, recovery_frontiers=True,
                        candidate_validator=lambda p: policy.evaluate(nav, current, p))
        self.assertTrue(plan.success, plan.reason)
        self.assertTrue(policy.evaluate(nav, current, plan)['accepted'])
        self.assertTrue(plan.diagnostics['observation_frontier']['observable'])
        self.assertTrue(nav.path_is_safe(current, plan.waypoints_ned_m))


if __name__ == '__main__':
    unittest.main()
