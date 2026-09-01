"""Busca limitada deve validar o caminho executavel de cada candidato."""
import unittest
from unittest.mock import Mock
import numpy as np

from spatial_mapping.execution import RecoveryCandidatePolicy, RecoveryProgressGuard
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator, SpatialPlan


class AlternativeSearchTest(unittest.TestCase):
    def test_diversity_reaches_exit_beyond_first_sixteen_ranked_voxels(self):
        nav, _, _, _ = self.scene()
        cluster = [(10, 0, 0)] + [
            (x, y, z) for x in (10, 11) for y in (-1, 0, 1) for z in (-1, 0, 1)
            if (x, y, z) != (10, 0, 0)
        ]
        lateral_exit = (0, -20, 0)
        candidates = cluster + [lateral_exit]
        indices = nav._spatial_candidate_indices(candidates, 16)
        self.assertEqual(indices[:3], [0, len(cluster), 1])
        self.assertGreater(len(cluster), 16)  # Old prefix missed this exit.
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(candidates, cluster + [lateral_exit])

    def test_diversity_ties_use_original_rank_and_are_repeatable(self):
        nav, _, _, _ = self.scene()
        candidates = [(0, 0, 0), (1, 0, 0), (0, 5, 0), (0, -5, 0)]
        expected = [0, 2, 1, 3]
        for _ in range(3):
            self.assertEqual(
                nav._spatial_candidate_indices(candidates, 16), expected
            )
        self.assertEqual(nav._spatial_candidate_indices(candidates, 1), [0])
        self.assertEqual(nav._spatial_candidate_indices([], 16), [])

    def test_diversity_prefers_nearer_ranked_exit_over_farthest_endpoint(self):
        nav, _, _, _ = self.scene()
        candidates = [(10, 0, 0), (10, 1, 0), (10, 3, 0), (0, 20, 0)]
        self.assertEqual(nav._spatial_candidate_indices(candidates, 2), [0, 2])

    def test_no_separated_candidate_falls_back_without_duplicates(self):
        nav, _, _, _ = self.scene()
        candidates = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        self.assertEqual(nav._spatial_candidate_indices(candidates, 16), [0, 1, 2])

    def test_search_records_original_rank_and_obeys_budget_when_all_vetoed(self):
        nav, current, goal, _ = self.scene()
        result = nav.plan(current, goal, max_candidates=16,
            candidate_validator=lambda p: {"accepted": False, "reason": "test_veto"})
        search = result.diagnostics["candidate_search"]
        self.assertEqual(search["selection_strategy"], "ranked_spatial_separation_v1")
        self.assertEqual(search["selection_spacing_m"], 2.5)
        ranks = [a["original_rank"] for a in search["attempts"]]
        self.assertEqual(ranks[0], 1)
        self.assertEqual(len(ranks), 16)
        self.assertEqual(len(set(ranks)), 16)
        self.assertGreater(max(ranks), 16)
        self.assertFalse(result.success)
        self.assertEqual(result.waypoints_ned_m, [])

    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., frontier_standoff_m=2.5,
            lock_path_altitude_to_goal=True, goal_acceptance_radius_m=1.5,
        ))
        for x in range(7):
            for y in range(5):
                nav.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        current = np.array([.5, .5, .5])
        goal = np.array([10.5, .5, .5])
        guard = RecoveryProgressGuard(0, tuple(goal), (4., .5, .5), (4., .5, .5))
        return nav, current, goal, RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5)

    def test_search_preserves_safety_and_full_frontier_standoff(self):
        nav, current, goal, policy = self.scene()
        first = nav.plan(current, goal)
        self.assertFalse(policy.evaluate(nav, current, first)["accepted"])
        result = nav.plan(current, goal,
            candidate_validator=lambda p: policy.evaluate(nav, current, p))
        self.assertTrue(result.success, result.reason)
        self.assertTrue(policy.evaluate(nav, current, result)["accepted"])
        self.assertTrue(nav.path_is_safe(current, result.waypoints_ned_m))
        self.assertAlmostEqual(result.frontier_standoff_applied_m, 2.5)
        search = result.diagnostics["candidate_search"]
        self.assertGreater(search["selected_candidate_index"], 0)
        self.assertLessEqual(len(search["attempts"]), 16)

    def test_budget_exhaustion_never_returns_rejected_primary_as_success(self):
        nav, current, goal, policy = self.scene()
        result = nav.plan(current, goal, max_candidates=1,
            candidate_validator=lambda p: policy.evaluate(nav, current, p))
        self.assertFalse(result.success)
        self.assertEqual(result.waypoints_ned_m, [])
        self.assertTrue(result.diagnostics["candidate_search"]["budget_exhausted"])
        self.assertEqual(len(result.diagnostics["candidate_search"]["attempts"]), 1)

    def test_generic_veto_is_called_only_on_safe_complete_candidates(self):
        nav, current, goal, _ = self.scene()
        calls = []
        def veto(plan):
            self.assertTrue(plan.diagnostics["final_path_safety"]["safe"])
            self.assertTrue(nav.path_is_safe(current, plan.waypoints_ned_m))
            calls.append(plan)
            return {"accepted": False, "reason": "test_veto"}
        result = nav.plan(current, goal, candidate_validator=veto, max_candidates=3)
        self.assertFalse(result.success)
        self.assertEqual(len(calls), 3)

    def test_first_acceptable_candidate_matches_ordinary_plan(self):
        nav, current, goal, _ = self.scene()
        ordinary = nav.plan(current, goal)
        filtered = nav.plan(current, goal,
            candidate_validator=lambda p: {"accepted": True})
        self.assertEqual(ordinary.path_voxels, filtered.path_voxels)
        self.assertEqual(ordinary.waypoints_ned_m, filtered.waypoints_ned_m)
        self.assertEqual(filtered.diagnostics["candidate_search"]["selected_candidate_index"], 0)

    def test_reference_veto_is_not_bypassed_by_trying_more_candidates(self):
        nav, current, goal, policy = self.scene()
        policy.reference_required = True
        reference = Mock()
        reference.path_avoids_obstacles_allowing_initial_escape.return_value = False
        result = nav.plan(current, goal,
            candidate_validator=lambda p: policy.evaluate(nav, current, p, reference))
        self.assertFalse(result.success)
        reference.path_avoids_obstacles_allowing_initial_escape.assert_called()
        self.assertIn("reference_veto", [
            a["validation"]["reason"] for a in result.diagnostics["candidate_search"]["attempts"]
        ])

    def test_connected_component_built_only_once(self):
        from unittest.mock import patch
        from spatial_mapping.astar import AStar3D
        nav, current, goal, _ = self.scene()
        original = AStar3D.reachable_from
        calls = []
        def counted(planner, start):
            calls.append(start)
            return original(planner, start)
        with patch.object(AStar3D, "reachable_from", counted):
            nav.plan(current, goal, max_candidates=4,
                     candidate_validator=lambda p: {"accepted": False})
        self.assertEqual(len(calls), 1)

    def test_policy_rechecks_pruned_path_and_retains_safe_entry(self):
        _, current, goal, policy = self.scene()
        nav = Mock()
        original = [
            (0.6, .5, .5), (0.6, 2.5, .5),
            (7., 2.5, .5), (7., .5, .5),
        ]
        nav.path_is_safe.side_effect = lambda a, path: len(path) == 4
        plan = SpatialPlan(True, "local_subgoal", tuple(goal), waypoints_ned_m=original)
        check = policy.evaluate(nav, current, plan)
        self.assertTrue(check["accepted"])
        self.assertTrue(check["strict_entry_waypoint_required"])
        self.assertEqual(plan.waypoints_ned_m, original)

    def test_unsafe_or_short_candidate_cannot_pass_policy(self):
        _, current, goal, policy = self.scene()
        nav = Mock()
        plan = SpatialPlan(True, "local_subgoal", tuple(goal),
                           waypoints_ned_m=[(7., .5, .5)])
        nav.path_is_safe.return_value = False
        self.assertEqual(policy.evaluate(nav, current, plan)["reason"], "unsafe_executable_path")
        nav.path_is_safe.return_value = True
        plan.waypoints_ned_m = [(1.4, .5, .5)]
        self.assertEqual(policy.evaluate(nav, current, plan)["reason"],
                         "insufficient_executable_length")


if __name__ == "__main__":
    unittest.main()
