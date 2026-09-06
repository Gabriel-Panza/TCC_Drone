"""Regressoes do limite do subobjetivo; sem ROS ou simulador."""

import unittest

import numpy as np

from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator


class SubgoalBoundsTest(unittest.TestCase):
    def setUp(self):
        self.navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0,
            local_plan_radius_m=25.0,
            min_subgoal_progress_m=0.5,
        ))
        self.current = np.array([0.5, 0.5, 0.5])
        self.goal = np.array([4.5, 0.5, 0.5])

    def test_does_not_choose_voxel_beyond_unknown_goal(self):
        selected = self.navigator._select_local_subgoal(
            self.current, self.goal, {(3, 0, 0), (8, 0, 0)}
        )
        self.assertEqual(selected, (3, 0, 0))

    def test_no_candidate_before_goal_returns_none(self):
        self.assertIsNone(self.navigator._select_local_subgoal(
            self.current, self.goal, {(8, 0, 0)}
        ))

    def test_prefers_nearest_goal_not_far_lateral_projection(self):
        selected = self.navigator._select_local_subgoal(
            self.current, self.goal, {(3, 0, 0), (4, 2, 0)}
        )
        self.assertEqual(selected, (3, 0, 0))

    def test_tie_is_independent_of_candidate_iteration_order(self):
        candidates = [(3, 1, 0), (3, -1, 0)]
        a = self.navigator._select_local_subgoal(
            self.current, self.goal, candidates
        )
        b = self.navigator._select_local_subgoal(
            self.current, self.goal, list(reversed(candidates))
        )
        self.assertEqual(a, b)

    def test_radius_and_minimum_progress_still_apply(self):
        self.assertIsNone(self.navigator._select_local_subgoal(
            self.current, np.array([40.5, 0.5, 0.5]),
            {(0, 0, 0), (-1, 0, 0), (30, 0, 0)},
        ))

    def test_empty_candidates_and_arrival_return_none(self):
        self.assertIsNone(self.navigator._select_local_subgoal(
            self.current, self.goal, set()
        ))
        self.assertIsNone(self.navigator._select_local_subgoal(
            self.current, self.current.copy(), {(1, 0, 0)}
        ))

    def test_astar_can_detour_beyond_goal_plane_and_return(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0,
            lock_path_altitude_to_goal=True,
            connectivity=6,
            frontier_standoff_m=0.5,
        ))
        # A known U-shaped corridor. Only its endpoint is bounded, not the
        # route used to reach it; the goal voxel (4, 1, 0) is unknown.
        corridor = (
            {(0, 1, 0), (0, 2, 0)}
            | {(x, 3, 0) for x in range(7)}
            | {(6, y, 0) for y in range(3)}
            | {(x, 0, 0) for x in range(4, 7)}
        )
        for voxel in corridor:
            navigator.grid.mark_free_sphere(
                navigator.grid.voxel_to_world(voxel), 0.1
            )
        current = np.array([0.5, 1.5, 0.5])
        goal = np.array([4.5, 1.5, 0.5])
        plan = navigator.plan(current, goal)
        self.assertTrue(plan.success, plan.reason)
        self.assertEqual(plan.diagnostics["selected_goal_voxel"], (4, 0, 0))
        self.assertGreater(max(v[0] for v in plan.path_voxels), 4)
        self.assertTrue(navigator.path_is_safe(current, plan.waypoints_ned_m))

    def test_arrival_region_keeps_known_safe_path_without_frontier_trim(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0,
            goal_acceptance_radius_m=1.5,
            frontier_standoff_m=2.5,
        ))
        for x in range(4):
            navigator.grid.mark_free_sphere((x + 0.5, 0.5, 0.5), 0.1)
        current = np.array([0.5, 0.5, 0.5])
        goal = np.array([4.6, 0.5, 0.5])
        plan = navigator.plan(current, goal)
        self.assertTrue(plan.success)
        self.assertEqual(plan.reason, "goal_region_observed")
        self.assertEqual(plan.frontier_standoff_applied_m, 0.0)
        self.assertGreaterEqual(plan.path_length_m, 1.5)
        self.assertEqual(navigator.grid.state((4, 0, 0)), "unknown")
        self.assertTrue(navigator.path_is_safe(current, plan.waypoints_ned_m))
        self.assertLessEqual(
            np.linalg.norm(np.asarray(plan.waypoints_ned_m[-1]) - goal), 1.5
        )

    def test_outside_arrival_region_preserves_frontier_trim(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0,
            goal_acceptance_radius_m=1.0,
            frontier_standoff_m=2.5,
        ))
        for x in range(4):
            navigator.grid.mark_free_sphere((x + 0.5, 0.5, 0.5), 0.1)
        plan = navigator.plan([0.5, 0.5, 0.5], [4.8, 0.5, 0.5])
        self.assertEqual(plan.reason, "local_subgoal")
        self.assertAlmostEqual(plan.frontier_standoff_applied_m, 2.5)
        self.assertAlmostEqual(plan.path_length_m, 0.5)

    def test_interior_arrival_avoids_zero_progress_near_obstacle(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=0.75, goal_acceptance_radius_m=1.5,
            lock_path_altitude_to_goal=True, frontier_standoff_m=2.5,
        ))
        # Geometry of old_41: center is 1.63m from goal but an inset point
        # in that same known cell is 1.44m away. Adjacent cells stay unknown.
        current = np.array([-29.625, 25.125, -1.6152359])
        goal = np.array([-24.99771809, 24.9861211, -1.6152359])
        for x in range(-40, -35):
            navigator.grid.mark_ego_voxel_free(
                navigator.grid.voxel_to_world((x, 33, -3))
            )
        navigator._inflated_obstacles = lambda: {
            (x, 34, -3) for x in range(-40, -35)
        }
        before = navigator.grid.free_voxels().copy()
        plan = navigator.plan(current, goal)
        self.assertTrue(plan.success, plan.reason)
        self.assertEqual(plan.reason, "goal_region_observed")
        self.assertEqual(plan.frontier_standoff_applied_m, 0.0)
        self.assertGreaterEqual(plan.path_length_m, 1.5)
        self.assertEqual(before, navigator.grid.free_voxels())
        self.assertEqual(
            navigator.grid.world_to_voxel(plan.waypoints_ned_m[-1]),
            (-36, 33, -3),
        )
        self.assertLessEqual(
            np.linalg.norm(np.asarray(plan.waypoints_ned_m[-1]) - goal), 1.5
        )
        self.assertTrue(navigator.path_is_safe(current, plan.waypoints_ned_m))

    def test_interior_arrival_still_rejects_unknown_connector(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=0.75, goal_acceptance_radius_m=1.5,
            lock_path_altitude_to_goal=True,
        ))
        for x in (-40, -36):
            navigator.grid.mark_ego_voxel_free(
                navigator.grid.voxel_to_world((x, 33, -3))
            )
        plan = navigator.plan(
            [-29.625, 25.125, -1.615], [-24.9977, 24.9861, -1.615]
        )
        self.assertFalse(plan.success)

    def test_arrival_tolerance_does_not_make_inflated_endpoint_free(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0, goal_acceptance_radius_m=1.5,
        ))
        for x in range(4):
            navigator.grid.mark_free_sphere((x + 0.5, 0.5, 0.5), 0.1)
        navigator._inflated_obstacles = lambda: {(3, 0, 0)}
        plan = navigator.plan([0.5, 0.5, 0.5], [4.6, 0.5, 0.5])
        self.assertNotEqual(plan.reason, "goal_region_observed")

    def test_locked_altitude_uses_commanded_point(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0,
            lock_path_altitude_to_goal=True,
        ))
        # The goal is not at the center of its voxel layer.
        current = np.array([0.5, 0.5, 0.6])
        goal = np.array([4.5, 0.5, 0.2])
        self.assertEqual(navigator._select_local_subgoal(
            current, goal, {(4, 1, 0), (5, 0, 0)}
        ), (4, 1, 0))


if __name__ == "__main__":
    unittest.main()
