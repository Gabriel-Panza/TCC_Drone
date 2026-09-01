"""Regressoes da reserva experimental e da troca de waypoint."""
import copy
import unittest
import numpy as np

from spatial_mapping.clearance import path_voxel_clearance
from spatial_mapping.execution_reserve import ExecutionReserveNavigator, waypoint_switch_diagnostics, _ReserveAStar
from spatial_mapping.astar import PathNotFoundError
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator


class ExecutionReserveTest(unittest.TestCase):
    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., drone_clearance_radius_m=0.,
            drone_vertical_clearance_m=.1, lock_path_altitude_to_goal=True,
            frontier_standoff_m=0., max_waypoint_spacing_m=2.,
        ))
        for x in range(8):
            for y in range(5):
                nav.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        nav.grid._log_odds[(3, 0, 0)] = 3.
        nav.grid._confirmed_occupied.add((3, 0, 0))
        return nav

    def test_zero_reserve_reproduces_existing_path_and_does_not_change_source(self):
        source = self.scene()
        before = copy.deepcopy(source.grid._log_odds)
        base = source.plan([.5, 1.5, .5], [6.5, 1.5, .5])
        candidate = ExecutionReserveNavigator(source, 0.).plan([.5, 1.5, .5], [6.5, 1.5, .5])
        self.assertEqual(base.waypoints_ned_m, candidate.waypoints_ned_m)
        self.assertEqual(base.path_voxels, candidate.path_voxels)
        self.assertEqual(source.grid._log_odds, before)

    def test_reserve_reroutes_around_obstacle_without_removing_inflation(self):
        source = self.scene()
        nav = ExecutionReserveNavigator(source, .6)
        start, goal = [.5, 1.5, .5], [6.5, 1.5, .5]
        result = nav.plan(start, goal)
        self.assertTrue(result.success, result.reason)
        self.assertEqual(source._inflated_obstacles(), nav._inflated_obstacles())
        self.assertGreater(path_voxel_clearance(
            [start, *result.waypoints_ned_m], source._inflated_obstacles(), 1.)["distance_m"], .6)
        self.assertTrue(source.path_is_safe(start, result.waypoints_ned_m))

    def test_unsafe_start_not_snapped_to_fake_safe_position(self):
        nav = ExecutionReserveNavigator(self.scene(), .6)
        result = nav.plan([3.5, 1.5, .5], [6.5, 1.5, .5])
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "execution_reserve_unavailable_at_start")
        self.assertEqual(result.waypoints_ned_m, [])

    def test_too_narrow_corridor_cannot_be_crossed_by_reducing_reserve(self):
        source = self.scene()
        source.grid._log_odds = {v: n for v, n in source.grid._log_odds.items()
                               if v[1] <= 1}
        nav = ExecutionReserveNavigator(source, .6)
        result = nav.plan([.5, 1.5, .5], [6.5, 1.5, .5])
        self.assertEqual(result.reason, "local_subgoal")
        self.assertLess(result.waypoints_ned_m[-1][0], 3.)
        self.assertTrue(nav.path_is_safe([.5, 1.5, .5], result.waypoints_ned_m))

    def test_reserve_is_checked_between_endpoints(self):
        nav = ExecutionReserveNavigator(self.scene(), .6)
        # Both endpoints have reserve; interior passes only 0.5m from the box.
        self.assertTrue(nav._has_reserve([[.5, 1.5, .5]]))
        self.assertTrue(nav._has_reserve([[6.5, 1.5, .5]]))
        self.assertFalse(nav._has_reserve([[.5, 1.5, .5], [6.5, 1.5, .5]]))

    def test_reserve_does_not_certify_unknown_space(self):
        source = self.scene()
        source.grid._log_odds.pop((1, 1, 0))
        nav = ExecutionReserveNavigator(source, .2)
        self.assertEqual(nav.path_safety_diagnostics(
            [.5, 1.5, .5], [[2.5, 1.5, .5]])["failure_reason"], "unknown")

    def test_altitude_locked_segments_not_just_voxel_centers_have_reserve(self):
        source = self.scene()
        source.grid._confirmed_occupied.clear()
        source.grid._log_odds[(3, 0, 0)] = -1.
        source.grid._log_odds[(3, 1, 1)] = 3.
        source.grid._confirmed_occupied.add((3, 1, 1))
        nav = ExecutionReserveNavigator(source, .2)
        start, goal = [.5, 1.5, .9], [6.5, 1.5, .9]
        result = nav.plan(start, goal)
        self.assertTrue(result.success, result.reason)
        self.assertTrue(all(p[2] == .9 for p in result.waypoints_ned_m))
        self.assertGreater(path_voxel_clearance(
            [start, *result.waypoints_ned_m], source._inflated_obstacles(), 1.)["distance_m"], .2)

    def test_finite_inputs_required(self):
        for radius in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                ExecutionReserveNavigator(self.scene(), radius)
        with self.assertRaises(ValueError):
            ExecutionReserveNavigator(self.scene(), .2).plan([np.nan, 0, 0], [1, 1, 1])


class InitialConnectionsTest(unittest.TestCase):
    def test_real_pose_has_reserve_but_voxel_center_does_not(self):
        source = ExecutionReserveTest().scene()
        before = copy.deepcopy(source.grid._log_odds)
        current, goal = [3.5, 1.7, .5], [6.5, 1.5, .5]
        legacy = ExecutionReserveNavigator(source, .6, use_start_connections=False)
        old = legacy.plan(current, goal)
        self.assertFalse(old.success)
        self.assertEqual(old.diagnostics['reachable_voxels'], 1)
        nav = ExecutionReserveNavigator(source, .6)
        result = nav.plan(current, goal)
        self.assertTrue(result.success, result.reason)
        self.assertGreater(result.path_length_m, 1.5)
        self.assertTrue(nav.path_is_safe(current, result.waypoints_ned_m))
        selected = result.diagnostics['selected_initial_connection']
        self.assertTrue(selected['safety']['safe'])
        self.assertNotEqual(selected['voxel'], source.grid.world_to_voxel(current))
        self.assertEqual(result.diagnostics['start_voxel'], result.path_voxels[0])
        self.assertGreater(path_voxel_clearance([current, *result.waypoints_ned_m],
                           source._inflated_obstacles(), 1.)['distance_m'], .6)
        self.assertEqual(source.grid._log_odds, before)

    def test_connector_cannot_cross_unknown_even_with_free_endpoint(self):
        source = ExecutionReserveTest().scene()
        source.grid._log_odds.pop((3, 2, 0))
        result = ExecutionReserveNavigator(source, .6).plan([3.5, 1.7, .5], [6.5, 1.5, .5])
        attempts = result.diagnostics['initial_connections']['attempts']
        rejected = next(a for a in attempts if a['voxel'] == (4, 2, 0))
        self.assertFalse(rejected['safety']['safe'])
        self.assertEqual(rejected['safety']['failure_reason'], 'unknown')

    def test_connections_are_bounded_and_all_accepted_segments_keep_reserve(self):
        source = ExecutionReserveTest().scene()
        current = np.asarray([3.5, 1.7, .5])
        nav = ExecutionReserveNavigator(source, .6)
        plan = nav.plan(current, [6.5, 1.5, .5])
        diagnostic = plan.diagnostics['initial_connections']
        self.assertLessEqual(len(diagnostic['attempts']), 27)
        origin = np.asarray(nav.grid.world_to_voxel(current))
        for a in diagnostic['attempts']:
            self.assertLessEqual(np.max(np.abs(np.asarray(a['voxel']) - origin)), 1)
            if a['safety']['safe']:
                self.assertTrue(nav.path_is_safe(current, [a['target_ned_m']]))
                self.assertGreater(path_voxel_clearance([current, a['target_ned_m']],
                                   source._inflated_obstacles(), 1.)['distance_m'], .6)

    def test_search_can_use_non_nearest_connected_component(self):
        planner = _ReserveAStar({(0,0,0), (3,0,0), (4,0,0)}, set(), 6,
                               lambda a,b: True,
                               initial_edges={(0,0,0): .1, (3,0,0): 1.})
        self.assertEqual(planner.reachable_from((0,0,0)), planner.traversable)
        self.assertEqual(planner.plan((0,0,0), (4,0,0)), [(3,0,0), (4,0,0)])

    def test_connector_distance_is_part_of_astar_cost(self):
        planner = _ReserveAStar({(i,0,0) for i in range(5)}, set(), 6,
                               lambda a,b: True,
                               initial_edges={(0,0,0): .1, (3,0,0): 100.})
        self.assertEqual(planner.plan((0,0,0), (4,0,0))[0], (0,0,0))

    def test_no_safe_connector_does_not_fall_back_to_unchecked_start(self):
        planner = _ReserveAStar({(0,0,0), (1,0,0)}, set(), 6,
                               lambda a,b: True, initial_edges={})
        self.assertEqual(planner.reachable_from((0,0,0)), set())
        with self.assertRaises(PathNotFoundError):
            planner.plan((0,0,0), (1,0,0))


class WaypointSwitchTest(unittest.TestCase):
    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.))
        for voxel in [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]:
            nav.grid.mark_ego_voxel_free(nav.grid.voxel_to_world(voxel))
        return nav

    def test_early_switch_to_unknown_corner_is_vetoed(self):
        result = waypoint_switch_diagnostics(
            self.scene(), [.5, .7, .5], [[.5, 1.5, .5], [2.5, 1.5, .5]], .8)
        self.assertEqual(result["action"], "keep_target")
        self.assertFalse(result["switch_allowed"])
        self.assertEqual(result["outgoing_safety"]["failure_reason"], "unknown")

    def test_switch_allowed_after_reaching_safe_side(self):
        result = waypoint_switch_diagnostics(
            self.scene(), [.5, 1.3, .5], [[.5, 1.5, .5], [2.5, 1.5, .5]], .8)
        self.assertEqual(result["action"], "advance")
        self.assertTrue(result["switch_allowed"])

    def test_unsafe_retained_path_requires_replan_not_blind_keep(self):
        nav = self.scene()
        nav.grid._log_odds.pop((0, 1, 0))
        result = waypoint_switch_diagnostics(
            nav, [.5, .7, .5], [[.5, 1.5, .5], [2.5, 1.5, .5]], .8)
        self.assertEqual(result["action"], "replan_required")
        self.assertFalse(result["switch_allowed"])

    def test_terminal_and_outside_radius_are_not_consumed(self):
        nav = self.scene()
        self.assertEqual(waypoint_switch_diagnostics(nav, [.5, .5, .5],
                         [[.5, 1.5, .5]], .8)["action"], "terminal_rules")
        self.assertEqual(waypoint_switch_diagnostics(nav, [.5, .5, .5],
                         [[.5, 1.5, .5], [2.5, 1.5, .5]], .8)["action"],
                         "outside_acceptance")


if __name__ == "__main__":
    unittest.main()
