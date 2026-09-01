"""Politica anti-repeticao e integracao real do controlador, sem ROS."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time
import unittest
import numpy as np

from spatial_mapping.execution import RecoveryProgressGuard
from spatial_mapping.execution import RecoveryCandidatePolicy
from spatial_mapping.execution import evaluate_path_handoff
from dataclasses import replace
from spatial_mapping.execution import plan_publication_decision
from spatial_mapping.navigation import SpatialPlan


class RecoveryProgressTest(unittest.TestCase):
    def guard(self):
        return RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.),
                                     (0., 0., 0.), observation_started_s=10.)

    def evaluate(self, points, current=(-2., 0., 0.), safe=True):
        return self.guard().evaluate(
            current, points, path_safe=safe, extension_m=1.,
            blocked_radius_m=.8, arrival_radius_m=1.5,
        )

    def test_retreat_does_not_make_same_endpoint_new_progress(self):
        result = self.evaluate([(0., 0., 0.)], current=(-5., 0., 0.))
        self.assertFalse(result["allowed"])
        self.assertEqual(result["endpoint_progress_m"], 0.)

    def test_new_forward_exit_is_accepted_only_around_blockage(self):
        result = self.evaluate([(-2., 1., 0.), (2., 1., 0.)])
        self.assertTrue(result["allowed"])
        self.assertEqual(result["reason"], "new_endpoint_progress")

    def test_endpoint_progress_cannot_cross_recovery_blockage(self):
        result = self.evaluate([(2., 0., 0.)])
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "reentry_into_recovery_blockage")
        self.assertLess(result["minimum_distance_to_stop_m"], .8)

    def test_run_175841_rejects_return_through_recovery_blockage(self):
        guard = RecoveryProgressGuard(
            1,
            (-50.0356659889, 69.9711473621, -1.5299087524),
            (-32.2678565979, 43.0775489807, -1.5404930115),
            (-32.2678565979, 43.0775489807, -1.5404930115),
        )
        result = guard.evaluate(
            (-32.5587234497, 41.0960426331, -1.5774765015),
            [
                (-31.875, 43.875, -1.5299087524),
                (-31.875, 48.9375, -1.5299087524),
                (-31.875, 61.625, -1.5299087524),
            ],
            path_safe=True,
            extension_m=1.,
            blocked_radius_m=.8,
            arrival_radius_m=1.5,
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "reentry_into_recovery_blockage")
        self.assertAlmostEqual(result["minimum_distance_to_stop_m"], .1910, places=3)

    def test_safe_lateral_detour_away_from_goal_is_allowed(self):
        result = self.evaluate([(-2., 2., 0.), (0., 2., 0.)])
        self.assertTrue(result["allowed"])
        self.assertEqual(result["reason"], "safe_lateral_detour")
        self.assertLess(result["endpoint_progress_m"], 0.)

    def test_lateral_endpoint_via_same_blockage_is_not_a_detour(self):
        self.assertFalse(self.evaluate([(0., 0., 0.), (0., 2., 0.)])["allowed"])

    def test_unsafe_or_unknown_path_never_bypasses_guard(self):
        self.assertFalse(self.evaluate([(2., 0., 0.)], safe=False)["allowed"])

    def test_arrival_region_does_not_require_one_more_meter(self):
        guard = RecoveryProgressGuard(0, (1.6, 0., 0.), (0., 0., 0.), (0., 0., 0.))
        result = guard.evaluate(
            (-2., 1., 0.), [(0.4, .8, 0.)], path_safe=True,
            extension_m=1., blocked_radius_m=.8, arrival_radius_m=1.5,
        )
        self.assertTrue(result["allowed"])

    def test_accepting_path_does_not_clear_memory_or_reset_timeout(self):
        guard = self.guard()
        for _ in range(5):
            guard.evaluate((-2., 0., 0.), [(0., 0., 0.)], path_safe=True,
                           extension_m=1., blocked_radius_m=.8, arrival_radius_m=1.5)
        self.assertEqual(guard.observation_started_s, 10.)
        self.assertFalse(guard.observation_expired(27.9, 18.))
        self.assertTrue(guard.observation_expired(28., 18.))
        self.assertFalse(guard.actual_progress_reached((-2., 0., 0.), 1., 1.5))
        self.assertTrue(guard.actual_progress_reached((1.1, 0., 0.), 1., 1.5))


class ControllerProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "drone_controller.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "DroneOffboardNode")
        names = {"_calcular_plano_espacial", "atualizar_memoria_recuperacao",
                 "distancia_restante_no_caminho"}
        methods = [
            n for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        scope = {
            "np": np, "time": time,
            "RecoveryProgressGuard": RecoveryProgressGuard,
            "RecoveryCandidatePolicy": RecoveryCandidatePolicy,
            "replace": replace,
            "evaluate_path_handoff": evaluate_path_handoff,
            "plan_publication_decision": plan_publication_decision,
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), "controller_guard_test", "exec"), scope)
        cls.calculate = staticmethod(scope["_calcular_plano_espacial"])
        cls.observe = staticmethod(scope["atualizar_memoria_recuperacao"])
        cls.length = staticmethod(scope["distancia_restante_no_caminho"])

    def node(self, endpoint=(0., 0., 0.)):
        nav = Mock()
        nav.plan.return_value = SpatialPlan(
            True, "local_subgoal", (10., 0., 0.), waypoints_ned_m=[endpoint])
        nav.path_safety_diagnostics.return_value = {"safe": True}
        nav.path_safety_allowing_dynamic_initial_escape.side_effect = (
            lambda position, path: nav.path_safety_diagnostics(position, path)
        )
        nav.position_is_safe.return_value = True
        nav.observation_frontier_diagnostics.return_value = {'observable': True}
        return SimpleNamespace(
            spatial_estimated_navigator=nav, spatial_reference_navigator=nav,
            spatial_reference_safety_veto=True, spatial_recorder=None,
            spatial_lock=nullcontext(), spatial_plan_lock=nullcontext(),
            spatial_recovery_guard=RecoveryProgressGuard(
                0, (10., 0., 0.), (0., 0., 0.), (0., 0., 0.), 10.),
            spatial_recovery_abort_requested=None,
            spatial_recovery_active=False, spatial_wait_scan_active=False,
            spatial_wait_scan_exhausted=False, spatial_wait_scan_started_s=None,
            spatial_wait_scan_max_duration_s=18.,
            spatial_path_waypoints=[], spatial_path_index=0,
            spatial_strict_entry_waypoint=False, spatial_consecutive_short_plans=0,
            spatial_short_plan_recovery_threshold=3,
            spatial_recovery_candidate_limit=16,
            spatial_waypoint_acceptance_radius_m=.8,
            spatial_strict_entry_acceptance_radius_m=.2,
            spatial_min_executable_path_m=1.5, spatial_replan_min_extension_m=1.,
            spatial_global_goal_acceptance_radius_m=1.5,
            spatial_plan_successes=0, spatial_plan_failures=0,
            current_x=-2., current_y=0., current_z=0.,
            wp_atual_index=0, missao_concluida=False,
            distancia_restante_no_caminho=self.length,
            construir_caminho_de_recuperacao=Mock(return_value=[]),
            get_logger=lambda: Mock(),
        )

    def test_repeated_candidate_waits_without_another_recovery(self):
        node = self.node()
        for now in (11., 12., 13., 14.):
            node.spatial_estimated_navigator.plan.return_value.success = True
            self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), now)
        self.assertFalse(node.spatial_current_plan.adopted_for_execution)
        node.construir_caminho_de_recuperacao.assert_not_called()
        self.assertTrue(node.spatial_wait_scan_active)
        self.assertEqual(node.spatial_wait_scan_started_s, 11.)
        self.assertEqual(node.spatial_recovery_guard.observation_started_s, 10.)

    def test_forward_path_adopted_but_memory_waits_for_real_movement(self):
        node = self.node((2., 1., 0.))
        node.current_y = 1.
        self.calculate(node, np.array([-2., 1., 0.]), np.array([10., 0., 0.]), 11.)
        self.assertTrue(node.spatial_current_plan.adopted_for_execution)
        self.assertIsNotNone(node.spatial_recovery_guard)
        self.observe(node, np.array([1.1, 1., 0.]), 12., False)
        self.assertIsNone(node.spatial_recovery_guard)

    def test_consuming_old_path_does_not_erase_visited_corridor(self):
        node = self.node((-2., 2., 0.))
        self.observe(node, np.array([-2., 0., 0.]), 10., False)
        self.observe(node, np.array([-2., 5., 0.]), 12., True)
        node.current_y = 5.
        self.calculate(node, np.array([-2., 5., 0.]), np.array([10., 0., 0.]), 12.)
        plan = node.spatial_estimated_navigator.plan.return_value
        self.assertFalse(plan.adopted_for_execution)
        self.assertEqual(plan.diagnostics['recovery_return_guard']['reason'],
                         'revisited_recovery_corridor')
        self.assertEqual(node.spatial_path_waypoints, [])

    def test_latest_history_vetoes_candidate_at_live_handoff(self):
        node = self.node((-2., 2., 0.))
        calls = []
        def safety(current, waypoints):
            calls.append(1)
            if len(calls) == 3:  # After the worker's initial guard check.
                node.spatial_recovery_guard.record_position((-2., 2., 0.))
            return {'safe': True}
        node.spatial_estimated_navigator.path_safety_diagnostics.side_effect = safety
        self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 11.)
        plan = node.spatial_estimated_navigator.plan.return_value
        self.assertFalse(plan.adopted_for_execution)
        self.assertFalse(plan.diagnostics['path_handoff']['recovery_guard_allowed'])
        self.assertEqual(plan.diagnostics['recovery_return_guard']['reason'],
                         'revisited_recovery_corridor')

    def test_controller_revalidates_observation_frontier_before_adoption(self):
        node = self.node((-2., 3., 0.))
        plan = node.spatial_estimated_navigator.plan.return_value
        plan.diagnostics['observation_frontier'] = {'observable': True}
        node.spatial_estimated_navigator.observation_frontier_diagnostics.return_value = {
            'observable': False, 'reason': 'frontier_occluded_after_standoff'}
        self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 11.)
        self.assertFalse(plan.adopted_for_execution)
        self.assertFalse(plan.diagnostics['path_handoff']['observation_frontier']['observable'])

    def test_timeout_aborts_without_recording_success(self):
        node = self.node()
        node.spatial_recorder = Mock()
        self.assertTrue(self.observe(node, np.array([-2., 0., 0.]), 28., True))
        self.assertTrue(node.missao_concluida)
        self.assertEqual(node.spatial_recorder.record_state.call_args.args[1],
                         "mission_aborted")
        self.assertFalse(node.spatial_wait_scan_active)

    def test_total_timeout_also_bounds_active_path_without_real_progress(self):
        node = self.node()
        self.observe(node, np.array([-2., 0., 0.]), 10., False)
        node.spatial_recorder = Mock()
        self.assertTrue(self.observe(node, np.array([-2., 2., 0.]), 28., False))
        self.assertTrue(node.missao_concluida)
        self.assertEqual(node.spatial_recorder.record_state.call_args.kwargs['reason'],
                         'recovery_no_progress_timeout')

    def test_incomplete_recovery_is_not_cleared_by_apparent_progress(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "recovery_incomplete"
        node.spatial_recorder = Mock()
        self.assertTrue(self.observe(node, np.array([1.1, 0., 0.]), 12., True))
        self.assertEqual(node.spatial_recorder.record_state.call_args.kwargs["reason"],
                         "recovery_incomplete")

    def advancing_exit_node(self):
        node = self.node()
        node.spatial_path_waypoints = [(-2., 5., 0.), (3., 5., 0.)]
        self.observe(node, np.array([-2., 0., 0.]), 26., False)
        node.current_y = 1.2
        self.observe(node, np.array([-2., 1.2, 0.]), 27.5, False)
        node.spatial_recorder = Mock()
        return node

    def test_safe_advancing_exit_gets_one_grace_event(self):
        node = self.advancing_exit_node()
        self.assertFalse(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))
        self.assertFalse(self.observe(node, np.array([-2., 1.2, 0.]), 28.1, False))
        self.assertFalse(node.missao_concluida)
        self.assertEqual(node.spatial_recovery_guard.exit_window.grace_deadline_s, 34.)
        self.assertEqual(node.spatial_recorder.record_state.call_count, 1)
        self.assertEqual(node.spatial_recorder.record_state.call_args.args[1],
                         'recovery_exit_grace_started')

    def test_unsafe_remaining_path_overrides_measured_progress(self):
        node = self.advancing_exit_node()
        node.spatial_estimated_navigator.path_safety_diagnostics.return_value = {
            'safe': False, 'failure_reason': 'inflated'}
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))
        check = node.spatial_recorder.record_state.call_args.kwargs['exit_window_decision']
        self.assertEqual(check['reason'], 'exit_path_unsafe')

    def test_reference_veto_overrides_grace(self):
        node = self.advancing_exit_node()
        node.spatial_reference_navigator = Mock()
        node.spatial_reference_navigator.path_avoids_obstacles_allowing_initial_escape.return_value = False
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))

    def test_grace_rechecks_current_pose_after_map_lock(self):
        node = self.advancing_exit_node()
        node.current_y = 0.
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))
        check = node.spatial_recorder.record_state.call_args.kwargs['exit_window_decision']
        self.assertEqual(check['reason'], 'exit_regressed')
        np.testing.assert_array_equal(
            node.spatial_estimated_navigator.path_safety_diagnostics.call_args.args[0],
            [-2., 0., 0.])

    def test_grace_cannot_mask_asynchronous_abort(self):
        node = self.advancing_exit_node()
        def safety(*args):
            node.spatial_recovery_abort_requested = 'recovery_incomplete'
            return {'safe': True}
        node.spatial_estimated_navigator.path_safety_diagnostics.side_effect = safety
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))
        self.assertEqual(node.spatial_recorder.record_state.call_args.kwargs['reason'],
                         'recovery_incomplete')

    def test_path_changed_during_grace_validation_fails_closed(self):
        node = self.advancing_exit_node()
        def safety(*args):
            node.spatial_path_waypoints = [(-2., 6., 0.), (3., 6., 0.)]
            return {'safe': True}
        node.spatial_estimated_navigator.path_safety_diagnostics.side_effect = safety
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 28., False))
        check = node.spatial_recorder.record_state.call_args.kwargs['exit_window_decision']
        self.assertFalse(check['active_path_unchanged'])

    def test_grace_stall_aborts_and_clears_waypoints(self):
        node = self.advancing_exit_node()
        self.observe(node, np.array([-2., 1.2, 0.]), 28., False)
        self.assertTrue(self.observe(node, np.array([-2., 1.2, 0.]), 30.5, False))
        self.assertEqual(node.spatial_path_waypoints, [])
        check = node.spatial_recorder.record_state.call_args.kwargs['exit_window_decision']
        self.assertEqual(check['reason'], 'exit_stalled')

    def test_new_leg_clears_memory_even_if_goal_coordinates_repeat(self):
        node = self.node()
        node.wp_atual_index = 2
        self.observe(node, np.array([-2., 0., 0.]), 30., True)
        self.assertIsNone(node.spatial_recovery_guard)
        self.assertFalse(node.missao_concluida)

    def test_obsolete_worker_cannot_adopt_or_start_recovery(self):
        node = self.node((2., 0., 0.))
        node.missao_concluida = True
        self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 30.)
        self.assertEqual(node.spatial_path_waypoints, [])
        self.assertFalse(node.spatial_estimated_navigator.plan.return_value.adopted_for_execution)
        node.construir_caminho_de_recuperacao.assert_not_called()

    def test_controller_adopts_safe_alternative_from_real_navigator(self):
        from spatial_mapping.navigation import SpatialNavigator, SpatialNavigationConfig
        node = self.node()
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., frontier_standoff_m=2.5,
            lock_path_altitude_to_goal=True, goal_acceptance_radius_m=1.5,
        ))
        for x in range(7):
            for y in range(5):
                nav.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        current, goal = np.array([.5, .5, .5]), np.array([10.5, .5, .5])
        node.current_x, node.current_y, node.current_z = current
        node.spatial_estimated_navigator = nav
        node.spatial_reference_navigator = nav
        node.spatial_recovery_guard = RecoveryProgressGuard(
            0, tuple(goal), (4., .5, .5), (4., .5, .5), 10.)
        self.calculate(node, current, goal, 11.)
        plan = node.spatial_current_plan
        self.assertTrue(plan.adopted_for_execution)
        self.assertGreater(plan.diagnostics["candidate_search"]["selected_candidate_index"], 0)
        self.assertTrue(plan.diagnostics["recovery_return_guard"]["allowed"])
        self.assertTrue(nav.path_is_safe(current, node.spatial_path_waypoints))
        self.assertIsNotNone(node.spatial_recovery_guard)

    def test_safe_reverse_handoff_preserves_active_path(self):
        node = self.node((3., 0., 0.))
        node.current_x = 0.
        node.spatial_path_waypoints = [(-2., 0., 0.)]
        self.calculate(node, np.zeros(3), np.array([10., 0., 0.]), 11.,
                       preserve_path_on_failure=True)
        self.assertEqual(node.spatial_path_waypoints, [(-2., 0., 0.)])
        plan = node.spatial_estimated_navigator.plan.return_value
        self.assertFalse(plan.adopted_for_execution)

    def test_delayed_unsafe_abort_is_cancelled_only_if_current_position_is_safe(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "unsafe_after_recovery"
        node.spatial_recorder = Mock()
        self.assertFalse(self.observe(node, np.array([-2., 0., 0.]), 12., True))
        self.assertIsNone(node.spatial_recovery_abort_requested)
        self.assertEqual(node.spatial_recovery_guard.observation_started_s, 10.)
        self.assertEqual(node.spatial_path_waypoints, [])

    def test_unsafe_position_still_aborts_after_revalidation(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "unsafe_after_recovery"
        node.spatial_estimated_navigator.position_is_safe.return_value = False
        self.assertTrue(self.observe(node, np.array([-2., 0., 0.]), 12., True))
        self.assertTrue(node.missao_concluida)

    def test_safe_revalidation_does_not_extend_observation_deadline(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "unsafe_after_recovery"
        node.spatial_recorder = Mock()
        self.assertTrue(self.observe(node, np.array([-2., 0., 0.]), 28., True))
        self.assertEqual(node.spatial_recorder.record_state.call_args.kwargs["reason"],
                         "recovery_observation_timeout")

    def test_reference_unsafe_position_prevents_cancelling_abort(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "unsafe_after_recovery"
        node.spatial_reference_navigator = Mock()
        node.spatial_reference_navigator.position_is_safe.return_value = False
        self.assertTrue(self.observe(node, np.array([-2., 0., 0.]), 12., True))

    def test_abort_uses_latest_telemetry_not_pose_passed_before_map_lock(self):
        node = self.node()
        node.spatial_recovery_abort_requested = "unsafe_after_recovery"
        node.current_x = -4.
        node.spatial_estimated_navigator.position_is_safe.side_effect = lambda p: p[0] == -4.
        self.assertFalse(self.observe(node, np.array([-2., 0., 0.]), 12., True))
        np.testing.assert_array_equal(
            node.spatial_estimated_navigator.position_is_safe.call_args.args[0], [-4., 0., 0.]
        )

    def test_controller_rechecks_candidate_from_live_pose_before_adoption(self):
        node = self.node((3., 0., 0.))
        node.current_x = -4.
        node.spatial_estimated_navigator.path_safety_diagnostics.side_effect = (
            lambda p, path: {"safe": p[0] == -2.}
        )
        self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 11.)
        self.assertFalse(node.spatial_current_plan.adopted_for_execution)
        self.assertEqual(node.spatial_path_waypoints, [])
        self.assertEqual(node.spatial_current_plan.diagnostics["path_handoff"]["action"], "reject")

    def test_handoff_revalidates_normal_initial_escape_from_live_pose(self):
        node = self.node((3., 0., 0.))
        node.spatial_recovery_guard = None
        node.current_x = -2.4
        plan = node.spatial_estimated_navigator.plan.return_value
        plan.diagnostics['initial_escape_required'] = True
        node.spatial_estimated_navigator.path_safety_diagnostics.return_value = {
            'safe': False, 'failure_reason': 'inflated'
        }
        node.spatial_estimated_navigator.initial_escape_path_safety_diagnostics.return_value = {
            'safe': True,
            'failure_reason': None,
            'policy': 'known_free_initial_escape_no_reentry',
        }

        self.calculate(
            node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 11.
        )

        self.assertTrue(node.spatial_current_plan.adopted_for_execution)
        handoff = node.spatial_current_plan.diagnostics['path_handoff']
        self.assertEqual(handoff['action'], 'adopt')
        self.assertTrue(handoff['candidate_initial_escape_revalidated'])
        node.spatial_estimated_navigator.initial_escape_path_safety_diagnostics.assert_called()

    def test_controller_does_not_keep_unsafe_existing_path_on_reverse(self):
        node = self.node((3., 0., 0.))
        node.current_x = 0.
        node.spatial_path_waypoints = [(-2., 0., 0.)]
        node.spatial_estimated_navigator.path_safety_diagnostics.side_effect = (
            lambda p, path: {"safe": bool(path and path[-1][0] == 3.)}
        )
        self.calculate(node, np.zeros(3), np.array([10., 0., 0.]), 11.,
                       preserve_path_on_failure=True)
        self.assertTrue(node.spatial_current_plan.adopted_for_execution)
        self.assertEqual(node.spatial_path_waypoints, [(3., 0., 0.)])

    def test_reference_veto_prevents_preserving_existing_path(self):
        node = self.node((3., 0., 0.))
        node.current_x = 0.
        node.spatial_path_waypoints = [(-2., 0., 0.)]
        node.spatial_reference_navigator = Mock()
        node.spatial_reference_navigator.path_avoids_obstacles_allowing_initial_escape.side_effect = (
            lambda p, path: bool(path and path[-1][0] == 3.)
        )
        self.calculate(node, np.zeros(3), np.array([10., 0., 0.]), 11.,
                       preserve_path_on_failure=True)
        self.assertTrue(node.spatial_current_plan.adopted_for_execution)

    def test_failed_worker_requests_abort_only_for_live_unsafe_pose(self):
        for live_safe in (False, True):
            with self.subTest(live_safe=live_safe):
                node = self.node()
                node.current_x = -4.
                node.spatial_estimated_navigator.plan.return_value.success = False
                node.spatial_estimated_navigator.position_is_safe.side_effect = (
                    lambda p: live_safe if p[0] == -4. else not live_safe
                )
                self.calculate(node, np.array([-2., 0., 0.]), np.array([10., 0., 0.]), 11.)
                self.assertEqual(node.spatial_recovery_abort_requested,
                                 None if live_safe else "unsafe_after_recovery")
                self.assertEqual(
                    node.spatial_current_plan.diagnostics["failure_current_position_ned_m"],
                    (-4., 0., 0.),
                )


if __name__ == "__main__":
    unittest.main()
