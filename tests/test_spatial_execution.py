"""Regressoes da recuperacao e do consumo do setpoint terminal."""
import unittest
import ast
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from spatial_mapping.execution import select_recovery_path, terminal_arrival_pending


class ExecutionPolicyTest(unittest.TestCase):
    def test_recorded_zigzag_is_not_a_real_retreat(self):
        current = [-7.94268274307251, 8.56118106842041, -1.6431236267089844]
        old_waypoints = [
            [-8.27978801727295, 8.988301277160645, -1.6435432434082031],
            [-7.535299777984619, 8.288877487182617, -1.6404380798339844],
        ]
        selected, diag = select_recovery_path(
            current, list(reversed(old_waypoints)), 1.5, 1.0, 0.2
        )
        self.assertEqual(selected, [])
        self.assertGreater(diag["path_length_m"], 1.5)
        self.assertAlmostEqual(diag["target_displacement_m"], 0.490017686)
        self.assertFalse(diag["selected"])

    def test_continues_to_older_point_for_net_retreat(self):
        history = [(-3., 0., 0.), (-2., 0., 0.), (-1., 0., 0.), (0., 0., 0.)]
        selected, diag = select_recovery_path((0., 0., 0.), history, 1.5, 1., 0.2)
        self.assertEqual(selected[-1], (-2., 0., 0.))
        self.assertTrue(diag["selected"])
        self.assertGreaterEqual(
            diag["target_displacement_m"] - diag["final_acceptance_radius_m"],
            diag["minimum_displacement_m"],
        )

    def test_includes_final_acceptance_in_required_displacement(self):
        selected, diag = select_recovery_path(
            (0., 0., 0.), [(-2., 0., 0.), (-1., 0., 0.)], 1.5, 1., 0.8
        )
        self.assertEqual(selected, [])
        self.assertAlmostEqual(diag["required_target_displacement_m"], 2.3)

    def test_empty_history_rejected(self):
        selected, diag = select_recovery_path((0., 0., 0.), [], 1.5, 1., 0.2)
        self.assertEqual(selected, [])
        self.assertFalse(diag["selected"])

    def test_terminal_target_not_consumed_outside_arrival_region(self):
        # Close enough for the ordinary 0.8m waypoint radius, but not arrived.
        self.assertTrue(terminal_arrival_pending(
            np.array([1.7, 0., 0.]), np.array([1.2, 0., 0.]),
            np.zeros(3), 1.5,
        ))

    def test_terminal_target_consumed_only_after_arrival(self):
        self.assertFalse(terminal_arrival_pending(
            [1.4, 0., 0.], [1.2, 0., 0.], [0., 0., 0.], 1.5
        ))

    def test_nonterminal_subgoal_still_uses_normal_acceptance(self):
        self.assertFalse(terminal_arrival_pending(
            [4., 0., 0.], [3.5, 0., 0.], [0., 0., 0.], 1.5
        ))


class ControllerRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual controller methods without importing ROS/PX4.
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "drone_controller.py").read_text()
        )
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "DroneOffboardNode")
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
                   and n.name in {
                       "construir_caminho_de_recuperacao",
                       "finalizar_recuperacao_espacial",
                   }]
        scope = {"np": np, "deque": deque, "select_recovery_path": select_recovery_path}
        exec(compile(ast.Module(body=methods, type_ignores=[]),
                     "controller_recovery_test", "exec"), scope)
        cls.build = staticmethod(scope["construir_caminho_de_recuperacao"])
        cls.finish = staticmethod(scope["finalizar_recuperacao_espacial"])

    def node(self):
        navigator = Mock()
        navigator.position_is_safe.return_value = True
        navigator.path_is_safe.return_value = True
        navigator.path_avoids_obstacles_allowing_initial_escape.return_value = True
        return SimpleNamespace(
            spatial_safe_position_history=deque([(-3., 0., 0.), (-2., 0., 0.),
                                                (-1., 0., 0.), (0., 0., 0.)]),
            spatial_recovery_retreat_distance_m=1.5,
            spatial_recovery_history_spacing_m=1.,
            spatial_waypoint_acceptance_radius_m=0.8,
            spatial_strict_entry_acceptance_radius_m=0.2,
            spatial_lock=nullcontext(),
            spatial_plan_lock=nullcontext(),
            spatial_estimated_navigator=navigator,
            spatial_reference_navigator=navigator,
            spatial_reference_safety_veto=True,
            spatial_recovery_active=True,
            spatial_recovery_origin=np.zeros(3),
            spatial_recovery_guard=None,
            spatial_recovery_abort_requested=None,
            spatial_recorder=Mock(),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)
            ),
        )

    def test_controller_uses_net_retreat_and_checks_endpoint(self):
        node = self.node()
        self.assertEqual(self.build(node, np.zeros(3))[-1], (-2., 0., 0.))
        node.spatial_estimated_navigator.position_is_safe.return_value = False
        self.assertEqual(self.build(node, np.zeros(3)), [])

    def test_no_movement_is_not_recorded_as_recovery_complete(self):
        node = self.node()
        self.finish(node, np.zeros(3))
        call = node.spatial_recorder.record_state.call_args
        self.assertEqual(call.args[1], "recovery_incomplete")
        self.assertEqual(call.kwargs["displacement_m"], 0.0)

    def test_real_safe_retreat_is_recorded_as_complete(self):
        node = self.node()
        self.finish(node, np.array([-2., 0., 0.]))
        self.assertEqual(node.spatial_recorder.record_state.call_args.args[1],
                         "recovery_complete")

    def test_unsafe_destination_is_not_recorded_as_complete(self):
        node = self.node()
        node.spatial_estimated_navigator.position_is_safe.return_value = False
        self.finish(node, np.array([-2., 0., 0.]))
        self.assertEqual(node.spatial_recorder.record_state.call_args.args[1],
                         "recovery_incomplete")


if __name__ == "__main__":
    unittest.main()
