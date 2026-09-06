import unittest

from spatial_mapping.execution import recovery_brake_decision


class RecoveryBrakeDecisionTest(unittest.TestCase):
    def test_waits_for_minimum_and_speed(self):
        self.assertTrue(
            recovery_brake_decision(.5, .2, .0, 1., 3., .6, .4)["waiting"]
        )
        self.assertTrue(
            recovery_brake_decision(1.2, 2., .1, 1., 3., .6, .4)["waiting"]
        )

    def test_brake_uses_zero_horizontal_velocity_and_fixed_altitude(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "drone_controller.py"
        ).read_text(encoding="utf-8")
        start = source.index("def publicar_setpoint_frenagem(")
        end = source.index("def publicar_setpoint_posicao(", start)
        publisher = source[start:end]
        self.assertIn("msg.position = [", publisher)
        self.assertIn("float(altitude_ned_m)", publisher)
        self.assertIn(
            "msg.velocity = [0.0, 0.0, float('nan')]",
            publisher,
        )
        waiting = source[
            source.index("if brake['waiting']:"):
            source.index("pre_brake_waypoints", source.index("if brake['waiting']:"))
        ]
        self.assertIn("self.publicar_setpoint_frenagem(", waiting)
        self.assertNotIn("self.publicar_setpoint_posicao(", waiting)

    def test_failed_run_224157_cannot_reverse_while_fast(self):
        result = recovery_brake_decision(
            .32, 3.636, .258, 1., 3., .6, .4
        )
        self.assertTrue(result["waiting"])
        self.assertFalse(result["ready"])
        self.assertFalse(result["abort"])

    def test_releases_only_after_slowing(self):
        result = recovery_brake_decision(1.2, .4, .1, 1., 3., .6, .4)
        self.assertTrue(result["ready"])
        self.assertFalse(result["abort"])

    def test_controller_arms_brake_before_recovery_path(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "drone_controller.py"
        ).read_text(encoding="utf-8")
        brake_check = source.index(
            "brake = recovery_brake_decision("
        )
        path_snapshot = source.index(
            "path = list(self.spatial_path_waypoints)",
            brake_check,
        )
        recovery_start = source.index(
            "self.spatial_recovery_active = True"
        )
        self.assertLess(brake_check, path_snapshot)
        settled_rebuild = source.index(
            "self.construir_caminho_de_recuperacao(current)",
            brake_check,
        )
        brake_clear = source.index(
            "self.spatial_recovery_brake_started_s = None",
            settled_rebuild,
        )
        self.assertLess(settled_rebuild, brake_clear)
        self.assertIn(
            "self.spatial_path_waypoints = settled_recovery_waypoints",
            source[settled_rebuild:settled_rebuild + 2500],
        )
        self.assertIn(
            "self.spatial_recovery_brake_started_s = timestamp_s",
            source[recovery_start:recovery_start + 800],
        )
        self.assertIn(
            "self.spatial_recovery_brake_position = np.asarray(",
            source[recovery_start:recovery_start + 800],
        )

    def test_active_path_veto_brakes_before_replanning(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "drone_controller.py"
        ).read_text(encoding="utf-8")
        veto = source.index(
            "reason='active_path_unsafe_from_actual_pose'"
        )
        block = source[veto:veto + 1800]
        self.assertIn("self.iniciar_frenagem_execucao(", block)
        self.assertIn("self.publicar_setpoint_frenagem(", block)
        self.assertNotIn("self.publicar_setpoint_posicao(current)", block)
        brake = source.index(
            "if execution_brake['waiting']:"
        )
        replan = source.index(
            "self.solicitar_plano_espacial(", brake
        )
        self.assertLess(brake, replan)
        waiting = source[brake:replan]
        self.assertIn("self.publicar_setpoint_frenagem(", waiting)
        self.assertIn("return", waiting)

    def test_reference_transition_veto_also_brakes(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "drone_controller.py"
        ).read_text(encoding="utf-8")
        start = source.index("if transition['action'] == 'hold_and_replan':")
        block = source[start:start + 1300]
        self.assertIn("self.iniciar_frenagem_execucao(", block)
        self.assertIn("self.publicar_setpoint_frenagem(", block)
        self.assertNotIn("self.publicar_setpoint_posicao(current)", block)

    def test_collision_run_would_remain_in_braking_state(self):
        result = recovery_brake_decision(
            1.08, 4.389, 0.172, 1.0, 3.0, 0.6, 0.4
        )
        self.assertTrue(result["waiting"])
        self.assertFalse(result["ready"])

    def test_timeout_and_vertical_drift_abort(self):
        timeout = recovery_brake_decision(3., 2., .1, 1., 3., .6, .4)
        self.assertTrue(timeout["abort"])
        self.assertEqual(timeout["abort_reason"], "recovery_brake_timeout")
        drift = recovery_brake_decision(1., .2, .41, 1., 3., .6, .4)
        self.assertTrue(drift["abort"])
        self.assertEqual(
            drift["abort_reason"], "recovery_brake_altitude_drift"
        )


if __name__ == "__main__":
    unittest.main()
