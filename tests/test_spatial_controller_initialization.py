import unittest
from pathlib import Path

class ControllerInitializationOrderTest(unittest.TestCase):
    def test_execution_validation_does_not_read_dt_before_assignment(self):
        source = (Path(__file__).resolve().parents[1] / 'drone_controller.py').read_text()
        parameter = source.index('self.spatial_execution_validation_interval_s = max(')
        dt_assignment = source.index('self.dt = 0.04')
        self.assertLess(parameter, dt_assignment)
        block = source[parameter:source.index(')', parameter) + 1]
        self.assertNotIn('self.dt', block)

class AsyncPlanningIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).resolve().parents[1] / 'drone_controller.py'
        ).read_text()

    def test_reference_guardian_is_independent_when_veto_is_enabled(self):
        creation = self.source[
            self.source.index('reference_config = replace('):
            self.source.index('self.spatial_lock = threading.RLock()')
        ]
        self.assertIn("'spatial_reference_obstacle_vertical_band_m'", creation)
        self.assertIn("'spatial_reference_vertical_clearance_m'", creation)
        self.assertIn('if self.spatial_reference_safety_veto', creation)
        self.assertNotIn("if self.spatial_depth_source == 'ground_truth_debug'", creation)

    def test_ground_truth_plans_on_physical_guardian_not_only_vetoes_it(self):
        worker = self.source[
            self.source.index('def _calcular_plano_espacial('):
            self.source.index('def publicar_setpoint_frenagem(')
        ]
        self.assertIn("getattr(self, 'spatial_depth_source', None) == 'ground_truth_debug'", worker)
        self.assertIn('planning_navigator = (', worker)
        self.assertIn('plan = planning_navigator.plan(', worker)
        self.assertIn("'physical_reference_guardian'", worker)

    def test_recovery_candidate_budget_is_configured_and_recorded(self):
        worker = self.source[
            self.source.index('def _calcular_plano_espacial('):
            self.source.index('def publicar_setpoint_frenagem(')
        ]
        self.assertIn('max_candidates=self.spatial_recovery_candidate_limit', worker)
        debug = (Path(__file__).resolve().parents[1] / 'config' / 'spatial_debug.yaml').read_text()
        monocular = (Path(__file__).resolve().parents[1] / 'config' / 'spatial_monocular.yaml').read_text()
        self.assertIn('spatial_recovery_candidate_limit: 32', debug)
        self.assertIn('spatial_recovery_candidate_limit: 32', monocular)

    def test_recovery_invalidates_generation_and_blocks_new_request(self):
        request = self.source[
            self.source.index('def solicitar_plano_espacial('):
            self.source.index('def _calcular_plano_espacial(')
        ]
        self.assertIn('if self.spatial_recovery_active:', request)
        self.assertIn('self.spatial_plan_generation += 1', request)

        recovery = self.source[
            self.source.index('if recovery_waypoints:'):
            self.source.index("if self.spatial_recorder is not None:",
                              self.source.index('if recovery_waypoints:'))
        ]
        self.assertIn("getattr(self, 'spatial_plan_generation', 0) + 1", recovery)

    def test_obsolete_result_has_a_side_effect_free_branch(self):
        start = self.source.index('obsolete_goal_result = bool(')
        success = self.source.index('elif plan.success:', start)
        branch = self.source[start:success]
        self.assertIn('if stale_plan_result or obsolete_goal_result:', branch)
        self.assertIn("getattr(self, 'spatial_stale_plan_results', 0) + 1", branch)
        self.assertNotIn('self.spatial_path_waypoints =', branch)
        self.assertNotIn('construir_caminho_de_recuperacao', branch)

    def test_goal_transition_invalidates_inflight_plan_and_settles(self):
        start = self.source.index(
            'self.spatial_goal_settle_started_s = now_s'
        )
        block = self.source[start:start + 1500]
        self.assertIn('self.wp_atual_index += 1', block)
        self.assertIn('self.spatial_plan_generation += 1', block)
        self.assertIn('self.spatial_path_waypoints = []', block)

if __name__ == '__main__':
    unittest.main()
