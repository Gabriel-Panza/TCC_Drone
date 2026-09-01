import unittest
from pathlib import Path


class SpatialPhysicalGuardianConfigTest(unittest.TestCase):
    def test_debug_keeps_thin_nominal_map_and_wider_independent_guardian(self):
        text = (
            Path(__file__).resolve().parents[1] / 'config' / 'spatial_debug.yaml'
        ).read_text()
        self.assertIn('spatial_reference_safety_veto: true', text)
        self.assertIn('spatial_obstacle_vertical_band_m: 0.1', text)
        self.assertIn('spatial_vertical_clearance_m: 0.1', text)
        self.assertIn('spatial_reference_obstacle_vertical_band_m: 0.2', text)
        self.assertIn('spatial_reference_vertical_clearance_m: 0.2', text)


class SpatialBatteryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_spatial_battery.sh"
        ).read_text(encoding="utf-8")

    def test_px4_parameters_are_applied_before_readiness_gate(self):
        launch = self.script.index("param set EKF2_MAG_CHK_STR 0.25")
        data_link = self.script.index("param set NAV_DLL_ACT 0")
        save = self.script.index("param save", data_link)
        readiness = self.script.index("Ready for takeoff!", save)
        self.assertLess(launch, data_link)
        self.assertLess(data_link, save)
        self.assertLess(save, readiness)

    def test_safe_aborts_are_recorded_without_relaxing_other_failures(self):
        self.assertIn('BATTERY_CONTINUE_SAFE_ABORTS:-1', self.script)
        self.assertIn('"recovery_observation_timeout"', self.script)
        self.assertIn('"recovery_no_progress_timeout"', self.script)
        self.assertIn('"recovery_brake_no_safe_path"', self.script)
        self.assertIn('outcome = "safe_abort" if reason in safe_reasons else "unsafe_abort"', self.script)
        self.assertIn('elif [[ "$controller_exit" -ne 0 ]]', self.script)

    def test_summary_separates_completion_abort_and_infrastructure_status(self):
        self.assertIn(r'mission_complete\toutcome\tabort_reason\tdataset', self.script)
        self.assertIn('"$outcome" "$abort_reason" "$dataset"', self.script)

    def test_cleanup_does_not_target_micro_xrce_agent(self):
        cleanup_start = self.script.index("stop_residual_gazebo()")
        cleanup_end = self.script.index("cleanup_run()", cleanup_start)
        cleanup = self.script[cleanup_start:cleanup_end]
        self.assertNotIn("MicroXRCEAgent", cleanup)


if __name__ == "__main__":
    unittest.main()
