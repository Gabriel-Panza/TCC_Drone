"""Testes do relatorio de folga, sem ROS/PX4/Gazebo."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_spatial_execution_clearance import audit_command, audit_recovery, audit_run
from spatial_mapping.navigation import SpatialNavigator, SpatialNavigationConfig


class ExecutionClearanceTest(unittest.TestCase):
    def scene(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.))
        for x in range(-2, 5):
            for y in range(-2, 5):
                nav.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        diag = {
            "planning_snapshot_file": "synthetic.npz",
            "current_position_ned_m": [.5, .5, .5],
            "waypoint_acceptance_radius_m": .8,
            "strict_entry_acceptance_radius_m": .2,
            "strict_entry_waypoint_required": False,
            "path_handoff": {
                "current_position_ned_m": [.5, .5, .5],
                "candidate_waypoints_ned_m": [[2.5, .5, .5], [2.5, 2.5, .5]],
            },
        }
        return nav, diag

    def test_safe_nominal_does_not_certify_acceptance_sphere(self):
        nav, diag = self.scene()
        before = copy.deepcopy(nav.grid._log_odds)
        with patch.object(nav, "_inflated_obstacles", return_value={(3, 0, 0)}):
            result = audit_command(nav, diag, [0., .2, .8])
        self.assertTrue(result["nominal_safety"]["safe"])
        self.assertAlmostEqual(result["nominal_clearance"]["distance_m"], .5)
        self.assertEqual(result["sphere_overlap_count"], 1)
        self.assertEqual([r["touches_inflation"] for r in result["extra_tube_comparisons"]],
                         [False, False, True])
        self.assertEqual(nav.grid._log_odds, before)
        json.dumps(result)

    def test_strict_entry_and_final_waypoint_not_conflated(self):
        nav, diag = self.scene()
        diag["strict_entry_waypoint_required"] = True
        with patch.object(nav, "_inflated_obstacles", return_value={(3, 0, 0)}):
            result = audit_command(nav, diag, [.2])
        self.assertEqual(len(result["intermediate_waypoints"]), 1)
        self.assertEqual(result["intermediate_waypoints"][0]["acceptance_radius_m"], .2)
        self.assertEqual(result["sphere_overlap_count"], 0)

    def test_no_obstacle_is_not_known_free_certificate(self):
        nav, diag = self.scene()
        nav.grid._log_odds.pop((1, 0, 0))
        with patch.object(nav, "_inflated_obstacles", return_value=set()):
            result = audit_command(nav, diag, [.8])
        self.assertIsNone(result["nominal_clearance"]["distance_m"])
        self.assertFalse(result["nominal_safety"]["safe"])
        self.assertEqual(result["nominal_safety"]["failure_reason"], "unknown")

    def test_missing_or_empty_recorded_command_raises(self):
        nav, diag = self.scene()
        with self.assertRaises(KeyError):
            audit_command(nav, {}, [.2])
        diag["path_handoff"]["candidate_waypoints_ned_m"] = []
        with self.assertRaises(ValueError):
            audit_command(nav, diag, [.2])

    def test_recovery_matches_timestamp_even_before_failed_plan_line(self):
        nav, old = self.scene()
        failed = copy.deepcopy(old)
        failed["path_handoff"]["existing_path_index"] = 0
        recovery = {"event": "mission_state", "state": "recovery_started",
                    "timestamp_s": 2., "position_ned_m": [1.5, 1.5, .5]}
        events = [
            {"event": "plan", "map_kind": "estimated", "timestamp_s": 1., "plan_id": 21,
             "plan": {"adopted_for_execution": True, "diagnostics": old}},
            recovery,
            {"event": "plan", "map_kind": "estimated", "timestamp_s": 2., "plan_id": 23,
             "plan": {"adopted_for_execution": False, "diagnostics": failed}},
        ]
        with patch("scripts.run_spatial_execution_clearance.restored_navigator", return_value=nav), \
             patch.object(nav, "_inflated_obstacles", return_value={(1, 1, 0)}):
            result = audit_recovery(Path("."), events, recovery)
        self.assertEqual(result["previous_adopted_plan_id"], 21)
        self.assertEqual(result["rejected_plan_id"], 23)
        self.assertTrue(result["inflation_already_present_before_recovery"])
        self.assertAlmostEqual(result["deviation_from_active_nominal_segment"]["distance_m"], 1.)
        self.assertEqual(result["active_command_segment_index"], 0)
        json.dumps(result)

    def test_missing_snapshot_evidence_is_reported_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            events = [{"event": "plan", "map_kind": "estimated", "plan_id": 1,
                       "plan": {"adopted_for_execution": True, "diagnostics": {}}}]
            (run / "events.jsonl").write_text(json.dumps(events[0]))
            records, recoveries, errors = audit_run(run, [.2])
        self.assertEqual(records, [])
        self.assertEqual(recoveries, [])
        self.assertEqual(errors[0]["plan_id"], 1)
        self.assertIn("planning_snapshot_file", errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
