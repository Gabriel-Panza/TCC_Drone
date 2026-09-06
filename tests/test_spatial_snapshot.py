"""Round-trip do mapa exato e replay seletivo sem imagens/ROS/Gazebo."""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from estudos_e_analises.diagnosticar_planejamento_espacial import main as replay_main
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.execution import RecoveryProgressGuard, RecoveryCandidatePolicy
from spatial_mapping.recorder import SpatialRunRecorder
from spatial_mapping.snapshot import (
    capture_planning_snapshot, restore_planning_snapshot,
    write_planning_snapshot, load_planning_snapshot,
)


class PlanningSnapshotTest(unittest.TestCase):
    def make_navigator(self):
        navigator = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=0.75, occupied_observations_required=2,
        ))
        grid = navigator.grid
        # One confirmed obstacle, one pending, free-ray and pending-clear evidence.
        for _ in range(2):
            grid.integrate_points([0, 0, 0], [[4.0, 0, 0]])
        grid.integrate_points([0, 0, 0], [[0, 4.0, 0]])
        grid.integrate_rays([0, 0, 0], [[0, 5.0, 0]], [False])
        navigator.frames_integrated = grid._integration_frame
        return navigator

    def assert_same_state(self, left, right):
        a, b = capture_planning_snapshot(left), capture_planning_snapshot(right)
        self.assertEqual(set(a), set(b))
        for name in a:
            # Set ordering is not part of the state.
            if name in {"confirmed_occupied", "evidence_frames"}:
                self.assertEqual(set(map(tuple, a[name])), set(map(tuple, b[name])))
            else:
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        self.assertEqual(left.grid.free_voxels(), right.grid.free_voxels())
        self.assertEqual(left._inflated_obstacles(), right._inflated_obstacles())

    def test_roundtrip_preserves_full_temporal_state_and_next_integration(self):
        nav = self.make_navigator()
        restored = restore_planning_snapshot(capture_planning_snapshot(nav))
        self.assert_same_state(nav, restored)
        for obj in (nav, restored):
            obj.grid.integrate_points([0, 0, 0], [[0, 4.0, 0]])
        self.assert_same_state(nav, restored)

    def test_empty_map_roundtrip(self):
        nav = SpatialNavigator()
        self.assert_same_state(nav, restore_planning_snapshot(capture_planning_snapshot(nav)))

    def test_copy_is_detached_from_live_updates(self):
        nav = self.make_navigator()
        snapshot = capture_planning_snapshot(nav)
        restored = restore_planning_snapshot(snapshot)
        nav.grid.mark_ego_voxel_free([100, 100, 100])
        self.assertNotEqual(nav.grid.free_voxels(), restored.grid.free_voxels())
        self.assert_same_state(restored, restore_planning_snapshot(snapshot))

    def test_disk_roundtrip_does_not_overwrite(self):
        nav = self.make_navigator()
        with tempfile.TemporaryDirectory() as directory:
            a = write_planning_snapshot(directory, capture_planning_snapshot(nav))
            b = write_planning_snapshot(directory, capture_planning_snapshot(nav))
            self.assertNotEqual(a, b)
            self.assert_same_state(nav, load_planning_snapshot(directory, a))

    def test_rejects_wrong_schema_and_outside_run_path(self):
        snapshot = capture_planning_snapshot(SpatialNavigator())
        snapshot["schema"] = np.asarray("bad")
        with self.assertRaises(ValueError):
            restore_planning_snapshot(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_planning_snapshot(directory, "../other.npz")

    def test_replay_rejects_safe_path_returning_to_exhausted_frontier(self):
        nav = SpatialNavigator(SpatialNavigationConfig(voxel_resolution_m=1.0))
        for x in range(8):
            nav.grid.mark_ego_voxel_free([x + .5, .5, .5])
        current, goal = [.5, .5, .5], [10.5, .5, .5]
        snapshot = capture_planning_snapshot(nav)
        online = nav.plan(current, goal)
        self.assertTrue(online.success)
        endpoint = tuple(online.waypoints_ned_m[-1])
        guard = RecoveryProgressGuard(0, tuple(goal), endpoint, endpoint, 1.)
        online.diagnostics["recovery_return_guard"] = guard.evaluate(
            current, online.waypoints_ned_m, path_safe=True, extension_m=1.,
            blocked_radius_m=.8, arrival_radius_m=0.,
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = SpatialRunRecorder(
                directory, metadata={"spatial_config": asdict(nav.config)},
                save_frames=False,
            )
            online.diagnostics["planning_snapshot_file"] = write_planning_snapshot(
                recorder.run_dir, snapshot
            )
            online.diagnostics["planning_snapshot_phase"] = "before_plan"
            recorder.record_plan(1., online, "estimated")
            output = io.StringIO()
            with patch("sys.argv", ["replay", str(recorder.run_dir), "--map-source", "snapshot"]):
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
                    replay_main()
            self.assertEqual(error.exception.code, 2)
        result = json.loads(output.getvalue())
        self.assertTrue(result["plan"]["executable_path_is_safe"])
        self.assertFalse(result["gate"]["passed"])
        self.assertFalse(result["plan"]["recovery_return_guard"]["allowed"])

    def test_replay_reproduces_recorded_alternative_search(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1., frontier_standoff_m=2.5,
            lock_path_altitude_to_goal=True, goal_acceptance_radius_m=1.5,
        ))
        for x in range(7):
            for y in range(5):
                nav.grid.mark_ego_voxel_free([x + .5, y + .5, .5])
        current, goal = [.5, .5, .5], [10.5, .5, .5]
        policy = RecoveryCandidatePolicy(
            RecoveryProgressGuard(0, tuple(goal), (4., .5, .5), (4., .5, .5)),
            .8, 1.5, 1., 1.5,
        )
        snapshot = capture_planning_snapshot(nav)
        online = nav.plan(current, goal,
                          candidate_validator=lambda p: policy.evaluate(nav, current, p))
        self.assertTrue(online.success)
        online.diagnostics["recovery_candidate_policy"] = policy.as_dict()
        with tempfile.TemporaryDirectory() as directory:
            recorder = SpatialRunRecorder(
                directory, metadata={"spatial_config": asdict(nav.config)}, save_frames=False)
            online.diagnostics["planning_snapshot_file"] = write_planning_snapshot(
                recorder.run_dir, snapshot)
            online.diagnostics["planning_snapshot_phase"] = "before_plan"
            recorder.record_plan(1., online, "estimated")
            output = io.StringIO()
            with patch("sys.argv", ["replay", str(recorder.run_dir), "--map-source", "snapshot"]):
                with contextlib.redirect_stdout(output):
                    replay_main()
        result = json.loads(output.getvalue())
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["plan"]["diagnostics"]["candidate_search"]["selected_candidate_index"],
            online.diagnostics["candidate_search"]["selected_candidate_index"],
        )
        np.testing.assert_allclose(result["plan"]["waypoints_ned_m"], online.waypoints_ned_m)

    def test_selective_replay_uses_exact_snapshot_without_any_saved_frames(self):
        nav = SpatialNavigator(SpatialNavigationConfig(
            voxel_resolution_m=1.0, lock_path_altitude_to_goal=True,
        ))
        for x in range(6):
            nav.grid.mark_ego_voxel_free([x + .5, .5, .5])
        current, goal = [.5, .5, .5], [5.5, .5, .5]
        snapshot = capture_planning_snapshot(nav)
        online = nav.plan(current, goal)
        with tempfile.TemporaryDirectory() as directory:
            recorder = SpatialRunRecorder(
                directory, metadata={"spatial_config": asdict(nav.config)},
                save_frames=False,
            )
            online.diagnostics["planning_snapshot_file"] = write_planning_snapshot(
                recorder.run_dir, snapshot
            )
            online.diagnostics["planning_snapshot_phase"] = "before_plan"
            recorder.record_plan(1.0, online, "estimated")
            output = io.StringIO()
            with patch("sys.argv", ["replay", str(recorder.run_dir), "--plan-id", "1",
                                    "--map-source", "snapshot"]):
                with contextlib.redirect_stdout(output):
                    replay_main()
        result = json.loads(output.getvalue())
        self.assertTrue(result["online_comparison"]["exact_online_map_reproduced"])
        self.assertEqual(
            result["online_comparison"]["saved_reachable_voxels"],
            result["online_comparison"]["replayed_reachable_voxels"],
        )
        self.assertEqual(
            result["plan"]["waypoints_ned_m"], [list(v) for v in online.waypoints_ned_m]
        )
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["saved_frame_replay"]["frames"], 0)


if __name__ == "__main__":
    unittest.main()
