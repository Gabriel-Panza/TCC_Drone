"""Testes do nucleo geometrico e do planejador, sem dependencia do ROS 2."""

import json
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from spatial_mapping import (
    AStar3D,
    CameraIntrinsics,
    OccupancyGrid3D,
    OccupancyGridConfig,
    PathNotFoundError,
    SpatialNavigationConfig,
    SpatialNavigator,
    backproject_depth,
    camera_to_ned_transform,
    transform_points,
)
from spatial_mapping.metrics import depth_metrics, occupancy_metrics
from spatial_mapping.recorder import SpatialRunRecorder

try:
    from spatial_mapping.depth_model import MetricDepthOnnx
except ModuleNotFoundError:
    MetricDepthOnnx = None


class GeometryTest(unittest.TestCase):
    def test_backprojects_center_pixel_on_optical_axis(self):
        depth = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        intrinsics = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0)

        points = backproject_depth(depth, intrinsics)

        np.testing.assert_allclose(points, [[0.0, 0.0, 2.0]])

    def test_applies_homogeneous_transform(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]

        points = transform_points([[0.0, 0.0, 2.0]], transform)

        np.testing.assert_allclose(points, [[1.0, 2.0, 5.0]])

    def test_front_camera_optical_axis_maps_to_body_forward(self):
        transform = camera_to_ned_transform(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
        )

        points = transform_points([[0.0, 0.0, 2.0]], transform)

        np.testing.assert_allclose(points, [[3.0, 2.0, 3.0]])


class OccupancyAndPlanningTest(unittest.TestCase):
    def test_depth_integration_filters_points_outside_vertical_band(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                depth_stride=1,
                min_depth_m=0.1,
                max_depth_m=5.0,
                obstacle_vertical_band_m=0.5,
            )
        )
        depth = np.full((3, 1), 2.0, dtype=float)
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=1.0)
        camera_to_ned = camera_to_ned_transform(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )

        stats = navigator.integrate_depth(depth, intrinsics, camera_to_ned)

        self.assertEqual(stats["points_integrated"], 1)
        self.assertEqual(stats["points_rejected_vertical"], 2)

    def test_marks_ray_as_free_and_endpoint_as_occupied(self):
        config = OccupancyGridConfig(
            resolution_m=1.0,
            occupied_threshold=0.5,
            free_threshold=-0.3,
        )
        grid = OccupancyGrid3D(config)

        grid.integrate_points([0.1, 0.1, 0.1], [[3.1, 0.1, 0.1]])

        self.assertEqual(grid.state((1, 0, 0)), "free")
        self.assertEqual(grid.state((3, 0, 0)), "occupied")

    def test_astar_passes_through_gap_in_wall(self):
        traversable = {
            (x, y, 0)
            for x in range(5)
            for y in range(5)
        }
        wall = {(2, y, 0) for y in range(5) if y != 2}
        planner = AStar3D(traversable, wall, connectivity=6)

        path = planner.plan((0, 0, 0), (4, 4, 0))

        self.assertIn((2, 2, 0), path)
        self.assertFalse(any(voxel in wall for voxel in path))

    def test_astar_does_not_cut_blocked_corner(self):
        traversable = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
        planner = AStar3D(
            traversable,
            blocked_voxels={(1, 0, 0), (0, 1, 0)},
            connectivity=26,
        )

        with self.assertRaises(PathNotFoundError):
            planner.plan((0, 0, 0), (1, 1, 0))

    def test_reachable_component_excludes_disconnected_free_voxels(self):
        planner = AStar3D(
            {(0, 0, 0), (1, 0, 0), (5, 0, 0)},
            connectivity=6,
        )

        reachable = planner.reachable_from((0, 0, 0))

        self.assertEqual(reachable, {(0, 0, 0), (1, 0, 0)})

    def test_navigator_selects_observed_local_subgoal(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                min_subgoal_progress_m=1.0,
                local_plan_radius_m=10.0,
            )
        )
        for x in range(6):
            navigator.grid.mark_free_sphere((x + 0.1, 0.1, 0.1), 0.1)

        plan = navigator.plan((0.1, 0.1, 0.1), (20.0, 0.1, 0.1))

        self.assertTrue(plan.success)
        self.assertEqual(plan.reason, "local_subgoal")
        self.assertGreater(plan.selected_goal_ned_m[0], 4.0)
        self.assertGreater(plan.planning_time_ms, 0.0)
        distances = [
            np.linalg.norm(np.asarray(current) - np.asarray(previous))
            for previous, current in zip(
                plan.waypoints_ned_m,
                plan.waypoints_ned_m[1:],
            )
        ]
        self.assertLessEqual(
            max(distances),
            navigator.config.max_waypoint_spacing_m + 1e-9,
        )

    def test_failed_local_subgoal_reports_reachable_progress(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                min_subgoal_progress_m=2.0,
            )
        )
        navigator.grid.mark_free_sphere((0.1, 0.1, 0.1), 0.1)
        navigator.grid.mark_free_sphere((1.1, 0.1, 0.1), 0.1)

        plan = navigator.plan((0.1, 0.1, 0.1), (20.0, 0.1, 0.1))

        self.assertFalse(plan.success)
        self.assertIn("alcancaveis=2", plan.reason)
        self.assertIn("max_progresso=1.40m", plan.reason)

    def test_path_safety_rejects_new_obstacle(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                drone_clearance_radius_m=0.1,
            )
        )
        for x in range(5):
            navigator.grid.mark_free_sphere((x + 0.1, 0.1, 0.1), 0.1)

        self.assertTrue(
            navigator.path_is_safe((0.1, 0.1, 0.1), [(4.1, 0.1, 0.1)])
        )

        navigator.grid.integrate_points(
            (1.1, 0.1, 0.1),
            [(2.1, 0.1, 0.1)],
        )
        navigator.grid.integrate_points(
            (1.1, 0.1, 0.1),
            [(2.1, 0.1, 0.1)],
        )

        self.assertFalse(
            navigator.path_is_safe((0.1, 0.1, 0.1), [(4.1, 0.1, 0.1)])
        )

    def test_planning_includes_adjacent_current_and_goal_altitude_layers(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=0.75,
                min_subgoal_progress_m=0.5,
                vertical_tolerance_m=0.5,
                drone_clearance_radius_m=0.1,
            )
        )
        for x_index in range(8):
            x = x_index * 0.75 + 0.1
            navigator.grid.mark_free_sphere((x, 0.1, -1.2), 0.1)
            navigator.grid.mark_free_sphere((x, 0.1, -1.65), 0.1)

        plan = navigator.plan(
            (0.1, 0.1, -1.22),
            (5.35, 0.1, -1.66),
        )

        self.assertTrue(plan.success)

    def test_shortcut_removes_voxel_zigzag_in_open_space(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(voxel_resolution_m=1.0)
        )
        traversable = {
            (x, y, 0)
            for x in range(5)
            for y in range(5)
        }
        path = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 4, 0)]

        shortcut = navigator._shortcut_path(path, traversable, set())

        self.assertEqual(shortcut, [(0, 0, 0), (4, 4, 0)])


class MetricsTest(unittest.TestCase):
    def test_depth_metrics_use_common_valid_pixels(self):
        metrics = depth_metrics(
            [[1.0, 2.0], [0.0, 4.0]],
            [[1.0, 3.0], [2.0, 4.0]],
        )

        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertAlmostEqual(metrics["mae_m"], 1.0 / 3.0)

    def test_occupancy_metrics_report_false_free(self):
        config = OccupancyGridConfig(
            resolution_m=1.0,
            occupied_threshold=0.5,
            free_threshold=-0.3,
        )
        estimated = OccupancyGrid3D(config)
        reference = OccupancyGrid3D(config)
        reference.integrate_points([0.1, 0.1, 0.1], [[2.1, 0.1, 0.1]])
        estimated.mark_free_sphere([2.1, 0.1, 0.1], 0.1)

        metrics = occupancy_metrics(estimated, reference)

        self.assertEqual(metrics["false_free_rate"], 1.0)


class DepthModelAndRecorderTest(unittest.TestCase):
    @unittest.skipIf(MetricDepthOnnx is None, "OpenCV indisponivel neste ambiente")
    def test_inverse_depth_output_is_converted_to_meters(self):
        model = MetricDepthOnnx.__new__(MetricDepthOnnx)
        model.output_representation = "inverse_depth"
        model.output_scale = 1.0
        model.output_shift = 0.0
        model.min_depth_m = 0.1
        model.max_depth_m = 50.0

        depth = model.postprocess(np.array([[[[0.5]]]], dtype=np.float32), (1, 1))

        self.assertAlmostEqual(float(depth[0, 0]), 2.0)

    @unittest.skipIf(MetricDepthOnnx is None, "OpenCV indisponivel neste ambiente")
    def test_metric_depth_accepts_common_onnx_output_shapes(self):
        model = MetricDepthOnnx.__new__(MetricDepthOnnx)
        model.output_representation = "metric_depth"
        model.output_scale = 1.0
        model.output_shift = 0.0
        model.min_depth_m = 0.1
        model.max_depth_m = 50.0

        for shape in ((1, 1, 2, 2), (1, 2, 2), (2, 2)):
            with self.subTest(shape=shape):
                raw = np.full(shape, 3.0, dtype=np.float32)
                depth = model.postprocess(raw, (2, 2))
                self.assertEqual(depth.shape, (2, 2))
                self.assertTrue(np.allclose(depth, 3.0))

    def test_recorder_writes_manifest_frames_and_maps(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(voxel_resolution_m=1.0)
        )
        navigator.grid.mark_free_sphere((0.1, 0.1, 0.1), 0.1)
        with TemporaryDirectory() as temp_dir:
            recorder = SpatialRunRecorder(temp_dir, save_frames=True)
            recorder.record_frame(
                timestamp_s=1.0,
                rgb_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                estimated_depth_m=np.ones((2, 2), dtype=np.float32),
                reference_depth_m=np.ones((2, 2), dtype=np.float32),
                camera_to_ned=np.eye(4),
                intrinsics=CameraIntrinsics(1.0, 1.0, 0.5, 0.5),
                source="ground_truth_debug",
                map_stats={"free_voxels": 1},
            )
            recorder.record_state(
                1.5,
                "takeoff_complete",
                position_ned_m=[0.0, 0.0, -1.65],
            )
            recorder.close({"estimated_map": navigator})

            self.assertTrue((recorder.run_dir / "manifest.json").is_file())
            self.assertTrue((recorder.run_dir / "frames/frame_000001.npz").is_file())
            self.assertTrue((recorder.run_dir / "estimated_map.npz").is_file())
            events = [
                json.loads(line)
                for line in recorder.events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                any(event.get("state") == "takeoff_complete" for event in events)
            )


if __name__ == "__main__":
    unittest.main()
