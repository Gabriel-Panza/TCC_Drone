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
    compress_collinear_path,
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

    def test_edge_sampling_preserves_thin_depth_discontinuity(self):
        depth = np.full((6, 6), 10.0, dtype=float)
        depth[:, 1] = 2.0
        intrinsics = CameraIntrinsics(fx=2.0, fy=2.0, cx=0.0, cy=0.0)

        regular = backproject_depth(depth, intrinsics, stride=4)
        adaptive = backproject_depth(
            depth,
            intrinsics,
            stride=4,
            edge_stride=1,
            edge_relative_threshold=0.10,
        )

        self.assertNotIn(2.0, regular[:, 2])
        self.assertIn(2.0, adaptive[:, 2])
        self.assertGreater(len(adaptive), len(regular))

    def test_edge_sampling_parameters_must_be_positive(self):
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        with self.assertRaises(ValueError):
            backproject_depth([[1.0]], intrinsics, edge_stride=0)
        with self.assertRaises(ValueError):
            backproject_depth(
                [[1.0]], intrinsics, edge_relative_threshold=0.0
            )

    def test_external_edge_mask_preserves_a_thin_rgb_edge(self):
        depth = np.full((6, 6), 10.0, dtype=float)
        depth[:, 1] = 2.0
        rgb_edges = np.zeros_like(depth, dtype=bool)
        rgb_edges[:, 1] = True
        intrinsics = CameraIntrinsics(fx=2.0, fy=2.0, cx=0.0, cy=0.0)

        points = backproject_depth(
            depth,
            intrinsics,
            stride=4,
            edge_stride=1,
            edge_relative_threshold=10.0,
            sampling_edge_mask=rgb_edges,
        )

        self.assertIn(2.0, points[:, 2])

    def test_external_edge_mask_must_match_depth_shape(self):
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        with self.assertRaises(ValueError):
            backproject_depth(
                np.ones((2, 2)),
                intrinsics,
                edge_stride=1,
                sampling_edge_mask=np.ones((1, 2), dtype=bool),
            )

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
    def test_free_voxel_requires_distinct_viewpoint_sectors_when_configured(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                free_threshold=-0.35,
                free_viewpoint_sectors_required=2,
                free_viewpoint_sector_deg=45.0,
            )
        )
        target = (0, 0, 0)
        grid.integrate_rays(
            [-1.1, 0.1, 0.1],
            [[2.1, 0.1, 0.1]],
            endpoint_is_occupied=[False],
        )
        self.assertEqual(grid.state(target), "unknown")
        grid.integrate_rays(
            [2.1, 0.1, 0.1],
            [[-1.1, 0.1, 0.1]],
            endpoint_is_occupied=[False],
        )
        self.assertEqual(grid.state(target), "free")
        self.assertEqual(len(grid._free_viewpoint_sectors[target]), 2)

    def test_known_free_sphere_preserves_confirmed_obstacle(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=0.75,
                occupied_threshold=0.6,
            )
        )
        obstacle = np.array([2.1, 0.1, 0.1])
        grid.integrate_points([0.1, 0.1, 0.1], [obstacle])
        obstacle_voxel = grid.world_to_voxel(obstacle)

        grid.mark_free_sphere(obstacle, 0.9)

        self.assertEqual(grid.state(obstacle_voxel), "occupied")

    def test_ego_position_clears_only_physically_occupied_voxel(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_threshold=0.6,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        ego_obstacle = np.array([2.1, 0.1, 0.1])
        neighbor_obstacle = np.array([2.1, 1.1, 0.1])
        grid.integrate_points(origin, [ego_obstacle, neighbor_obstacle])

        cleared = grid.mark_ego_voxel_free(ego_obstacle)

        self.assertTrue(cleared)
        self.assertEqual(
            grid.state(grid.world_to_voxel(ego_obstacle)),
            "free",
        )
        self.assertEqual(
            grid.state(grid.world_to_voxel(neighbor_obstacle)),
            "occupied",
        )

    def test_ego_evidence_wins_over_same_frame_monocular_endpoint(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                min_depth_m=0.1,
                known_free_radius_m=0.0,
                obstacle_vertical_band_m=1.0,
            )
        )
        stats = navigator.integrate_depth(
            np.array([[0.5]]),
            CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            np.eye(4),
        )

        self.assertTrue(stats["ego_voxel_cleared_occupied"])
        self.assertEqual(
            navigator.grid.state(
                navigator.grid.world_to_voxel(np.zeros(3))
            ),
            "free",
        )

    def test_planner_clears_body_voxel_but_preserves_neighbor_obstacle(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                known_free_radius_m=0.0,
            )
        )
        current = np.array([2.1, 0.1, 0.1])
        neighbor = np.array([2.1, 1.1, 0.1])
        navigator.grid.integrate_points(
            np.array([0.1, 0.1, 0.1]),
            [current, neighbor],
        )

        plan = navigator.plan(current, np.array([8.1, 0.1, 0.1]))

        self.assertTrue(plan.diagnostics["ego_voxel_cleared_occupied_at_plan"])
        self.assertEqual(navigator.grid.state((2, 0, 0)), "free")
        self.assertEqual(navigator.grid.state((2, 1, 0)), "occupied")

    def test_obstacle_inflation_uses_euclidean_radius(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=0.75,
                occupied_threshold=0.6,
            )
        )
        grid.integrate_points([-1.0, 0.1, 0.1], [[0.1, 0.1, 0.1]])

        inflated = grid.inflated_occupied_voxels(1.25)

        self.assertIn((1, 1, 0), inflated)
        self.assertNotIn((1, 1, 1), inflated)
        self.assertNotIn((2, 0, 0), inflated)

    def test_anisotropic_inflation_preserves_horizontal_clearance(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=0.75,
                occupied_threshold=0.6,
            )
        )
        grid.integrate_points([-1.0, 0.1, 0.1], [[0.1, 0.1, 0.1]])

        inflated = grid.inflated_occupied_voxels(
            1.25,
            vertical_radius_m=0.4,
        )

        self.assertIn((1, 0, 0), inflated)
        self.assertNotIn((0, 0, 1), inflated)

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
        self.assertEqual(stats["free_only_rays"], 2)
        self.assertEqual(stats["total_valid_rays"], 3)
        self.assertEqual(navigator.grid.state((1, 0, -1)), "free")
        self.assertEqual(navigator.grid.state((2, 0, -2)), "unknown")

    def test_free_only_rays_preserve_safe_exit_near_obstacle(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                drone_clearance_radius_m=1.25,
                min_subgoal_progress_m=0.5,
                local_plan_radius_m=12.0,
                max_waypoint_spacing_m=20.0,
                frontier_standoff_m=2.5,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        safe_exit_end = np.array([8.1, 0.1, 0.1])
        nearby_obstacle = np.array([2.1, 2.1, 0.1])

        navigator.grid.integrate_rays(
            origin,
            [safe_exit_end],
            endpoint_is_occupied=[False],
        )
        navigator.grid.integrate_points(origin, [nearby_obstacle])

        plan = navigator.plan(origin, (20.0, 0.1, 0.1))

        self.assertTrue(plan.success)
        self.assertEqual(plan.reason, "local_subgoal")
        self.assertGreaterEqual(plan.path_length_m, 1.5)
        self.assertGreater(plan.diagnostics["max_reachable_progress_m"], 6.0)
        self.assertGreater(
            plan.diagnostics["post_standoff_path_length_m"],
            1.5,
        )
        self.assertEqual(
            navigator.grid.state(navigator.grid.world_to_voxel(nearby_obstacle)),
            "occupied",
        )

    def test_free_only_ray_does_not_clear_confirmed_obstacle(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_threshold=0.5,
                free_threshold=-0.3,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        obstacle = np.array([2.1, 0.1, 0.1])
        grid.integrate_points(origin, [obstacle])

        grid.integrate_rays(
            origin,
            [[4.1, 0.1, 0.1]],
            endpoint_is_occupied=[False],
        )

        self.assertEqual(grid.state(grid.world_to_voxel(obstacle)), "occupied")

    def test_navigator_requires_configured_free_observations(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                known_free_radius_m=0.0,
                free_observations_required=2,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        endpoint = np.array([3.1, 0.1, 0.1])

        navigator.grid.integrate_rays(
            origin,
            [endpoint],
            endpoint_is_occupied=[False],
        )
        self.assertEqual(navigator.grid.state((1, 0, 0)), "unknown")

        navigator.grid.integrate_rays(
            origin,
            [endpoint],
            endpoint_is_occupied=[False],
        )
        self.assertEqual(navigator.grid.state((1, 0, 0)), "free")

    def test_free_observations_required_must_be_positive(self):
        with self.assertRaises(ValueError):
            SpatialNavigationConfig(free_observations_required=0)

    def test_navigator_requires_configured_occupied_observations(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                known_free_radius_m=0.0,
                occupied_observations_required=2,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        endpoint = np.array([3.1, 0.1, 0.1])
        endpoint_voxel = navigator.grid.world_to_voxel(endpoint)

        navigator.grid.integrate_rays(
            origin,
            [endpoint],
            endpoint_is_occupied=[True],
        )
        self.assertEqual(navigator.grid.state(endpoint_voxel), "unknown")
        self.assertIn(
            endpoint_voxel,
            navigator.grid.pending_occupied_voxels(),
        )

        navigator.grid.integrate_rays(
            origin,
            [endpoint],
            endpoint_is_occupied=[True],
        )
        self.assertEqual(navigator.grid.state(endpoint_voxel), "occupied")
        self.assertNotIn(
            endpoint_voxel,
            navigator.grid.pending_occupied_voxels(),
        )

    def test_pending_obstacle_is_not_cleared_by_one_free_observation(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_observations_required=2,
                pending_clear_free_observations_required=3,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        obstacle = np.array([3.1, 0.1, 0.1])
        voxel = grid.world_to_voxel(obstacle)
        grid.integrate_rays(origin, [obstacle], [True])
        grid.integrate_rays(origin, [[5.1, 0.1, 0.1]], [False])

        self.assertEqual(grid.state(voxel), "unknown")
        self.assertIn(voxel, grid.pending_occupied_voxels())

    def test_pending_obstacle_requires_repeated_free_evidence_to_expire(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_observations_required=2,
                pending_clear_free_observations_required=3,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        obstacle = np.array([3.1, 0.1, 0.1])
        voxel = grid.world_to_voxel(obstacle)
        grid.integrate_rays(origin, [obstacle], [True])
        for _ in range(3):
            grid.integrate_rays(origin, [[5.1, 0.1, 0.1]], [False])

        self.assertNotIn(voxel, grid.pending_occupied_voxels())

    def test_neighboring_observations_confirm_with_spatial_support(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_observations_required=2,
                occupied_support_radius_voxels=1,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        first = np.array([3.1, 0.1, 0.1])
        second = np.array([3.1, 1.1, 0.1])
        grid.integrate_rays(origin, [first], [True])
        grid.integrate_rays(origin, [second], [True])

        self.assertEqual(grid.state(grid.world_to_voxel(second)), "occupied")

    def test_neighboring_observations_do_not_confirm_without_support(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_observations_required=2,
                occupied_support_radius_voxels=0,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        first = np.array([3.1, 0.1, 0.1])
        second = np.array([3.1, 1.1, 0.1])
        grid.integrate_rays(origin, [first], [True])
        grid.integrate_rays(origin, [second], [True])

        self.assertEqual(grid.state(grid.world_to_voxel(second)), "unknown")

    def test_occupied_observations_required_has_supported_range(self):
        with self.assertRaises(ValueError):
            SpatialNavigationConfig(occupied_observations_required=0)
        with self.assertRaises(ValueError):
            SpatialNavigationConfig(occupied_observations_required=5)

    def test_depth_stats_distinguish_pending_and_confirmed_occupancy(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                known_free_radius_m=0.0,
                occupied_observations_required=2,
                obstacle_vertical_band_m=3.0,
            )
        )
        depth = np.array([[2.0]])
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

        first = navigator.integrate_depth(depth, intrinsics, np.eye(4))
        second = navigator.integrate_depth(depth, intrinsics, np.eye(4))

        self.assertEqual(first["occupied_voxels"], 0)
        self.assertEqual(first["pending_occupied_voxels"], 1)
        self.assertEqual(second["occupied_voxels"], 1)
        self.assertEqual(second["pending_occupied_voxels"], 0)
        self.assertEqual(second["occupied_observations_required"], 2)

    def test_overlapping_rays_count_once_per_integration_frame(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                free_threshold=-0.75,
            )
        )
        origin = np.array([0.1, 0.1, 0.1])
        endpoints = np.array(
            [
                [3.1, 0.1, 0.1],
                [3.1, 0.2, 0.1],
                [3.1, 0.3, 0.1],
            ]
        )

        grid.integrate_rays(
            origin,
            endpoints,
            endpoint_is_occupied=[False, False, False],
        )
        self.assertEqual(grid.state((1, 0, 0)), "unknown")

        grid.integrate_rays(
            origin,
            endpoints,
            endpoint_is_occupied=[False, False, False],
        )
        self.assertEqual(grid.state((1, 0, 0)), "free")

    def test_occupied_uncertainty_marks_near_side_of_surface(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_threshold=0.5,
            )
        )
        grid.integrate_rays(
            [0.1, 0.1, 0.1],
            [[5.1, 0.1, 0.1]],
            endpoint_is_occupied=[True],
            free_space_margin_m=2.0,
            occupied_uncertainty_m=1.5,
        )

        self.assertEqual(grid.state((4, 0, 0)), "occupied")
        self.assertEqual(grid.state((5, 0, 0)), "occupied")

    def test_occupied_uncertainty_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            OccupancyGrid3D().integrate_rays(
                [0.0, 0.0, 0.0],
                [[1.0, 0.0, 0.0]],
                endpoint_is_occupied=[True],
                occupied_uncertainty_m=-0.1,
            )

    def test_free_space_margin_keeps_uncertain_voxels_unknown(self):
        grid = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_threshold=0.5,
                free_threshold=-0.3,
            )
        )

        grid.integrate_rays(
            [0.1, 0.1, 0.1],
            [[5.1, 0.1, 0.1]],
            endpoint_is_occupied=[True],
            free_space_margin_m=2.0,
        )

        self.assertEqual(grid.state((3, 0, 0)), "free")
        self.assertEqual(grid.state((4, 0, 0)), "unknown")
        self.assertEqual(grid.state((5, 0, 0)), "occupied")

    def test_distance_scaled_free_space_margin_is_more_conservative_far_away(self):
        near = OccupancyGrid3D(
            OccupancyGridConfig(
                resolution_m=1.0,
                occupied_threshold=0.5,
                free_threshold=-0.3,
            )
        )
        far = OccupancyGrid3D(near.config)
        kwargs = {
            "endpoint_is_occupied": [True],
            "free_space_margin_m": 1.0,
            "free_space_margin_ratio": 0.4,
            "free_space_margin_max_m": 8.0,
        }
        near.integrate_rays([0.1, 0.1, 0.1], [[5.1, 0.1, 0.1]], **kwargs)
        far.integrate_rays([0.1, 0.1, 0.1], [[15.1, 0.1, 0.1]], **kwargs)

        self.assertEqual(near.state((2, 0, 0)), "free")
        self.assertEqual(near.state((3, 0, 0)), "unknown")
        self.assertEqual(far.state((7, 0, 0)), "free")
        self.assertEqual(far.state((8, 0, 0)), "free")
        self.assertEqual(far.state((9, 0, 0)), "unknown")

    def test_free_space_margin_ratio_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            OccupancyGrid3D().integrate_rays(
                [0.0, 0.0, 0.0],
                [[1.0, 0.0, 0.0]],
                endpoint_is_occupied=[True],
                free_space_margin_ratio=-0.1,
            )

    def test_free_space_margin_max_cannot_be_below_base(self):
        with self.assertRaises(ValueError):
            OccupancyGrid3D().integrate_rays(
                [0.0, 0.0, 0.0],
                [[5.0, 0.0, 0.0]],
                endpoint_is_occupied=[True],
                free_space_margin_m=2.0,
                free_space_margin_max_m=1.0,
            )

    def test_free_space_margin_cannot_be_negative(self):
        grid = OccupancyGrid3D()

        with self.assertRaises(ValueError):
            grid.integrate_rays(
                [0.0, 0.0, 0.0],
                [[1.0, 0.0, 0.0]],
                endpoint_is_occupied=[True],
                free_space_margin_m=-0.1,
            )

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

    def test_local_subgoal_keeps_standoff_from_observed_frontier(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                local_plan_radius_m=12.0,
                min_subgoal_progress_m=0.5,
                max_waypoint_spacing_m=20.0,
                frontier_standoff_m=2.5,
            )
        )
        for x in range(10):
            navigator.grid.mark_free_sphere((x + 0.1, 0.1, 0.1), 0.1)

        current = np.array([0.1, 0.1, 0.1])
        plan = navigator.plan(current, (20.0, 0.1, 0.1))

        self.assertTrue(plan.success)
        self.assertEqual(plan.reason, "local_subgoal")
        self.assertAlmostEqual(plan.frontier_standoff_applied_m, 2.5)
        selected_distance = np.linalg.norm(
            np.asarray(plan.selected_goal_ned_m) - current
        )
        executed_distance = np.linalg.norm(
            np.asarray(plan.waypoints_ned_m[-1]) - current
        )
        self.assertAlmostEqual(
            selected_distance - executed_distance,
            2.5,
            delta=0.01,
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

    def test_reference_guardian_allows_initial_escape_but_not_reentry(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=1.0,
                drone_clearance_radius_m=0.0,
                drone_vertical_clearance_m=0.4,
            )
        )
        current = np.array([0.1, 0.1, 0.1])
        navigator.grid.integrate_points([-1.1, 0.1, 0.1], [current])

        self.assertTrue(
            navigator.path_avoids_obstacles_allowing_initial_escape(
                current,
                [[3.1, 0.1, 0.1]],
            )
        )

        navigator.grid.integrate_points(
            [1.1, 0.1, 0.1],
            [[2.1, 0.1, 0.1]],
        )

        self.assertFalse(
            navigator.path_avoids_obstacles_allowing_initial_escape(
                current,
                [[3.1, 0.1, 0.1]],
            )
        )

        self.assertEqual(
            navigator.first_obstacle_reentry_voxel(
                current,
                [[3.1, 0.1, 0.1]],
            ),
            (2, 0, 0),
        )

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

    def test_level_route_keeps_continuous_goal_altitude_across_voxel_boundary(self):
        navigator = SpatialNavigator(
            SpatialNavigationConfig(
                voxel_resolution_m=0.75,
                min_subgoal_progress_m=0.5,
                vertical_tolerance_m=0.5,
                drone_clearance_radius_m=0.1,
                frontier_standoff_m=0.0,
                lock_path_altitude_to_goal=True,
            )
        )
        for x_index in range(8):
            x = x_index * 0.75 + 0.1
            navigator.grid.mark_free_sphere((x, 0.1, -1.13), 0.1)
            navigator.grid.mark_free_sphere((x, 0.1, -1.60), 0.1)

        requested_altitude = -1.60
        plan = navigator.plan(
            (0.1, 0.1, -1.13),
            (5.35, 0.1, requested_altitude),
        )

        self.assertTrue(plan.success)
        self.assertTrue(plan.diagnostics["path_altitude_locked_to_goal"])
        self.assertEqual(plan.diagnostics["vertical_layer_range"], (-3, -3))
        self.assertTrue(plan.waypoints_ned_m)
        self.assertTrue(
            all(
                abs(point[2] - requested_altitude) < 1e-9
                for point in plan.waypoints_ned_m
            )
        )

    def test_collinear_compression_preserves_changes_in_slope(self):
        path = [
            (0, 0, 0),
            (1, 1, 0),
            (3, 2, 0),
            (5, 3, 0),
        ]

        compressed = compress_collinear_path(path)

        self.assertEqual(
            compressed,
            [(0, 0, 0), (1, 1, 0), (5, 3, 0)],
        )

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
    def test_depth_model_preprocess_preserves_expected_camera_aspect(self):
        model = MetricDepthOnnx.__new__(MetricDepthOnnx)
        model.input_width = 686
        model.input_height = 518
        model.input_aspect_tolerance = 0.03
        model.mean = np.asarray(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        model.std = np.asarray(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        blob = model.preprocess(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual(blob.shape, (1, 3, 518, 686))
        self.assertEqual(blob.dtype, np.float32)

    @unittest.skipIf(MetricDepthOnnx is None, "OpenCV indisponivel neste ambiente")
    def test_depth_model_rejects_unexpected_camera_aspect(self):
        model = MetricDepthOnnx.__new__(MetricDepthOnnx)
        model.input_width = 686
        model.input_height = 518
        model.input_aspect_tolerance = 0.03
        model.mean = np.zeros((1, 1, 3), dtype=np.float32)
        model.std = np.ones((1, 1, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            model.preprocess(np.zeros((180, 320, 3), dtype=np.uint8))

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

class VerticalObstacleEnvelopeTest(unittest.TestCase):
    def test_body_envelope_excludes_canopy_and_keeps_flight_level_obstacle(self):
        from spatial_mapping.navigation import vertical_obstacle_mask
        points = np.array([
            [4.0, 0.0, -1.0],
            [4.0, 0.0, -0.2],
            [4.0, 0.0, 0.3],
            [4.0, 0.0, 0.7],
        ])
        mask = vertical_obstacle_mask(points, reference_ned_z=0.0, band_m=0.4)
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_vertical_envelope_rejects_invalid_reference(self):
        from spatial_mapping.navigation import vertical_obstacle_mask
        with self.assertRaises(ValueError):
            vertical_obstacle_mask(np.zeros((1, 3)), np.nan, 0.4)

    def test_controller_passes_body_altitude_to_both_maps(self):
        path = __import__('pathlib').Path(__file__).resolve().parents[1]
        source = (path / 'drone_controller.py').read_text()
        self.assertGreaterEqual(source.count('obstacle_reference_ned_z=position[2]'), 2)


if __name__ == "__main__":
    unittest.main()
