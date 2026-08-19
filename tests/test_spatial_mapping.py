"""Testes do nucleo geometrico e do planejador, sem dependencia do ROS 2."""

import unittest

import numpy as np

from spatial_mapping import (
    AStar3D,
    CameraIntrinsics,
    OccupancyGrid3D,
    OccupancyGridConfig,
    PathNotFoundError,
    backproject_depth,
    transform_points,
)


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


class OccupancyAndPlanningTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
