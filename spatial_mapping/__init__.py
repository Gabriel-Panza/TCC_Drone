"""Ferramentas de representacao espacial usadas na validacao complementar."""

from .astar import AStar3D, PathNotFoundError, compress_collinear_path
from .geometry import (
    CameraIntrinsics,
    backproject_depth,
    camera_to_ned_transform,
    euler_xyz_rotation_matrix,
    quaternion_to_rotation_matrix,
    transform_points,
)
from .navigation import SpatialNavigationConfig, SpatialNavigator, SpatialPlan
from .occupancy import OccupancyGrid3D, OccupancyGridConfig

__all__ = [
    "AStar3D",
    "CameraIntrinsics",
    "OccupancyGrid3D",
    "OccupancyGridConfig",
    "PathNotFoundError",
    "SpatialNavigationConfig",
    "SpatialNavigator",
    "SpatialPlan",
    "backproject_depth",
    "camera_to_ned_transform",
    "compress_collinear_path",
    "euler_xyz_rotation_matrix",
    "quaternion_to_rotation_matrix",
    "transform_points",
]
