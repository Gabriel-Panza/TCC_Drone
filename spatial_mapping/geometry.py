"""Conversoes geometricas entre imagem, camera e mapa tridimensional."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Parametros intrinsecos de uma camera pinhole."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx e fy devem ser positivos")


def backproject_depth(
    depth_m,
    intrinsics,
    *,
    stride=1,
    min_depth_m=0.1,
    max_depth_m=np.inf,
):
    """Reprojeta um mapa de profundidade em pontos no referencial optico da camera.

    O referencial segue a convencao optica do ROS: X aponta para a direita, Y para
    baixo e Z para a frente. Pixels invalidos ou fora do intervalo configurado sao
    descartados.
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m deve possuir duas dimensoes")
    if stride < 1:
        raise ValueError("stride deve ser maior ou igual a 1")

    rows, cols = np.indices(depth.shape)
    rows = rows[::stride, ::stride]
    cols = cols[::stride, ::stride]
    sampled_depth = depth[::stride, ::stride]

    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_m)
        & (sampled_depth <= max_depth_m)
    )
    z = sampled_depth[valid]
    x = (cols[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (rows[valid] - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def transform_points(points, transform):
    """Aplica uma transformacao homogenea 4x4 a pontos tridimensionais."""

    points_array = np.asarray(points, dtype=np.float64)
    transform_array = np.asarray(transform, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points deve possuir formato (N, 3)")
    if transform_array.shape != (4, 4):
        raise ValueError("transform deve possuir formato (4, 4)")

    homogeneous = np.column_stack((points_array, np.ones(len(points_array))))
    transformed = homogeneous @ transform_array.T
    return transformed[:, :3]
