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

    def scaled(self, scale_x, scale_y=None):
        """Ajusta os parametros para uma imagem redimensionada."""

        if scale_y is None:
            scale_y = scale_x
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("as escalas devem ser positivas")
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )


def backproject_depth(
    depth_m,
    intrinsics,
    *,
    stride=1,
    edge_stride=None,
    edge_relative_threshold=0.10,
    sampling_edge_mask=None,
    min_depth_m=0.1,
    max_depth_m=np.inf,
):
    """Reprojeta um mapa de profundidade em pontos no referencial optico da camera.

    O referencial segue a convencao optica do ROS: X aponta para a direita, Y para
    baixo e Z para a frente. Pixels invalidos ou fora do intervalo configurado sao
    descartados. Quando edge_stride e informado, descontinuidades relativas de
    profundidade recebem amostragem adicional para preservar superficies finas.
    Uma mascara externa (por exemplo, bordas RGB) pode complementar essas bordas.
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m deve possuir duas dimensoes")
    if stride < 1:
        raise ValueError("stride deve ser maior ou igual a 1")
    if edge_stride is not None and edge_stride < 1:
        raise ValueError("edge_stride deve ser maior ou igual a 1")
    if edge_relative_threshold <= 0:
        raise ValueError("edge_relative_threshold deve ser positivo")
    if sampling_edge_mask is not None:
        sampling_edge_mask = np.asarray(sampling_edge_mask, dtype=bool)
        if sampling_edge_mask.shape != depth.shape:
            raise ValueError("sampling_edge_mask deve possuir o formato de depth_m")

    rows, cols = np.indices(depth.shape)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    sampled = np.zeros(depth.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    if edge_stride is not None:
        edge_mask = np.zeros(depth.shape, dtype=bool)
        horizontal_valid = valid[:, 1:] & valid[:, :-1]
        horizontal_scale = np.minimum(depth[:, 1:], depth[:, :-1])
        horizontal_edge = horizontal_valid & (
            np.abs(depth[:, 1:] - depth[:, :-1])
            / np.maximum(horizontal_scale, min_depth_m)
            >= edge_relative_threshold
        )
        edge_mask[:, 1:] |= horizontal_edge
        edge_mask[:, :-1] |= horizontal_edge

        vertical_valid = valid[1:, :] & valid[:-1, :]
        vertical_scale = np.minimum(depth[1:, :], depth[:-1, :])
        vertical_edge = vertical_valid & (
            np.abs(depth[1:, :] - depth[:-1, :])
            / np.maximum(vertical_scale, min_depth_m)
            >= edge_relative_threshold
        )
        edge_mask[1:, :] |= vertical_edge
        edge_mask[:-1, :] |= vertical_edge

        if sampling_edge_mask is not None:
            edge_mask |= sampling_edge_mask

        edge_lattice = np.zeros(depth.shape, dtype=bool)
        edge_lattice[::edge_stride, ::edge_stride] = True
        sampled |= edge_mask & edge_lattice

    selected = valid & sampled
    z = depth[selected]
    x = (cols[selected] - intrinsics.cx) * z / intrinsics.fx
    y = (rows[selected] - intrinsics.cy) * z / intrinsics.fy
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


def quaternion_to_rotation_matrix(quaternion_wxyz):
    """Converte um quaternion Hamilton (w, x, y, z) em matriz de rotacao."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion deve possuir quatro valores finitos")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion nao pode possuir norma zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_rotation_matrix(roll_rad, pitch_rad, yaw_rad):
    """Retorna Rz(yaw) Ry(pitch) Rx(roll) para uma montagem rigida."""

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rotation_z @ rotation_y @ rotation_x


def camera_to_ned_transform(
    position_ned_m,
    attitude_body_to_ned_wxyz,
    camera_translation_body_m=(0.0, 0.0, 0.0),
    camera_rotation_body_from_optical=None,
):
    """Monta a transformacao da camera optica para o referencial local NED.

    A rotacao padrao considera uma camera frontal alinhada ao corpo FRD: o eixo Z
    optico aponta para a frente do drone, X optico para a direita e Y optico para
    baixo. Uma montagem diferente deve fornecer sua matriz extrinseca.
    """

    position = np.asarray(position_ned_m, dtype=np.float64)
    translation_body = np.asarray(camera_translation_body_m, dtype=np.float64)
    if position.shape != (3,) or translation_body.shape != (3,):
        raise ValueError("posicao e translacao devem possuir tres coordenadas")

    if camera_rotation_body_from_optical is None:
        rotation_body_from_optical = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
    else:
        rotation_body_from_optical = np.asarray(
            camera_rotation_body_from_optical,
            dtype=np.float64,
        )
    if rotation_body_from_optical.shape != (3, 3):
        raise ValueError("a rotacao extrinseca deve possuir formato (3, 3)")

    rotation_ned_from_body = quaternion_to_rotation_matrix(attitude_body_to_ned_wxyz)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_ned_from_body @ rotation_body_from_optical
    transform[:3, 3] = position + rotation_ned_from_body @ translation_body
    return transform
