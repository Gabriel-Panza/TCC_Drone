"""Visualizacoes leves para acompanhar o mapa durante a simulacao."""

import cv2
import numpy as np


def render_top_down(
    grid,
    *,
    center_ned_m,
    target_ned_m=None,
    path_ned_m=(),
    extent_m=35.0,
    image_size=720,
    vertical_band_m=1.5,
):
    return render_top_down_voxels(
        grid.free_voxels(), grid.occupied_voxels(),
        resolution_m=grid.config.resolution_m, center_ned_m=center_ned_m,
        target_ned_m=target_ned_m, path_ned_m=path_ned_m, extent_m=extent_m,
        image_size=image_size, vertical_band_m=vertical_band_m,
    )


def render_top_down_voxels(
    free_voxels, occupied_voxels, *, resolution_m, center_ned_m,
    target_ned_m=None, path_ned_m=(), extent_m=35.0, image_size=720,
    vertical_band_m=1.5, map_age_s=None,
):
    """Projeta uma cópia do mapa sem acessar a grade durante o desenho."""

    center = np.asarray(center_ned_m, dtype=float)
    image = np.full((image_size, image_size, 3), 248, dtype=np.uint8)
    scale = image_size / (2.0 * extent_m)

    def pixel(point):
        point = np.asarray(point, dtype=float)
        east = point[1] - center[1]
        north = point[0] - center[0]
        return (
            int(round(image_size / 2 + east * scale)),
            int(round(image_size / 2 - north * scale)),
        )

    def visible_pixels(voxels):
        indices = np.asarray(list(voxels), dtype=float).reshape(-1, 3)
        offset = (indices + 0.5) * resolution_m - center
        in_band = np.abs(offset[:, 2]) <= vertical_band_m
        offset = offset[in_band]
        xy = np.rint(image_size / 2 + offset[:, [1, 0]] * [scale, -scale]).astype(int)
        visible = np.all((0 <= xy) & (xy < image_size), axis=1)
        return xy[visible], indices[in_band, 2][visible]

    free_pixels, _ = visible_pixels(free_voxels)
    image[free_pixels[:, 1], free_pixels[:, 0]] = (225, 238, 225)
    occupied_pixels, occupied_layers = visible_pixels(occupied_voxels)
    current_layer = np.floor(center[2] / resolution_m)
    same_layer = occupied_layers == current_layer
    for x, y in occupied_pixels[~same_layer]:
        cv2.circle(image, (int(x), int(y)), 3, (155, 155, 155), 1)
    for x, y in occupied_pixels[same_layer]:
        cv2.circle(image, (int(x), int(y)), 4, (40, 40, 210), -1)

    path_pixels = [pixel(point) for point in path_ned_m]
    for start, end in zip(path_pixels, path_pixels[1:]):
        cv2.line(image, start, end, (40, 160, 40), 2, cv2.LINE_AA)
    for point in path_pixels:
        cv2.circle(image, point, 3, (40, 160, 40), -1)

    current_pixel = pixel(center)
    cv2.circle(image, current_pixel, 6, (220, 100, 20), -1)
    if target_ned_m is not None:
        target_pixel = pixel(target_ned_m)
        if 0 <= target_pixel[0] < image_size and 0 <= target_pixel[1] < image_size:
            cv2.drawMarker(
                image,
                target_pixel,
                (0, 170, 220),
                cv2.MARKER_STAR,
                14,
                2,
            )

    cv2.putText(
        image,
        ("Mapa 3D projetado no plano NED" if map_age_s is None
         else f"Mapa NED | ultima integracao ha {map_age_s:.1f} s"),
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        (
            "vermelho: ocupado na camada atual | cinza: ocupado em outra altura"
        ),
        (14, image_size - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "verde: caminho | azul: drone",
        (14, image_size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    return image
