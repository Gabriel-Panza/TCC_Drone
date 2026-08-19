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
    """Projeta os voxels 3D no plano horizontal ao redor do drone."""

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

    visible_free = 0
    for voxel in grid.free_voxels():
        world = grid.voxel_to_world(voxel)
        if abs(float(world[2] - center[2])) > vertical_band_m:
            continue
        x, y = pixel(world)
        if 0 <= x < image_size and 0 <= y < image_size:
            image[y, x] = (225, 238, 225)
            visible_free += 1

    visible_occupied = 0
    for voxel in grid.occupied_voxels():
        world = grid.voxel_to_world(voxel)
        if abs(float(world[2] - center[2])) > vertical_band_m:
            continue
        x, y = pixel(world)
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(image, (x, y), 4, (40, 40, 210), -1)
            visible_occupied += 1

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
        "Mapa 3D projetado no plano NED",
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
            "vermelho: ocupado | verde: caminho | azul: drone | "
            f"visiveis: {visible_occupied} ocupados, {visible_free} livres"
        ),
        (14, image_size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    return image
