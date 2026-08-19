"""Grade probabilistica simples para representar ocupacao em tres dimensoes."""

from dataclasses import dataclass
from itertools import product

import numpy as np


Voxel = tuple[int, int, int]


@dataclass(frozen=True)
class OccupancyGridConfig:
    """Configura resolucao e atualizacao probabilistica da grade."""

    resolution_m: float = 0.5
    occupied_increment: float = 0.85
    free_decrement: float = 0.4
    occupied_threshold: float = 0.6
    free_threshold: float = -0.6
    min_log_odds: float = -2.0
    max_log_odds: float = 3.5

    def __post_init__(self):
        if self.resolution_m <= 0:
            raise ValueError("resolution_m deve ser positiva")


class OccupancyGrid3D:
    """Acumula raios de profundidade em uma grade esparsa de voxels."""

    def __init__(self, config=None):
        self.config = config or OccupancyGridConfig()
        self._log_odds: dict[Voxel, float] = {}

    def world_to_voxel(self, point):
        point_array = np.asarray(point, dtype=np.float64)
        return tuple(np.floor(point_array / self.config.resolution_m).astype(int))

    def voxel_to_world(self, voxel):
        voxel_array = np.asarray(voxel, dtype=np.float64)
        return (voxel_array + 0.5) * self.config.resolution_m

    def integrate_points(self, sensor_origin, obstacle_points, max_range_m=np.inf):
        """Marca como livres os raios observados e como ocupados seus extremos."""

        origin = np.asarray(sensor_origin, dtype=np.float64)
        points = np.asarray(obstacle_points, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("sensor_origin deve possuir tres coordenadas")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("obstacle_points deve possuir formato (N, 3)")

        for point in points:
            distance = float(np.linalg.norm(point - origin))
            if not np.isfinite(distance) or distance == 0 or distance > max_range_m:
                continue
            ray = self._ray_voxels(origin, point)
            for voxel in ray[:-1]:
                self._update(voxel, -self.config.free_decrement)
            self._update(ray[-1], self.config.occupied_increment)

    def occupied_voxels(self):
        return {
            voxel
            for voxel, value in self._log_odds.items()
            if value >= self.config.occupied_threshold
        }

    def free_voxels(self):
        return {
            voxel
            for voxel, value in self._log_odds.items()
            if value <= self.config.free_threshold
        }

    def observed_voxels(self):
        return set(self._log_odds)

    def mark_free_sphere(self, center, radius_m):
        """Marca a vizinhanca conhecida do drone como livre."""

        if radius_m < 0:
            raise ValueError("radius_m nao pode ser negativo")
        center_voxel = self.world_to_voxel(center)
        radius_voxels = int(np.ceil(radius_m / self.config.resolution_m))
        evidence = min(
            self.config.free_threshold - 1e-6,
            -self.config.free_decrement,
        )
        for dx, dy, dz in product(
            range(-radius_voxels, radius_voxels + 1),
            repeat=3,
        ):
            if (
                self.config.resolution_m * np.sqrt(dx * dx + dy * dy + dz * dz)
                > radius_m + 1e-9
            ):
                continue
            voxel = (
                center_voxel[0] + dx,
                center_voxel[1] + dy,
                center_voxel[2] + dz,
            )
            if (
                self._log_odds.get(voxel, 0.0)
                >= self.config.occupied_threshold
            ):
                continue
            self._log_odds[voxel] = min(self._log_odds.get(voxel, 0.0), evidence)

    def export_arrays(self):
        """Retorna coordenadas e log-odds em arrays adequados para NPZ."""

        if not self._log_odds:
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        items = sorted(self._log_odds.items())
        voxels = np.asarray([item[0] for item in items], dtype=np.int32)
        log_odds = np.asarray([item[1] for item in items], dtype=np.float32)
        return voxels, log_odds

    def inflated_occupied_voxels(self, radius_m):
        """Expande obstaculos para considerar dimensoes e margem do drone."""

        if radius_m < 0:
            raise ValueError("radius_m nao pode ser negativo")
        radius_voxels = int(np.ceil(radius_m / self.config.resolution_m))
        offsets = list(product(range(-radius_voxels, radius_voxels + 1), repeat=3))
        return {
            (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            for voxel in self.occupied_voxels()
            for dx, dy, dz in offsets
            if self.config.resolution_m * np.sqrt(dx * dx + dy * dy + dz * dz)
            <= radius_m + 1e-9
        }

    def state(self, voxel):
        value = self._log_odds.get(tuple(voxel), 0.0)
        if value >= self.config.occupied_threshold:
            return "occupied"
        if value <= self.config.free_threshold:
            return "free"
        return "unknown"

    def _update(self, voxel, delta):
        current = self._log_odds.get(voxel, 0.0)
        self._log_odds[voxel] = float(
            np.clip(
                current + delta,
                self.config.min_log_odds,
                self.config.max_log_odds,
            )
        )

    def _ray_voxels(self, start, end):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance / (self.config.resolution_m * 0.5))))
        samples = np.linspace(start, end, steps + 1)
        voxels = []
        for sample in samples:
            voxel = self.world_to_voxel(sample)
            if not voxels or voxel != voxels[-1]:
                voxels.append(voxel)
        return voxels
