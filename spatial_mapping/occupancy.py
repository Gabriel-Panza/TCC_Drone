"""Grade probabilistica simples para representar ocupacao em tres dimensoes."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from .traversal import segment_voxels


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
    occupied_observations_required: int = 1
    occupied_support_radius_voxels: int = 0
    pending_clear_free_observations_required: int = 3
    occupied_evidence_window_frames: int = 6
    free_viewpoint_sectors_required: int = 1
    free_viewpoint_sector_deg: float = 45.0

    def __post_init__(self):
        if self.resolution_m <= 0:
            raise ValueError("resolution_m deve ser positiva")
        if self.occupied_observations_required < 1:
            raise ValueError("occupied_observations_required deve ser positivo")
        if self.occupied_support_radius_voxels < 0:
            raise ValueError("occupied_support_radius_voxels nao pode ser negativo")
        if self.pending_clear_free_observations_required < 1:
            raise ValueError(
                "pending_clear_free_observations_required deve ser positivo"
            )
        if self.occupied_evidence_window_frames < 1:
            raise ValueError("occupied_evidence_window_frames deve ser positivo")
        if self.free_viewpoint_sectors_required < 1:
            raise ValueError("free_viewpoint_sectors_required deve ser positivo")
        if not 0 < self.free_viewpoint_sector_deg <= 360:
            raise ValueError("free_viewpoint_sector_deg deve estar no intervalo (0, 360]")


class OccupancyGrid3D:
    """Acumula raios de profundidade em uma grade esparsa de voxels."""

    def __init__(self, config=None):
        self.config = config or OccupancyGridConfig()
        self._log_odds: dict[Voxel, float] = {}
        self._integration_frame = 0
        self._occupied_evidence_frames: dict[Voxel, set[int]] = {}
        self._pending_free_observations: dict[Voxel, int] = {}
        self._free_viewpoint_sectors: dict[Voxel, set[int]] = {}
        self._confirmed_occupied: set[Voxel] = set()

    def world_to_voxel(self, point):
        point_array = np.asarray(point, dtype=np.float64)
        return tuple(np.floor(point_array / self.config.resolution_m).astype(int))

    def voxel_to_world(self, voxel):
        voxel_array = np.asarray(voxel, dtype=np.float64)
        return (voxel_array + 0.5) * self.config.resolution_m

    def integrate_points(self, sensor_origin, obstacle_points, max_range_m=np.inf):
        """Marca como livres os raios observados e como ocupados seus extremos."""

        points = np.asarray(obstacle_points, dtype=np.float64)
        self.integrate_rays(
            sensor_origin,
            points,
            endpoint_is_occupied=np.ones(len(points), dtype=bool),
            max_range_m=max_range_m,
        )

    def integrate_rays(
        self,
        sensor_origin,
        endpoints,
        endpoint_is_occupied,
        max_range_m=np.inf,
        free_space_margin_m=0.0,
        free_space_margin_ratio=0.0,
        free_space_margin_max_m=None,
        occupied_uncertainty_m=0.0,
    ):
        """Integra espaco livre observado e ocupa apenas extremos selecionados.

        Um retorno de profundidade pode terminar fora da faixa vertical usada para
        representar obstaculos de voo. Nesse caso, o trecho anterior ao retorno
        continua sendo uma observacao valida de espaco livre, mas o extremo nao
        deve ser inserido como obstaculo nessa representacao.
        """

        origin = np.asarray(sensor_origin, dtype=np.float64)
        points = np.asarray(endpoints, dtype=np.float64)
        occupied_mask = np.asarray(endpoint_is_occupied, dtype=bool)
        if origin.shape != (3,):
            raise ValueError("sensor_origin deve possuir tres coordenadas")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("endpoints deve possuir formato (N, 3)")
        if occupied_mask.shape != (len(points),):
            raise ValueError("endpoint_is_occupied deve possuir formato (N,)")
        if free_space_margin_m < 0:
            raise ValueError("free_space_margin_m nao pode ser negativo")
        if free_space_margin_ratio < 0:
            raise ValueError("free_space_margin_ratio nao pode ser negativo")
        if (
            free_space_margin_max_m is not None
            and free_space_margin_max_m < free_space_margin_m
        ):
            raise ValueError(
                "free_space_margin_max_m deve ser maior ou igual a margem base"
            )
        if occupied_uncertainty_m < 0:
            raise ValueError("occupied_uncertainty_m nao pode ser negativo")

        self._integration_frame += 1
        self._expire_old_occupied_evidence()
        free_observed_this_frame = set()
        occupied_observed_this_frame = set()
        for point, mark_endpoint_occupied in zip(points, occupied_mask):
            distance = float(np.linalg.norm(point - origin))
            if not np.isfinite(distance) or distance == 0 or distance > max_range_m:
                continue
            ray = self._ray_voxels(origin, point)
            effective_free_margin_m = (
                free_space_margin_m + free_space_margin_ratio * distance
            )
            if free_space_margin_max_m is not None:
                effective_free_margin_m = min(
                    effective_free_margin_m,
                    free_space_margin_max_m,
                )
            if effective_free_margin_m > 0:
                free_distance = max(0.0, distance - effective_free_margin_m)
                free_endpoint = origin + (point - origin) * (
                    free_distance / distance
                )
                free_voxels = self._ray_voxels(origin, free_endpoint)
            else:
                free_voxels = ray[:-1]
            free_observed_this_frame.update(free_voxels)
            if mark_endpoint_occupied:
                occupied_start_distance = max(
                    0.0,
                    distance - occupied_uncertainty_m,
                )
                occupied_start = origin + (point - origin) * (
                    occupied_start_distance / distance
                )
                occupied_observed_this_frame.update(
                    self._ray_voxels(occupied_start, point)
                )

        for voxel in free_observed_this_frame - occupied_observed_this_frame:
            if voxel in self._confirmed_occupied:
                continue
            sector = self._free_viewpoint_sector(voxel, origin)
            sectors = self._free_viewpoint_sectors.setdefault(voxel, set())
            sectors.add(sector)
            if len(sectors) < self.config.free_viewpoint_sectors_required:
                continue
            if voxel in self._occupied_evidence_frames:
                free_count = self._pending_free_observations.get(voxel, 0) + 1
                self._pending_free_observations[voxel] = free_count
                if (
                    free_count
                    < self.config.pending_clear_free_observations_required
                ):
                    continue
                self._occupied_evidence_frames.pop(voxel, None)
                self._pending_free_observations.pop(voxel, None)
            self._update(voxel, -self.config.free_decrement)
        for voxel in occupied_observed_this_frame:
            self._free_viewpoint_sectors.pop(voxel, None)
            if voxel not in self._confirmed_occupied:
                self._occupied_evidence_frames.setdefault(voxel, set()).add(
                    self._integration_frame
                )
                self._pending_free_observations.pop(voxel, None)
            self._update(voxel, self.config.occupied_increment)
        for voxel in occupied_observed_this_frame:
            self._confirm_if_supported(voxel)

    def occupied_voxels(self):
        return set(self._confirmed_occupied)

    def free_voxels(self):
        unavailable = self._confirmed_occupied | set(self._occupied_evidence_frames)
        return {
            voxel
            for voxel, value in self._log_odds.items()
            if value <= self.config.free_threshold and voxel not in unavailable
        }

    def observed_voxels(self):
        return set(self._log_odds)

    def pending_occupied_voxels(self):
        """Retorna evidencias positivas ainda insuficientes para ocupacao."""

        return set(self._occupied_evidence_frames) - self._confirmed_occupied

    def mark_ego_voxel_free(self, position):
        """Usa a presenca fisica do sensor para liberar somente seu voxel."""

        voxel = self.world_to_voxel(position)
        was_confirmed = voxel in self._confirmed_occupied
        self._confirmed_occupied.discard(voxel)
        self._occupied_evidence_frames.pop(voxel, None)
        self._pending_free_observations.pop(voxel, None)
        self._free_viewpoint_sectors.pop(voxel, None)
        evidence = min(
            self.config.free_threshold - 1e-6,
            -self.config.free_decrement,
        )
        self._log_odds[voxel] = evidence
        return was_confirmed

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

    def _free_viewpoint_sector(self, voxel, sensor_origin):
        """Discretiza a direcao horizontal de observacao de um voxel."""

        center = self.voxel_to_world(voxel)
        direction = np.asarray(sensor_origin, dtype=float) - center
        angle_deg = (np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0
        return int(angle_deg // self.config.free_viewpoint_sector_deg)

    def occupied_evidence_count(self, voxel):
        """Conta frames distintos de evidencia no suporte espacial do voxel."""

        voxel = tuple(voxel)
        radius = self.config.occupied_support_radius_voxels
        frames = set()
        for dx, dy, dz in product(range(-radius, radius + 1), repeat=3):
            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            frames.update(self._occupied_evidence_frames.get(neighbor, ()))
        return len(frames)

    def _confirm_if_supported(self, voxel):
        if voxel in self._confirmed_occupied:
            return
        if (
            self.occupied_evidence_count(voxel)
            < self.config.occupied_observations_required
        ):
            return
        self._confirmed_occupied.add(voxel)
        self._log_odds[voxel] = max(
            self._log_odds.get(voxel, 0.0),
            self.config.occupied_threshold,
        )

    def _expire_old_occupied_evidence(self):
        oldest_frame = (
            self._integration_frame
            - self.config.occupied_evidence_window_frames
            + 1
        )
        for voxel, frames in list(self._occupied_evidence_frames.items()):
            frames.intersection_update(
                frame for frame in frames if frame >= oldest_frame
            )
            if not frames and voxel not in self._confirmed_occupied:
                self._occupied_evidence_frames.pop(voxel, None)
                self._pending_free_observations.pop(voxel, None)

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

    def inflated_occupied_voxels(
        self,
        radius_m,
        vertical_radius_m=None,
    ):
        """Expande obstaculos com raios horizontal e vertical independentes."""

        if radius_m < 0:
            raise ValueError("radius_m nao pode ser negativo")
        if vertical_radius_m is None:
            vertical_radius_m = radius_m
        if vertical_radius_m <= 0:
            raise ValueError("vertical_radius_m deve ser positivo")
        if radius_m == 0:
            return set(self.occupied_voxels())

        resolution = self.config.resolution_m
        horizontal_voxels = int(np.ceil(radius_m / resolution))
        vertical_voxels = int(np.ceil(vertical_radius_m / resolution))
        offsets = product(
            range(-horizontal_voxels, horizontal_voxels + 1),
            range(-horizontal_voxels, horizontal_voxels + 1),
            range(-vertical_voxels, vertical_voxels + 1),
        )
        valid_offsets = [
            (dx, dy, dz)
            for dx, dy, dz in offsets
            if (
                (resolution * dx / radius_m) ** 2
                + (resolution * dy / radius_m) ** 2
                + (resolution * dz / vertical_radius_m) ** 2
                <= 1.0 + 1e-9
            )
        ]
        return {
            (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            for voxel in self.occupied_voxels()
            for dx, dy, dz in valid_offsets
        }

    def state(self, voxel):
        voxel = tuple(voxel)
        if voxel in self._confirmed_occupied:
            return "occupied"
        if voxel in self._occupied_evidence_frames:
            return "unknown"
        value = self._log_odds.get(voxel, 0.0)
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
        """Amostragem legada para integrar profundidade; nao usar para seguranca."""
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance / (self.config.resolution_m * 0.5))))
        samples = np.linspace(start, end, steps + 1)
        voxels = []
        for sample in samples:
            voxel = self.world_to_voxel(sample)
            if not voxels or voxel != voxels[-1]:
                voxels.append(voxel)
        return voxels

    def segment_voxels(self, start, end):
        """Travessia conservadora completa para validacao de caminhos."""
        return segment_voxels(start, end, self.config.resolution_m)
