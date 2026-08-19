"""Orquestracao independente de ROS para mapeamento e planejamento espacial."""

from dataclasses import dataclass, field
from math import sqrt
from time import perf_counter

import numpy as np

from .astar import AStar3D, PathNotFoundError, compress_collinear_path
from .geometry import CameraIntrinsics, backproject_depth, transform_points
from .occupancy import OccupancyGrid3D, OccupancyGridConfig


@dataclass(frozen=True)
class SpatialNavigationConfig:
    """Parametros que devem permanecer iguais entre mapa estimado e mapa ideal."""

    voxel_resolution_m: float = 0.75
    depth_stride: int = 40
    min_depth_m: float = 0.5
    max_depth_m: float = 30.0
    obstacle_vertical_band_m: float = 1.5
    drone_clearance_radius_m: float = 1.25
    known_free_radius_m: float = 0.8
    local_plan_radius_m: float = 20.0
    min_subgoal_progress_m: float = 0.5
    vertical_tolerance_m: float = 1.5
    max_waypoint_spacing_m: float = 5.0
    frontier_standoff_m: float = 2.5
    connectivity: int = 26

    def __post_init__(self):
        if self.depth_stride < 1:
            raise ValueError("depth_stride deve ser maior ou igual a 1")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m deve ser maior que min_depth_m")
        if self.obstacle_vertical_band_m <= 0:
            raise ValueError("obstacle_vertical_band_m deve ser positivo")
        if self.local_plan_radius_m <= 0:
            raise ValueError("local_plan_radius_m deve ser positivo")
        if self.vertical_tolerance_m <= 0:
            raise ValueError("vertical_tolerance_m deve ser positivo")
        if self.max_waypoint_spacing_m <= 0:
            raise ValueError("max_waypoint_spacing_m deve ser positivo")
        if self.frontier_standoff_m < 0:
            raise ValueError("frontier_standoff_m nao pode ser negativo")


@dataclass
class SpatialPlan:
    """Resultado de uma tentativa de planejamento local."""

    success: bool
    reason: str
    requested_goal_ned_m: tuple[float, float, float]
    selected_goal_ned_m: tuple[float, float, float] | None = None
    path_voxels: list[tuple[int, int, int]] = field(default_factory=list)
    waypoints_ned_m: list[tuple[float, float, float]] = field(default_factory=list)
    path_length_m: float = 0.0
    raw_path_length_m: float = 0.0
    frontier_standoff_applied_m: float = 0.0
    planning_time_ms: float = 0.0


class SpatialNavigator:
    """Constroi uma grade de ocupacao e planeja subcaminhos no espaco observado."""

    def __init__(self, config=None):
        self.config = config or SpatialNavigationConfig()
        occupancy_config = OccupancyGridConfig(
            resolution_m=self.config.voxel_resolution_m,
            free_threshold=-0.35,
        )
        self.grid = OccupancyGrid3D(occupancy_config)
        self.frames_integrated = 0

    def integrate_depth(self, depth_m, intrinsics, camera_to_ned):
        """Reprojeta e integra um frame, retornando estatisticas da atualizacao."""

        if not isinstance(intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics deve ser CameraIntrinsics")
        points_camera = backproject_depth(
            depth_m,
            intrinsics,
            stride=self.config.depth_stride,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        camera_to_ned = np.asarray(camera_to_ned, dtype=np.float64)
        points_ned = transform_points(points_camera, camera_to_ned)
        camera_origin_ned = camera_to_ned[:3, 3]
        points_before_vertical_filter = len(points_ned)
        points_ned = points_ned[
            np.abs(points_ned[:, 2] - camera_origin_ned[2])
            <= self.config.obstacle_vertical_band_m
        ]
        self.grid.mark_free_sphere(
            camera_origin_ned,
            self.config.known_free_radius_m,
        )
        self.grid.integrate_points(
            camera_origin_ned,
            points_ned,
            max_range_m=self.config.max_depth_m,
        )
        self.frames_integrated += 1
        return {
            "frame_index": self.frames_integrated,
            "points_integrated": int(len(points_ned)),
            "points_rejected_vertical": int(
                points_before_vertical_filter - len(points_ned)
            ),
            "free_voxels": int(len(self.grid.free_voxels())),
            "occupied_voxels": int(len(self.grid.occupied_voxels())),
        }

    def plan(self, current_position_ned_m, requested_goal_ned_m):
        """Planeja ate o destino ou ate o melhor subobjetivo observado na direcao dele."""

        started = perf_counter()
        current = np.asarray(current_position_ned_m, dtype=np.float64)
        requested_goal = np.asarray(requested_goal_ned_m, dtype=np.float64)
        if current.shape != (3,) or requested_goal.shape != (3,):
            raise ValueError("posicao e destino devem possuir tres coordenadas")

        blocked = self.grid.inflated_occupied_voxels(
            self.config.drone_clearance_radius_m
        )
        current_layer = self.grid.world_to_voxel(current)[2]
        goal_layer = self.grid.world_to_voxel(requested_goal)[2]
        layer_margin = int(
            np.floor(
                self.config.vertical_tolerance_m
                / self.config.voxel_resolution_m
            )
        )
        min_layer = min(current_layer, goal_layer) - layer_margin
        max_layer = max(current_layer, goal_layer) + layer_margin
        traversable = {
            voxel
            for voxel in self.grid.free_voxels() - blocked
            if min_layer <= voxel[2] <= max_layer
        }
        start = self._nearest_voxel(self.grid.world_to_voxel(current), traversable)
        if start is None:
            return self._failure(
                requested_goal,
                "sem espaco livre ao redor do drone",
                self._elapsed_ms(started),
            )

        planner = AStar3D(
            traversable,
            blocked,
            connectivity=self.config.connectivity,
        )
        reachable = planner.reachable_from(start)

        requested_goal_voxel = self.grid.world_to_voxel(requested_goal)
        if requested_goal_voxel in reachable:
            selected_goal = requested_goal_voxel
        else:
            selected_goal = self._select_local_subgoal(
                current,
                requested_goal,
                reachable,
            )
        if selected_goal is None or selected_goal == start:
            max_progress = self._max_progress(
                current,
                requested_goal,
                reachable,
            )
            return self._failure(
                requested_goal,
                (
                    "nenhum subobjetivo observado com progresso "
                    f"(livres={len(traversable)}, alcancaveis={len(reachable)}, "
                    f"max_progresso={max_progress:.2f}m)"
                ),
                self._elapsed_ms(started),
            )

        try:
            path = planner.plan(start, selected_goal)
        except PathNotFoundError as error:
            return self._failure(
                requested_goal,
                str(error),
                self._elapsed_ms(started),
            )

        shortcut = self._shortcut_path(path, traversable, blocked)
        compressed = compress_collinear_path(shortcut)
        waypoints = self._densify_waypoints(
            [self.grid.voxel_to_world(voxel) for voxel in compressed]
        )
        reason = (
            "goal_observed"
            if selected_goal == requested_goal_voxel
            else "local_subgoal"
        )
        frontier_standoff_applied_m = 0.0
        if reason == "local_subgoal":
            waypoints, frontier_standoff_applied_m = self._reserve_frontier(
                current,
                waypoints,
            )
        return SpatialPlan(
            success=True,
            reason=reason,
            requested_goal_ned_m=tuple(requested_goal),
            selected_goal_ned_m=tuple(self.grid.voxel_to_world(selected_goal)),
            path_voxels=path,
            waypoints_ned_m=waypoints,
            path_length_m=self._waypoint_length([current, *waypoints]),
            raw_path_length_m=self._path_length(path),
            frontier_standoff_applied_m=frontier_standoff_applied_m,
            planning_time_ms=self._elapsed_ms(started),
        )

    def export_map(self):
        voxels, log_odds = self.grid.export_arrays()
        return {
            "voxels": voxels,
            "log_odds": log_odds,
            "resolution_m": np.asarray(self.config.voxel_resolution_m),
            "occupied_threshold": np.asarray(self.grid.config.occupied_threshold),
            "free_threshold": np.asarray(self.grid.config.free_threshold),
            "frames_integrated": np.asarray(self.frames_integrated),
        }

    def path_is_safe(self, current_position_ned_m, waypoints_ned_m):
        """Verifica se todo o caminho restante continua observado e desocupado."""

        points = [np.asarray(current_position_ned_m, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints_ned_m
        ]
        if len(points) < 2:
            return False

        blocked = self.grid.inflated_occupied_voxels(
            self.config.drone_clearance_radius_m
        )
        free = self.grid.free_voxels()
        for start, end in zip(points, points[1:]):
            for voxel in self.grid._ray_voxels(start, end):
                if voxel in blocked or voxel not in free:
                    return False
        return True

    def position_is_safe(self, position_ned_m):
        """Confirma que uma posicao pertence ao espaco livre fora da inflacao."""

        voxel = self.grid.world_to_voxel(position_ned_m)
        return (
            voxel in self.grid.free_voxels()
            and voxel
            not in self.grid.inflated_occupied_voxels(
                self.config.drone_clearance_radius_m
            )
        )

    def _select_local_subgoal(self, current, requested_goal, traversable):
        direction = requested_goal - current
        target_distance = float(np.linalg.norm(direction))
        if target_distance <= 1e-9:
            return None
        direction /= target_distance

        best_voxel = None
        best_score = -np.inf
        for voxel in traversable:
            point = self.grid.voxel_to_world(voxel)
            offset = point - current
            distance = float(np.linalg.norm(offset))
            if distance > self.config.local_plan_radius_m:
                continue
            progress = float(np.dot(offset, direction))
            if progress < self.config.min_subgoal_progress_m:
                continue
            lateral = float(np.linalg.norm(offset - progress * direction))
            vertical_error = abs(float(point[2] - requested_goal[2]))
            score = progress - 0.35 * lateral - 0.2 * vertical_error
            if score > best_score:
                best_score = score
                best_voxel = voxel
        return best_voxel

    def _max_progress(self, current, requested_goal, traversable):
        direction = requested_goal - current
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9 or not traversable:
            return 0.0
        direction /= norm
        return max(
            float(np.dot(self.grid.voxel_to_world(voxel) - current, direction))
            for voxel in traversable
        )

    @staticmethod
    def _nearest_voxel(requested, candidates, max_offset=2):
        if requested in candidates:
            return requested
        best = None
        best_distance = float("inf")
        for voxel in candidates:
            distance = sqrt(sum((a - b) ** 2 for a, b in zip(voxel, requested)))
            if distance <= max_offset and distance < best_distance:
                best = voxel
                best_distance = distance
        return best

    def _path_length(self, path):
        return self.config.voxel_resolution_m * sum(
            sqrt(sum((a - b) ** 2 for a, b in zip(current, previous)))
            for previous, current in zip(path, path[1:])
        )

    @staticmethod
    def _waypoint_length(waypoints):
        return sum(
            float(np.linalg.norm(np.asarray(current) - np.asarray(previous)))
            for previous, current in zip(waypoints, waypoints[1:])
        )

    def _shortcut_path(self, path, traversable, blocked):
        """Remove curvas da grade quando a linha direta permanece conhecida e livre."""

        path = [tuple(voxel) for voxel in path]
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            next_index = anchor + 1
            for candidate in range(len(path) - 1, anchor, -1):
                start = self.grid.voxel_to_world(path[anchor])
                end = self.grid.voxel_to_world(path[candidate])
                segment = self.grid._ray_voxels(start, end)
                if all(
                    voxel in traversable and voxel not in blocked
                    for voxel in segment
                ):
                    next_index = candidate
                    break
            simplified.append(path[next_index])
            anchor = next_index
        return simplified

    def _densify_waypoints(self, points):
        """Limita saltos entre setpoints sem alterar a geometria do caminho."""

        if not points:
            return []
        dense = [np.asarray(points[0], dtype=float)]
        for endpoint in points[1:]:
            start = dense[-1]
            endpoint = np.asarray(endpoint, dtype=float)
            distance = float(np.linalg.norm(endpoint - start))
            steps = max(1, int(np.ceil(distance / self.config.max_waypoint_spacing_m)))
            dense.extend(
                start + (endpoint - start) * (step / steps)
                for step in range(1, steps + 1)
            )
        return [tuple(point) for point in dense]

    def _reserve_frontier(self, current, waypoints):
        """Encurta um caminho local para o drone observar a fronteira a distancia."""

        points = [np.asarray(current, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints
        ]
        if len(points) < 2 or self.config.frontier_standoff_m <= 0:
            return list(waypoints), 0.0

        lengths = [
            float(np.linalg.norm(end - start))
            for start, end in zip(points, points[1:])
        ]
        total_length = sum(lengths)
        travel_length = max(
            self.config.min_subgoal_progress_m,
            total_length - self.config.frontier_standoff_m,
        )
        travel_length = min(total_length, travel_length)
        applied = max(0.0, total_length - travel_length)
        if applied <= 1e-9:
            return list(waypoints), 0.0

        reserved = []
        traveled = 0.0
        for start, end, length in zip(points, points[1:], lengths):
            if length <= 1e-9:
                continue
            if traveled + length < travel_length - 1e-9:
                reserved.append(tuple(end))
                traveled += length
                continue
            fraction = (travel_length - traveled) / length
            reserved.append(tuple(start + fraction * (end - start)))
            break
        return reserved, applied

    @staticmethod
    def _failure(requested_goal, reason, planning_time_ms=0.0):
        return SpatialPlan(
            success=False,
            reason=reason,
            requested_goal_ned_m=tuple(requested_goal),
            planning_time_ms=float(planning_time_ms),
        )

    @staticmethod
    def _elapsed_ms(started):
        return (perf_counter() - started) * 1000.0
