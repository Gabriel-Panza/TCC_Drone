"""Orquestracao independente de ROS para mapeamento e planejamento espacial."""

from dataclasses import dataclass, field
from itertools import product
from math import sqrt
from time import perf_counter

import numpy as np

from .astar import AStar3D, PathNotFoundError, compress_collinear_path
from .geometry import CameraIntrinsics, backproject_depth, transform_points
from .occupancy import OccupancyGrid3D, OccupancyGridConfig


def vertical_obstacle_mask(points_ned, reference_ned_z, band_m):
    """Seleciona endpoints no envelope vertical do corpo, em coordenadas NED."""
    points = np.asarray(points_ned, dtype=float)
    reference = float(reference_ned_z)
    band = float(band_m)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ned deve possuir shape (N, 3)")
    if not np.isfinite(reference) or not np.isfinite(band) or band <= 0:
        raise ValueError("referencia e banda vertical devem ser finitas e positivas")
    return np.abs(points[:, 2] - reference) <= band


@dataclass(frozen=True)
class SpatialNavigationConfig:
    """Parametros que devem permanecer iguais entre mapa estimado e mapa ideal."""

    voxel_resolution_m: float = 0.75
    depth_stride: int = 40
    depth_edge_stride: int | None = None
    depth_edge_relative_threshold: float = 0.10
    min_depth_m: float = 0.5
    max_depth_m: float = 30.0
    depth_free_space_margin_m: float = 0.0
    depth_free_space_margin_ratio: float = 0.0
    depth_free_space_margin_max_m: float | None = None
    depth_occupied_uncertainty_m: float = 0.0
    free_observations_required: int = 1
    free_viewpoint_sectors_required: int = 1
    free_viewpoint_sector_deg: float = 45.0
    occupied_observations_required: int = 1
    occupied_support_radius_voxels: int = 0
    pending_clear_free_observations_required: int = 3
    occupied_evidence_window_frames: int = 6
    obstacle_vertical_band_m: float = 1.5
    drone_clearance_radius_m: float = 1.25
    drone_vertical_clearance_m: float | None = None
    known_free_radius_m: float = 0.8
    local_plan_radius_m: float = 20.0
    min_subgoal_progress_m: float = 0.5
    vertical_tolerance_m: float = 1.5
    lock_path_altitude_to_goal: bool = False
    max_waypoint_spacing_m: float = 5.0
    frontier_standoff_m: float = 2.5
    goal_acceptance_radius_m: float = 0.0
    connectivity: int = 26

    def __post_init__(self):
        if self.depth_stride < 1:
            raise ValueError("depth_stride deve ser maior ou igual a 1")
        if self.depth_edge_stride is not None and self.depth_edge_stride < 1:
            raise ValueError("depth_edge_stride deve ser maior ou igual a 1")
        if self.depth_edge_relative_threshold <= 0:
            raise ValueError(
                "depth_edge_relative_threshold deve ser positivo"
            )
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m deve ser maior que min_depth_m")
        if self.depth_free_space_margin_m < 0:
            raise ValueError("depth_free_space_margin_m nao pode ser negativo")
        if self.depth_free_space_margin_ratio < 0:
            raise ValueError("depth_free_space_margin_ratio nao pode ser negativo")
        if (
            self.depth_free_space_margin_max_m is not None
            and self.depth_free_space_margin_max_m
            < self.depth_free_space_margin_m
        ):
            raise ValueError(
                "depth_free_space_margin_max_m deve ser maior ou igual a base"
            )
        if self.depth_occupied_uncertainty_m < 0:
            raise ValueError("depth_occupied_uncertainty_m nao pode ser negativo")
        if self.free_observations_required < 1:
            raise ValueError("free_observations_required deve ser maior ou igual a 1")
        if self.free_viewpoint_sectors_required < 1:
            raise ValueError("free_viewpoint_sectors_required deve ser maior ou igual a 1")
        if not 0 < self.free_viewpoint_sector_deg <= 360:
            raise ValueError("free_viewpoint_sector_deg deve estar no intervalo (0, 360]")
        if not 1 <= self.occupied_observations_required <= 4:
            raise ValueError(
                "occupied_observations_required deve estar entre 1 e 4"
            )
        if self.occupied_support_radius_voxels < 0:
            raise ValueError("occupied_support_radius_voxels nao pode ser negativo")
        if self.pending_clear_free_observations_required < 1:
            raise ValueError(
                "pending_clear_free_observations_required deve ser positivo"
            )
        if self.occupied_evidence_window_frames < 1:
            raise ValueError("occupied_evidence_window_frames deve ser positivo")
        if self.obstacle_vertical_band_m <= 0:
            raise ValueError("obstacle_vertical_band_m deve ser positivo")
        if (
            self.drone_vertical_clearance_m is not None
            and self.drone_vertical_clearance_m <= 0
        ):
            raise ValueError("drone_vertical_clearance_m deve ser positivo")
        if self.local_plan_radius_m <= 0:
            raise ValueError("local_plan_radius_m deve ser positivo")
        if self.vertical_tolerance_m <= 0:
            raise ValueError("vertical_tolerance_m deve ser positivo")
        if self.max_waypoint_spacing_m <= 0:
            raise ValueError("max_waypoint_spacing_m deve ser positivo")
        if self.frontier_standoff_m < 0:
            raise ValueError("frontier_standoff_m nao pode ser negativo")
        if (
            not np.isfinite(self.goal_acceptance_radius_m)
            or self.goal_acceptance_radius_m < 0
        ):
            raise ValueError("goal_acceptance_radius_m deve ser finito e nao negativo")


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
    adopted_for_execution: bool | None = None
    planning_time_ms: float = 0.0
    diagnostics: dict = field(default_factory=dict)


class SpatialNavigator:
    """Constroi uma grade de ocupacao e planeja subcaminhos no espaco observado."""

    def __init__(self, config=None):
        self.config = config or SpatialNavigationConfig()
        free_decrement = OccupancyGridConfig.free_decrement
        occupied_increment = OccupancyGridConfig.occupied_increment
        occupancy_config = OccupancyGridConfig(
            resolution_m=self.config.voxel_resolution_m,
            occupied_threshold=(
                OccupancyGridConfig.occupied_threshold
                + (self.config.occupied_observations_required - 1)
                * occupied_increment
            ),
            free_threshold=(
                -0.35
                - (self.config.free_observations_required - 1) * free_decrement
            ),
            occupied_observations_required=(
                self.config.occupied_observations_required
            ),
            occupied_support_radius_voxels=(
                self.config.occupied_support_radius_voxels
            ),
            pending_clear_free_observations_required=(
                self.config.pending_clear_free_observations_required
            ),
            occupied_evidence_window_frames=self.config.occupied_evidence_window_frames,
            free_viewpoint_sectors_required=self.config.free_viewpoint_sectors_required,
            free_viewpoint_sector_deg=self.config.free_viewpoint_sector_deg,
        )
        self.grid = OccupancyGrid3D(occupancy_config)
        self.frames_integrated = 0

    def integrate_depth(
        self,
        depth_m,
        intrinsics,
        camera_to_ned,
        sampling_edge_mask=None,
        obstacle_reference_ned_z=None,
    ):
        """Reprojeta e integra um frame, retornando estatisticas da atualizacao."""

        if not isinstance(intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics deve ser CameraIntrinsics")
        points_camera = backproject_depth(
            depth_m,
            intrinsics,
            stride=self.config.depth_stride,
            edge_stride=self.config.depth_edge_stride,
            edge_relative_threshold=self.config.depth_edge_relative_threshold,
            sampling_edge_mask=sampling_edge_mask,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        camera_to_ned = np.asarray(camera_to_ned, dtype=np.float64)
        points_ned = transform_points(points_camera, camera_to_ned)
        camera_origin_ned = camera_to_ned[:3, 3]
        obstacle_reference_ned_z = (
            camera_origin_ned[2] if obstacle_reference_ned_z is None
            else float(obstacle_reference_ned_z)
        )
        obstacle_endpoint_mask = vertical_obstacle_mask(
            points_ned, obstacle_reference_ned_z,
            self.config.obstacle_vertical_band_m,
        )
        self.grid.mark_free_sphere(
            camera_origin_ned,
            self.config.known_free_radius_m,
        )
        self.grid.integrate_rays(
            camera_origin_ned,
            points_ned,
            obstacle_endpoint_mask,
            max_range_m=self.config.max_depth_m,
            free_space_margin_m=self.config.depth_free_space_margin_m,
            free_space_margin_ratio=self.config.depth_free_space_margin_ratio,
            free_space_margin_max_m=self.config.depth_free_space_margin_max_m,
            occupied_uncertainty_m=self.config.depth_occupied_uncertainty_m,
        )
        ego_voxel_cleared_occupied = self.grid.mark_ego_voxel_free(
            camera_origin_ned
        )
        points_integrated = int(np.count_nonzero(obstacle_endpoint_mask))
        free_only_rays = int(len(points_ned) - points_integrated)
        self.frames_integrated += 1
        return {
            "frame_index": self.frames_integrated,
            "points_integrated": points_integrated,
            "points_rejected_vertical": free_only_rays,
            "obstacle_reference_ned_z": obstacle_reference_ned_z,
            "free_only_rays": free_only_rays,
            "total_valid_rays": int(len(points_ned)),
            "ego_voxel_cleared_occupied": bool(
                ego_voxel_cleared_occupied
            ),
            "free_voxels": int(len(self.grid.free_voxels())),
            "occupied_voxels": int(len(self.grid.occupied_voxels())),
            "pending_occupied_voxels": int(
                len(self.grid.pending_occupied_voxels())
            ),
            "occupied_observations_required": int(
                self.config.occupied_observations_required
            ),
            "occupied_support_radius_voxels": int(
                self.config.occupied_support_radius_voxels
            ),
            "pending_clear_free_observations_required": int(
                self.config.pending_clear_free_observations_required
            ),
            "occupied_evidence_window_frames": int(
                self.config.occupied_evidence_window_frames
            ),
        }

    def plan(self, current_position_ned_m, requested_goal_ned_m, *,
             candidate_validator=None, max_candidates=16, recovery_frontiers=False, allow_initial_escape=False):
        """Planeja ate o destino ou ate o melhor subobjetivo observado na direcao dele."""

        started = perf_counter()
        if not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates deve ser inteiro positivo")
        if recovery_frontiers and candidate_validator is None:
            raise ValueError('exploracao de fronteiras exige validador de recuperacao')
        current = np.asarray(current_position_ned_m, dtype=np.float64)
        requested_goal = np.asarray(requested_goal_ned_m, dtype=np.float64)
        if current.shape != (3,) or requested_goal.shape != (3,):
            raise ValueError("posicao e destino devem possuir tres coordenadas")

        ego_voxel_cleared_occupied = self.grid.mark_ego_voxel_free(current)
        blocked = self._inflated_obstacles()
        current_voxel = self.grid.world_to_voxel(current)
        neighbor_states = {"free": 0, "occupied": 0, "unknown": 0}
        blocked_neighbors = 0
        for offset in product((-1, 0, 1), repeat=3):
            if offset == (0, 0, 0):
                continue
            neighbor = tuple(
                current_voxel[axis] + offset[axis] for axis in range(3)
            )
            neighbor_states[self.grid.state(neighbor)] += 1
            blocked_neighbors += int(neighbor in blocked)
        diagnostics = {
            "segment_traversal": "closed_voxel_supercover_v1",
            "mapping_frames_integrated_at_plan": self.frames_integrated,
            "current_position_ned_m": tuple(current),
            "requested_goal_ned_m": tuple(requested_goal),
            "current_voxel": current_voxel,
            "current_voxel_state": self.grid.state(current_voxel),
            "current_voxel_inflated": current_voxel in blocked,
            "ego_voxel_cleared_occupied_at_plan": bool(
                ego_voxel_cleared_occupied
            ),
            "clearance_horizontal_m": self.config.drone_clearance_radius_m,
            "clearance_vertical_m": (
                self.config.drone_vertical_clearance_m
                if self.config.drone_vertical_clearance_m is not None
                else self.config.drone_clearance_radius_m
            ),
            "neighbor_states_26": neighbor_states,
            "blocked_neighbors_26": blocked_neighbors,
        }
        current_layer = current_voxel[2]
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
        if self.config.lock_path_altitude_to_goal:
            min_layer = goal_layer
            max_layer = goal_layer
            traversable = {
                voxel for voxel in traversable if voxel[2] == goal_layer
            }
        diagnostics["traversable_voxels"] = len(traversable)
        diagnostics["vertical_layer_range"] = (min_layer, max_layer)
        diagnostics["path_altitude_locked_to_goal"] = bool(
            self.config.lock_path_altitude_to_goal
        )
        start = self._nearest_voxel(current_voxel, traversable)
        diagnostics["start_voxel"] = start
        if start is None:
            return self._failure(
                requested_goal,
                "sem espaco livre ao redor do drone",
                self._elapsed_ms(started),
                diagnostics,
            )

        planner = self._make_planner(traversable, blocked)
        reachable = planner.reachable_from(start)
        max_progress = self._max_progress(current, requested_goal, reachable)
        diagnostics["reachable_voxels"] = len(reachable)
        diagnostics["max_reachable_progress_m"] = max_progress

        requested_goal_voxel = self.grid.world_to_voxel(requested_goal)
        diagnostics["requested_goal_voxel_state"] = self.grid.state(
            requested_goal_voxel
        )
        diagnostics["requested_goal_voxel_inflated"] = (
            requested_goal_voxel in blocked
        )
        diagnostics["goal_acceptance_radius_m"] = self.config.goal_acceptance_radius_m
        if requested_goal_voxel in reachable:
            selected_goal = requested_goal_voxel
        else:
            selected_goal = self._select_local_subgoal(
                current,
                requested_goal,
                reachable,
            )
        frontier_candidates = (
            self._rank_observation_frontiers(current, requested_goal, reachable)
            if recovery_frontiers else []
        )
        if recovery_frontiers and (selected_goal is None or selected_goal == start):
            selected_goal = next((v for v in frontier_candidates if v != start), None)
        if selected_goal is None or selected_goal == start:
            return self._failure(
                requested_goal,
                (
                    "nenhum subobjetivo observado com progresso "
                    f"(livres={len(traversable)}, alcancaveis={len(reachable)}, "
                    f"max_progresso={max_progress:.2f}m)"
                ),
                self._elapsed_ms(started),
                diagnostics,
            )

        if candidate_validator is None:
            return self._plan_to_subgoal(
                current, requested_goal, selected_goal, planner,
                traversable, blocked, started, diagnostics,
                allow_initial_escape=allow_initial_escape,
            )

        # Same map and connected component for every attempt. A deterministic
        # count budget avoids machine-load-dependent choices in snapshot replay.
        alternatives = (frontier_candidates if recovery_frontiers else
                        self._rank_local_subgoals(current, requested_goal, reachable))
        candidates = [selected_goal] + [voxel for voxel in alternatives
                                      if voxel not in (selected_goal, start)]
        search = {
            "candidate_limit": max_candidates,
            "available_candidates": len(candidates),
            "selection_strategy": "ranked_spatial_separation_v1",
            "selection_spacing_m": max(
                self.config.voxel_resolution_m, self.config.frontier_standoff_m
            ),
            "attempts": [],
            "selected_candidate_index": None,
            "budget_exhausted": False,
            "candidate_pool": ('connected_observation_frontiers_v1' if recovery_frontiers
                               else 'forward_subgoals'),
            "frontier_candidates": len(frontier_candidates),
            "backward_exploration_enabled": bool(recovery_frontiers),
        }
        candidate_indices = self._spatial_candidate_indices(candidates, max_candidates)
        for index, original_index in enumerate(candidate_indices):
            candidate = candidates[original_index]
            result = self._plan_to_subgoal(
                current, requested_goal, candidate, planner,
                traversable, blocked, started, dict(diagnostics),
                allow_initial_escape=allow_initial_escape,
            )
            if recovery_frontiers:
                result.diagnostics['subgoal_selection_policy'] = 'recovery_observation_frontiers_v1'
                result.diagnostics['backward_exploration_selected'] = bool(
                    result.diagnostics.get('selected_goal_projection_m', 0.) < 0.)
            check = (
                candidate_validator(result) if result.success
                else {"accepted": False, "reason": result.reason}
            )
            search["attempts"].append({
                "candidate_voxel": tuple(candidate),
                "original_rank": original_index + 1,
                "plan_success": result.success,
                "plan_reason": result.reason,
                "validation": check,
            })
            if result.success and check["accepted"]:
                search["selected_candidate_index"] = index
                result.diagnostics["candidate_search"] = search
                result.planning_time_ms = self._elapsed_ms(started)
                return result
        search["budget_exhausted"] = len(candidates) > max_candidates
        diagnostics["candidate_search"] = search
        return self._failure(
            requested_goal, "nenhum subobjetivo executavel aprovado na busca limitada",
            self._elapsed_ms(started), diagnostics,
        )

    def _make_planner(self, traversable, blocked):
        """Ponto de extensao para experimentos offline; padrao permanece igual."""
        return AStar3D(traversable, blocked, connectivity=self.config.connectivity)

    def _spatial_candidate_indices(self, candidates, limit):
        """Intercala prioridade ao objetivo e cobertura espacial, sem novos alvos.

        Mantem o principal primeiro. Nas tentativas pares (2, 4, ...), escolhe
        o primeiro voxel do ranking separado dos ja selecionados pela distancia
        de recuo da fronteira (no minimo um voxel); nas impares, retoma o ranking
        original. Se nao houver alternativa separada, usa o proximo do ranking.
        """
        if not candidates or limit <= 0:
            return []
        # Equal voxel dimensions: squared distances in index space preserve
        # metric ordering. O(N * limit), without an all-pairs distance matrix.
        points = np.asarray(candidates, dtype=np.float64)
        available = np.ones(len(candidates), dtype=bool)
        available[0] = False
        selected = [0]
        min_distance_sq = np.sum((points - points[0]) ** 2, axis=1)
        separation_sq = max(
            1.0, self.config.frontier_standoff_m / self.config.voxel_resolution_m
        ) ** 2
        while len(selected) < min(limit, len(candidates)):
            if len(selected) % 2:
                separated = np.flatnonzero(available & (min_distance_sq >= separation_sq))
                index = int(separated[0]) if len(separated) else int(np.flatnonzero(available)[0])
            else:
                index = int(np.flatnonzero(available)[0])
            selected.append(index)
            available[index] = False
            min_distance_sq = np.minimum(
                min_distance_sq, np.sum((points - points[index]) ** 2, axis=1)
            )
        # Diversity is only a scheduling decision: every selected candidate
        # still needs A*, post-processing, known-free safety and the caller veto.
        return selected

    def _plan_to_subgoal(self, current, requested_goal, selected_goal, planner,
                         traversable, blocked, started, diagnostics,
                         allow_initial_escape=False):
        """A* e TODOS os pos-processamentos para um unico candidato."""
        requested_goal_voxel = self.grid.world_to_voxel(requested_goal)
        start = diagnostics["start_voxel"]
        try:
            path = planner.plan(start, selected_goal)
        except PathNotFoundError as error:
            return self._failure(
                requested_goal,
                str(error),
                self._elapsed_ms(started),
                diagnostics,
            )

        shortcut = self._shortcut_path(path, traversable, blocked)
        compressed = compress_collinear_path(shortcut)
        arrival_point = self._arrival_point_in_voxel(
            current, requested_goal, selected_goal
        )
        world_points = [self.grid.voxel_to_world(voxel) for voxel in compressed]
        if arrival_point is not None:
            world_points.append(arrival_point)
        waypoints = self._densify_waypoints(world_points)
        diagnostics["arrival_point_inside_selected_voxel"] = (
            tuple(arrival_point) if arrival_point is not None else None
        )
        if self.config.lock_path_altitude_to_goal:
            waypoints = [
                (float(point[0]), float(point[1]), float(requested_goal[2]))
                for point in waypoints
            ]
            diagnostics["commanded_altitude_ned_m"] = float(requested_goal[2])
        raw_path_length_m = self._path_length(path)
        shortcut_path_length_m = self._path_length(shortcut)
        smoothed_path_length_m = self._waypoint_length([current, *waypoints])
        terminal_point = np.asarray(waypoints[-1], dtype=float)
        within_arrival_region = (
            self.config.goal_acceptance_radius_m > 0
            and np.linalg.norm(terminal_point - requested_goal)
            <= self.config.goal_acceptance_radius_m
        )
        reason = (
            "goal_observed" if selected_goal == requested_goal_voxel
            else "goal_region_observed" if within_arrival_region
            else "local_subgoal"
        )
        diagnostics["terminal_goal_region_observed"] = bool(within_arrival_region)
        frontier_standoff_applied_m = 0.0
        if reason == "local_subgoal":
            waypoints, frontier_standoff_applied_m = self._reserve_frontier(
                current,
                waypoints,
            )
        fallback_waypoints = [
            tuple(self.grid.voxel_to_world(voxel)) for voxel in path
        ]
        if arrival_point is not None:
            fallback_waypoints.append(tuple(arrival_point))
        if self.config.lock_path_altitude_to_goal:
            fallback_waypoints = [
                (
                    float(point[0]),
                    float(point[1]),
                    float(requested_goal[2]),
                )
                for point in fallback_waypoints
            ]

        fallback_standoff_applied_m = 0.0
        if reason == "local_subgoal":
            (
                fallback_waypoints,
                fallback_standoff_applied_m,
            ) = self._reserve_frontier(
                current,
                fallback_waypoints,
            )

        (
            waypoints,
            raw_voxel_fallback_used,
            postprocessed_path_safety,
            final_path_safety,
        ) = self._select_safe_waypoint_representation(
            current,
            waypoints,
            fallback_waypoints,
            allow_initial_escape=allow_initial_escape,
        )

        if raw_voxel_fallback_used:
            frontier_standoff_applied_m = fallback_standoff_applied_m

        diagnostics["postprocessed_path_safety"] = (
            postprocessed_path_safety
        )
        diagnostics["final_path_safety"] = final_path_safety
        diagnostics["initial_escape_required"] = bool(
            final_path_safety.get('policy')
            == 'known_free_initial_escape_no_reentry'
        )
        diagnostics["raw_voxel_fallback_used"] = (
            raw_voxel_fallback_used
        )

        if not final_path_safety["safe"]:
            return self._failure(
                requested_goal,
                (
                    "caminho inseguro apos pos-processamento e "
                    "fallback pelos centros dos voxels"
                ),
                self._elapsed_ms(started),
                diagnostics,
            )

        post_standoff_path_length_m = self._waypoint_length(
            [current, *waypoints]
        )
        diagnostics.update(
            {
                "selected_goal_voxel": selected_goal,
                "raw_path_voxels": len(path),
                "shortcut_path_voxels": len(shortcut),
                "compressed_path_voxels": len(compressed),
                "shortcut_path_length_m": shortcut_path_length_m,
                "smoothed_path_length_m": smoothed_path_length_m,
                "post_standoff_path_length_m": post_standoff_path_length_m,
            }
        )
        selected_goal_ned_m = self.grid.voxel_to_world(selected_goal)
        if self.config.lock_path_altitude_to_goal:
            selected_goal_ned_m = selected_goal_ned_m.copy()
            selected_goal_ned_m[2] = requested_goal[2]
        if arrival_point is not None:
            selected_goal_ned_m = arrival_point.copy()
        goal_direction = requested_goal - current
        remaining_goal_distance = float(np.linalg.norm(goal_direction))
        diagnostics["subgoal_selection_policy"] = "bounded_nearest_goal"
        diagnostics["remaining_global_goal_distance_m"] = remaining_goal_distance
        diagnostics["selected_goal_projection_m"] = (
            float(np.dot(selected_goal_ned_m - current, goal_direction))
            / remaining_goal_distance
            if remaining_goal_distance > 1e-9 else 0.0
        )
        diagnostics["selected_goal_distance_to_global_m"] = float(
            np.linalg.norm(selected_goal_ned_m - requested_goal)
        )
        return SpatialPlan(
            success=True,
            reason=reason,
            requested_goal_ned_m=tuple(requested_goal),
            selected_goal_ned_m=tuple(selected_goal_ned_m),
            path_voxels=path,
            waypoints_ned_m=waypoints,
            path_length_m=post_standoff_path_length_m,
            raw_path_length_m=raw_path_length_m,
            frontier_standoff_applied_m=frontier_standoff_applied_m,
            planning_time_ms=self._elapsed_ms(started),
            diagnostics=diagnostics,
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

    def path_safety_diagnostics(
        self, current_position_ned_m, waypoints_ned_m
    ):
        """Explica a primeira falha da verificacao continua do caminho."""

        points = [np.asarray(current_position_ned_m, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints_ned_m
        ]
        if len(points) < 2:
            return {
                "safe": False,
                "failure_reason": "empty_path",
                "segment_traversal": "closed_voxel_supercover_v1",
                "checked_voxels": 0,
                "segment_index": None,
                "first_unsafe_voxel": None,
                "first_unsafe_state": None,
                "first_unsafe_is_inflated": False,
            }

        blocked = self._inflated_obstacles()
        occupied = self.grid.occupied_voxels()
        free = self.grid.free_voxels()
        checked_voxels = 0
        previous_voxel = None

        for segment_index, (start, end) in enumerate(
            zip(points, points[1:])
        ):
            for voxel in self.grid.segment_voxels(start, end):
                if voxel == previous_voxel:
                    continue
                previous_voxel = voxel
                checked_voxels += 1

                if voxel in occupied:
                    state = "occupied"
                elif voxel in blocked:
                    state = "inflated"
                elif voxel not in free:
                    state = "unknown"
                else:
                    continue

                return {
                    "safe": False,
                    "failure_reason": state,
                    "segment_traversal": "closed_voxel_supercover_v1",
                    "checked_voxels": checked_voxels,
                    "segment_index": segment_index,
                    "segment_start_ned_m": start.tolist(),
                    "segment_end_ned_m": end.tolist(),
                    "first_unsafe_voxel": list(voxel),
                    "first_unsafe_state": state,
                    "first_unsafe_is_inflated": voxel in blocked,
                }

        return {
            "safe": True,
            "failure_reason": None,
            "segment_traversal": "closed_voxel_supercover_v1",
            "checked_voxels": checked_voxels,
            "segment_index": None,
            "first_unsafe_voxel": None,
            "first_unsafe_state": None,
            "first_unsafe_is_inflated": False,
        }

    def path_is_safe(self, current_position_ned_m, waypoints_ned_m):
        """Verifica se todo o caminho restante continua observado e desocupado."""

        return bool(
            self.path_safety_diagnostics(
                current_position_ned_m,
                waypoints_ned_m,
            )["safe"]
        )

    def path_safety_allowing_dynamic_initial_escape(
        self, current_position_ned_m, waypoints_ned_m
    ):
        """Tolera apenas inflacao surgida no voxel atual durante a execucao."""
        strict = self.path_safety_diagnostics(
            current_position_ned_m, waypoints_ned_m
        )
        result = dict(strict)
        result["dynamic_initial_escape_used"] = False
        if strict.get("safe") or strict.get("failure_reason") != "inflated":
            return result
        current_voxel = self.grid.world_to_voxel(current_position_ned_m)
        first_unsafe = strict.get("first_unsafe_voxel")
        if first_unsafe is None or tuple(first_unsafe) != tuple(current_voxel):
            return result
        escape = self.initial_escape_path_safety_diagnostics(
            current_position_ned_m, waypoints_ned_m
        )
        if not escape.get("safe"):
            result["dynamic_initial_escape_attempt"] = escape
            return result
        result = dict(escape)
        result["dynamic_initial_escape_used"] = True
        result["strict_path_safety"] = strict
        return result

    def initial_escape_path_safety_diagnostics(
        self, current_position_ned_m, waypoints_ned_m
    ):
        # Permite apenas sair da inflacao inicial por espaco conhecido.
        points = [np.asarray(current_position_ned_m, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints_ned_m
        ]
        blocked = self._inflated_obstacles()
        free = self.grid.free_voxels()
        start_voxel = self.grid.world_to_voxel(points[0])
        result = {
            'safe': False,
            'failure_reason': 'start_not_inflated',
            'policy': 'known_free_initial_escape_no_reentry',
            'checked_voxels': 0,
            'first_unsafe_voxel': None,
        }
        if len(points) < 2:
            result['failure_reason'] = 'empty_path'
            return result
        if start_voxel not in blocked:
            return self.path_safety_diagnostics(points[0], points[1:])
        left_inflation = False
        previous = None
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            for voxel in self.grid.segment_voxels(start, end):
                if voxel == previous:
                    continue
                previous = voxel
                result['checked_voxels'] += 1
                if voxel in blocked:
                    if left_inflation:
                        result.update(
                            failure_reason='obstacle_reentry',
                            first_unsafe_voxel=list(voxel),
                            segment_index=segment_index,
                        )
                        return result
                    continue
                left_inflation = True
                if voxel not in free:
                    result.update(
                        failure_reason='unknown',
                        first_unsafe_voxel=list(voxel),
                        segment_index=segment_index,
                    )
                    return result
        result['safe'] = bool(left_inflation)
        result['failure_reason'] = None if left_inflation else 'never_left_inflation'
        return result

    def path_avoids_obstacles_allowing_initial_escape(
        self, current_position_ned_m, waypoints_ned_m
    ):
        """Permite sair de inflacao inicial, mas proibe reentrada posterior."""

        points = [np.asarray(current_position_ned_m, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints_ned_m
        ]
        if len(points) < 2:
            return False
        path_voxels = []
        for start, end in zip(points, points[1:]):
            for voxel in self.grid.segment_voxels(start, end):
                if not path_voxels or voxel != path_voxels[-1]:
                    path_voxels.append(voxel)
        blocked = self._inflated_obstacles()
        reached_free = False
        for voxel in path_voxels:
            if voxel in blocked:
                if reached_free:
                    return False
            else:
                reached_free = True
        return reached_free

    def first_obstacle_reentry_voxel(
        self, current_position_ned_m, waypoints_ned_m
    ):
        """Retorna o primeiro voxel inflado reentrado apos sair da inflacao."""

        points = [np.asarray(current_position_ned_m, dtype=float)] + [
            np.asarray(point, dtype=float) for point in waypoints_ned_m
        ]
        if len(points) < 2:
            return None
        blocked = self._inflated_obstacles()
        reached_free = False
        previous_voxel = None
        for start, end in zip(points, points[1:]):
            for voxel in self.grid.segment_voxels(start, end):
                if voxel == previous_voxel:
                    continue
                previous_voxel = voxel
                if voxel in blocked:
                    if reached_free:
                        return voxel
                else:
                    reached_free = True
        return None

    def position_is_safe(self, position_ned_m):
        """Confirma que uma posicao pertence ao espaco livre fora da inflacao."""

        voxel = self.grid.world_to_voxel(position_ned_m)
        return (
            voxel in self.grid.free_voxels()
            and voxel not in self._inflated_obstacles()
        )

    def _inflated_obstacles(self):
        return self.grid.inflated_occupied_voxels(
            self.config.drone_clearance_radius_m,
            self.config.drone_vertical_clearance_m,
        )

    def _arrival_point_in_voxel(self, current, requested_goal, voxel):
        """Encontra chegada no interior do voxel selecionado, sem abrir celulas."""

        radius = self.config.goal_acceptance_radius_m
        if radius <= 0:
            return None
        center = self.grid.voxel_to_world(voxel)
        commanded_center = center.copy()
        if self.config.lock_path_altitude_to_goal:
            commanded_center[2] = requested_goal[2]
        if np.linalg.norm(commanded_center - requested_goal) <= radius:
            return None
        # Keep a quarter-voxel inset from each face, avoiding boundary rounding.
        inset = self.config.voxel_resolution_m * 0.25
        target = np.clip(requested_goal, center - inset, center + inset)
        if self.config.lock_path_altitude_to_goal:
            target[2] = requested_goal[2]
        direction = requested_goal - current
        if (
            self.grid.world_to_voxel(target) != tuple(voxel)
            or np.linalg.norm(target - requested_goal) > radius
            or np.dot(target - current, direction) > np.dot(direction, direction)
        ):
            return None
        # This is a geometric candidate only. The complete commanded path,
        # including this last segment, still passes the ordinary safety gate.
        return target

    def _select_local_subgoal(self, current, requested_goal, traversable):
        ranked = self._rank_local_subgoals(current, requested_goal, traversable)
        return ranked[0] if ranked else None

    def _frontier_unknown_faces(self, voxel, blocked):
        """Vizinhos desconhecidos sao pistas de observacao, nunca espaco livre."""
        if self.grid.state(voxel) != 'free' or voxel in blocked:
            return []
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        if not self.config.lock_path_altitude_to_goal:
            offsets += [(0, 0, 1), (0, 0, -1)]
        neighbors = [tuple(voxel[i] + offset[i] for i in range(3)) for offset in offsets]
        return [v for v in neighbors if v not in blocked and self.grid.state(v) == 'unknown']

    def _rank_observation_frontiers(self, current, goal, reachable):
        """Alternativas de recuperacao, inclusive atras, no componente observado.

        Mantem raio local e limite de ultrapassagem do objetivo. Prioriza a
        proximidade ao objetivo; faces desconhecidas desempatam a prioridade.
        O recuo e a visibilidade do ponto final sao validados DEPOIS do A*.
        """
        direction = goal - current
        distance_to_goal = float(np.linalg.norm(direction))
        if distance_to_goal <= 1e-9:
            return []
        direction /= distance_to_goal
        blocked = self._inflated_obstacles()
        ranked = []
        for voxel in reachable:
            faces = self._frontier_unknown_faces(voxel, blocked)
            if not faces:
                continue
            point = self.grid.voxel_to_world(voxel)
            if self.config.lock_path_altitude_to_goal:
                point[2] = goal[2]
            offset = point - current
            distance = float(np.linalg.norm(offset))
            if (distance > self.config.local_plan_radius_m
                or distance < self.config.frontier_standoff_m + self.config.min_subgoal_progress_m
                or np.dot(offset, direction) > distance_to_goal + 1e-9):
                continue
            rank = (float(np.linalg.norm(point - goal)), -len(faces), distance, tuple(voxel))
            ranked.append((rank, tuple(voxel)))
        return [voxel for _, voxel in sorted(ranked)]

    def observation_frontier_diagnostics(self, current, waypoints, target):
        """Proxy geometrico de observabilidade; nao promete ganho real da camera."""
        result = {'observable': False, 'reason': 'not_observation_frontier',
                  'unknown_space_traversed': False,
                  'evidence_kind': 'known_free_line_to_unknown_boundary_not_measured_gain'}
        if target is None or not waypoints:
            return result
        voxel = self.grid.world_to_voxel(target)
        neighbors = self._frontier_unknown_faces(voxel, self._inflated_obstacles())
        result.update(frontier_voxel=voxel, unknown_neighbor_voxels=neighbors,
                      potential_unknown_faces=len(neighbors))
        if not neighbors:
            return result
        endpoint = waypoints[-1]
        sight = self.path_safety_diagnostics(endpoint, [target])
        result.update(
            observable=bool(sight['safe']),
            reason='observable_frontier' if sight['safe'] else 'frontier_occluded_after_standoff',
            endpoint_to_frontier_safety=sight,
            visible_from_current=self.path_is_safe(current, [target]),
            endpoint_to_frontier_m=float(np.linalg.norm(np.asarray(endpoint) - target)),
        )
        return result

    def _rank_local_subgoals(self, current, requested_goal, traversable):
        """Escolhe um alvo observado sem projetar alem do objetivo global.

        O limite vale para o subobjetivo, nao para os voxels intermediarios
        do A*: desvios necessarios ao redor de obstaculos continuam permitidos.
        """

        direction = requested_goal - current
        target_distance = float(np.linalg.norm(direction))
        if target_distance <= 1e-9:
            return []
        direction /= target_distance

        ranked = []
        for voxel in traversable:
            point = self.grid.voxel_to_world(voxel).copy()
            if self.config.lock_path_altitude_to_goal:
                point[2] = requested_goal[2]
            offset = point - current
            distance = float(np.linalg.norm(offset))
            if distance > self.config.local_plan_radius_m:
                continue
            progress = float(np.dot(offset, direction))
            if progress < self.config.min_subgoal_progress_m:
                continue
            if progress > target_distance + 1e-9:
                continue
            lateral = float(np.linalg.norm(offset - progress * direction))
            vertical_error = abs(float(point[2] - requested_goal[2]))
            remaining_distance = float(np.linalg.norm(requested_goal - point))
            rank = (remaining_distance, vertical_error, lateral, tuple(voxel))
            ranked.append((rank, tuple(voxel)))
        return [voxel for _, voxel in sorted(ranked)]

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

    def _select_safe_waypoint_representation(
        self,
        current_position_ned_m,
        postprocessed_waypoints_ned_m,
        raw_voxel_waypoints_ned_m,
        *,
        allow_initial_escape=False,
    ):
        """Prefere o caminho processado e recua para centros de voxel."""

        postprocessed = list(postprocessed_waypoints_ned_m)
        fallback = list(raw_voxel_waypoints_ned_m)

        postprocessed_safety = self.path_safety_diagnostics(
            current_position_ned_m,
            postprocessed,
        )
        if postprocessed_safety["safe"]:
            return (
                postprocessed,
                False,
                postprocessed_safety,
                postprocessed_safety,
            )

        fallback_safety = self.path_safety_diagnostics(
            current_position_ned_m,
            fallback,
        )
        if allow_initial_escape and not fallback_safety["safe"]:
            postprocessed_escape = self.initial_escape_path_safety_diagnostics(
                current_position_ned_m, postprocessed
            )
            fallback_escape = self.initial_escape_path_safety_diagnostics(
                current_position_ned_m, fallback
            )
            if postprocessed_escape["safe"]:
                return (
                    postprocessed, False,
                    postprocessed_safety, postprocessed_escape,
                )
            if fallback_escape["safe"]:
                return (
                    fallback, True,
                    postprocessed_safety, fallback_escape,
                )
        if fallback_safety["safe"]:
            return (
                fallback,
                True,
                postprocessed_safety,
                fallback_safety,
            )

        return [], False, postprocessed_safety, fallback_safety

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
                segment = self.grid.segment_voxels(start, end)
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
    def _failure(requested_goal, reason, planning_time_ms=0.0, diagnostics=None):
        return SpatialPlan(
            success=False,
            reason=reason,
            requested_goal_ned_m=tuple(requested_goal),
            planning_time_ms=float(planning_time_ms),
            diagnostics=dict(diagnostics or {}),
        )

    @staticmethod
    def _elapsed_ms(started):
        return (perf_counter() - started) * 1000.0
