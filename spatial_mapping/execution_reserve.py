"""Reserva de execucao EXPERIMENTAL para snapshots; nao ligada ao controlador.

A reserva e distancia adicional as caixas ja infladas. Nao redefine ocupacao,
nao libera espaco desconhecido nem promete que o PX4 respeite esse limite.
"""
from copy import deepcopy
from heapq import heappop, heappush
from itertools import product
from time import perf_counter
import numpy as np

from .astar import AStar3D, PathNotFoundError
from .clearance import path_voxel_clearance
from .navigation import SpatialNavigator


class _ReserveAStar(AStar3D):
    def __init__(self, traversable, blocked, connectivity, segment_allowed, initial_edges=None):
        super().__init__(traversable, blocked, connectivity)
        self.segment_allowed = segment_allowed
        self.edge_cache = {}
        self.initial_edges = initial_edges

    def reachable_from(self, start):
        if self.initial_edges is None:
            return super().reachable_from(start)
        reachable = set(self.initial_edges)
        frontier = sorted(reachable)
        while frontier:
            current = frontier.pop()
            for neighbor, _ in self._valid_neighbors(current):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        return reachable

    def plan(self, start, goal):
        if self.initial_edges is None:
            return super().plan(start, goal)
        goal = tuple(goal)
        if goal not in self.traversable or goal in self.blocked:
            raise PathNotFoundError('destino deve pertencer ao espaco livre')
        # A virtual real-pose source has ONLY validated outgoing connectors.
        # Connector costs use voxel units, like the ordinary A* edge costs.
        costs = dict(self.initial_edges)
        frontier, came_from = [], {}
        for voxel, cost in sorted(costs.items()):
            heappush(frontier, (cost + self._heuristic(voxel, goal), cost, voxel))
        while frontier:
            _, cost, current = heappop(frontier)
            if cost > costs[current]:
                continue
            if current == goal:
                return self._reconstruct(came_from, current)
            for neighbor, move_cost in self._valid_neighbors(current):
                new_cost = cost + move_cost
                if new_cost < costs.get(neighbor, float('inf')):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    heappush(frontier, (new_cost + self._heuristic(neighbor, goal),
                                        new_cost, neighbor))
        raise PathNotFoundError('nenhum caminho a partir dos conectores iniciais seguros')

    def _valid_neighbors(self, voxel):
        for neighbor, cost in super()._valid_neighbors(voxel):
            key = tuple(sorted((tuple(voxel), tuple(neighbor))))
            if key not in self.edge_cache:
                self.edge_cache[key] = self.segment_allowed(*key)
            if self.edge_cache[key]:
                yield neighbor, cost


class ExecutionReserveNavigator(SpatialNavigator):
    """Copia isolada do mapa; raio zero reproduz o planejador existente."""
    def __init__(self, source, extra_radius_m, *, use_start_connections=True):
        if not np.isfinite(extra_radius_m) or extra_radius_m < 0:
            raise ValueError("reserva deve ser finita e nao negativa")
        super().__init__(source.config)
        self.grid = deepcopy(source.grid)
        self.frames_integrated = source.frames_integrated
        self.extra_radius_m = float(extra_radius_m)
        self._commanded_z = None
        self._reserve_cells = None
        self.use_start_connections = bool(use_start_connections)
        self._connection_current = None
        self._connection_diagnostics = None

    def _world(self, voxel):
        point = self.grid.voxel_to_world(voxel)
        if self._commanded_z is not None:
            point[2] = self._commanded_z
        return point

    def _has_reserve(self, points):
        if self.extra_radius_m == 0:
            return True
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            return False
        cells = self._reserve_cells
        if cells is None:
            cells = np.asarray(sorted(self._inflated_obstacles()), dtype=int).reshape(-1, 3)
        if not len(cells):
            return True
        resolution = self.config.voxel_resolution_m
        # Exact broad phase: boxes outside this expanded bounding box cannot
        # touch the reserve tube. Narrow phase uses analytic segment/box distance.
        lo = points.min(axis=0) - self.extra_radius_m - 1e-9
        hi = points.max(axis=0) + self.extra_radius_m + 1e-9
        nearby = cells[np.all((cells + 1) * resolution >= lo, axis=1)
                       & np.all(cells * resolution <= hi, axis=1)]
        if not len(nearby):
            return True
        distance = path_voxel_clearance(points, nearby, resolution)["distance_m"]
        return distance > self.extra_radius_m + 1e-9

    def plan(self, current_position_ned_m, requested_goal_ned_m, **kwargs):
        """Planeja em NED exigindo a reserva extra ao redor de cada segmento."""
        started = perf_counter()
        current = np.asarray(current_position_ned_m, dtype=float)
        goal = np.asarray(requested_goal_ned_m, dtype=float)
        if current.shape != (3,) or goal.shape != (3,) or not np.isfinite([current, goal]).all():
            raise ValueError("posicao e destino devem possuir tres coordenadas finitas")
        self._commanded_z = float(goal[2]) if self.config.lock_path_altitude_to_goal else None
        self._connection_current = current.copy()
        self._connection_diagnostics = None
        self.grid.mark_ego_voxel_free(current)
        self._reserve_cells = np.asarray(sorted(self._inflated_obstacles()), dtype=int).reshape(-1, 3)
        try:
            if self.extra_radius_m and not self._has_reserve([current]):
                return self._failure(goal, "execution_reserve_unavailable_at_start",
                                     self._elapsed_ms(started),
                                     {"extra_execution_radius_m": self.extra_radius_m,
                                      "current_position_ned_m": tuple(current),
                                      "start_clearance": path_voxel_clearance(
                                          [current], self._reserve_cells, self.config.voxel_resolution_m)})
            result = super().plan(current, goal, **kwargs)
            result.diagnostics["extra_execution_radius_m"] = self.extra_radius_m
            result.diagnostics["execution_reserve_experimental"] = True
            if self._connection_diagnostics is not None:
                result.diagnostics['initial_connections'] = self._connection_diagnostics
            return result
        finally:
            self._reserve_cells = None


    def plan_executable(self, current, goal, *, acceptance_m, minimum_m,
                        candidate_validator=None, max_candidates=16, recovery_frontiers=False):
        """Busca limitada; rejeita microcaminhos sem mudar margem ou recuo."""
        if any(not np.isfinite(v) or v < 0 for v in (acceptance_m, minimum_m)):
            raise ValueError('tolerancia e minimo invalidos')
        if recovery_frontiers and candidate_validator is None:
            raise ValueError('exploracao de recuperacao exige seu validador')
        def validate(candidate):
            check = executable_candidate_diagnostics(
                self, current, candidate, acceptance_m=acceptance_m, minimum_m=minimum_m,
                require_goal_progress=candidate_validator is None,
            )
            if check['accepted'] and candidate_validator is not None:
                external = candidate_validator(candidate)
                check['recovery_validation'] = external
                if not external['accepted']:
                    check.update(accepted=False, reason=external['reason'])
            candidate.diagnostics['executable_candidate_validation'] = check
            return check
        result = self.plan(
            current, goal, candidate_validator=validate, max_candidates=max_candidates,
            recovery_frontiers=recovery_frontiers)
        result.diagnostics['executable_selection_policy'] = 'bounded_post_standoff_progress_v1'
        return result

    def _initial_connections(self, traversable):
        current = self._connection_current
        origin = self.grid.world_to_voxel(current)
        attempts, edges = [], {}
        for offset in product((-1, 0, 1), repeat=3):
            voxel = tuple(origin[i] + offset[i] for i in range(3))
            if voxel not in traversable:
                continue
            target = self._world(voxel)
            check = self.path_safety_diagnostics(current, [target])
            length = float(np.linalg.norm(target - current))
            attempts.append({'voxel': voxel, 'target_ned_m': tuple(target),
                             'length_m': length, 'safety': check})
            if check['safe']:
                edges[voxel] = length / self.config.voxel_resolution_m
        self._connection_diagnostics = {
            'policy': 'real_pose_validated_local_connectors_v1',
            'maximum_chebyshev_offset_voxels': 1,
            'current_position_ned_m': tuple(current),
            'accepted_count': len(edges), 'attempts': attempts,
        }
        return edges

    def _make_planner(self, traversable, blocked):
        if self.extra_radius_m == 0:
            return super()._make_planner(traversable, blocked)
        return _ReserveAStar(
            traversable, blocked, self.config.connectivity,
            lambda a, b: self._has_reserve([self._world(a), self._world(b)]),
            initial_edges=(self._initial_connections(traversable)
                           if self.use_start_connections else None),
        )

    def _plan_to_subgoal(self, current, requested_goal, selected_goal, planner,
                         traversable, blocked, started, diagnostics,
                         allow_initial_escape=False):
        result = super()._plan_to_subgoal(
            current, requested_goal, selected_goal, planner,
            traversable, blocked, started, diagnostics,
            allow_initial_escape=allow_initial_escape)
        if self._connection_diagnostics is not None:
            result.diagnostics['initial_connections'] = self._connection_diagnostics
            if result.path_voxels:
                entry = np.asarray(self._world(result.path_voxels[0]))
                result.diagnostics['legacy_start_voxel'] = diagnostics['start_voxel']
                result.diagnostics['start_voxel'] = result.path_voxels[0]
                result.diagnostics['selected_initial_connection'] = {
                    'voxel': result.path_voxels[0], 'target_ned_m': tuple(entry),
                    'length_m': float(np.linalg.norm(entry - current)),
                    'safety': self.path_safety_diagnostics(current, [entry]),
                }
        return result

    def _shortcut_path(self, path, traversable, blocked):
        if self.extra_radius_m == 0:
            return super()._shortcut_path(path, traversable, blocked)
        path = list(map(tuple, path))
        if not path:
            return []
        simplified, anchor = [path[0]], 0
        while anchor < len(path) - 1:
            chosen = anchor + 1
            for candidate in range(len(path) - 1, anchor, -1):
                a, b = self._world(path[anchor]), self._world(path[candidate])
                if (all(v in traversable and v not in blocked
                        for v in self.grid.segment_voxels(a, b))
                        and self._has_reserve([a, b])):
                    chosen = candidate
                    break
            simplified.append(path[chosen])
            anchor = chosen
        return simplified

    def path_safety_diagnostics(self, current_position_ned_m, waypoints_ned_m):
        """Combina seguranca conhecida com a reserva geometrica adicional."""
        result = super().path_safety_diagnostics(current_position_ned_m, waypoints_ned_m)
        if result["safe"] and not self._has_reserve([current_position_ned_m, *waypoints_ned_m]):
            result.update(safe=False, failure_reason="execution_reserve",
                          extra_execution_radius_m=self.extra_radius_m)
        return result



def prepare_executable_command(navigator, current, plan, acceptance_m, minimum_m):
    """Mesma poda/entrada estrita do replay; aplicada APOS todo pos-processamento."""
    if any(not np.isfinite(v) or v < 0 for v in (acceptance_m, minimum_m)):
        raise ValueError('tolerancia e comprimento minimo devem ser finitos e nao negativos')
    current = np.asarray(current, dtype=float)
    original = list(plan.waypoints_ned_m) if plan.success else []
    command = list(original)
    while command and np.linalg.norm(np.asarray(command[0]) - current) <= acceptance_m:
        command.pop(0)
    strict = False
    if original and not navigator.path_is_safe(current, command) and navigator.path_is_safe(current, original):
        command, strict = original, True
    safety = navigator.path_safety_diagnostics(current, command)
    points = [current, *command]
    length = sum(float(np.linalg.norm(np.asarray(b) - a)) for a, b in zip(points, points[1:]))
    enough = plan.reason != 'local_subgoal' or length >= minimum_m
    return command, strict, safety, length, bool(plan.success and safety['safe'] and enough)


def executable_candidate_diagnostics(navigator, current, plan, *, acceptance_m,
                                     minimum_m, require_goal_progress=True):
    """Explica por que um plano pos-processado pode ou nao ser executado."""
    command, strict, safety, length, executable = prepare_executable_command(
        navigator, current, plan, acceptance_m, minimum_m)
    current = np.asarray(current, dtype=float)
    goal = np.asarray(plan.requested_goal_ned_m, dtype=float)
    before = float(np.linalg.norm(current - goal))
    remaining = float(np.linalg.norm(np.asarray(command[-1]) - goal)) if command else None
    progress = before - remaining if remaining is not None else None
    result = {
        'accepted': False, 'reason': 'planner_failure',
        'executable_path_length_m': length, 'minimum_executable_m': minimum_m,
        'strict_entry_waypoint_required': strict,
        'executable_path_safety': safety,
        'endpoint_goal_distance_reduction_m': progress,
        'remaining_goal_distance_m': remaining,
        'minimum_endpoint_progress_m': navigator.config.min_subgoal_progress_m,
        'progress_policy': 'endpoint_distance_reduction' if require_goal_progress else 'delegated_recovery_guard',
    }
    if not plan.success:
        return result
    if not safety['safe']:
        result['reason'] = 'unsafe_executable_path'
    elif not executable:
        result['reason'] = 'insufficient_executable_length_after_standoff'
    elif (require_goal_progress and plan.reason == 'local_subgoal'
          and progress + 1e-9 < navigator.config.min_subgoal_progress_m):
        # Long arclength is not evidence of approaching the global objective.
        result['reason'] = 'insufficient_endpoint_progress_after_standoff'
    else:
        result.update(accepted=True, reason='executable_candidate')
    return result


def waypoint_switch_diagnostics(navigator, current, remaining, acceptance_radius_m):
    """Avalia a troca usando a POSE atual, nao a reta nominal nem so a distancia.

    Helper experimental: manter o alvo tambem exige caminho seguro. Nenhuma
    decisao retorna permissao para seguir um segmento conhecido como inseguro.
    """
    if not np.isfinite(acceptance_radius_m) or acceptance_radius_m < 0:
        raise ValueError("raio de aceitacao invalido")
    points = np.asarray([current, *remaining], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("posicoes invalidas")
    if len(remaining) < 2:
        return {"action": "terminal_rules", "switch_allowed": False}
    distance = float(np.linalg.norm(points[0] - points[1]))
    if distance > acceptance_radius_m + 1e-9:
        return {"action": "outside_acceptance", "switch_allowed": False}
    skipped = navigator.path_safety_diagnostics(current, remaining[1:])
    if skipped["safe"]:
        return {"action": "advance", "switch_allowed": True, "outgoing_safety": skipped}
    retained = navigator.path_safety_diagnostics(current, remaining)
    return {"action": "keep_target" if retained["safe"] else "replan_required",
            "switch_allowed": False, "outgoing_safety": skipped,
            "retained_path_safety": retained}
