"""Politicas geometricas de execucao, testaveis sem ROS ou PX4."""

from dataclasses import dataclass, asdict, field, replace

import numpy as np


def _distance_to_polyline(point, points):
    if len(points) == 1:
        return float(np.linalg.norm(point - points[0]))
    minimum = float("inf")
    for start, end in zip(points, points[1:]):
        delta = end - start
        square = float(np.dot(delta, delta))
        t = np.clip(np.dot(point - start, delta) / square, 0, 1) if square else 0
        minimum = min(minimum, float(np.linalg.norm(point - (start + t * delta))))
    return minimum


def segment_tracking_diagnostics(current, start, end, velocity=None):
    """Mede a pose contra UM segmento comandado, sem usar waypoints futuros."""
    current, start, end = (
        np.asarray(value, dtype=float) for value in (current, start, end)
    )
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (current, start, end)):
        raise ValueError('segmento/pose devem ser vetores 3D finitos')
    delta = end - start
    square = float(np.dot(delta, delta))
    raw_fraction = (
        float(np.dot(current - start, delta) / square) if square > 0 else 0.0
    )
    fraction = float(np.clip(raw_fraction, 0.0, 1.0))
    projection = start + fraction * delta
    result = {
        'segment_start_ned_m': tuple(start),
        'segment_end_ned_m': tuple(end),
        'segment_length_m': float(np.sqrt(square)),
        'projection_fraction': fraction,
        'unclipped_projection_fraction': raw_fraction,
        'along_track_m': fraction * float(np.sqrt(square)),
        'cross_track_m': float(np.linalg.norm(current - projection)),
        'distance_to_target_m': float(np.linalg.norm(current - end)),
        'overshot_segment': bool(raw_fraction > 1.0),
    }
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('velocidade deve ser vetor 3D finito')
        result.update(
            velocity_ned_m_s=tuple(velocity),
            speed_m_s=float(np.linalg.norm(velocity)),
        )
    return result


def waypoint_advance_decision(
    navigator, current, path, index, acceptance_radius_m, *,
    terminal_arrival_pending=False, allow_initial_escape=False,
):
    """So troca o alvo se o caminho restante for seguro desde a pose REAL."""
    if not np.isfinite(acceptance_radius_m) or acceptance_radius_m < 0:
        raise ValueError('raio de aceitacao invalido')
    current = np.asarray(current, dtype=float)
    points = [np.asarray(point, dtype=float) for point in path]
    if (current.shape != (3,) or not np.isfinite(current).all()
            or any(p.shape != (3,) or not np.isfinite(p).all() for p in points)):
        raise ValueError('pose/caminho invalidos')
    if not 0 <= index < len(points):
        return {'action': 'no_target', 'advance': False, 'reason': 'path_exhausted'}
    distance = float(np.linalg.norm(current - points[index]))
    result = {
        'action': 'keep_target', 'advance': False,
        'reason': 'outside_acceptance', 'target_index': int(index),
        'distance_to_target_m': distance,
        'acceptance_radius_m': float(acceptance_radius_m),
    }
    if distance > acceptance_radius_m + 1e-9:
        return result
    if terminal_arrival_pending:
        result['reason'] = 'terminal_arrival_pending'
        return result
    if index == len(points) - 1:
        result.update(action='advance', advance=True, reason='final_target_accepted')
        return result
    def path_safety(remaining):
        if not allow_initial_escape:
            return navigator.path_safety_diagnostics(current, remaining)
        safe = bool(
            navigator.path_avoids_obstacles_allowing_initial_escape(
                current, remaining
            )
        )
        return {
            'safe': safe,
            'failure_reason': None if safe else 'obstacle_reentry',
            'policy': 'initial_escape_no_reentry',
        }

    outgoing = path_safety(points[index + 1:])
    result['outgoing_path_safety'] = outgoing
    if outgoing['safe']:
        result.update(action='advance', advance=True, reason='outgoing_path_safe')
        return result
    retained = path_safety(points[index:])
    result['retained_path_safety'] = retained
    if retained['safe']:
        result['reason'] = 'unsafe_early_switch_keep_target'
    else:
        result.update(action='hold_and_replan', reason='active_path_unsafe')
    return result


@dataclass(frozen=True)
class RecoveryExitWindow:
    """Extensao unica para uma saida ativa, com progresso medido, nao previsto.

    A polilinha fica fixa durante a execucao. Consumir waypoints, alternar a
    pose ou receber um novo plano nao soma progresso nem renova o prazo.
    """
    path_key: tuple = ()
    points: tuple = ()
    progress_m: float = 0.0
    best_progress_m: float = 0.0
    milestone_m: float = 0.0
    cross_track_m: float = 0.0
    last_progress_s: float | None = None
    grace_deadline_s: float | None = None
    grace_path_key: tuple = ()

    GRACE_LIMIT_S = 6.0
    STALL_LIMIT_S = 3.0

    def __post_init__(self):
        for name in ('path_key', 'points', 'grace_path_key'):
            object.__setattr__(self, name, tuple(tuple(p) for p in getattr(self, name)))

    def observe(self, current, path, index, now_s, progress_step_m):
        """Atualiza progresso medido na polilinha NED sem reiniciar o prazo."""
        key = tuple(tuple(float(v) for v in p) for p in path) if index < len(path) else ()
        if (not key or not np.isfinite([current, *key]).all()
                or not np.isfinite(now_s) or progress_step_m <= 0):
            return replace(self, path_key=(), points=(), progress_m=0.,
                           best_progress_m=0., milestone_m=0., last_progress_s=None)
        current = np.asarray(current, dtype=float)
        if key != self.path_key:
            return replace(self, path_key=key, points=(tuple(current), *key[index:]),
                           progress_m=0., best_progress_m=0., milestone_m=0.,
                           cross_track_m=0., last_progress_s=float(now_s))
        nearest_distance, progress, arc = float('inf'), 0., 0.
        for a, b in zip(self.points, self.points[1:]):
            start, end = np.asarray(a), np.asarray(b)
            delta = end - start
            square = float(np.dot(delta, delta))
            t = float(np.clip(np.dot(current - start, delta) / square, 0., 1.)) if square else 0.
            distance = float(np.linalg.norm(current - start - t * delta))
            length = float(np.sqrt(square))
            # Earliest projection wins ties at crossings: conservative evidence.
            if distance < nearest_distance - 1e-9:
                nearest_distance, progress = distance, arc + t * length
            arc += length
        best = max(self.best_progress_m, progress)
        advanced = best >= self.milestone_m + progress_step_m
        return replace(self, progress_m=progress, best_progress_m=best,
                       cross_track_m=nearest_distance,
                       milestone_m=best if advanced else self.milestone_m,
                       last_progress_s=float(now_s) if advanced else self.last_progress_s)

    def decide(self, now_s, *, base_deadline_s, path_safe, endpoint_progress_m,
               arrival, min_progress_m, corridor_m):
        """Decide se a janela limitada de saida continua segura e produtiva."""
        deadline = (self.grace_deadline_s if self.grace_deadline_s is not None
                    else base_deadline_s + self.GRACE_LIMIT_S)
        reason = (
            'exit_grace_expired' if now_s >= deadline
            else 'exit_path_unsafe' if not path_safe
            else 'exit_path_missing' if not self.path_key
            else 'exit_path_changed' if (self.grace_deadline_s is not None
                                         and self.path_key != self.grace_path_key)
            else 'exit_without_goal_progress' if not arrival and endpoint_progress_m < min_progress_m
            else 'exit_off_corridor' if self.cross_track_m > corridor_m
            else 'exit_regressed' if self.best_progress_m - self.progress_m > corridor_m
            else 'exit_no_measured_progress' if self.best_progress_m < min_progress_m
            else 'exit_stalled' if (self.last_progress_s is None
                                    or now_s - self.last_progress_s >= self.STALL_LIMIT_S)
            else 'active_exit_grace'
        )
        allowed = reason == 'active_exit_grace'
        granted = allowed and self.grace_deadline_s is None
        updated = (replace(self, grace_deadline_s=deadline, grace_path_key=self.path_key)
                   if granted else self)
        return updated, {
            'allowed': allowed, 'reason': reason, 'granted_now': granted,
            'base_deadline_s': base_deadline_s, 'grace_deadline_s': deadline,
            'grace_limit_s': self.GRACE_LIMIT_S, 'stall_limit_s': self.STALL_LIMIT_S,
            'path_safe': bool(path_safe), 'endpoint_progress_m': float(endpoint_progress_m),
            'measured_path_progress_m': self.progress_m,
            'best_path_progress_m': self.best_progress_m,
            'cross_track_m': self.cross_track_m, 'last_progress_s': self.last_progress_s,
        }


@dataclass
class RecoveryProgressGuard:
    """Memoria de fronteira esgotada; aceitar um plano nao apaga a memoria."""

    goal_index: int
    goal_ned_m: tuple
    stopped_position_ned_m: tuple
    stopped_endpoint_ned_m: tuple
    observation_started_s: float | None = None
    # Immutable tuples allow replace(guard) to freeze a worker's evidence.
    # Only measured positions belong here, never unexecuted planned waypoints.
    visited_positions_ned_m: tuple = ()
    history_saturated: bool = False
    waiting_elapsed_s: float = 0.0
    active_path_elapsed_s: float = 0.0
    last_observation_s: float | None = None
    last_path_missing: bool = True
    exit_window: RecoveryExitWindow = field(default_factory=RecoveryExitWindow)

    MAX_TRACE_POINTS = 512

    def __post_init__(self):
        if isinstance(self.exit_window, dict):
            self.exit_window = RecoveryExitWindow(**self.exit_window)
        self.visited_positions_ned_m = tuple(
            tuple(float(v) for v in p) for p in self.visited_positions_ned_m
        )

    def record_position(self, current, spacing_m=.2):
        """Memoria limitada sem expulsar corredores antigos (falha fechada)."""
        point = np.asarray(current, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or spacing_m <= 0:
            raise ValueError('pose/espacamento invalido na memoria de recuperacao')
        if (self.visited_positions_ned_m and
            np.linalg.norm(point - self.visited_positions_ned_m[-1]) < spacing_m):
            return
        if len(self.visited_positions_ned_m) >= self.MAX_TRACE_POINTS:
            self.history_saturated = True
            return
        self.visited_positions_ned_m += (tuple(float(v) for v in point),)

    def observe(self, current, now_s, *, path_missing, spacing_m=.2):
        """Separa espera de caminho ativo; nenhum deles reinicia o prazo total."""
        if self.observation_started_s is None:
            return
        previous = (self.last_observation_s if self.last_observation_s is not None
                    else self.observation_started_s)
        if now_s < previous:
            return
        elapsed = now_s - previous
        if self.last_path_missing:
            self.waiting_elapsed_s += elapsed
        else:
            self.active_path_elapsed_s += elapsed
        self.last_observation_s = float(now_s)
        self.last_path_missing = bool(path_missing)
        self.record_position(current, spacing_m=spacing_m)

    def as_dict(self):
        return asdict(self)

    def endpoint_progress(self, endpoint):
        """Retorna a reducao, em metros, da distancia ao objetivo NED."""
        goal = np.asarray(self.goal_ned_m, dtype=float)
        baseline = min(
            np.linalg.norm(np.asarray(self.stopped_position_ned_m) - goal),
            np.linalg.norm(np.asarray(self.stopped_endpoint_ned_m) - goal),
        )
        return float(baseline - np.linalg.norm(np.asarray(endpoint) - goal))

    def actual_progress_reached(self, current, extension_m, arrival_radius_m):
        """Confirma extensao mensuravel ou chegada ao objetivo global."""
        return bool(
            self.endpoint_progress(current) >= extension_m
            or np.linalg.norm(np.asarray(current) - self.goal_ned_m) <= arrival_radius_m
        )

    def observation_expired(self, now_s, limit_s):
        """Indica se o prazo total de observacao terminou, sem reinicios."""
        return (
            self.observation_started_s is not None
            and now_s - self.observation_started_s >= limit_s
        )

    def evaluate(self, current, waypoints, *, path_safe, extension_m,
                 blocked_radius_m, arrival_radius_m):
        """Exige extensao real ou desvio lateral sem revisitar o bloqueio.

        Nao substitui o gate de espaco conhecido/inflacao. Frame novo, caminho
        mais comprido ou tempo decorrido, por si so, nao representam progresso.
        """
        result = {**self.as_dict(), "allowed": False, "reason": "unsafe_or_empty_path"}
        if not path_safe or not waypoints:
            return result
        points = [np.asarray(current, dtype=float)] + [
            np.asarray(v, dtype=float) for v in waypoints
        ]
        if not np.isfinite(points).all():
            return result
        stop = np.asarray(self.stopped_position_ned_m, dtype=float)
        goal = np.asarray(self.goal_ned_m, dtype=float)
        endpoint = points[-1]
        direction = goal - stop
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else np.zeros(3)
        offset = endpoint - stop
        lateral = float(np.linalg.norm(offset - np.dot(offset, direction) * direction))
        separation = _distance_to_polyline(stop, points)
        endpoint_shift = float(np.linalg.norm(endpoint - self.stopped_endpoint_ned_m))
        progress = self.endpoint_progress(endpoint)
        arrived = np.linalg.norm(endpoint - goal) <= arrival_radius_m
        visited = [np.asarray(p) for p in self.visited_positions_ned_m]
        endpoint_novelty = _distance_to_polyline(endpoint, visited) if visited else None
        revisited = endpoint_novelty is not None and endpoint_novelty < extension_m
        detour_geometry = (lateral >= extension_m and endpoint_shift >= extension_m
                           and separation > blocked_radius_m)
        detour = detour_geometry and not revisited and not self.history_saturated
        start_distance = float(np.linalg.norm(points[0] - stop))
        endpoint_distance = float(np.linalg.norm(endpoint - stop))
        initial_escape = (
            start_distance <= blocked_radius_m
            and endpoint_distance > blocked_radius_m
        )
        corridor_clear = separation > blocked_radius_m or initial_escape
        progress_exit = progress >= extension_m and corridor_clear
        arrival_exit = arrived and corridor_clear
        reason = (
            "terminal_arrival" if arrival_exit
            else "reentry_into_recovery_blockage" if not corridor_clear
            else "new_endpoint_progress" if progress_exit
            else "safe_lateral_detour" if detour
            else "recovery_history_limit" if self.history_saturated
            else "revisited_recovery_corridor" if detour_geometry and revisited
            else "same_exhausted_frontier"
        )
        result.update(
            allowed=bool(arrival_exit or progress_exit or detour),
            reason=reason, endpoint_progress_m=progress,
            lateral_displacement_m=lateral, endpoint_shift_m=endpoint_shift,
            minimum_distance_to_stop_m=separation,
            recovery_blockage_cleared=bool(corridor_clear),
            recovery_initial_escape=bool(initial_escape),
            recovery_start_distance_m=start_distance,
            recovery_endpoint_distance_m=endpoint_distance,
            required_extension_m=float(extension_m),
            blocked_radius_m=float(blocked_radius_m),
            visited_position_count=len(visited),
            endpoint_distance_to_visited_m=endpoint_novelty,
        )
        return result



@dataclass
class RecoveryCandidatePolicy:
    """Validacao compartilhada entre busca online e replay de snapshots."""

    guard: RecoveryProgressGuard
    waypoint_acceptance_m: float
    min_executable_m: float
    extension_m: float
    goal_acceptance_m: float
    reference_required: bool = False
    max_candidates: int = 16
    explore_frontiers: bool = False  # Legacy snapshots retain their original policy.

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["guard"] = RecoveryProgressGuard(**values["guard"])
        return cls(**values)

    def evaluate(self, navigator, current, plan, reference_navigator=None):
        """Aplica os gates de execucao e referencia a um plano candidato."""
        result = {"accepted": False, "reason": "planner_failure"}
        if not plan.success:
            return result
        current = np.asarray(current, dtype=float)
        original = list(plan.waypoints_ned_m)
        waypoints = list(original)
        while (waypoints and
               np.linalg.norm(np.asarray(waypoints[0]) - current)
               <= self.waypoint_acceptance_m):
            waypoints.pop(0)
        pruned_safe = bool(waypoints and navigator.path_is_safe(current, waypoints))
        original_safe = bool(original and navigator.path_is_safe(current, original))
        strict_entry = not pruned_safe and len(waypoints) < len(original) and original_safe
        if strict_entry:
            waypoints = original
        safe = bool(waypoints and (original_safe if strict_entry else pruned_safe))
        points = [current] + [np.asarray(p) for p in waypoints]
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
        result.update(
            executable_path_is_safe=safe, executable_path_length_m=length,
            strict_entry_waypoint_required=bool(strict_entry),
        )
        if not safe:
            result["reason"] = "unsafe_executable_path"
            return result
        if plan.reason == "local_subgoal" and length < self.min_executable_m:
            result["reason"] = "insufficient_executable_length"
            return result
        check = self.guard.evaluate(
            current, waypoints, path_safe=safe, extension_m=self.extension_m,
            blocked_radius_m=self.waypoint_acceptance_m,
            arrival_radius_m=self.goal_acceptance_m,
        )
        result["recovery_return_guard"] = check
        if not check["allowed"]:
            result["reason"] = check["reason"]
            return result
        if self.explore_frontiers and check['reason'] == 'safe_lateral_detour':
            frontier = navigator.observation_frontier_diagnostics(
                current, waypoints, plan.selected_goal_ned_m)
            result['observation_frontier'] = frontier
            plan.diagnostics['observation_frontier'] = frontier
            if not frontier['observable']:
                result['reason'] = frontier['reason']
                return result
        if self.reference_required:
            reference_safe = (
                reference_navigator is not None
                and reference_navigator.path_avoids_obstacles_allowing_initial_escape(
                    current, waypoints
                )
            )
            result["reference_path_avoids_obstacles"] = bool(reference_safe)
            if not reference_safe:
                result["reason"] = "reference_veto"
                return result
        result.update(accepted=True, reason=check["reason"])
        return result


def terminal_arrival_pending(current, target, goal, radius_m):
    """Nao consumir o ultimo setpoint de chegada antes de entrar na regiao."""
    return bool(
        np.linalg.norm(np.asarray(target) - np.asarray(goal)) <= radius_m
        and np.linalg.norm(np.asarray(current) - np.asarray(goal)) > radius_m
    )


def evaluate_path_handoff(
    current, existing, candidate, goal, *, existing_safe, candidate_safe,
    acceptance_m, min_extension_m, require_extension,
):
    """Preserva continuidade somente enquanto o caminho restante e seguro.

    Nao modela o controlador PX4: evita trocar um trecho ainda ativo por uma
    direcao oposta (>90 graus). A seguranca vem das verificacoes do chamador
    no mapa/pose atuais, nunca de um resultado antigo do planejador.
    """
    current = np.asarray(current, dtype=float)
    old = [np.asarray(p, dtype=float) for p in existing]
    new = [np.asarray(p, dtype=float) for p in candidate]
    finite = all(np.isfinite(p).all() for p in [current, *old, *new, np.asarray(goal)])
    old_safe = bool(finite and old and existing_safe)
    new_safe = bool(finite and new and candidate_safe)
    result = {
        "action": "reject", "reason": "no_safe_path",
        "existing_path_safe": old_safe, "candidate_path_safe": new_safe,
        "turn_angle_deg": None, "endpoint_progress_gain_m": None,
    }
    if not new_safe:
        if old_safe:
            result.update(action="preserve", reason="candidate_unsafe_keep_safe_path")
        return result
    result.update(action="adopt", reason="safe_candidate")
    if not old_safe:
        return result
    gain = float(np.linalg.norm(old[-1] - goal) - np.linalg.norm(new[-1] - goal))
    result["endpoint_progress_gain_m"] = gain
    # Ignore already-reached waypoints for heading comparison, NOT for safety.
    old_direction = next((p[:2] - current[:2] for p in old
                          if np.linalg.norm(p[:2] - current[:2]) > acceptance_m), None)
    new_direction = next((p[:2] - current[:2] for p in new
                          if np.linalg.norm(p[:2] - current[:2]) > acceptance_m), None)
    if old_direction is not None and new_direction is not None:
        cosine = float(np.dot(old_direction, new_direction)
                       / (np.linalg.norm(old_direction) * np.linalg.norm(new_direction)))
        result["turn_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        if cosine < -1e-9:
            result.update(action="preserve", reason="opposite_direction_keep_safe_path")
            return result
    if require_extension and gain < min_extension_m:
        result.update(action="preserve", reason="insufficient_extension_keep_safe_path")
    return result


def select_recovery_path(
    current, history, retreat_m, history_spacing_m, final_acceptance_m
):
    """Escolhe recuo por deslocamento liquido, incluindo a tolerancia final.

    O chamador ainda precisa validar o caminho e o destino no mapa atual.
    Somar segmentos de um historico em zigue-zague nao comprova afastamento.
    """
    current = np.asarray(current, dtype=float)
    previous = current
    waypoints = []
    arc = 0.0
    displacement = 0.0
    required_target = retreat_m + final_acceptance_m
    selected = False
    for saved in reversed(list(history)):
        saved = np.asarray(saved, dtype=float)
        if not np.all(np.isfinite(saved)):
            continue
        segment = float(np.linalg.norm(saved - previous))
        if segment < history_spacing_m * 0.5:
            continue
        waypoints.append(tuple(saved))
        arc += segment
        previous = saved
        displacement = float(np.linalg.norm(saved - current))
        if displacement >= required_target:
            selected = True
            break
    return (waypoints if selected else []), {
        "selected": selected,
        "path_length_m": arc,
        "target_displacement_m": displacement,
        "minimum_displacement_m": float(retreat_m),
        "required_target_displacement_m": float(required_target),
        "final_acceptance_radius_m": float(final_acceptance_m),
        "path_evidence": "previously_flown_history",
    }

def recovery_brake_decision(
    elapsed_s, speed_m_s, altitude_drift_m, minimum_s, maximum_s,
    speed_threshold_m_s, max_altitude_drift_m,
):
    """Freia antes do recuo; prazo/deriva falham fechado em vez de inverter."""
    settle = goal_settle_decision(
        elapsed_s, speed_m_s, minimum_s, maximum_s, speed_threshold_m_s,
    )
    if not np.isfinite(altitude_drift_m) or altitude_drift_m < 0:
        raise ValueError("deriva vertical deve ser finita e nao negativa")
    if not np.isfinite(max_altitude_drift_m) or max_altitude_drift_m <= 0:
        raise ValueError("limite de deriva vertical invalido")
    abort_reason = (
        "recovery_brake_altitude_drift"
        if altitude_drift_m > max_altitude_drift_m
        else "recovery_brake_timeout"
        if settle["deadline_reached"] and not settle["speed_below_threshold"]
        else None
    )
    return {
        **settle,
        "waiting": abort_reason is None and settle["waiting"],
        "ready": abort_reason is None and not settle["waiting"],
        "abort": abort_reason is not None,
        "abort_reason": abort_reason,
        "altitude_drift_m": float(altitude_drift_m),
        "max_altitude_drift_m": float(max_altitude_drift_m),
    }


def goal_settle_decision(
    elapsed_s, speed_m_s, minimum_s, maximum_s, speed_threshold_m_s
):
    """Decide uma espera limitada antes de inverter/trocar a perna global."""

    values = np.asarray(
        [elapsed_s, speed_m_s, minimum_s, maximum_s, speed_threshold_m_s],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("parametros de estabilizacao devem ser finitos")
    if elapsed_s < 0 or speed_m_s < 0 or minimum_s < 0:
        raise ValueError("tempo e velocidade nao podem ser negativos")
    if maximum_s < minimum_s or speed_threshold_m_s <= 0:
        raise ValueError("limites de estabilizacao invalidos")
    minimum_elapsed = elapsed_s >= minimum_s
    stopped = speed_m_s <= speed_threshold_m_s
    deadline_reached = elapsed_s >= maximum_s
    waiting = not minimum_elapsed or (not stopped and not deadline_reached)
    return {
        "waiting": waiting,
        "minimum_elapsed": minimum_elapsed,
        "speed_below_threshold": stopped,
        "deadline_reached": deadline_reached,
        "elapsed_s": float(elapsed_s),
        "speed_m_s": float(speed_m_s),
    }


def plan_publication_decision(
    request_generation, live_generation, recovery_active, path_available
):
    """Impede que resultado assíncrono antigo substitua uma recuperação."""

    request_generation = int(request_generation)
    live_generation = int(live_generation)
    stale = request_generation != live_generation or bool(recovery_active)
    return {
        "publish": not stale,
        "stale": stale,
        "preserve_active_path": bool(stale and path_available),
        "reason": (
            "recovery_active" if recovery_active
            else "generation_changed" if request_generation != live_generation
            else "current"
        ),
        "request_generation": request_generation,
        "live_generation": live_generation,
        "recovery_active": bool(recovery_active),
        "path_available": bool(path_available),
    }
