"""Reexecuta offline uma run espacial e aplica um gate antes dos testes SITL."""

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_mapping.geometry import CameraIntrinsics
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.snapshot import load_planning_snapshot
from spatial_mapping.execution import RecoveryProgressGuard, RecoveryCandidatePolicy


def latest_run(base_dir):
    runs = sorted(base_dir.glob("run_*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"nenhuma run_* encontrada em {base_dir}")
    return runs[-1]


def load_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def navigation_config(
    manifest, replay_stride, vertical_clearance_m=None,
    goal_acceptance_radius_m=None, obstacle_vertical_band_m=None,
):
    saved = manifest.get("metadata", {}).get("spatial_config", {})
    allowed = {item.name for item in fields(SpatialNavigationConfig)}
    values = {key: value for key, value in saved.items() if key in allowed}
    config = SpatialNavigationConfig(**values)
    overrides = {"depth_stride": replay_stride}
    if vertical_clearance_m is not None:
        overrides["drone_vertical_clearance_m"] = vertical_clearance_m
    if obstacle_vertical_band_m is not None:
        overrides["obstacle_vertical_band_m"] = obstacle_vertical_band_m
    if goal_acceptance_radius_m is not None:
        overrides["goal_acceptance_radius_m"] = goal_acceptance_radius_m
    return replace(config, **overrides)


def body_position_from_frame(transform, metadata):
    translation = metadata.get("camera_translation_body_m")
    rotation_body_from_optical = metadata.get(
        "camera_rotation_body_from_optical"
    )
    if translation is None or rotation_body_from_optical is None:
        return np.asarray(transform[:3, 3], dtype=float), True

    translation = np.asarray(translation, dtype=float)
    rotation_body_from_optical = np.asarray(
        rotation_body_from_optical,
        dtype=float,
    )
    rotation_ned_from_body = (
        np.asarray(transform[:3, :3], dtype=float)
        @ rotation_body_from_optical.T
    )
    position = (
        np.asarray(transform[:3, 3], dtype=float)
        - rotation_ned_from_body @ translation
    )
    return position, False


def waypoint_length(current, waypoints):
    points = [np.asarray(current, dtype=float)] + [
        np.asarray(point, dtype=float) for point in waypoints
    ]
    return sum(
        float(np.linalg.norm(end - start))
        for start, end in zip(points, points[1:])
    )


def replay_event_prefix(navigator, run_dir, events, metadata=None):
    """Reproduz frames e mutacoes ego dos planos, na ordem registrada."""
    metadata = metadata or {}

    totals = {
        "points_integrated": 0,
        "free_only_rays": 0,
        "total_valid_rays": 0,
    }
    ego_updates = 0
    ego_positions_missing = 0
    last_transform = None
    for event in events:
        if event.get("event") == "plan" and event.get("map_kind") == "estimated":
            position = ((event.get("plan") or {}).get("diagnostics") or {}).get(
                "current_position_ned_m"
            )
            if position is None:
                ego_positions_missing += 1
            else:
                # plan() mutates only the ego cell before querying the map.
                # Do not rerun historical planning or infer free connecting rays.
                navigator.grid.mark_ego_voxel_free(position)
                ego_updates += 1
        if event.get("event") != "frame" or not event.get("file"):
            continue
        with np.load(run_dir / event["file"]) as frame:
            transform = np.asarray(frame["camera_to_ned"], dtype=float)
            intrinsics = CameraIntrinsics(
                *np.asarray(frame["intrinsics"], dtype=float)
            )
            stats = navigator.integrate_depth(
                np.asarray(frame["estimated_depth_m"], dtype=float),
                intrinsics,
                transform,
                obstacle_reference_ned_z=body_position_from_frame(transform, metadata)[0][2],
            )
        last_transform = transform
        for key in totals:
            totals[key] += int(stats[key])
    return last_transform, {
        **totals,
        "historical_ego_updates": ego_updates,
        "historical_ego_positions_missing": ego_positions_missing,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reprocessa frames salvos com o codigo atual e falha se o ultimo "
            "objetivo nao produzir um caminho executavel."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="diretorio run_*; por padrao usa a run mais recente",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("datasets/spatial_mapping"),
    )
    parser.add_argument(
        "--replay-stride",
        type=int,
        default=9,
        help=(
            "stride nos frames 320x240 salvos; 9 aproxima o stride 35 "
            "usado na imagem original 4x maior"
        ),
    )
    parser.add_argument(
        "--vertical-clearance-m",
        type=float,
        default=None,
        help=(
            "sobrescreve apenas a inflacao vertical para analise de "
            "sensibilidade; os demais parametros permanecem iguais"
        ),
    )
    parser.add_argument(
        "--obstacle-vertical-band-m",
        type=float,
        default=None,
        help="reintegra frames com banda vertical centrada no corpo",
    )
    parser.add_argument(
        "--require-executable-m",
        type=float,
        default=None,
        help="comprimento minimo; por padrao usa o valor do manifest",
    )
    parser.add_argument(
        "--plan-id",
        type=int,
        default=None,
        help="seleciona um plano estimado especifico; por padrao usa o ultimo",
    )
    parser.add_argument(
        "--goal-acceptance-radius-m", type=float, default=None,
        help="tolerancia global explicita para manifests antigos que nao a gravam",
    )
    parser.add_argument(
        "--map-source", choices=("auto", "frames", "snapshot"), default="auto",
        help="auto usa snapshot exato quando gravado; frames e aproximado",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run(args.base_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    metadata = manifest.get("metadata", {})
    execution = metadata.get("execution_safety", {})
    minimum = (
        float(args.require_executable_m)
        if args.require_executable_m is not None
        else float(execution.get("min_executable_path_m", 1.5))
    )
    acceptance = float(execution.get("waypoint_acceptance_radius_m", 0.8))
    config = navigation_config(
        manifest,
        args.replay_stride,
        vertical_clearance_m=args.vertical_clearance_m,
        obstacle_vertical_band_m=args.obstacle_vertical_band_m,
        goal_acceptance_radius_m=args.goal_acceptance_radius_m,
    )
    navigator = SpatialNavigator(config)
    events = load_events(run_dir / "events.jsonl")
    estimated_plan_events = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "plan"
        and event.get("map_kind") == "estimated"
    ]
    if not estimated_plan_events:
        raise RuntimeError("run sem planos estimados")
    def recorded_plan_id(event):
        saved = event.get("plan") or {}
        diagnostics = saved.get("diagnostics") or {}
        for value in (
            event.get("plan_id"),
            saved.get("plan_id"),
            diagnostics.get("plan_id"),
        ):
            if value is not None:
                return int(value)
        return None

    if args.plan_id is None:
        plan_event_index, selected_plan_event = estimated_plan_events[-1]
    else:
        matching = [
            item
            for item in estimated_plan_events
            if recorded_plan_id(item[1]) == args.plan_id
        ]
        if not matching:
            available = [
                recorded_plan_id(event)
                for _, event in estimated_plan_events
            ]
            raise RuntimeError(
                f"plan-id {args.plan_id} nao encontrado; "
                f"disponiveis: {available}"
            )
        plan_event_index, selected_plan_event = matching[-1]

    saved_plan = selected_plan_event["plan"]
    selected_plan_id = recorded_plan_id(selected_plan_event)
    requested_goal = np.asarray(
        saved_plan["requested_goal_ned_m"],
        dtype=float,
    )

    frame_events = [
        event
        for event in events[:plan_event_index]
        if event.get("event") == "frame" and event.get("file")
    ]
    saved_diagnostics = saved_plan.get("diagnostics") or {}
    snapshot_file = saved_diagnostics.get("planning_snapshot_file")
    if args.obstacle_vertical_band_m is not None and args.map_source == "snapshot":
        raise RuntimeError(
            "banda vertical exige reintegracao por frames; snapshot ja contem o mapa"
        )
    use_snapshot = (
        args.map_source != "frames"
        and args.obstacle_vertical_band_m is None
        and bool(snapshot_file)
    )
    if args.map_source == "snapshot" and not snapshot_file:
        raise RuntimeError("plano historico sem snapshot exato; nao use mapa final")
    last_transform = None
    totals = {}
    if use_snapshot:
        if saved_diagnostics.get("planning_snapshot_phase") != "before_plan":
            raise RuntimeError("snapshot nao identificado como anterior ao plano")
        navigator = load_planning_snapshot(run_dir, snapshot_file)
        overrides = {}
        if args.vertical_clearance_m is not None:
            overrides["drone_vertical_clearance_m"] = args.vertical_clearance_m
        if args.goal_acceptance_radius_m is not None:
            overrides["goal_acceptance_radius_m"] = args.goal_acceptance_radius_m
        navigator.config = replace(navigator.config, **overrides)
    else:
        if not frame_events:
            raise RuntimeError("run sem frames salvos")
        last_transform, totals = replay_event_prefix(
            navigator, run_dir, events[:plan_event_index], metadata
        )

    saved_current = (saved_plan.get("diagnostics") or {}).get(
        "current_position_ned_m"
    )
    if saved_current is None:
        if use_snapshot:
            raise RuntimeError("snapshot sem posicao registrada para o plano")
        current, approximate_position = body_position_from_frame(
            last_transform,
            metadata,
        )
    else:
        current = np.asarray(saved_current, dtype=float)
        approximate_position = False
    recorded_policy = saved_diagnostics.get("recovery_candidate_policy")
    if recorded_policy is None:
        plan = navigator.plan(
            current, requested_goal, allow_initial_escape=True
        )
    else:
        policy = RecoveryCandidatePolicy.from_dict(recorded_policy)
        if args.goal_acceptance_radius_m is not None:
            policy.goal_acceptance_m = navigator.config.goal_acceptance_radius_m
        if args.require_executable_m is not None:
            policy.min_executable_m = minimum
        reference_navigator = None
        if policy.reference_required:
            reference_file = saved_diagnostics.get("candidate_reference_snapshot_file")
            if not use_snapshot or not reference_file:
                raise RuntimeError("busca com veto exige snapshot de referencia")
            reference_navigator = load_planning_snapshot(run_dir, reference_file)
        plan = navigator.plan(
            current, requested_goal,
            candidate_validator=lambda candidate: policy.evaluate(
                navigator, current, candidate, reference_navigator
            ),
            max_candidates=policy.max_candidates,
            recovery_frontiers=policy.explore_frontiers,
        )
        plan.diagnostics["recovery_candidate_policy"] = policy.as_dict()
    saved_diagnostics = saved_plan.get("diagnostics") or {}
    saved_path_in_replay = navigator.path_safety_diagnostics(
        current, saved_plan.get("waypoints_ned_m") or []
    )
    original_waypoints = list(plan.waypoints_ned_m) if plan.success else []
    waypoints = list(original_waypoints)
    while (
        waypoints
        and np.linalg.norm(np.asarray(waypoints[0]) - current) <= acceptance
    ):
        waypoints.pop(0)
    attempted_pruned = len(original_waypoints) - len(waypoints)
    initial_escape_required = bool(
        plan.diagnostics.get("initial_escape_required")
    )
    safety_check = (
        navigator.initial_escape_path_safety_diagnostics
        if initial_escape_required
        else navigator.path_safety_diagnostics
    )
    pruned_path_safety = safety_check(current, waypoints)
    original_path_safety = safety_check(current, original_waypoints)
    pruned_path_is_safe = bool(pruned_path_safety["safe"])
    original_path_is_safe = bool(original_path_safety["safe"])
    strict_entry_required = bool(initial_escape_required)
    if (
        not pruned_path_is_safe
        and attempted_pruned > 0
        and original_path_is_safe
    ):
        waypoints = original_waypoints
        strict_entry_required = True
    executable_path_safety = (
        original_path_safety
        if strict_entry_required
        else pruned_path_safety
    )
    executable_path_is_safe = bool(executable_path_safety["safe"])
    executable = waypoint_length(current, waypoints)
    saved_guard = saved_diagnostics.get("recovery_return_guard")
    recovery_guard_check = None
    if saved_guard is not None:
        guard = RecoveryProgressGuard(**{
            item.name: saved_guard[item.name]
            for item in fields(RecoveryProgressGuard)
            if item.name in saved_guard
        })
        recovery_guard_check = guard.evaluate(
            current, waypoints, path_safe=executable_path_is_safe,
            extension_m=float(saved_guard["required_extension_m"]),
            blocked_radius_m=float(saved_guard["blocked_radius_m"]),
            arrival_radius_m=navigator.config.goal_acceptance_radius_m,
        )
    gate_passed = bool(
        plan.success
        and executable_path_is_safe
        and executable >= minimum
        and (recovery_guard_check is None or recovery_guard_check["allowed"])
    )

    result = {
        "run_dir": str(run_dir),
        "diagnostic_only": True,
        "selected_plan_id": selected_plan_id,
        "online_comparison": {
            "exact_online_map_reproduced": use_snapshot,
            "map_source": "snapshot" if use_snapshot else "frames",
            "snapshot_file": snapshot_file if use_snapshot else None,
            "note": (
                "Estado exato do mapa ANTES do A*; nao reproduz voo/controlador."
                if use_snapshot else
                "Frames reduzidos e ordem de gravacao podem divergir da "
                "integracao online; ausencia de conectividade no replay "
                "nao comprova ausencia durante o voo."
            ),
            "saved_plan_success": saved_plan.get("success"),
            "saved_plan_adopted": saved_plan.get("adopted_for_execution"),
            "saved_reachable_voxels": saved_diagnostics.get("reachable_voxels"),
            "replayed_reachable_voxels": plan.diagnostics.get("reachable_voxels"),
            "saved_mapping_frames_integrated": saved_diagnostics.get(
                "mapping_frames_integrated_at_plan"
            ),
            "replayed_mapping_frames_integrated": navigator.frames_integrated,
            "saved_path_safety_in_replayed_map": saved_path_in_replay,
            "last_saved_camera_distance_to_current_m": (
                float(np.linalg.norm(last_transform[:3, 3] - current))
                if last_transform is not None else None
            ),
        },
        "saved_frame_replay": {
            "frames": 0 if use_snapshot else len(frame_events),
            "stride": None if use_snapshot else args.replay_stride,
            "obstacle_vertical_reference": (
                "saved_body_position_ned_z_v1" if not use_snapshot
                else "preintegrated_snapshot"
            ),
            "obstacle_vertical_band_override_m": args.obstacle_vertical_band_m,
            "vertical_clearance_override_m": args.vertical_clearance_m,
            "goal_acceptance_radius_override_m": args.goal_acceptance_radius_m,
            "position_uses_camera_origin_fallback": approximate_position,
            **totals,
        },
        "map": {
            "free_voxels": len(navigator.grid.free_voxels()),
            "occupied_voxels": len(navigator.grid.occupied_voxels()),
        },
        "plan": {
            **asdict(plan),
            "waypoints_pruned_by_acceptance_attempted": attempted_pruned,
            "waypoints_pruned_by_acceptance": (
                len(original_waypoints) - len(waypoints)
            ),
            "pruned_path_is_safe": pruned_path_is_safe,
            "original_path_is_safe": original_path_is_safe,
            "pruned_path_safety": pruned_path_safety,
            "original_path_safety": original_path_safety,
            "executable_path_safety": executable_path_safety,
            "strict_entry_waypoint_required": strict_entry_required,
            "executable_path_is_safe": executable_path_is_safe,
            "executable_path_length_m": executable,
            "recovery_return_guard": recovery_guard_check,
        },
        "gate": {
            "minimum_executable_path_m": minimum,
            "waypoint_acceptance_radius_m": acceptance,
            "requires_safe_known_path": True,
            "requires_post_recovery_progress": saved_guard is not None,
            "passed": gate_passed,
        },
    }
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else value.tolist()
                if isinstance(value, np.ndarray)
                else str(value)
            ),
        )
    )
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
