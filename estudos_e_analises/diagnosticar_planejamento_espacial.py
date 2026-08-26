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


def navigation_config(manifest, replay_stride, vertical_clearance_m=None):
    saved = manifest.get("metadata", {}).get("spatial_config", {})
    allowed = {item.name for item in fields(SpatialNavigationConfig)}
    values = {key: value for key, value in saved.items() if key in allowed}
    config = SpatialNavigationConfig(**values)
    overrides = {"depth_stride": replay_stride}
    if vertical_clearance_m is not None:
        overrides["drone_vertical_clearance_m"] = vertical_clearance_m
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
        "--require-executable-m",
        type=float,
        default=None,
        help="comprimento minimo; por padrao usa o valor do manifest",
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
    plan_event_index, last_plan_event = estimated_plan_events[-1]
    saved_plan = last_plan_event["plan"]
    requested_goal = np.asarray(
        saved_plan["requested_goal_ned_m"],
        dtype=float,
    )

    frame_events = [
        event
        for event in events[:plan_event_index]
        if event.get("event") == "frame" and event.get("file")
    ]
    if not frame_events:
        raise RuntimeError("run sem frames salvos")

    totals = {
        "points_integrated": 0,
        "free_only_rays": 0,
        "total_valid_rays": 0,
    }
    last_transform = None
    for event in frame_events:
        with np.load(run_dir / event["file"]) as frame:
            transform = np.asarray(frame["camera_to_ned"], dtype=float)
            intrinsics = CameraIntrinsics(
                *np.asarray(frame["intrinsics"], dtype=float)
            )
            stats = navigator.integrate_depth(
                np.asarray(frame["estimated_depth_m"], dtype=float),
                intrinsics,
                transform,
            )
        last_transform = transform
        for key in totals:
            totals[key] += int(stats[key])

    saved_current = (saved_plan.get("diagnostics") or {}).get(
        "current_position_ned_m"
    )
    if saved_current is None:
        current, approximate_position = body_position_from_frame(
            last_transform,
            metadata,
        )
    else:
        current = np.asarray(saved_current, dtype=float)
        approximate_position = False
    plan = navigator.plan(current, requested_goal)
    original_waypoints = list(plan.waypoints_ned_m) if plan.success else []
    waypoints = list(original_waypoints)
    while (
        waypoints
        and np.linalg.norm(np.asarray(waypoints[0]) - current) <= acceptance
    ):
        waypoints.pop(0)
    attempted_pruned = len(original_waypoints) - len(waypoints)
    pruned_path_is_safe = bool(
        waypoints and navigator.path_is_safe(current, waypoints)
    )
    original_path_is_safe = bool(
        original_waypoints
        and navigator.path_is_safe(current, original_waypoints)
    )
    strict_entry_required = False
    if (
        not pruned_path_is_safe
        and attempted_pruned > 0
        and original_path_is_safe
    ):
        waypoints = original_waypoints
        strict_entry_required = True
    executable_path_is_safe = bool(
        waypoints and navigator.path_is_safe(current, waypoints)
    )
    executable = waypoint_length(current, waypoints)
    gate_passed = bool(
        plan.success
        and executable_path_is_safe
        and executable >= minimum
    )

    result = {
        "run_dir": str(run_dir),
        "diagnostic_only": True,
        "saved_frame_replay": {
            "frames": len(frame_events),
            "stride": args.replay_stride,
            "vertical_clearance_override_m": args.vertical_clearance_m,
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
            "strict_entry_waypoint_required": strict_entry_required,
            "executable_path_is_safe": executable_path_is_safe,
            "executable_path_length_m": executable,
        },
        "gate": {
            "minimum_executable_path_m": minimum,
            "waypoint_acceptance_radius_m": acceptance,
            "requires_safe_known_path": True,
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
