#!/usr/bin/env python3
"""Reconstroi mapas e planos monoculares contra referencia sem executar voo."""

import argparse
import hashlib
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_mapping.depth_model import MetricDepthOnnx
from spatial_mapping.geometry import CameraIntrinsics
from spatial_mapping.metrics import occupancy_metrics
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator


def load_config(
    manifest,
    replay_stride,
    free_space_margin_m,
    edge_stride=None,
    edge_relative_threshold=None,
    obstacle_vertical_band_m=None,
    free_observations_required=None,
    occupied_observations_required=None,
    occupied_support_radius_voxels=None,
    pending_clear_free_observations_required=None,
    occupied_evidence_window_frames=None,
    occupied_uncertainty_m=None,
    lock_path_altitude_to_goal=False,
):
    saved = manifest.get("metadata", {}).get("spatial_config", {})
    allowed = {item.name for item in fields(SpatialNavigationConfig)}
    config = SpatialNavigationConfig(
        **{key: value for key, value in saved.items() if key in allowed}
    )
    return replace(
        config,
        depth_stride=replay_stride,
        depth_edge_stride=(
            config.depth_edge_stride
            if edge_stride is None
            else edge_stride
        ),
        depth_edge_relative_threshold=(
            config.depth_edge_relative_threshold
            if edge_relative_threshold is None
            else edge_relative_threshold
        ),
        depth_free_space_margin_m=free_space_margin_m,
        obstacle_vertical_band_m=(
            config.obstacle_vertical_band_m
            if obstacle_vertical_band_m is None
            else obstacle_vertical_band_m
        ),
        free_observations_required=(
            config.free_observations_required
            if free_observations_required is None
            else free_observations_required
        ),
        occupied_observations_required=(
            config.occupied_observations_required
            if occupied_observations_required is None
            else occupied_observations_required
        ),
        occupied_support_radius_voxels=(
            config.occupied_support_radius_voxels
            if occupied_support_radius_voxels is None
            else occupied_support_radius_voxels
        ),
        pending_clear_free_observations_required=(
            config.pending_clear_free_observations_required
            if pending_clear_free_observations_required is None
            else pending_clear_free_observations_required
        ),
        occupied_evidence_window_frames=(
            config.occupied_evidence_window_frames
            if occupied_evidence_window_frames is None
            else occupied_evidence_window_frames
        ),
        depth_occupied_uncertainty_m=(
            config.depth_occupied_uncertainty_m
            if occupied_uncertainty_m is None
            else occupied_uncertainty_m
        ),
        lock_path_altitude_to_goal=lock_path_altitude_to_goal,
    )


def depth_error_counts(predicted_depth, reference_depth):
    predicted = np.asarray(predicted_depth, dtype=float)
    reference = np.asarray(reference_depth, dtype=float)
    if predicted.shape != reference.shape:
        predicted = cv2.resize(
            predicted.astype(np.float32),
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(float)
    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > 0.0)
        & (reference > 0.0)
    )
    error = predicted - reference
    vertical = np.zeros_like(valid)
    horizontal = np.zeros_like(valid)
    vertical[1:, :] = (
        np.abs(reference[1:, :] - reference[:-1, :])
        > 0.10 * np.maximum(reference[1:, :], 1e-6)
    )
    horizontal[:, 1:] = (
        np.abs(reference[:, 1:] - reference[:, :-1])
        > 0.10 * np.maximum(reference[:, 1:], 1e-6)
    )
    edges = valid & (vertical | horizontal)
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "absolute_error_sum_m": float(np.abs(error[valid]).sum()),
        "signed_error_sum_m": float(error[valid].sum()),
        "dangerous_far_pixels_1m": int(np.count_nonzero(valid & (error > 1.0))),
        "edge_pixels": int(np.count_nonzero(edges)),
        "dangerous_far_edge_pixels_1m": int(
            np.count_nonzero(edges & (error > 1.0))
        ),
    }


def summarize_depth_error(counts):
    valid = sum(item["valid_pixels"] for item in counts)
    edges = sum(item["edge_pixels"] for item in counts)
    return {
        "frames": len(counts),
        "valid_pixels": valid,
        "mae_m": (
            sum(item["absolute_error_sum_m"] for item in counts) / valid
            if valid else None
        ),
        "mean_signed_error_m": (
            sum(item["signed_error_sum_m"] for item in counts) / valid
            if valid else None
        ),
        "dangerous_far_rate_1m": (
            sum(item["dangerous_far_pixels_1m"] for item in counts) / valid
            if valid else None
        ),
        "dangerous_far_edge_rate_1m": (
            sum(item["dangerous_far_edge_pixels_1m"] for item in counts) / edges
            if edges else None
        ),
    }


def obstacle_proximity(navigator, position, radius_m=5.0):
    occupied = navigator.grid.occupied_voxels()
    if not occupied:
        return {"occupied_within_radius": 0, "nearest_distance_m": None}
    position = np.asarray(position, dtype=float)
    voxels = list(occupied)
    centers = np.asarray(
        [navigator.grid.voxel_to_world(voxel) for voxel in voxels],
        dtype=float,
    )
    offsets = centers - position
    distances = np.linalg.norm(offsets, axis=1)
    nearest_index = int(np.argmin(distances))
    nearby_dz = offsets[distances <= radius_m, 2]
    return {
        "occupied_within_radius": int(len(nearby_dz)),
        "nearest_distance_m": float(distances[nearest_index]),
        "nearest_voxel": [int(value) for value in voxels[nearest_index]],
        "nearest_world_ned_m": centers[nearest_index].tolist(),
        "vertical_counts_within_radius": {
            "above_more_than_0_4m": int(np.count_nonzero(nearby_dz < -0.4)),
            "flight_band_abs_0_4m": int(np.count_nonzero(np.abs(nearby_dz) <= 0.4)),
            "below_more_than_0_4m": int(np.count_nonzero(nearby_dz > 0.4)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prediction-cache-dir",
        type=Path,
        help="Reutiliza previsoes .npy do mesmo modelo/run entre configuracoes.",
    )
    parser.add_argument("--replay-stride", type=int, default=9)
    parser.add_argument("--edge-stride", type=int)
    parser.add_argument("--edge-relative-threshold", type=float)
    parser.add_argument(
        "--rgb-edge-sampling",
        action="store_true",
        help="Complementa bordas de profundidade com Canny sobre o RGB.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--depth-output-scale", type=float, default=1.0)
    parser.add_argument("--free-space-margin-m", type=float, default=0.0)
    parser.add_argument("--conservative-depth-shift-m", type=float, default=0.0)
    parser.add_argument("--obstacle-vertical-band-m", type=float)
    parser.add_argument("--free-observations-required", type=int)
    parser.add_argument("--occupied-observations-required", type=int)
    parser.add_argument("--occupied-support-radius-voxels", type=int)
    parser.add_argument("--pending-clear-free-observations-required", type=int)
    parser.add_argument("--occupied-evidence-window-frames", type=int)
    parser.add_argument("--occupied-uncertainty-m", type=float, default=0.0)
    parser.add_argument(
        "--minimum-executable-path-m",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--lock-path-altitude-to-goal",
        action="store_true",
    )
    args = parser.parse_args()
    if args.depth_output_scale <= 0:
        parser.error("--depth-output-scale deve ser positivo")

    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    estimated_config = load_config(
        manifest,
        args.replay_stride,
        args.free_space_margin_m,
        args.edge_stride,
        args.edge_relative_threshold,
        args.obstacle_vertical_band_m,
        args.free_observations_required,
        args.occupied_observations_required,
        args.occupied_support_radius_voxels,
        args.pending_clear_free_observations_required,
        args.occupied_evidence_window_frames,
        args.occupied_uncertainty_m,
        args.lock_path_altitude_to_goal,
    )
    reference_config = load_config(
        manifest,
        args.replay_stride,
        0.0,
        args.edge_stride,
        args.edge_relative_threshold,
        args.obstacle_vertical_band_m,
        args.free_observations_required,
        1,
        0,
        args.pending_clear_free_observations_required,
        args.occupied_evidence_window_frames,
        0.0,
        args.lock_path_altitude_to_goal,
    )
    estimated = SpatialNavigator(estimated_config)
    reference = SpatialNavigator(reference_config)
    model = MetricDepthOnnx(
        args.model,
        input_width=686,
        input_height=518,
        backend="onnxruntime",
        max_depth_m=80.0,
        input_aspect_tolerance=0.03,
    )
    events = [
        json.loads(line)
        for line in (args.run_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    frames = 0
    plans = []
    depth_errors_since_plan = []
    for event in events:
        if event.get("event") == "frame" and event.get("file"):
            if args.max_frames is not None and frames >= args.max_frames:
                break
            with np.load(args.run_dir / event["file"]) as frame:
                bgr = frame["rgb_bgr"]
                reference_depth = frame["reference_depth_m"].astype(float)
                transform = frame["camera_to_ned"].astype(float)
                intrinsics = CameraIntrinsics(
                    *frame["intrinsics"].astype(float)
                )
            cache_file = None
            if args.prediction_cache_dir is not None:
                cache_file = (
                    args.prediction_cache_dir
                    / f"{Path(event['file']).stem}.npy"
                )
            if cache_file is not None and cache_file.is_file():
                predicted_depth = np.load(cache_file).astype(float)
            else:
                predicted_depth = model.predict(bgr)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_file.with_suffix(".tmp.npy")
                    np.save(temporary, predicted_depth.astype(np.float32))
                    temporary.replace(cache_file)
            predicted_depth = predicted_depth * args.depth_output_scale
            predicted_edges = None
            reference_edges = None
            if args.rgb_edge_sampling:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                rgb_edges = cv2.Canny(gray, 50, 150) > 0
                rgb_edges = cv2.dilate(
                    rgb_edges.astype(np.uint8), np.ones((3, 3), np.uint8)
                ).astype(bool)
                predicted_edges = cv2.resize(
                    rgb_edges.astype(np.uint8),
                    (predicted_depth.shape[1], predicted_depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                reference_edges = cv2.resize(
                    rgb_edges.astype(np.uint8),
                    (reference_depth.shape[1], reference_depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if args.conservative_depth_shift_m > 0:
                valid_prediction = predicted_depth > 0.0
                predicted_depth[valid_prediction] = np.maximum(
                    predicted_depth[valid_prediction]
                    - args.conservative_depth_shift_m,
                    0.0,
                )
            depth_errors_since_plan.append(
                depth_error_counts(predicted_depth, reference_depth)
            )
            estimated.integrate_depth(
                predicted_depth,
                intrinsics,
                transform,
                sampling_edge_mask=predicted_edges,
            )
            reference.integrate_depth(
                reference_depth,
                intrinsics,
                transform,
                sampling_edge_mask=reference_edges,
            )
            frames += 1
        elif event.get("event") == "plan" and event.get("map_kind") == "estimated":
            saved_plan = event.get("plan") or {}
            diagnostics = saved_plan.get("diagnostics") or {}
            current = diagnostics.get("current_position_ned_m")
            goal = saved_plan.get("requested_goal_ned_m")
            if current is None or goal is None or frames < 1:
                continue
            plan = estimated.plan(current, goal)
            safe_reference = bool(
                plan.success
                and plan.waypoints_ned_m
                and reference.path_is_safe(current, plan.waypoints_ned_m)
            )
            collision_free_reference = False
            first_collision_voxel = None
            first_collision_world_ned_m = None
            if plan.success and plan.waypoints_ned_m:
                blocked_reference = reference._inflated_obstacles()
                points = [np.asarray(current, dtype=float)] + [
                    np.asarray(point, dtype=float)
                    for point in plan.waypoints_ned_m
                ]
                path_voxels = []
                for start, end in zip(points, points[1:]):
                    for voxel in reference.grid._ray_voxels(start, end):
                        if not path_voxels or voxel != path_voxels[-1]:
                            path_voxels.append(voxel)
                reached_free = False
                reentered_blocked = False
                for voxel in path_voxels:
                    if voxel in blocked_reference:
                        if reached_free:
                            reentered_blocked = True
                            first_collision_voxel = voxel
                            first_collision_world_ned_m = (
                                reference.grid.voxel_to_world(voxel).tolist()
                            )
                            break
                    else:
                        reached_free = True
                collision_free_reference = reached_free and not reentered_blocked
            estimated_proximity = obstacle_proximity(estimated, current)
            reference_proximity = obstacle_proximity(reference, current)
            depth_error = summarize_depth_error(depth_errors_since_plan)
            executable = bool(
                plan.success
                and plan.path_length_m >= args.minimum_executable_path_m
            )
            plans.append(
                {
                    "frames_integrated": frames,
                    "free_space_margin_m": args.free_space_margin_m,
                    "conservative_depth_shift_m": (
                        args.conservative_depth_shift_m
                    ),
                    "success": plan.success,
                    "executable": executable,
                    "reason": plan.reason,
                    "path_length_m": plan.path_length_m,
                    "current_position_ned_m": current,
                    "requested_goal_ned_m": goal,
                    "waypoints_ned_m": plan.waypoints_ned_m,
                    "safe_in_reference": safe_reference,
                    "collision_free_in_reference": collision_free_reference,
                    "first_collision_voxel": (
                        None
                        if first_collision_voxel is None
                        else [int(value) for value in first_collision_voxel]
                    ),
                    "first_collision_world_ned_m": first_collision_world_ned_m,
                    "estimated_state_at_collision": (
                        None
                        if first_collision_voxel is None
                        else estimated.grid.state(first_collision_voxel)
                    ),
                    "current_voxel_inflated": plan.diagnostics.get(
                        "current_voxel_inflated"
                    ),
                    "reachable_voxels": plan.diagnostics.get(
                        "reachable_voxels", 0
                    ),
                    "current_voxel_state": plan.diagnostics.get(
                        "current_voxel_state"
                    ),
                    "neighbor_states_26": plan.diagnostics.get(
                        "neighbor_states_26"
                    ),
                    "blocked_neighbors_26": plan.diagnostics.get(
                        "blocked_neighbors_26"
                    ),
                    "max_reachable_progress_m": plan.diagnostics.get(
                        "max_reachable_progress_m"
                    ),
                    "estimated_obstacle_proximity": estimated_proximity,
                    "reference_obstacle_proximity": reference_proximity,
                    "depth_error_since_previous_plan": depth_error,
                }
            )
            depth_errors_since_plan.clear()

    occupied = occupancy_metrics(estimated.grid, reference.grid)
    successful = [
        plan for plan in plans
        if plan["success"] and plan["executable"]
    ]
    collision_free = sum(
        int(plan["collision_free_in_reference"]) for plan in successful
    )
    limits = {
        "minimum_frames": 100,
        "minimum_executable_path_m": args.minimum_executable_path_m,
        "minimum_plan_success_rate": 0.80,
        "maximum_false_free_rate": 0.10,
        "require_all_successful_plans_collision_free": True,
    }
    success_rate = len(successful) / len(plans) if plans else 0.0
    checks = {
        "frames": frames >= limits["minimum_frames"],
        "plan_success_rate": success_rate
        >= limits["minimum_plan_success_rate"],
        "false_free_rate": occupied["false_free_rate"]
        <= limits["maximum_false_free_rate"],
        "collision_free_plans": collision_free == len(successful)
        and bool(successful),
    }
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    result = {
        "run": str(args.run_dir.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": digest,
        "frames_integrated": frames,
        "depth_output_scale": args.depth_output_scale,
        "free_space_margin_m": args.free_space_margin_m,
        "edge_stride": estimated_config.depth_edge_stride,
        "rgb_edge_sampling": args.rgb_edge_sampling,
        "edge_relative_threshold": estimated_config.depth_edge_relative_threshold,
        "obstacle_vertical_band_m": estimated_config.obstacle_vertical_band_m,
        "free_observations_required": (
            estimated_config.free_observations_required
        ),
        "occupied_observations_required": (
            estimated_config.occupied_observations_required
        ),
        "occupied_support_radius_voxels": (
            estimated_config.occupied_support_radius_voxels
        ),
        "pending_clear_free_observations_required": (
            estimated_config.pending_clear_free_observations_required
        ),
        "occupied_evidence_window_frames": (
            estimated_config.occupied_evidence_window_frames
        ),
        "lock_path_altitude_to_goal": (
            estimated_config.lock_path_altitude_to_goal
        ),
        "occupied_uncertainty_m": estimated_config.depth_occupied_uncertainty_m,
        "map": occupied,
        "estimated_map": {
            "free_voxels": len(estimated.grid.free_voxels()),
            "occupied_voxels": len(estimated.grid.occupied_voxels()),
            "pending_occupied_voxels": len(
                estimated.grid.pending_occupied_voxels()
            ),
        },
        "reference_map": {
            "free_voxels": len(reference.grid.free_voxels()),
            "occupied_voxels": len(reference.grid.occupied_voxels()),
            "pending_occupied_voxels": len(
                reference.grid.pending_occupied_voxels()
            ),
        },
        "planning": {
            "attempts": len(plans),
            "successes": len(successful),
            "safe_successes": sum(
                int(plan["safe_in_reference"]) for plan in successful
            ),
            "collision_free_successes": collision_free,
            "success_rate": success_rate,
            "plans": plans,
        },
        "qualification": {
            "passed": all(checks.values()),
            "limits": limits,
            "checks": checks,
        },
    }
    encoded = json.dumps(result, indent=2, allow_nan=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
