"""Gera resumo e visualizacao 3D de uma run do pipeline espacial."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_map(path):
    data = np.load(path)
    voxels = np.asarray(data["voxels"], dtype=np.int32)
    log_odds = np.asarray(data["log_odds"], dtype=float)
    resolution = float(data["resolution_m"])
    occupied_threshold = float(data["occupied_threshold"])
    free_threshold = float(data["free_threshold"])
    return {
        "voxels": voxels,
        "log_odds": log_odds,
        "resolution_m": resolution,
        "occupied": voxels[log_odds >= occupied_threshold],
        "free": voxels[log_odds <= free_threshold],
    }


def voxel_set(values):
    return {tuple(int(value) for value in row) for row in values}


def compare_maps(estimated, reference):
    estimated_occupied = voxel_set(estimated["occupied"])
    estimated_free = voxel_set(estimated["free"])
    reference_occupied = voxel_set(reference["occupied"])
    true_positive = len(estimated_occupied & reference_occupied)
    false_positive = len(estimated_occupied - reference_occupied)
    false_negative = len(reference_occupied - estimated_occupied)
    union = len(estimated_occupied | reference_occupied)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "iou": _ratio(true_positive, union),
        "false_free_rate": _ratio(
            len(reference_occupied & estimated_free),
            len(reference_occupied),
        ),
    }


def aggregate_depth_metrics(events):
    values = [
        event["depth_metrics"]
        for event in events
        if event.get("event") == "frame" and event.get("depth_metrics")
    ]
    if not values:
        return {}
    return {
        key: float(np.nanmean([value.get(key, np.nan) for value in values]))
        for key in ("mae_m", "rmse_m", "abs_rel")
    }


def aggregate_mapping_metrics(events):
    maps = [
        event.get("map") or {}
        for event in events
        if event.get("event") == "frame"
        and (event.get("map") or {}).get("integration_time_ms") is not None
    ]
    if not maps:
        return {}
    times = [float(mapping["integration_time_ms"]) for mapping in maps]
    summary = {
        "mean_integration_time_ms": float(np.mean(times)),
        "p95_integration_time_ms": float(np.percentile(times, 95)),
    }
    filtered = [
        mapping
        for mapping in maps
        if mapping.get("points_rejected_vertical") is not None
    ]
    if filtered:
        integrated = sum(int(mapping.get("points_integrated", 0)) for mapping in filtered)
        rejected = sum(
            int(mapping.get("points_rejected_vertical", 0))
            for mapping in filtered
        )
        summary.update(
            {
                "mean_points_integrated": float(
                    np.mean([mapping.get("points_integrated", 0) for mapping in filtered])
                ),
                "mean_points_rejected_vertical": float(
                    np.mean(
                        [mapping.get("points_rejected_vertical", 0) for mapping in filtered]
                    )
                ),
                "vertical_rejection_rate": _ratio(
                    rejected,
                    integrated + rejected,
                ),
            }
        )
    return summary


def load_trajectory(run_dir, events):
    """Carrega a posicao da camera salva em cada frame espacial."""

    frame_events = [
        event
        for event in events
        if event.get("event") == "frame" and event.get("file")
    ]
    positions = []
    timestamps = []
    for event in frame_events:
        with np.load(run_dir / event["file"]) as frame:
            positions.append(
                np.asarray(frame["camera_to_ned"], dtype=float)[:3, 3]
            )
        timestamps.append(float(event["timestamp_s"]))
    return np.asarray(positions, dtype=float), np.asarray(timestamps, dtype=float)


def movement_metrics(positions, timestamps, nominal_length=None):
    """Resume distancia, velocidade, altitude e mudancas de direcao."""

    if len(positions) < 2:
        return {}

    positions = np.asarray(positions)
    timestamps = np.asarray(timestamps)
    segments = np.diff(positions, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    traveled = float(np.sum(lengths))
    duration = max(0.0, timestamps[-1] - timestamps[0])

    turning_points = [positions[0]]
    for position in positions[1:-1]:
        if np.linalg.norm(position - turning_points[-1]) >= 0.25:
            turning_points.append(position)
    turning_points.append(positions[-1])
    moving = np.diff(np.asarray(turning_points), axis=0)
    moving = moving[np.linalg.norm(moving, axis=1) > 0.15]
    total_turn_deg = 0.0
    if len(moving) >= 2:
        unit = moving / np.linalg.norm(moving, axis=1, keepdims=True)
        cosines = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
        total_turn_deg = float(np.degrees(np.arccos(cosines)).sum())

    return {
        "duration_s": duration,
        "distance_traveled_m": traveled,
        "nominal_route_length_m": nominal_length,
        "distance_over_nominal": _ratio(traveled, nominal_length),
        "mean_sampled_speed_m_s": _ratio(traveled, duration),
        "altitude_range_m": float(np.ptp(-positions[:, 2])),
        "altitude_std_m": float(np.std(-positions[:, 2])),
        "total_turn_deg": total_turn_deg,
        "turn_deg_per_meter": _ratio(total_turn_deg, traveled),
    }


def mission_boundary_index(event, positions, timestamps, *, prefer_last=False):
    """Localiza um marco da missao mesmo com relogios de camera e ROS desalinhados."""

    if event is None:
        return len(positions) - 1 if prefer_last else 0

    event_position = event.get("position_ned_m")
    if event_position is not None:
        distances = np.linalg.norm(
            positions - np.asarray(event_position, dtype=float),
            axis=1,
        )
        tolerance = max(0.25, float(np.min(distances)) + 0.10)
        candidates = np.flatnonzero(distances <= tolerance)
        if len(candidates):
            return int(candidates[-1] if prefer_last else candidates[0])

    event_timestamp = float(event.get("timestamp_s", timestamps[-1]))
    return int(np.argmin(np.abs(timestamps - event_timestamp)))


def event_frame_index(events, target_event, frame_count, *, before_event=False):
    """Converte a posicao de um evento no JSONL para o indice de frame."""

    if target_event is None:
        return None
    seen_frames = 0
    for event in events:
        if event is target_event:
            index = seen_frames - 1 if before_event else seen_frames
            return int(np.clip(index, 0, frame_count - 1))
        if event.get("event") == "frame" and event.get("file"):
            seen_frames += 1
    return None


def trajectory_metrics(positions, timestamps, events, manifest):
    """Separa a missao completa do trecho entre decolagem e objetivo final."""

    if len(positions) < 2:
        return {}
    metadata = manifest.get("metadata", {})
    route = metadata.get("waypoints_relative_m") or []
    full_route = np.asarray([[0.0, 0.0, 0.0], *route], dtype=float)
    full_nominal = float(
        np.linalg.norm(np.diff(full_route, axis=0), axis=1).sum()
    )
    takeoff_altitude = float(
        metadata.get(
            "takeoff_altitude_m",
            abs(route[0][2]) if route else 0.0,
        )
    )
    cruise_route = np.asarray(
        [[0.0, 0.0, -takeoff_altitude], *route],
        dtype=float,
    )
    cruise_nominal = float(
        np.linalg.norm(np.diff(cruise_route, axis=0), axis=1).sum()
    )

    states = [event for event in events if event.get("event") == "mission_state"]
    takeoff = next(
        (event for event in states if event.get("state") == "takeoff_complete"),
        None,
    )
    mission_complete = next(
        (event for event in states if event.get("state") == "mission_complete"),
        None,
    )
    if takeoff is None:
        takeoff = next(
            (
                event
                for event in events
                if event.get("event") == "plan"
                and event.get("map_kind") == "estimated"
            ),
            None,
        )
    start_index = event_frame_index(
        events,
        takeoff,
        len(positions),
        before_event=False,
    )
    end_index = event_frame_index(
        events,
        mission_complete,
        len(positions),
        before_event=True,
    )
    bounds_source = "mission_event_order"
    if start_index is None:
        start_index = mission_boundary_index(
            takeoff,
            positions,
            timestamps,
            prefer_last=False,
        )
        bounds_source = "mission_state_position_or_time"
    if end_index is None:
        end_index = mission_boundary_index(
            mission_complete,
            positions,
            timestamps,
            prefer_last=True,
        )
        bounds_source = "mission_state_position_or_time"
    if end_index <= start_index:
        start_index = 0
        end_index = len(positions) - 1
        bounds_source = "full_mission_fallback"
    cruise_slice = slice(start_index, end_index + 1)

    return {
        "full_mission": movement_metrics(
            positions,
            timestamps,
            full_nominal,
        ),
        "cruise": movement_metrics(
            positions[cruise_slice],
            timestamps[cruise_slice],
            cruise_nominal,
        ),
        "cruise_bounds_source": bounds_source,
        "cruise_frame_range": [start_index, end_index],
    }


def last_successful_path(events, map_kind="estimated"):
    for event in reversed(events):
        plan = event.get("plan") or {}
        if (
            event.get("event") == "plan"
            and event.get("map_kind") == map_kind
            and plan.get("success")
        ):
            return np.asarray(plan.get("waypoints_ned_m", []), dtype=float)
    return np.empty((0, 3), dtype=float)


def planning_metrics(events, reference):
    plans = [event for event in events if event.get("event") == "plan"]
    summary = {}
    for map_kind in ("estimated", "reference"):
        selected = [event.get("plan") or {} for event in plans if event.get("map_kind") == map_kind]
        successful = [plan for plan in selected if plan.get("success")]
        adoption_decisions = [
            plan
            for plan in selected
            if plan.get("adopted_for_execution") is not None
        ]
        adopted = sum(
            plan.get("adopted_for_execution") is True
            for plan in adoption_decisions
        )
        failure_reasons = Counter(
            plan.get("reason", "motivo ausente")
            for plan in selected
            if not plan.get("success")
        )
        summary[map_kind] = {
            "attempts": len(selected),
            "successes": len(successful),
            "success_rate": _ratio(len(successful), len(selected)),
            "adopted_plans": adopted,
            "adoption_rate": _ratio(adopted, len(adoption_decisions)),
            "failure_reasons": dict(failure_reasons),
            "mean_path_length_m": (
                float(np.mean([plan.get("path_length_m", 0.0) for plan in successful]))
                if successful
                else None
            ),
            "mean_raw_path_length_m": (
                float(
                    np.mean(
                        [
                            plan.get("raw_path_length_m", plan.get("path_length_m", 0.0))
                            for plan in successful
                        ]
                    )
                )
                if successful
                else None
            ),
            "mean_smoothing_ratio": (
                float(
                    np.mean(
                        [
                            _ratio(
                                plan.get("path_length_m", 0.0),
                                plan.get("raw_path_length_m", 0.0),
                            )
                            for plan in successful
                            if plan.get("raw_path_length_m", 0.0) > 0.0
                        ]
                    )
                )
                if any(plan.get("raw_path_length_m", 0.0) > 0.0 for plan in successful)
                else None
            ),
            "mean_frontier_standoff_applied_m": (
                float(
                    np.mean(
                        [
                            plan.get("frontier_standoff_applied_m", 0.0)
                            for plan in successful
                        ]
                    )
                )
                if successful
                else None
            ),
            "mean_planning_time_ms": (
                float(np.mean([plan.get("planning_time_ms", 0.0) for plan in selected]))
                if selected
                else None
            ),
        }

    reference_occupied = voxel_set(reference["occupied"])
    estimated_successful = [
        event.get("plan") or {}
        for event in plans
        if event.get("map_kind") == "estimated"
        and (event.get("plan") or {}).get("success")
    ]
    collision_counts = [
        len(voxel_set(plan.get("path_voxels", [])) & reference_occupied)
        for plan in estimated_successful
    ]
    summary["estimated"]["reference_collision_free_rate"] = _ratio(
        sum(count == 0 for count in collision_counts),
        len(collision_counts),
    )
    summary["estimated"]["mean_reference_occupied_voxels_on_path"] = (
        float(np.mean(collision_counts)) if collision_counts else None
    )
    return summary


def recovery_metrics(events):
    """Resume retiradas acionadas depois de uma falha sem caminho seguro."""

    started = [
        event
        for event in events
        if event.get("event") == "mission_state"
        and event.get("state") == "recovery_started"
    ]
    completed = sum(
        event.get("event") == "mission_state"
        and event.get("state") == "recovery_complete"
        for event in events
    )
    return {
        "attempts": len(started),
        "completed": completed,
        "completion_rate": _ratio(completed, len(started)),
        "trigger_reasons": dict(
            Counter(
                event.get("plan_failure_reason", "motivo ausente")
                for event in started
            )
        ),
    }


def map_points(mapping, max_points=30000):
    occupied = mapping["occupied"]
    if len(occupied) > max_points:
        indices = np.linspace(0, len(occupied) - 1, max_points).astype(int)
        occupied = occupied[indices]
    return (occupied.astype(float) + 0.5) * mapping["resolution_m"]


def build_figure(
    estimated,
    reference,
    estimated_path,
    reference_path,
    flown_trajectory,
):
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Mapa estimado", "Mapa ideal do Gazebo"),
    )
    for column, mapping, path, color in (
        (1, estimated, estimated_path, "#d1495b"),
        (2, reference, reference_path, "#3066be"),
    ):
        points = map_points(mapping)
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0] if len(points) else [],
                y=points[:, 1] if len(points) else [],
                z=points[:, 2] if len(points) else [],
                mode="markers",
                marker={"size": 2, "color": color, "opacity": 0.55},
                name="voxels ocupados",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        if len(path):
            figure.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    line={"color": "#2a9d55", "width": 6},
                    marker={"size": 3},
                    name="ultimo caminho A*",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        if len(flown_trajectory):
            figure.add_trace(
                go.Scatter3d(
                    x=flown_trajectory[:, 0],
                    y=flown_trajectory[:, 1],
                    z=flown_trajectory[:, 2],
                    mode="lines",
                    line={"color": "#111111", "width": 5},
                    name="trajetoria voada",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
    figure.update_scenes(
        xaxis_title="Norte (m)",
        yaxis_title="Leste (m)",
        zaxis_title="Down (m)",
        aspectmode="data",
    )
    figure.update_layout(
        title="Validacao espacial: mapa estimado x referencia",
        height=760,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    estimated = load_map(run_dir / "estimated_map.npz")
    reference = load_map(run_dir / "reference_map.npz")
    events = load_events(run_dir / "events.jsonl")
    manifest = load_manifest(run_dir / "manifest.json")
    flown_trajectory, trajectory_timestamps = load_trajectory(run_dir, events)
    estimated_path = last_successful_path(events, "estimated")
    reference_path = last_successful_path(events, "reference")
    summary = {
        "run_dir": str(run_dir),
        "depth": aggregate_depth_metrics(events),
        "mapping": aggregate_mapping_metrics(events),
        "trajectory": trajectory_metrics(
            flown_trajectory,
            trajectory_timestamps,
            events,
            manifest,
        ),
        "occupancy": compare_maps(estimated, reference),
        "planning": planning_metrics(events, reference),
        "recovery": recovery_metrics(events),
        "frames": sum(event.get("event") == "frame" for event in events),
        "plans": sum(event.get("event") == "plan" for event in events),
        "successful_plans": sum(
            event.get("event") == "plan" and (event.get("plan") or {}).get("success")
            for event in events
        ),
    }

    (run_dir / "spatial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_figure(
        estimated,
        reference,
        estimated_path,
        reference_path,
        flown_trajectory,
    ).write_html(
        run_dir / "spatial_map_3d.html",
        include_plotlyjs=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"HTML: {run_dir / 'spatial_map_3d.html'}")


if __name__ == "__main__":
    main()
