"""Gera resumo e visualizacao 3D de uma run do pipeline espacial."""

import argparse
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
    times = [
        float((event.get("map") or {}).get("integration_time_ms"))
        for event in events
        if event.get("event") == "frame"
        and (event.get("map") or {}).get("integration_time_ms") is not None
    ]
    if not times:
        return {}
    return {
        "mean_integration_time_ms": float(np.mean(times)),
        "p95_integration_time_ms": float(np.percentile(times, 95)),
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
        summary[map_kind] = {
            "attempts": len(selected),
            "successes": len(successful),
            "success_rate": _ratio(len(successful), len(selected)),
            "mean_path_length_m": (
                float(np.mean([plan.get("path_length_m", 0.0) for plan in successful]))
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


def map_points(mapping, max_points=30000):
    occupied = mapping["occupied"]
    if len(occupied) > max_points:
        indices = np.linspace(0, len(occupied) - 1, max_points).astype(int)
        occupied = occupied[indices]
    return (occupied.astype(float) + 0.5) * mapping["resolution_m"]


def build_figure(estimated, reference, estimated_path, reference_path):
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
    estimated_path = last_successful_path(events, "estimated")
    reference_path = last_successful_path(events, "reference")
    summary = {
        "run_dir": str(run_dir),
        "depth": aggregate_depth_metrics(events),
        "mapping": aggregate_mapping_metrics(events),
        "occupancy": compare_maps(estimated, reference),
        "planning": planning_metrics(events, reference),
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
    ).write_html(
        run_dir / "spatial_map_3d.html",
        include_plotlyjs=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"HTML: {run_dir / 'spatial_map_3d.html'}")


if __name__ == "__main__":
    main()
