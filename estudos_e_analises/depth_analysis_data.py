"""Adaptador dos datasets RGB-profundidade atuais para as analises tabulares."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def list_analysis_run_dirs(project_root: Path) -> list[Path]:
    """Prioriza as runs espaciais completas e usa o memmap legado como fallback."""

    spatial_dir = Path(project_root) / "datasets" / "spatial_mapping"
    spatial = []
    for run_dir in sorted(spatial_dir.glob("run_*")):
        manifest = _manifest(run_dir)
        if (
            manifest.get("schema_version") == "spatial_mapping_v1"
            and manifest.get("complete") is True
            and (run_dir / "events.jsonl").exists()
            and len(list((run_dir / "frames").glob("frame_*.npz"))) >= 2
        ):
            spatial.append(run_dir)
    if spatial:
        return spatial

    legacy_dir = Path(project_root) / "datasets" / "depth_ground_truth"
    return sorted(path.parent for path in legacy_dir.glob("run_*/manifest.json"))


def _depth_stats(depth_m: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    values = np.asarray(depth_m[valid], dtype=np.float32)
    if not values.size:
        return {key: np.nan for key in (
            "min", "mean", "p10", "p50", "p90", "close_2", "close_5", "close_10"
        )} | {"valid_pct": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "close_2": float((values < 2.0).mean() * 100.0),
        "close_5": float((values < 5.0).mean() * 100.0),
        "close_10": float((values < 10.0).mean() * 100.0),
        "valid_pct": float(valid.mean() * 100.0),
    }


def _flow_features(previous_bgr: np.ndarray, current_bgr: np.ndarray) -> dict[str, float]:
    previous = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        previous, maxCorners=180, qualityLevel=0.01, minDistance=12, blockSize=8
    )
    empty = {
        "flow_valid_points": 0.0, "flow_active_points": 0.0,
        "flow_track_retention_pct": 0.0, "flow_mean_x_px": 0.0,
        "flow_mean_y_px": 0.0, "flow_mag_mean_px": 0.0,
        "flow_mag_p90_px": 0.0, "radial_flow_mean_px": 0.0,
        "radial_flow_p90_px": 0.0, "point_risk_mean_delta_basis": 0.0,
        "point_risk_p75_delta_basis": 0.0, "flow_vec_rel_std": 0.0,
        "flow_vec_xy_mean": 0.0, "flow_vec_radial_mean": 0.0,
        "flow_vec_risk_mean": 0.0,
    }
    if points is None:
        return empty
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.04),
    )
    if next_points is None or status is None:
        return empty
    old = points[status.ravel() == 1].reshape(-1, 2)
    new = next_points[status.ravel() == 1].reshape(-1, 2)
    height, width = previous.shape
    inside = (
        (new[:, 0] >= 0) & (new[:, 0] < width)
        & (new[:, 1] >= 0) & (new[:, 1] < height)
    )
    old, new = old[inside], new[inside]
    if not len(new):
        return empty
    flow = new - old
    magnitude = np.linalg.norm(flow, axis=1)
    center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
    rays = old - center
    ray_norm = np.linalg.norm(rays, axis=1)
    radial = np.sum(flow * rays, axis=1) / np.maximum(ray_norm, 1.0)
    risk = np.maximum(radial, 0.0)
    return {
        "flow_valid_points": float(len(flow)),
        "flow_active_points": float(np.count_nonzero(magnitude > 1.0)),
        "flow_track_retention_pct": float(len(flow) / max(len(points), 1) * 100.0),
        "flow_mean_x_px": float(flow[:, 0].mean()),
        "flow_mean_y_px": float(flow[:, 1].mean()),
        "flow_mag_mean_px": float(magnitude.mean()),
        "flow_mag_p90_px": float(np.percentile(magnitude, 90)),
        "radial_flow_mean_px": float(radial.mean()),
        "radial_flow_p90_px": float(np.percentile(radial, 90)),
        "point_risk_mean_delta_basis": float(risk.mean()),
        "point_risk_p75_delta_basis": float(np.percentile(risk, 75)),
        "flow_vec_rel_std": float(flow.std()),
        "flow_vec_xy_mean": float(flow.mean()),
        "flow_vec_radial_mean": float(radial.mean()),
        "flow_vec_risk_mean": float(risk.mean()),
    }


def _frame_events(run_dir: Path) -> list[dict]:
    events = []
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        frame_path = run_dir / str(event.get("file", ""))
        if event.get("event") == "frame" and frame_path.exists():
            events.append(event)
    return sorted(events, key=lambda event: (float(event.get("timestamp_s", 0.0)), int(event.get("frame_id", 0))))


def _load_frame(run_dir: Path, event: dict) -> dict:
    with np.load(run_dir / event["file"]) as frame:
        return {
            "rgb": np.asarray(frame["rgb_bgr"], dtype=np.uint8),
            "depth": np.asarray(
                frame["reference_depth_m"]
                if "reference_depth_m" in frame.files
                else frame["estimated_depth_m"],
                dtype=np.float32,
            ),
            "transform": np.asarray(frame["camera_to_ned"], dtype=float),
        }


def load_spatial_analysis_run(run_dir: Path) -> dict:
    """Converte frames espaciais consecutivos em intervalos RGB-depth compactos."""

    events = _frame_events(run_dir)
    rows, feature_rows = [], []
    if len(events) < 2:
        return {"run_dir": run_dir, "manifesto": _manifest(run_dir),
                "intervalos": pd.DataFrame(), "precomputed_features": pd.DataFrame()}

    previous_event = events[0]
    previous = _load_frame(run_dir, previous_event)
    previous_stats = _depth_stats(previous["depth"])
    for sample_id, current_event in enumerate(events[1:], start=1):
        current = _load_frame(run_dir, current_event)
        current_stats = _depth_stats(current["depth"])
        dt_s = float(current_event.get("timestamp_s", 0.0)) - float(previous_event.get("timestamp_s", 0.0))
        if not np.isfinite(dt_s) or not (0.0 < dt_s <= 10.0):
            previous_event, previous, previous_stats = current_event, current, current_stats
            continue

        translation = current["transform"][:3, 3] - previous["transform"][:3, 3]
        relative_rotation = previous["transform"][:3, :3].T @ current["transform"][:3, :3]
        rotation_vector, _ = cv2.Rodrigues(relative_rotation)
        rotation = rotation_vector.ravel()
        flow = _flow_features(previous["rgb"], current["rgb"])
        image_delta = current["rgb"].astype(np.float32) - previous["rgb"].astype(np.float32)
        absolute_delta = np.abs(image_delta)
        depth_age = float((current_event.get("map") or {}).get("estimated_age_s", 0.0))
        row = {
            "sample_id": sample_id, "dt_s": dt_s, "depth_age_s": depth_age,
            "depth_dt_s": dt_s, "delta_x_m": float(translation[0]),
            "delta_y_m": float(translation[1]), "delta_z_m": float(translation[2]),
            "delta_roll_rad": float(rotation[0]), "delta_pitch_rad": float(rotation[1]),
            "delta_yaw_heading_rad": float(rotation[2]),
            "delta_yaw_attitude_rad": float(rotation[2]),
            "pan_comp_delta_rad": 0.0, "source_format": "spatial_mapping_v1",
            **{key: float(value) for key, value in flow.items() if not key.startswith("flow_vec_")},
        }
        for axis, index in zip(("x", "y", "z"), range(3)):
            angular_rate = float(rotation[index] / dt_s)
            row[f"delta_gyro_{axis}_rad_s"] = angular_rate
            row[f"gyro_{axis}_integral_rad"] = float(rotation[index])
            row[f"delta_accel_{axis}_m_s2"] = 0.0
        for key in ("min", "mean", "p10", "p50", "p90", "close_2", "close_5", "close_10"):
            suffix = f"{key}_m" if key in {"min", "mean", "p10", "p50", "p90"} else f"{key}m_pp"
            row[f"delta_depth_{suffix}"] = float(current_stats[key] - previous_stats[key])
        row["delta_valid_px_pct"] = float(current_stats["valid_pct"] - previous_stats["valid_pct"])
        rows.append(row)
        feature_rows.append({
            "img_delta_mean": float(image_delta.mean()),
            "img_delta_std": float(image_delta.std()),
            "img_delta_abs_mean": float(absolute_delta.mean()),
            "img_delta_abs_p90": float(np.percentile(absolute_delta, 90)),
            **{key: value for key, value in flow.items() if key.startswith("flow_vec_")},
        })
        previous_event, previous, previous_stats = current_event, current, current_stats

    intervals = pd.DataFrame(rows)
    if not intervals.empty:
        intervals["run_id"] = run_dir.name
        intervals["ordem_intervalo"] = np.arange(len(intervals))
    return {
        "run_dir": run_dir,
        "manifesto": _manifest(run_dir),
        "intervalos": intervals,
        "precomputed_features": pd.DataFrame(feature_rows),
    }


def load_analysis_run(run_dir: Path) -> dict:
    manifest = _manifest(run_dir)
    if manifest.get("schema_version") == "spatial_mapping_v1":
        return load_spatial_analysis_run(run_dir)
    raise ValueError(f"Schema nao suportado pelo adaptador atual: {manifest.get('schema_version')}")


@lru_cache(maxsize=2)
def load_all_analysis_runs(project_root: str) -> list[dict]:
    runs = []
    for run_dir in list_analysis_run_dirs(Path(project_root)):
        try:
            run = load_analysis_run(run_dir)
        except Exception as error:
            print(f"Run ignorada em {run_dir.name}: {error}")
            continue
        if not run["intervalos"].empty:
            runs.append(run)
    return runs
