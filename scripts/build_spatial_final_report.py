#!/usr/bin/env python3
"""Build a reproducible final spatial experiment evidence bundle."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BATTERY = Path(
    "logs/spatial_battery/20260831_082359_ground_truth_debug/summary.tsv"
)
MODEL_INPUTS = {
    "v12": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v12_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v12_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v12_flight_envelope_01/"
            "reports/v12__v12_flight_envelope/heldout_old.json"
        ),
    },
    "v13": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v13_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v13_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v13_validation_01/"
            "reports/v13__v13_flight_envelope/heldout_old.json"
        ),
    },
    "v14": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v14_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v14_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v14_validation_01/"
            "reports/v14__v14_flight_envelope/reference_reserved_03.json"
        ),
    },
    "v15": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v15_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v15_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v15_validation_01/"
            "reports/v15__v15_flight_envelope/reference_reserved_03.json"
        ),
    },
    "v16": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v16_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v16_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v16_validation_01/"
            "reports/v16__v16_flight_envelope/reference_reserved_03.json"
        ),
    },
    "v17": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v17_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v17_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v17_validation_01/"
            "reports/v17__v17_flight_envelope/reference_reserved_03.json"
        ),
    },
    "v18": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v18_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v18_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v18_validation_01/"
            "reports/v18__v18_shift0p0/reference_reserved_03.json"
        ),
    },
    "v19": {
        "model": "models/depth_anything_v2_metric_baylands_vits_v19_final_686x518_fp32.onnx",
        "depth": "models/monocular_validation_v19_final_independent.json",
        "spatial": (
            "logs/spatial_offline_sweep/manual_v19_final_validation_01/"
            "reports/v19_final__v19_scale1p0_shift1p2/"
            "reference_reserved_03.json"
        ),
    },
}
FREEZE_INPUTS = [
    "config/spatial_debug.yaml",
    "config/spatial_monocular.yaml",
    "config/monocular_depth_onnx.yaml",
    "config/depth_v13_training.json",
    "config/depth_v14_training.json",
    "config/spatial_v12_validation_sweep.json",
    "config/spatial_v13_validation_sweep.json",
    "config/spatial_v14_validation_sweep.json",
    "config/depth_v15_training.json",
    "config/depth_v16_training.json",
    "config/depth_v17_training.json",
    "config/depth_v18_training.json",
    "config/depth_v19_final_training.json",
    "config/spatial_v15_validation_sweep.json",
    "config/spatial_v16_validation_sweep.json",
    "config/spatial_v17_validation_sweep.json",
    "config/spatial_v18_validation_sweep.json",
    "config/spatial_v19_final_validation_sweep.json",
    "logs/spatial_offline_sweep/manual_v19_final_validation_01/ranking.csv",
    "logs/spatial_offline_sweep/manual_v19_final_validation_01/summary.csv",
    "models/depth_anything_v2_metric_baylands_vits_v19_final.training.json",
    str(REFERENCE_BATTERY),
]


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def state_time(states, name):
    event = next((item for item in states if item.get("state") == name), None)
    return float(event["timestamp_s"]) if event and event.get("timestamp_s") else None


def reference_rows(summary_path):
    with summary_path.open(newline="", encoding="utf-8") as stream:
        battery = list(csv.DictReader(stream, delimiter="\t"))
    rows = []
    for item in battery:
        run_dir = Path(item["dataset"])
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        states = [
            event for event in events if event.get("event") == "mission_state"
        ]
        plans = [
            event["plan"]
            for event in events
            if event.get("event") == "plan"
            and event.get("map_kind") == "estimated"
        ]
        adopted = [
            plan for plan in plans if plan.get("adopted_for_execution")
        ]
        unsafe = [
            plan
            for plan in adopted
            if not ((plan.get("diagnostics") or {}).get(
                "final_path_safety"
            ) or {}).get("safe", False)
            or not (plan.get("diagnostics") or {}).get(
                "reference_path_avoids_obstacles", False
            )
        ]
        summaries = [
            event for event in events if event.get("event") == "summary"
        ]
        occupancy = (
            summaries[-1].get("occupancy_metrics", {}) if summaries else {}
        )
        takeoff = state_time(states, "takeoff_complete")
        landing = state_time(states, "landing_started")
        rows.append(
            {
                "run": int(item["run_index"]),
                "dataset": str(run_dir),
                "completed": item["mission_complete"].lower() == "true",
                "global_goals": sum(
                    state.get("state") == "global_goal_reached"
                    for state in states
                ),
                "plan_attempts": len(plans),
                "plan_successes": sum(
                    bool(plan.get("success")) for plan in plans
                ),
                "adopted_paths": len(adopted),
                "unsafe_adopted_paths": len(unsafe),
                "active_path_vetoes": sum(
                    state.get("state") == "active_path_veto"
                    for state in states
                ),
                "completed_brakes": sum(
                    state.get("state") == "execution_brake_complete"
                    for state in states
                ),
                "recoveries": sum(
                    state.get("state") == "recovery_started"
                    for state in states
                ),
                "duration_s": (
                    landing - takeoff
                    if takeoff is not None and landing is not None
                    else None
                ),
                "planning_mean_ms": (
                    float(np.mean([
                        float(plan["planning_time_ms"])
                        for plan in plans
                        if plan.get("planning_time_ms") is not None
                    ]))
                    if plans
                    else None
                ),
                "precision": occupancy.get("precision"),
                "recall": occupancy.get("recall"),
                "iou": occupancy.get("iou"),
                "false_free_rate": occupancy.get("false_free_rate"),
            }
        )
    return rows


def aggregate_reference(rows):
    attempts = sum(row["plan_attempts"] for row in rows)
    successes = sum(row["plan_successes"] for row in rows)
    durations = [row["duration_s"] for row in rows if row["duration_s"]]
    planning = [row["planning_mean_ms"] for row in rows if row["planning_mean_ms"]]
    return {
        "runs": len(rows),
        "completed_runs": sum(row["completed"] for row in rows),
        "global_goals_reached": sum(row["global_goals"] for row in rows),
        "plan_attempts": attempts,
        "plan_successes": successes,
        "plan_success_rate": successes / attempts if attempts else None,
        "adopted_paths": sum(row["adopted_paths"] for row in rows),
        "unsafe_adopted_paths": sum(
            row["unsafe_adopted_paths"] for row in rows
        ),
        "active_path_vetoes": sum(row["active_path_vetoes"] for row in rows),
        "completed_brakes": sum(row["completed_brakes"] for row in rows),
        "recoveries": sum(row["recoveries"] for row in rows),
        "duration_mean_s": float(np.mean(durations)),
        "duration_std_s": float(np.std(durations, ddof=1)),
        "duration_min_s": min(durations),
        "duration_max_s": max(durations),
        "planning_mean_ms": (
            sum(
                row["planning_mean_ms"] * row["plan_attempts"]
                for row in rows
                if row["planning_mean_ms"] is not None
            )
            / attempts
            if attempts
            else None
        ),
        "occupancy_precision_mean": float(np.mean([
            row["precision"] for row in rows if row["precision"] is not None
        ])),
        "occupancy_recall_mean": float(np.mean([
            row["recall"] for row in rows if row["recall"] is not None
        ])),
        "occupancy_iou_mean": float(np.mean([
            row["iou"] for row in rows if row["iou"] is not None
        ])),
        "false_free_rate_mean": float(np.mean([
            row["false_free_rate"]
            for row in rows
            if row["false_free_rate"] is not None
        ])),
    }


def model_rows():
    rows = []
    for version, paths in MODEL_INPUTS.items():
        depth = load_json(paths["depth"])
        spatial = load_json(paths["spatial"])
        planning = spatial["planning"]
        mapping = spatial["map"]
        qualification = spatial["qualification"]
        failed_spatial_checks = [
            name
            for name, passed in qualification.get("checks", {}).items()
            if not passed
        ]
        validation = depth["validation_raw"]
        rows.append(
            {
                "model": version,
                "model_sha256": sha256(ROOT / paths["model"]),
                "depth_dataset": depth["validation_run"],
                "depth_qualified": depth["qualification"]["passed"],
                "depth_mae_m": validation["mae_m"],
                "depth_abs_rel": validation["abs_rel"],
                "depth_bias_m": validation["bias_m"],
                "spatial_dataset": spatial["run"],
                "spatial_qualified": qualification["passed"],
                "failed_spatial_checks": ";".join(failed_spatial_checks),
                "frames": spatial["frames_integrated"],
                "plan_attempts": planning["attempts"],
                "plan_successes": planning["successes"],
                "plan_success_rate": planning["success_rate"],
                "collisions": planning.get(
                    "collisions",
                    planning["successes"]
                    - planning["collision_free_successes"],
                ),
                "false_free_rate": mapping["false_free_rate"],
                "precision": mapping["precision"],
                "recall": mapping["recall"],
                "decision": (
                    "rejected_spatial_gate"
                    if not qualification["passed"]
                    else "eligible_for_sitl"
                ),
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision_rows(reference, models):
    """Summarize the two evaluated pipelines without mixing their evidence."""

    v19 = next(row for row in models if row["model"] == "v19")
    return [
        {
            "pipeline": "Gazebo ground-truth depth + spatial map + A*",
            "evidence": "10-run SITL battery",
            "result": (
                f'{reference["completed_runs"]}/{reference["runs"]} complete; '
                f'{reference["unsafe_adopted_paths"]} unsafe adopted paths'
            ),
            "sitl_status": "authorized_and_completed",
            "justification": (
                "Reference-depth pipeline completed the fixed tree route "
                "without adopting a path rejected by the reference guardian."
            ),
        },
        {
            "pipeline": "Monocular Depth Anything V2 v19 + spatial map + A*",
            "evidence": "independent pixel test + reserved offline spatial replay",
            "result": (
                f'pixel_gate={v19["depth_qualified"]}; '
                f'spatial_gate={v19["spatial_qualified"]}; '
                f'failed={v19["failed_spatial_checks"]}'
            ),
            "sitl_status": "blocked",
            "justification": (
                "Pixel metrics passed, but the complete spatial gate failed; "
                "zero collisions in one primary replay does not compensate "
                "for insufficient observed-path availability and false-free "
                "space above the configured limit."
            ),
        },
    ]


def v19_split_rows():
    """Classify each final-gate run against v19 training provenance."""

    training = load_json(
        "models/depth_anything_v2_metric_baylands_vits_v19_final.training.json"
    )
    matrix = load_json("config/spatial_v19_final_validation_sweep.json")
    train = {str(Path(path).resolve()) for path in training["train_runs"]}
    validation = {
        str(Path(path).resolve()) for path in training.get("validation_runs", [])
    }
    model = matrix["models"]["v19_final"]
    primary = model["primary_dataset"]
    rows = []
    for name, item in matrix["datasets"].items():
        path = str((ROOT / item["path"]).resolve())
        rows.append(
            {
                "dataset": name,
                "path": path,
                "gate_stage": "primary" if name == primary else "secondary",
                "used_for_gradients": path in train,
                "used_for_training_validation": path in validation,
                "independent_test": path not in train and path not in validation,
                "executed_in_final_sweep": name == primary,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/spatial_final_analysis/final_20260831"),
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    references = reference_rows(ROOT / REFERENCE_BATTERY)
    models = model_rows()
    freeze_paths = FREEZE_INPUTS + [
        values["model"] for values in MODEL_INPUTS.values()
    ] + [
        values["depth"] for values in MODEL_INPUTS.values()
    ] + [
        values["spatial"] for values in MODEL_INPUTS.values()
    ]
    artifacts = []
    for relative in freeze_paths:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        artifacts.append(
            {
                "path": str(relative),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    reference = aggregate_reference(references)
    decisions = decision_rows(reference, models)
    split_audit = v19_split_rows()
    report = {
        "schema_version": 1,
        "scope": (
            "Spatial mapping and navigation experiment freeze; no thesis "
            "documentation, legacy notebook or dashboard modified."
        ),
        "reference": reference,
        "monocular_models": models,
        "decision": {
            "reference_baseline": "accepted",
            "monocular_sitl_battery": "not_authorized",
            "reason": (
            "All evaluated v12-v19 models failed the complete spatial gate. "
            "Some v19 candidates had zero collisions on the primary reserved "
            "replay, but still failed availability and false-free criteria "
            "and therefore were not eligible for monocular SITL."
            ),
            "interpretation": (
                "Pixel-level depth qualification did not imply safe occupancy "
                "mapping or executable path generation."
            ),
        },
        "artifacts": artifacts,
        "decision_table": decisions,
        "v19_split_audit": split_audit,
    }
    write_csv(output / "reference_runs.csv", references)
    write_csv(output / "monocular_comparison.csv", models)
    write_csv(output / "decision_table.csv", decisions)
    write_csv(output / "v19_split_audit.csv", split_audit)
    (output / "final_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["reference"], indent=2))
    print(json.dumps(report["monocular_models"], indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
