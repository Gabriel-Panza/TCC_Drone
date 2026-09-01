#!/usr/bin/env python3
"""Resume os planos criticos 5--8 dos replays de hipoteses espaciais."""

import argparse
import csv
import json
from pathlib import Path


def nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--first-plan", type=int, default=5)
    parser.add_argument("--last-plan", type=int, default=8)
    parser.add_argument(
        "--dataset",
        default="heldout_old",
        help="nome do dataset primario gravado em reports/<candidate>",
    )
    args = parser.parse_args()

    rows = []
    reports = sorted(
        (args.sweep_dir / "reports").glob(f"*/{args.dataset}.json")
    )
    if not reports:
        raise FileNotFoundError(
            f"nenhum relatorio {args.dataset!r} encontrado"
        )
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        candidate = report.parent.name
        plans = payload.get("planning", {}).get("plans", [])
        for plan_number in range(args.first_plan, args.last_plan + 1):
            if plan_number > len(plans):
                continue
            plan = plans[plan_number - 1]
            estimated = plan.get("estimated_obstacle_proximity") or {}
            reference = plan.get("reference_obstacle_proximity") or {}
            vertical = estimated.get("vertical_counts_within_radius") or {}
            depth = plan.get("depth_error_since_previous_plan") or {}
            rows.append(
                {
                    "candidate": candidate,
                    "plan": plan_number,
                    "frames_integrated": plan.get("frames_integrated"),
                    "success": plan.get("success"),
                    "executable": plan.get("executable"),
                    "collision_free": plan.get("collision_free_in_reference"),
                    "reason": plan.get("reason"),
                    "current_voxel_state": plan.get("current_voxel_state"),
                    "current_voxel_inflated": plan.get("current_voxel_inflated"),
                    "blocked_neighbors_26": plan.get("blocked_neighbors_26"),
                    "reachable_voxels": plan.get("reachable_voxels"),
                    "max_reachable_progress_m": plan.get(
                        "max_reachable_progress_m"
                    ),
                    "estimated_nearest_obstacle_m": estimated.get(
                        "nearest_distance_m"
                    ),
                    "reference_nearest_obstacle_m": reference.get(
                        "nearest_distance_m"
                    ),
                    "estimated_occupied_within_5m": estimated.get(
                        "occupied_within_radius"
                    ),
                    "estimated_above_0_4m": vertical.get(
                        "above_more_than_0_4m"
                    ),
                    "estimated_flight_band_abs_0_4m": vertical.get(
                        "flight_band_abs_0_4m"
                    ),
                    "estimated_below_0_4m": vertical.get(
                        "below_more_than_0_4m"
                    ),
                    "depth_mae_m": depth.get("mae_m"),
                    "depth_mean_signed_error_m": depth.get(
                        "mean_signed_error_m"
                    ),
                    "dangerous_far_rate_1m": depth.get(
                        "dangerous_far_rate_1m"
                    ),
                    "dangerous_far_edge_rate_1m": depth.get(
                        "dangerous_far_edge_rate_1m"
                    ),
                }
            )

    output = args.sweep_dir / "critical_plans_5_8.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Diagnostico critico: {output.resolve()}")
    print(f"Linhas: {len(rows)}")


if __name__ == "__main__":
    main()
