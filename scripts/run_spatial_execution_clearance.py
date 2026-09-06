#!/usr/bin/env python3
"""Auditoria offline de folga e tolerancia de waypoints; nao libera SITL."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.clearance import (
    acceptance_entry_point, path_voxel_clearance, point_to_path,
)
from spatial_mapping.snapshot import load_planning_snapshot


def restored_navigator(run, diag):
    nav = load_planning_snapshot(run, diag["planning_snapshot_file"])
    # Same deterministic ego update performed by plan() after snapshot capture.
    nav.grid.mark_ego_voxel_free(diag["current_position_ned_m"])
    return nav


def audit_command(nav, diag, margins):
    """Audit recorded command only; missing evidence must not become a pass."""
    handoff = diag["path_handoff"]
    current = handoff["current_position_ned_m"]
    waypoints = handoff["candidate_waypoints_ned_m"]
    if not waypoints:
        raise ValueError("comando adotado sem waypoints gravados")
    points = [current, *waypoints]
    blocked = nav._inflated_obstacles()
    resolution = nav.config.voxel_resolution_m

    def clearance(path):
        return path_voxel_clearance(path, blocked, resolution)

    nominal = clearance(points)
    distance = nominal["distance_m"]
    radius = float(diag["waypoint_acceptance_radius_m"])
    corners = []
    # Final waypoint has terminal-arrival rules: no outgoing shortcut to audit.
    for index, waypoint in enumerate(waypoints[:-1]):
        effective_radius = (
            float(diag["strict_entry_acceptance_radius_m"])
            if index == 0 and diag.get("strict_entry_waypoint_required") else radius
        )
        center = clearance([waypoint])
        entry = acceptance_entry_point(points[index], waypoint, effective_radius)
        shortcut = nav.path_safety_diagnostics(entry, [waypoints[index + 1]])
        corners.append({
            "waypoint_index": index,
            "acceptance_radius_m": effective_radius,
            "center_clearance": center,
            "sphere_touches_inflation": bool(
                center["distance_m"] is not None
                and center["distance_m"] <= effective_radius + 1e-9
            ),
            "nominal_sphere_entry_ned_m": entry,
            "hypothetical_outgoing_segment_safety": shortcut,
        })
    return {
        "nominal_safety": nav.path_safety_diagnostics(current, waypoints),
        "nominal_clearance": nominal,
        "extra_tube_comparisons": [
            {"extra_radius_m": float(margin),
             "touches_inflation": bool(distance is not None and distance <= margin + 1e-9)}
            for margin in margins
        ],
        "intermediate_waypoints": corners,
        "sphere_overlap_count": sum(c["sphere_touches_inflation"] for c in corners),
        "hypothetical_unsafe_shortcuts": sum(
            not c["hypothetical_outgoing_segment_safety"]["safe"] for c in corners
        ),
        "command_points_ned_m": points,
    }


def audit_recovery(run, events, recovery):
    """Locate the failed handoff by timestamp, not line order in events.jsonl."""
    plans = [e for e in events if e.get("event") == "plan"
             and e.get("map_kind") == "estimated"]
    timestamp = recovery["timestamp_s"]
    failed = [e for e in plans if e["timestamp_s"] == timestamp
              and not e["plan"].get("adopted_for_execution")]
    previous = [e for e in plans if e["timestamp_s"] < timestamp
                and e["plan"].get("adopted_for_execution")]
    if len(failed) != 1 or not previous:
        raise ValueError("recuperacao sem par inequivoco de plano anterior/handoff rejeitado")
    previous = max(previous, key=lambda e: e["timestamp_s"])
    failed = failed[0]
    old_diag = previous["plan"]["diagnostics"]
    new_diag = failed["plan"]["diagnostics"]
    old_handoff = old_diag["path_handoff"]
    index = int(new_diag["path_handoff"]["existing_path_index"])
    points = [old_handoff["current_position_ned_m"],
              *old_handoff["candidate_waypoints_ned_m"]]
    if not 0 <= index < len(points) - 1:
        raise ValueError("indice de segmento ativo fora do comando anterior")
    pose = recovery["position_ned_m"]
    projection = point_to_path(pose, points[index:index + 2])
    projection["segment_index"] = index
    states = {}
    for label, diag in (("previous_adoption_snapshot", old_diag),
                        ("rejected_plan_snapshot", new_diag)):
        nav = restored_navigator(run, diag)
        voxel = nav.grid.world_to_voxel(pose)
        blocked = nav._inflated_obstacles()
        states[label] = {
            "snapshot_file": diag["planning_snapshot_file"],
            "pose_voxel": [int(axis) for axis in voxel],
            "pose_voxel_inflated": voxel in blocked,
            "pose_voxel_known_free": voxel in nav.grid.free_voxels(),
            "pose_clearance": path_voxel_clearance(
                [pose], blocked, nav.config.voxel_resolution_m
            ),
        }
    return {
        "timestamp_s": timestamp,
        "previous_adopted_plan_id": previous["plan_id"],
        "rejected_plan_id": failed["plan_id"],
        "position_ned_m": pose,
        "active_command_segment_index": index,
        "deviation_from_active_nominal_segment": projection,
        "map_comparison": states,
        "inflation_already_present_before_recovery": states[
            "previous_adoption_snapshot"]["pose_voxel_inflated"],
    }


def audit_run(run, margins):
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
              if line.strip()]
    records, recoveries, errors = [], [], []
    for event in events:
        if event.get("event") == "plan" and event.get("map_kind") == "estimated":
            plan = event["plan"]
            if not plan.get("adopted_for_execution"):
                continue
            try:
                diag = plan["diagnostics"]
                nav = restored_navigator(run, diag)
                records.append({
                    "run": str(run), "plan_id": event["plan_id"],
                    "snapshot_file": diag["planning_snapshot_file"],
                    **audit_command(nav, diag, margins),
                })
            except (KeyError, OSError, ValueError, TypeError) as error:
                errors.append({"run": str(run), "plan_id": event.get("plan_id"),
                               "error": str(error)})
        elif event.get("event") == "mission_state" and event.get("state") == "recovery_started":
            try:
                recoveries.append({"run": str(run), **audit_recovery(run, events, event)})
            except (KeyError, OSError, ValueError, TypeError) as error:
                errors.append({"run": str(run), "recovery_timestamp_s": event.get("timestamp_s"),
                               "error": str(error)})
    if not records:
        errors.append({"run": str(run), "error": "nenhum comando adotado auditavel"})
    return records, recoveries, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--margins-m", nargs="+", type=float, default=[0., .2, .4, .8],
                        help="raios ADICIONAIS apenas para comparacao offline")
    args = parser.parse_args()
    if any(not np.isfinite(m) or m < 0 for m in args.margins_m):
        parser.error("margens devem ser finitas e nao negativas")
    output = args.output or ROOT / "logs/spatial_execution_clearance" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Saida: {output}", flush=True)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_spatial_mapping",
         "tests.test_spatial_segments", "tests.test_spatial_clearance",
         "tests.test_spatial_execution_clearance", "-q"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    (output / "tests.log").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        print(tests.stdout)
        print("Testes falharam; auditoria interrompida.")
        return 1
    print("Testes: OK", flush=True)
    records, recoveries, errors = [], [], []
    for run in args.run_dirs:
        run = run.resolve()
        try:
            r, recovery, err = audit_run(run, args.margins_m)
            records.extend(r)
            recoveries.extend(recovery)
            errors.extend(err)
            print(f"{run.name}: {len(r)} comandos, {len(recovery)} recuperacoes", flush=True)
        except (OSError, ValueError) as error:
            errors.append({"run": str(run), "error": str(error)})
    distances = [r["nominal_clearance"]["distance_m"] for r in records
                 if r["nominal_clearance"]["distance_m"] is not None]
    summary = {
        "adopted_commands_audited": len(records),
        "nominal_unsafe_on_snapshot": sum(not r["nominal_safety"]["safe"] for r in records),
        "minimum_extra_clearance_m": min(distances) if distances else None,
        "commands_touching_extra_tubes": {
            str(float(margin)): sum(
                r["extra_tube_comparisons"][i]["touches_inflation"] for r in records
            ) for i, margin in enumerate(args.margins_m)
        },
        "intermediate_acceptance_sphere_overlaps": sum(r["sphere_overlap_count"] for r in records),
        "hypothetical_unsafe_shortcuts": sum(r["hypothetical_unsafe_shortcuts"] for r in records),
        "recoveries_in_preexisting_inflation": sum(
            r["inflation_already_present_before_recovery"] for r in recoveries),
        "errors_or_missing_evidence": len(errors),
    }
    report = {
        "diagnostic_only": True, "releases_sitl": False,
        "limitations": [
            "Snapshots are before planning, not necessarily the adoption-time map.",
            "Clearance is to CLOSED inflated voxel boxes, not physical obstacles or their centers.",
            "Extra tubes and acceptance spheres are sensitivity scenarios, not measured tracking errors.",
            "Tube comparisons check inflated obstacles only; they do not certify unknown-space safety.",
            "Shortcut assumes nominal incoming motion; sphere overlap does not prove it was executed.",
            "Recovery analysis uses recorded poses only; it is not a complete flight trajectory audit.",
            "No parameters, datasets, controller or simulator processes are modified.",
        ],
        "summary": summary, "records": records, "recoveries": recoveries, "errors": errors,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for recovery in recoveries:
        distance = recovery["deviation_from_active_nominal_segment"]["distance_m"]
        print(f"Recuperacao plano={recovery['rejected_plan_id']}: desvio={distance:.3f}m "
              f"segmento={recovery['active_command_segment_index']} "
              f"inflacao_preexistente={recovery['inflation_already_present_before_recovery']}")
    print(f"Relatorio: {output / 'summary.json'}")
    print("Codigo 2 = achados para investigar; codigo 1 = erro/teste falhou.")
    print("Nenhum processo de simulacao iniciado/encerrado. Este diagnostico nao libera SITL.")
    if errors or not records:
        return 1
    findings = any(summary[key] for key in (
        "nominal_unsafe_on_snapshot", "intermediate_acceptance_sphere_overlaps",
        "hypothetical_unsafe_shortcuts", "recoveries_in_preexisting_inflation",
    ))
    tube_findings = any(summary["commands_touching_extra_tubes"].values())
    return 2 if findings or tube_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
