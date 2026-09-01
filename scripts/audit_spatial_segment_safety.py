#!/usr/bin/env python3
"""Reaudita caminhos adotados nos snapshots; nao libera voo nem altera datasets."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.snapshot import load_planning_snapshot
from spatial_mapping.execution import RecoveryCandidatePolicy


def compare_segment_checks(navigator, current, waypoints):
    exact = navigator.path_safety_diagnostics(current, waypoints)
    # Diagnostic-only reproduction of the former checker on the SAME map.
    with patch.object(navigator.grid, "segment_voxels", navigator.grid._ray_voxels):
        sampled = navigator.path_safety_diagnostics(current, waypoints)
    return {
        "sampled_safe": sampled["safe"], "exact": exact,
        "sampling_false_approval": bool(sampled["safe"] and not exact["safe"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "datasets/spatial_mapping")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replan", action="store_true",
                        help="replaneja os estados salvos com o codigo atual, sem voo")
    args = parser.parse_args()
    runs = args.run_dirs or sorted(p.parent for p in args.base_dir.glob("run_*/planning_maps"))
    output = args.output or ROOT / "logs/spatial_segment_audit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    records, errors = [], []
    for run in runs:
        try:
            events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
                      if line.strip()]
        except (OSError, ValueError) as error:
            errors.append({"run": str(run), "error": str(error)})
            continue
        count = 0
        for event in events:
            if event.get("event") != "plan" or event.get("map_kind") != "estimated":
                continue
            plan = event["plan"]
            if not plan.get("adopted_for_execution"):
                continue
            diag = plan.get("diagnostics") or {}
            try:
                nav = load_planning_snapshot(run, diag["planning_snapshot_file"])
                current = diag["current_position_ned_m"]
                # plan() applies this deterministic ego update after capture.
                nav.grid.mark_ego_voxel_free(current)
                checks = {"saved_plan": compare_segment_checks(nav, current, plan["waypoints_ned_m"])}
                handoff = diag.get("path_handoff")
                if handoff and handoff.get("candidate_waypoints_ned_m"):
                    checks["recorded_command"] = compare_segment_checks(
                        nav, handoff["current_position_ned_m"],
                        handoff["candidate_waypoints_ned_m"],
                    )
                record = {
                    "run": str(run), "plan_id": event["plan_id"], "checks": checks,
                    "sampling_false_approval": any(c["sampling_false_approval"] for c in checks.values()),
                    "any_unsafe": any(not c["exact"]["safe"] for c in checks.values()),
                    "snapshot_file": diag["planning_snapshot_file"],
                    "command_recorded": "recorded_command" in checks,
                }
                if args.replan:
                    policy_data = diag.get("recovery_candidate_policy")
                    if policy_data:
                        policy = RecoveryCandidatePolicy.from_dict(policy_data)
                        reference = None
                        if policy.reference_required:
                            reference = load_planning_snapshot(
                                run, diag["candidate_reference_snapshot_file"]
                            )
                        new_plan = nav.plan(
                            current, plan["requested_goal_ned_m"],
                            candidate_validator=lambda p: policy.evaluate(nav, current, p, reference),
                            max_candidates=policy.max_candidates,
                            recovery_frontiers=policy.explore_frontiers,
                        )
                    else:
                        new_plan = nav.plan(current, plan["requested_goal_ned_m"])
                    # Reproduce the controller's pruning/strict-entry fallback.
                    original = list(new_plan.waypoints_ned_m) if new_plan.success else []
                    command = list(original)
                    acceptance = float(diag.get("waypoint_acceptance_radius_m", .8))
                    while command and np.linalg.norm(np.asarray(command[0]) - current) <= acceptance:
                        command.pop(0)
                    if (not nav.path_is_safe(current, command)
                        and nav.path_is_safe(current, original)):
                        command = original
                    safety = nav.path_safety_diagnostics(current, command)
                    points = [np.asarray(current)] + [np.asarray(p) for p in command]
                    length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
                    executable = bool(
                        new_plan.success and safety["safe"] and
                        (new_plan.reason != "local_subgoal" or
                         length >= float(diag.get("min_executable_path_m", 1.5)))
                    )
                    record["replan"] = {
                        "planner_success": new_plan.success, "reason": new_plan.reason,
                        "executable": executable, "command_safety": safety,
                        "executable_length_m": length, "waypoints_ned_m": command,
                        "planning_time_ms": new_plan.planning_time_ms,
                    }
                records.append(record)
                count += 1
            except (KeyError, OSError, ValueError) as error:
                errors.append({"run": str(run), "plan_id": event.get("plan_id"), "error": str(error)})
        print(f"{run.name}: {count} planos adotados auditados", flush=True)
    findings = [r for r in records if r["any_unsafe"]]
    summary = {
        "runs_selected": len(runs), "adopted_plans_audited": len(records),
        "sampling_false_approvals": sum(r["sampling_false_approval"] for r in records),
        "unsafe_on_snapshot": len(findings), "errors_or_missing_evidence": len(errors),
        "plans_without_command_record": sum(not r["command_recorded"] for r in records),
        "first_failure_reasons": dict(Counter(
            c["exact"]["failure_reason"] for r in records for c in r["checks"].values()
            if not c["exact"]["safe"]
        )),
    }
    if args.replan:
        replans = [r["replan"] for r in records if "replan" in r]
        summary["replan"] = {
            "states": len(replans),
            "executable": sum(p["executable"] for p in replans),
            "rejected": sum(not p["executable"] for p in replans),
            "successful_planner_but_unsafe_command": sum(
                p["planner_success"] and not p["command_safety"]["safe"] for p in replans
            ),
        }
    report = {
        "diagnostic_only": True, "releases_sitl": False,
        "scope": "Estimated before-plan snapshots; no physical collision inference.",
        "limitations": (
            "Before-plan maps are not necessarily the map at adoption. Without a recorded "
            "command, only the saved plan is audited. Runs without snapshots are not selected "
            "by default. Each false approval compares both checkers on the same restored map."
        ),
        "summary": summary, "findings": findings, "errors": errors, "records": records,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Relatorio: {output / 'summary.json'}")
    print("Codigo 2: achados historicos/ausencia de evidencia; nao execute SITL por este relatorio.")
    return 2 if findings or errors or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
