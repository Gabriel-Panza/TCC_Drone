#!/usr/bin/env python3
"""Compara reserva de execucao em snapshots, sem iniciar/autorizar simulacao."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.clearance import path_voxel_clearance
from spatial_mapping.execution import RecoveryCandidatePolicy
from spatial_mapping.execution_reserve import (
    ExecutionReserveNavigator, waypoint_switch_diagnostics, prepare_executable_command,
)
from spatial_mapping.snapshot import load_planning_snapshot
from scripts.run_spatial_execution_clearance import audit_command, restored_navigator


def json_text(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    return json.dumps(value, indent=2, ensure_ascii=False, default=convert, allow_nan=False)


def executable_command(nav, current, plan, diag):
    return prepare_executable_command(
        nav, current, plan, diag['waypoint_acceptance_radius_m'], diag['min_executable_path_m'])


def compare_case(run, event, radius, *, use_start_connections=True, search_executable=True):
    saved = event["plan"]
    diag = saved["diagnostics"]
    source = restored_navigator(run, diag)
    nav = ExecutionReserveNavigator(source, radius, use_start_connections=use_start_connections)
    current = np.asarray(diag["current_position_ned_m"], dtype=float)
    policy_data = diag.get("recovery_candidate_policy")
    policy = RecoveryCandidatePolicy.from_dict(policy_data) if policy_data else None
    reference = None
    if policy and policy.reference_required:
        reference = load_planning_snapshot(run, diag["candidate_reference_snapshot_file"])
    started = perf_counter()
    options = dict(candidate_validator=lambda p: policy.evaluate(nav, current, p, reference),
                   max_candidates=policy.max_candidates, recovery_frontiers=policy.explore_frontiers) if policy else {}
    if radius > 0 and search_executable:
        plan = nav.plan_executable(
            current, saved['requested_goal_ned_m'],
            acceptance_m=diag['waypoint_acceptance_radius_m'],
            minimum_m=diag['min_executable_path_m'], **options)
    else:
        plan = nav.plan(current, saved['requested_goal_ned_m'], **options)
    planning_ms = (perf_counter() - started) * 1000.
    command, strict, safety, length, executable = executable_command(nav, current, plan, diag)
    handoff_current = diag["path_handoff"]["current_position_ned_m"]
    handoff_safety = nav.path_safety_diagnostics(handoff_current, command)
    clearance = path_voxel_clearance(
        [current, *command], source._inflated_obstacles(), source.config.voxel_resolution_m
    ) if command else None
    result = {
        "plan_id": event["plan_id"], "extra_radius_m": radius,
        "start_connections_enabled": bool(radius > 0 and use_start_connections),
        "executable_search_enabled": bool(radius > 0 and search_executable),
        "historically_adopted": bool(saved.get("adopted_for_execution")),
        "replayed_at": "planning_pose_on_before_plan_snapshot",
        "planner_success": plan.success, "reason": plan.reason,
        "planning_time_ms": planning_ms, "executable_at_planning_pose": executable,
        "planning_pose_safety": safety,
        "historical_handoff_pose_safety_on_same_snapshot": handoff_safety,
        "executable_length_m": length, "command_waypoints_ned_m": command,
        "strict_entry_required": strict, "clearance_to_original_inflation": clearance,
        "diagnostics": plan.diagnostics,
        "initial_extra_clearance_m": path_voxel_clearance(
            [current], source._inflated_obstacles(), source.config.voxel_resolution_m
        )["distance_m"],
    }
    if command:
        audit_diag = dict(diag, strict_entry_waypoint_required=strict,
                          path_handoff={"current_position_ned_m": current,
                                        "candidate_waypoints_ned_m": command})
        audit = audit_command(nav, audit_diag, [radius])
        switches = []
        for corner in audit["intermediate_waypoints"]:
            i = corner["waypoint_index"]
            switches.append({
                "waypoint_index": i,
                **waypoint_switch_diagnostics(
                    nav, corner["nominal_sphere_entry_ned_m"], command[i:],
                    corner["acceptance_radius_m"]),
            })
        result["hypothetical_switches"] = switches
    return result


def historical_switches(run, events):
    """Same recorded commands and same .8/.2 radii; vary only the switch check."""
    records = []
    for event in events:
        plan = event["plan"]
        if not plan.get("adopted_for_execution"):
            continue
        diag = plan["diagnostics"]
        source = restored_navigator(run, diag)
        audit = audit_command(source, diag, [0.])
        command = diag["path_handoff"]["candidate_waypoints_ned_m"]
        for corner in audit["intermediate_waypoints"]:
            i = corner["waypoint_index"]
            decision = waypoint_switch_diagnostics(
                source, corner["nominal_sphere_entry_ned_m"], command[i:],
                corner["acceptance_radius_m"])
            records.append({
                "plan_id": event["plan_id"], "waypoint_index": i,
                "scenario": "nominal_entry_not_observed_pose",
                "original_next_segment_safe": corner["hypothetical_outgoing_segment_safety"]["safe"],
                **decision,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-ids", type=int, nargs="+", help="teste seletivo; omita para todos")
    parser.add_argument("--radii-m", type=float, nargs="+", default=[0., .2, .4])
    parser.add_argument("--legacy-start", action="store_true",
                        help="reproduz inicio antigo pelo centro do voxel, apenas offline")
    parser.add_argument("--legacy-selection", action="store_true",
                        help="desativa novo filtro de candidatos; preserva a politica historica")
    args = parser.parse_args()
    if any(not np.isfinite(r) or r < 0 for r in args.radii_m):
        parser.error("reservas devem ser finitas e nao negativas")
    run = args.run_dir.resolve()
    output = args.output or ROOT / "logs/spatial_execution_reserve" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Saida: {output}", flush=True)
    test_modules = sorted("tests." + p.stem for p in (ROOT / "tests").glob("test_spatial*.py"))
    test_modules.append("tests.test_navigation_safety_diagnostics")
    tests = subprocess.run([sys.executable, "-m", "unittest", *test_modules, "-q"],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "tests.log").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        print(tests.stdout)
        return 1
    print("Testes: OK. Apenas replay offline; nenhum processo de simulacao sera alterado.", flush=True)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get("event") == "plan" and e.get("map_kind") == "estimated"]
    if args.plan_ids:
        missing = set(args.plan_ids) - {e["plan_id"] for e in events}
        if missing:
            parser.error(f"planos ausentes: {sorted(missing)}")
        events = [e for e in events if e["plan_id"] in args.plan_ids]
    records, errors = [], []
    switches = []
    try:
        switches = historical_switches(run, events)
    except (KeyError, OSError, ValueError, TypeError) as error:
        errors.append({"stage": "historical_switches", "error": str(error)})
    radii = list(dict.fromkeys(args.radii_m))
    for event in events:
        for radius in radii:
            try:
                result = compare_case(run, event, radius, use_start_connections=not args.legacy_start,
                                      search_executable=not args.legacy_selection)
                records.append(result)
                label = str(radius).replace(".", "p")
                (output / f"plan_{event['plan_id']}_reserve_{label}.json").write_text(
                    json_text(result), encoding="utf-8")
                print(f"plano={event['plan_id']} reserva={radius:.2f}m "
                      f"executavel={result['executable_at_planning_pose']} "
                      f"handoff_seguro={result['historical_handoff_pose_safety_on_same_snapshot']['safe']} "
                      f"comprimento={result['executable_length_m']:.2f}m "
                      f"conectores={result['diagnostics'].get('initial_connections', {}).get('accepted_count', 0)} "
                      f"tentativas={len(result['diagnostics'].get('candidate_search', {}).get('attempts', []))} "
                      f"motivo={result['reason']}", flush=True)
            except (KeyError, OSError, ValueError, TypeError) as error:
                errors.append({"plan_id": event["plan_id"], "extra_radius_m": radius, "error": str(error)})
                print(f"ERRO plano={event['plan_id']} reserva={radius}: {error}", flush=True)
    summaries = []
    for radius in radii:
        cases = [r for r in records if r["extra_radius_m"] == radius]
        summaries.append({
            "extra_radius_m": radius, "cases": len(cases),
            "executable": sum(c["executable_at_planning_pose"] for c in cases),
            "safe_at_historical_handoff_pose": sum(c["historical_handoff_pose_safety_on_same_snapshot"]["safe"] for c in cases),
            "executable_and_safe_at_historical_handoff_pose": sum(
                c["executable_at_planning_pose"] and c["historical_handoff_pose_safety_on_same_snapshot"]["safe"]
                for c in cases),
            "start_without_reserve": sum(c["reason"] == "execution_reserve_unavailable_at_start" for c in cases),
            "reasons": dict(Counter(c["reason"] for c in cases)),
        })
    report = {
        "diagnostic_only": True, "releases_sitl": False, "run_dir": str(run),
        "legacy_start": args.legacy_start,
        "legacy_selection": args.legacy_selection,
        "selected_plan_ids": [e["plan_id"] for e in events], "summary": summaries,
        "historical_switch_actions": dict(Counter(s["action"] for s in switches)),
        "historical_switch_scenarios": switches, "errors": errors,
        "limitations": [
            "Zero radius retains the original selection baseline; positive radii use the enabled experimental policies.",
            "Bounded candidate rejection does not prove no route exists; the search limit remains 16 for ordinary plans.",
            "One radius changed at a time; original inflation, known space and other parameters preserved.",
            "Reserve constrains the centerline relative to inflated boxes; surrounding unknown volume is not certified.",
            "Fixed historical snapshots/poses do not predict the closed-loop trajectory after a different plan.",
            "Historical handoff check uses the BEFORE-plan map, not an invented adoption-time map.",
            "Waypoint checks use hypothetical nominal sphere-entry poses, not observed switch telemetry.",
            "Switch veto is experimental and not connected to the live controller yet.",
            "No candidate is automatically promoted and no simulation is launched.",
        ],
    }
    (output / "summary.json").write_text(json_text(report), encoding="utf-8")
    print(json_text(summaries))
    print(f"Relatorio: {output / 'summary.json'}")
    print("Codigo 2 = alternativas rejeitadas; 1 = erro/testes; 0 nao libera SITL.")
    if errors or not records:
        return 1
    return 2 if any(not r["executable_at_planning_pose"] or not r[
        "historical_handoff_pose_safety_on_same_snapshot"]["safe"] for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
