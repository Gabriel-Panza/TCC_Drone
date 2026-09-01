#!/usr/bin/env python3
"""Gate anti-repeticao: testes e replay seletivo de snapshots reais, sem SITL."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.execution import (
    RecoveryProgressGuard, RecoveryCandidatePolicy, RecoveryExitWindow, evaluate_path_handoff,
)
from dataclasses import fields
from dataclasses import replace
from spatial_mapping.snapshot import load_planning_snapshot
from estudos_e_analises.diagnosticar_planejamento_espacial import body_position_from_frame


def replay_recovery_cycle():
    """Decisoes locais em snapshots; nao simula uma trajetoria contrafactual.

    O historico antigo nao gravava poses a cada tick. Usamos somente poses de
    frames anteriores ao snapshot (com extrinsecos), nunca waypoints comandados
    como se fossem trajetoria medida. A origem e declarada no relatorio.
    """
    run = ROOT / 'datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_175409_181223'
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()
              if line.strip()]
    metadata = json.loads((run / 'manifest.json').read_text())['metadata']
    recovery_index = next(i for i, e in enumerate(events)
                          if e.get('state') == 'recovery_complete')
    rows = []
    for event_index, event in enumerate(events):
        plan_id = event.get('plan_id')
        if (event.get('map_kind') != 'estimated' or event.get('event') != 'plan'
                or plan_id not in (33, 35, 37, 39)):
            continue
        saved = event['plan']
        diag = saved['diagnostics']
        nav = load_planning_snapshot(run, diag['planning_snapshot_file'])
        policy = RecoveryCandidatePolicy.from_dict(diag['recovery_candidate_policy'])
        guard = policy.guard
        guard.record_position(events[recovery_index]['position_ned_m'])
        source_frames = []
        for frame in events[recovery_index + 1:event_index]:
            if (frame.get('event') != 'frame'
                    or frame['frame_id'] > nav.frames_integrated):
                continue
            with np.load(run / frame['file'], allow_pickle=False) as data:
                position, approximate = body_position_from_frame(data['camera_to_ned'], metadata)
            if approximate:
                raise ValueError('gate de ciclo exige extrinsecos para a pose do corpo')
            guard.record_position(position)
            source_frames.append(frame['frame_id'])
        current = np.asarray(diag['current_position_ned_m'])
        guard.record_position(current)
        plan = nav.plan(
            current, saved['requested_goal_ned_m'],
            candidate_validator=lambda candidate: policy.evaluate(nav, current, candidate),
            max_candidates=policy.max_candidates,
        )
        handoff = diag['path_handoff']
        command_pose = handoff['current_position_ned_m']
        command = handoff['candidate_waypoints_ned_m']
        old_safe = nav.path_is_safe(command_pose, command)
        old_check = guard.evaluate(
            command_pose, command, path_safe=old_safe,
            extension_m=policy.extension_m, blocked_radius_m=policy.waypoint_acceptance_m,
            arrival_radius_m=policy.goal_acceptance_m,
        )
        check = policy.evaluate(nav, current, plan)
        live_check = policy.evaluate(nav, command_pose, plan)
        search = plan.diagnostics['candidate_search']
        # 39 is intentionally a fail-closed case, NOT an executable exit.
        expected_exit = plan_id != 39
        expected_old_allowed = plan_id in (33, 37)
        passed = bool(
            old_safe and old_check['allowed'] == expected_old_allowed
            and (expected_old_allowed or old_check['reason'] == 'revisited_recovery_corridor')
            and plan.success == expected_exit
            and (check['accepted'] and live_check['accepted'] if expected_exit
                 else not plan.waypoints_ned_m and not plan.adopted_for_execution)
            and len(search['attempts']) <= 16
        )
        rows.append({
            'case': f'recovery_cycle_{plan_id}', 'gate_passed': passed,
            'run_dir': str(run), 'plan_id': plan_id,
            'expected_outcome': 'safe_new_exit' if expected_exit else 'bounded_safe_rejection',
            'history_evidence': 'body poses from prior frames plus recorded current pose; not commands',
            'history_frame_ids': source_frames,
            'map_frame_limit': nav.frames_integrated,
            'old_command_validation': old_check,
            'new_validation': check, 'live_pose_validation': live_check,
            'candidate_search': search, 'planning_time_ms': plan.planning_time_ms,
            'waypoints_ned_m': plan.waypoints_ned_m,
        })
        print(f"ciclo={plan_id} anterior={old_check['reason']} "
              f"saida_nova={plan.success} esperado={'saida' if expected_exit else 'rejeicao_segura'} "
              f"tentativas={len(search['attempts'])}/16 gate={passed}", flush=True)
    if len(rows) != 4:
        raise ValueError('planos criticos ausentes no gate de ciclo')
    return rows


def replay_active_exit_deadline():
    """O abort real tinha uma saida segura em andamento (planos 39 -> 41)."""
    run = ROOT / 'datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_185158_116336'
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()
              if line.strip()]
    plans = {e['plan_id']: e['plan'] for e in events if e.get('map_kind') == 'estimated'}
    aborted = next(e for e in events if e.get('state') == 'mission_aborted')
    guard = RecoveryProgressGuard(**aborted['recovery_guard'])
    command = plans[39]['diagnostics']['path_handoff']['candidate_waypoints_ned_m']
    handoff = plans[41]['diagnostics']['path_handoff']
    window = RecoveryExitWindow()
    for plan_id in (39, 41):
        diag = plans[plan_id]['diagnostics']
        window = window.observe(
            diag['path_handoff']['current_position_ned_m'], command,
            0 if plan_id == 39 else handoff['existing_path_index'],
            diag['recovery_return_guard']['last_observation_s'], 1.,
        )
    window = window.observe(aborted['position_ned_m'], command,
                            handoff['existing_path_index'], aborted['timestamp_s'], 1.)
    nav = load_planning_snapshot(run, plans[41]['diagnostics']['planning_snapshot_file'])
    safety = nav.path_safety_diagnostics(
        aborted['position_ned_m'], handoff['existing_remaining_waypoints_ned_m'])
    args = dict(base_deadline_s=guard.observation_started_s + 18.,
                path_safe=safety['safe'], endpoint_progress_m=guard.endpoint_progress(command[-1]),
                arrival=False, min_progress_m=1., corridor_m=.8)
    window, check = window.decide(aborted['timestamp_s'], **args)
    # Hypothetical safeguards, explicitly NOT later flight observations.
    stalled_time = window.last_progress_s + window.STALL_LIMIT_S
    _, stalled = window.decide(stalled_time, **args)
    _, hard_stop = window.decide(args['base_deadline_s'] + window.GRACE_LIMIT_S, **args)
    _, vetoed = window.decide(aborted['timestamp_s'], **{**args, 'path_safe': False})
    passed = bool(
        guard.observation_expired(aborted['timestamp_s'], 18.)
        and plans[39]['adopted_for_execution'] and not plans[41]['adopted_for_execution']
        and handoff['reason'] == 'candidate_unsafe_keep_safe_path'
        and safety['safe'] and check['allowed'] and check['granted_now']
        and check['measured_path_progress_m'] >= 1.
        and stalled['reason'] == 'exit_stalled' and not stalled['allowed']
        and hard_stop['reason'] == 'exit_grace_expired' and not hard_stop['allowed']
        and not vetoed['allowed']
    )
    print(f"prazo=39->41 avancado={check['measured_path_progress_m']:.2f}m "
          f"janela={check['allowed']} max_extra={window.GRACE_LIMIT_S:.0f}s "
          f"parado_vetado={not stalled['allowed']} gate={passed}", flush=True)
    return {
        'case': 'active_exit_deadline_39_41', 'gate_passed': passed, 'run_dir': str(run),
        'map_source': plans[41]['diagnostics']['planning_snapshot_file'],
        'note': ('Recorded handoff/abort poses; handoff timing uses the last recorded guard tick. '
                 'Safety is checked on snapshot 41, not an unrecorded map at abort. '
                 'Stall/deadline/veto checks are synthetic; no later flight is predicted.'),
        'safety': safety, 'decision_at_abort': check, 'synthetic_stall': stalled,
        'synthetic_deadline': hard_stop, 'synthetic_veto': vetoed,
    }


def replay_observation_frontiers():
    """Compara a regra antiga e a exploracao no mesmo mapa, sem novos frames."""
    run = ROOT / 'datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_203548_111434'
    plans = {event['plan_id']: event['plan']
             for line in (run / 'events.jsonl').read_text().splitlines() if line.strip()
             for event in [json.loads(line)] if event.get('map_kind') == 'estimated'}
    rows = []
    for plan_id in (17, 21, 23, 25, 31):
        saved = plans[plan_id]
        diag = saved['diagnostics']
        current = np.asarray(diag['current_position_ned_m'])
        old_policy = RecoveryCandidatePolicy.from_dict(diag['recovery_candidate_policy'])
        baseline_nav = load_planning_snapshot(run, diag['planning_snapshot_file'])
        baseline = baseline_nav.plan(
            current, saved['requested_goal_ned_m'], max_candidates=64 if plan_id in (25, 31) else 16,
            candidate_validator=lambda p: old_policy.evaluate(baseline_nav, current, p),
        )
        policy = replace(old_policy, explore_frontiers=True)
        nav = load_planning_snapshot(run, diag['planning_snapshot_file'])
        plan = nav.plan(
            current, saved['requested_goal_ned_m'],
            candidate_validator=lambda p: policy.evaluate(nav, current, p),
            max_candidates=policy.max_candidates, recovery_frontiers=policy.explore_frontiers,
        )
        check = policy.evaluate(nav, current, plan)
        frontier = check.get('observation_frontier', {})
        search = plan.diagnostics.get('candidate_search', {})
        passed = bool(
            plan.success and check['accepted'] and frontier.get('observable')
            and frontier.get('unknown_space_traversed') is False
            and frontier['potential_unknown_faces'] > 0
            and nav.path_is_safe(current, plan.waypoints_ned_m)
            and plan.frontier_standoff_applied_m >= nav.config.frontier_standoff_m - 1e-9
            and check['executable_path_length_m'] >= policy.min_executable_m
            and len(search.get('attempts', [])) <= 16
            and nav.config == baseline_nav.config
            and plan.diagnostics['reachable_voxels'] == baseline.diagnostics['reachable_voxels']
            and (plan_id not in (25, 31) or (
                not baseline.success
                and not baseline.diagnostics['candidate_search']['budget_exhausted']
                and plan.diagnostics['backward_exploration_selected']))
        )
        rows.append({
            'case': f'observation_frontier_{plan_id}', 'gate_passed': passed,
            'run_dir': str(run), 'plan_id': plan_id, 'baseline_success': baseline.success,
            'baseline_search': baseline.diagnostics.get('candidate_search'),
            'validation': check, 'candidate_search': search,
            'backward_exploration_selected': plan.diagnostics.get('backward_exploration_selected'),
            'selected_goal_projection_m': plan.diagnostics.get('selected_goal_projection_m'),
            'waypoints_ned_m': plan.waypoints_ned_m, 'planning_time_ms': plan.planning_time_ms,
            'policy': policy.as_dict(),
            'note': 'Geometric observation opportunity, not measured future information gain or flight success.',
        })
        print(f"fronteira={plan_id} anterior={baseline.success} novo={plan.success} "
              f"visivel={frontier.get('observable')} "
              f"comprimento={check.get('executable_path_length_m', 0):.2f}m "
              f"tentativas={len(search.get('attempts', []))}/16 gate={passed}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "logs/spatial_recovery_gate" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Saida: {output}", flush=True)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_spatial_mapping",
         "tests.test_navigation_safety_diagnostics", "tests.test_spatial_execution",
         "tests.test_spatial_subgoal_bounds", "tests.test_spatial_replay",
         "tests.test_spatial_snapshot", "tests.test_spatial_recovery_progress",
         "tests.test_spatial_alternative_search", "tests.test_spatial_handoff",
         "tests.test_spatial_segments", "tests.test_spatial_recovery_memory",
         "tests.test_spatial_exit_window", "tests.test_spatial_frontiers",
         "tests.test_spatial_waypoint_advance",
         "tests.test_spatial_controller_initialization", "-q"],
        cwd=ROOT, text=True, capture_output=True,
    )
    (output / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        print(f"Testes falharam: {output / 'tests.log'}")
        return 1
    print("Testes: OK", flush=True)
    run = ROOT / "datasets/spatial_mapping/run_20260827_064750_916804"
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
              if line.strip()]
    plans = {e["plan_id"]: e["plan"] for e in events
             if e.get("event") == "plan" and e.get("map_kind") == "estimated"}
    failed = plans[11]
    first_recovery = next(e for e in events if e.get("state") == "recovery_complete")
    guard = RecoveryProgressGuard(
        0, tuple(failed["requested_goal_ned_m"]),
        tuple(failed["diagnostics"]["current_position_ned_m"]),
        tuple(failed["waypoints_ned_m"][-1]),
        first_recovery["timestamp_s"],
    )
    rows = []
    for plan_id, expected in ((13, False), (21, False), (25, True)):
        saved = plans[plan_id]
        diag = saved["diagnostics"]
        nav = load_planning_snapshot(run, diag["planning_snapshot_file"])
        current = np.asarray(diag["current_position_ned_m"])
        plan = nav.plan(current, saved["requested_goal_ned_m"])
        original = list(plan.waypoints_ned_m)
        waypoints = list(original)
        while waypoints and np.linalg.norm(np.asarray(waypoints[0]) - current) <= .8:
            waypoints.pop(0)
        if not nav.path_is_safe(current, waypoints) and nav.path_is_safe(current, original):
            waypoints = original
        safe = nav.path_is_safe(current, waypoints)
        check = guard.evaluate(
            current, waypoints, path_safe=safe, extension_m=1.,
            blocked_radius_m=.8, arrival_radius_m=nav.config.goal_acceptance_radius_m,
        )
        same_map = diag["reachable_voxels"] == plan.diagnostics["reachable_voxels"]
        points = [current, *map(np.asarray, waypoints)]
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
        passed = bool(plan.success and safe and same_map and length >= 1.5
                      and check["allowed"] == expected)
        rows.append({"plan_id": plan_id, "expected_allowed": expected,
                     "safe": safe, "same_reachable_component": same_map,
                     "executable_length_m": length, "gate_passed": passed,
                     "guard": check})
        print(f"plano={plan_id} safe={safe} permitido={check['allowed']} "
              f"esperado={expected} progresso_novo={check['endpoint_progress_m']:.2f}m "
              f"gate={passed}", flush=True)
    # The failed battery exposed a valid lateral exit that was not the first
    # ranked candidate. Apply the same policy/search used by the controller.
    alternative_run = ROOT / "datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_101854_360569"
    alternative_events = [
        json.loads(line) for line in (alternative_run / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    saved = next(e["plan"] for e in alternative_events if e.get("plan_id") == 75)
    diag = saved["diagnostics"]
    saved_guard = diag["recovery_return_guard"]
    guard = RecoveryProgressGuard(**{
        item.name: saved_guard[item.name] for item in fields(RecoveryProgressGuard)
        if item.name in saved_guard
    })
    nav = load_planning_snapshot(alternative_run, diag["planning_snapshot_file"])
    current = np.asarray(diag["current_position_ned_m"])
    policy = RecoveryCandidatePolicy(
        guard, diag["waypoint_acceptance_radius_m"], diag["min_executable_path_m"],
        saved_guard["required_extension_m"], nav.config.goal_acceptance_radius_m,
    )
    plan = nav.plan(
        current, saved["requested_goal_ned_m"],
        candidate_validator=lambda candidate: policy.evaluate(nav, current, candidate),
        max_candidates=policy.max_candidates,
    )
    search = plan.diagnostics.get("candidate_search", {})
    check = policy.evaluate(nav, current, plan)
    passed = bool(
        plan.success and check["accepted"]
        and search.get("selected_candidate_index", 0) > 0
        and len(search.get("attempts", [])) <= policy.max_candidates
        and search["attempts"][0]["validation"]["accepted"] is False
    )
    rows.append({
        "case": "alternative_75", "run_dir": str(alternative_run), "plan_id": 75,
        "gate_passed": passed, "validation": check, "candidate_search": search,
        "waypoints_ned_m": plan.waypoints_ned_m, "planning_time_ms": plan.planning_time_ms,
    })
    print(
        f"plano=75 alternativa={search.get('selected_candidate_index')} "
        f"tentativas={len(search.get('attempts', []))}/{policy.max_candidates} "
        f"comprimento={check.get('executable_path_length_m', 0):.2f}m "
        f"tempo={plan.planning_time_ms:.1f}ms gate={passed}", flush=True,
    )
    # Exact snapshots from the canopy blockage: the old rank-only prefix of
    # 16 exhausted its budget despite safe lateral exits elsewhere in the list.
    canopy_run = ROOT / "datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_135536_325462"
    canopy_plans = {
        event["plan_id"]: event["plan"]
        for line in (canopy_run / "events.jsonl").read_text().splitlines()
        if line.strip()
        for event in [json.loads(line)]
        if event.get("event") == "plan" and event.get("map_kind") == "estimated"
    }
    for plan_id in (19, 29, 39):
        saved = canopy_plans[plan_id]
        diag = saved["diagnostics"]
        policy = RecoveryCandidatePolicy.from_dict(diag["recovery_candidate_policy"])
        current = np.asarray(diag["current_position_ned_m"])
        baseline_nav = load_planning_snapshot(canopy_run, diag["planning_snapshot_file"])
        with patch.object(
            baseline_nav, "_spatial_candidate_indices",
            side_effect=lambda candidates, limit: list(range(min(len(candidates), limit))),
        ):
            baseline = baseline_nav.plan(
                current, saved["requested_goal_ned_m"],
                candidate_validator=lambda candidate: policy.evaluate(
                    baseline_nav, current, candidate
                ),
                max_candidates=policy.max_candidates,
            )
        nav = load_planning_snapshot(canopy_run, diag["planning_snapshot_file"])
        plan = nav.plan(
            current, saved["requested_goal_ned_m"],
            candidate_validator=lambda candidate: policy.evaluate(nav, current, candidate),
            max_candidates=policy.max_candidates,
        )
        search = plan.diagnostics.get("candidate_search", {})
        check = policy.evaluate(nav, current, plan)
        passed = bool(
            not baseline.success and plan.success and check["accepted"]
            and plan.diagnostics.get("final_path_safety", {}).get("safe") is True
            and plan.diagnostics["reachable_voxels"] == baseline.diagnostics["reachable_voxels"]
            and policy.max_candidates == 16
            and len(search.get("attempts", [])) <= 16
            and search["attempts"][0]["validation"]["accepted"] is False
            and plan.frontier_standoff_applied_m >= nav.config.frontier_standoff_m - 1e-9
        )
        rows.append({
            "case": f"canopy_{plan_id}", "run_dir": str(canopy_run), "plan_id": plan_id,
            "gate_passed": passed, "validation": check, "candidate_search": search,
            "baseline_rank_only_success": baseline.success,
            "baseline_planning_time_ms": baseline.planning_time_ms,
            "planning_time_ms": plan.planning_time_ms,
            "waypoints_ned_m": plan.waypoints_ned_m,
            "final_path_safety": plan.diagnostics.get("final_path_safety"),
            "frontier_standoff_applied_m": plan.frontier_standoff_applied_m,
            "navigation_config": vars(nav.config),
        })
        print(
            f"copa={plan_id} anterior={baseline.success} "
            f"tentativas={len(search.get('attempts', []))}/16 "
            f"comprimento={check.get('executable_path_length_m', 0):.2f}m "
            f"tempo={plan.planning_time_ms:.1f}ms gate={passed}", flush=True,
        )
    handoff_run = ROOT / "datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_152441_008261"
    handoff_events = [
        json.loads(line) for line in (handoff_run / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    handoff_plans = {e["plan_id"]: e["plan"] for e in handoff_events
                    if e.get("event") == "plan" and e.get("map_kind") == "estimated"}
    saved = handoff_plans[47]
    diag = saved["diagnostics"]
    nav = load_planning_snapshot(handoff_run, diag["planning_snapshot_file"])
    current = np.asarray(diag["current_position_ned_m"])
    # The old recorder did not store the active index at handoff. This case
    # explicitly evaluates the last segment, already approached in plan 47.
    old_remaining = handoff_plans[45]["waypoints_ned_m"][-1:]
    candidate = saved["waypoints_ned_m"]
    handoff = evaluate_path_handoff(
        current, old_remaining, candidate, saved["requested_goal_ned_m"],
        existing_safe=nav.path_is_safe(current, old_remaining),
        candidate_safe=nav.path_is_safe(current, candidate),
        acceptance_m=diag["waypoint_acceptance_radius_m"],
        min_extension_m=1., require_extension=True,
    )
    passed = bool(
        handoff["action"] == "preserve"
        and handoff["reason"] == "opposite_direction_keep_safe_path"
        and handoff["existing_path_safe"] and handoff["candidate_path_safe"]
        and handoff["endpoint_progress_gain_m"] > 1.
    )
    rows.append({
        "case": "handoff_45_47", "gate_passed": passed, "run_dir": str(handoff_run),
        "decision": handoff, "current_position_ned_m": current,
        "remaining_path_assumption": "last waypoint of plan 45; old active index not recorded",
        "existing_remaining_waypoints_ned_m": old_remaining,
        "candidate_waypoints_ned_m": candidate,
    })
    print(f"troca=45->47 decisao={handoff['action']} "
          f"angulo={handoff['turn_angle_deg']:.1f}graus gate={passed}", flush=True)

    saved = handoff_plans[49]
    nav = load_planning_snapshot(handoff_run, saved["diagnostics"]["planning_snapshot_file"])
    planned_position = np.asarray(saved["diagnostics"]["current_position_ned_m"])
    later_position = np.asarray(next(e["position_ned_m"] for e in handoff_events
                                    if e.get("state") == "mission_aborted"))
    old_pose_safe = nav.position_is_safe(planned_position)
    later_pose_safe = nav.position_is_safe(later_position)
    # Do NOT turn this failed historical plan into success. Its starting voxel
    # must still be vetoed; only the delayed abort decision uses the newer pose.
    vetoed = nav.plan(planned_position, saved["requested_goal_ned_m"])
    passed = bool(not old_pose_safe and later_pose_safe and not vetoed.success
                  and vetoed.diagnostics["final_path_safety"]["failure_reason"] == "inflated")
    rows.append({
        "case": "abort_pose_49", "gate_passed": passed, "run_dir": str(handoff_run),
        "map_source": saved["diagnostics"]["planning_snapshot_file"],
        "planning_position_ned_m": planned_position, "later_position_ned_m": later_position,
        "planning_position_safe": old_pose_safe, "later_position_safe": later_pose_safe,
        "unsafe_initial_voxel_still_vetoed": not vetoed.success,
        "note": "Two recorded poses checked on map 49, not a counterfactual flight replay.",
    })
    print(f"abort=49 pose_antiga_segura={old_pose_safe} pose_posterior_segura={later_pose_safe} "
          f"veto_inflacao_mantido={not vetoed.success} gate={passed}", flush=True)
    segment_run = ROOT / "datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260827_165729_539035"
    segment_events = [
        json.loads(line) for line in (segment_run / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    saved = next(e["plan"] for e in segment_events
                 if e.get("map_kind") == "estimated" and e.get("plan_id") == 51)
    diag = saved["diagnostics"]
    nav = load_planning_snapshot(segment_run, diag["planning_snapshot_file"])
    handoff = diag["path_handoff"]
    old_command_safety = nav.path_safety_diagnostics(
        handoff["current_position_ned_m"], handoff["candidate_waypoints_ned_m"]
    )
    with patch.object(nav.grid, "segment_voxels", nav.grid._ray_voxels):
        old_sampled_safe = nav.path_is_safe(
            handoff["current_position_ned_m"], handoff["candidate_waypoints_ned_m"]
        )
    policy = RecoveryCandidatePolicy.from_dict(diag["recovery_candidate_policy"])
    current = np.asarray(diag["current_position_ned_m"])
    plan = nav.plan(
        current, saved["requested_goal_ned_m"],
        candidate_validator=lambda candidate: policy.evaluate(nav, current, candidate),
        max_candidates=policy.max_candidates,
    )
    check = policy.evaluate(nav, current, plan)
    later_pose_check = policy.evaluate(nav, handoff["current_position_ned_m"], plan)
    passed = bool(
        old_sampled_safe and not old_command_safety["safe"]
        and old_command_safety["first_unsafe_voxel"] == [-41, 68, -3]
        and old_command_safety["failure_reason"] == "inflated"
        and plan.success and check["accepted"] and later_pose_check["accepted"]
        and len(plan.diagnostics["candidate_search"]["attempts"]) <= 16
    )
    rows.append({
        "case": "segment_51", "gate_passed": passed, "run_dir": str(segment_run),
        "old_sampled_safe": old_sampled_safe, "old_command_safety": old_command_safety,
        "new_validation": check, "later_pose_validation": later_pose_check,
        "waypoints_ned_m": plan.waypoints_ned_m,
        "candidate_search": plan.diagnostics.get("candidate_search"),
        "planning_time_ms": plan.planning_time_ms,
    })
    print(f"segmento=51 antigo_seguro={old_sampled_safe} "
          f"veto_exato={old_command_safety['failure_reason']} "
          f"novo_comprimento={check.get('executable_path_length_m', 0):.2f}m gate={passed}",
          flush=True)
    rows.extend(replay_recovery_cycle())
    rows.append(replay_active_exit_deadline())
    rows.extend(replay_observation_frontiers())
    report = {
        "diagnostic_only": True, "closed_loop_validation": False,
        "run_dir": str(run), "all_gates_passed": all(r["gate_passed"] for r in rows),
        "note": (
            "Testa decisoes sobre mapas/poses realmente registrados. Nao garante "
            "que o drone parado apos recuar observaria os mesmos frames futuros."
        ),
        "cases": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, default=lambda value: (
            value.item() if isinstance(value, np.generic) else value.tolist()
        )), encoding="utf-8",
    )
    print("Nenhum processo de simulacao iniciado ou encerrado.")
    print("Gate offline nao libera bateria; proximo passo e uma run supervisionada.")
    return 0 if report["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
