#!/usr/bin/env python3
"""Gate offline de regressao do subobjetivo: nao inicia nem encerra o SITL."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("old_37", "run_20260825_221643_307790", 37),
    ("old_41", "run_20260825_221643_307790", 41),
    ("near_goal_79", "run_20260826_084137_260838", 79),
    ("end_93", "run_20260826_084137_260838", 93),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--goal-acceptance-radius-m", type=float, default=1.5,
        help="tolerancia global dos quatro runs historicos (nao salva no manifest)",
    )
    args = parser.parse_args()
    output = args.output or Path(
        "logs/spatial_path_safety_diag"
    ) / ("subgoal_gate_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    print(f"Saida: {output}", flush=True)
    print(f"Tolerancia global explicita: {args.goal_acceptance_radius_m}m")

    tests = subprocess.run(
        [sys.executable, "-m", "unittest",
         "tests.test_spatial_mapping",
         "tests.test_navigation_safety_diagnostics",
         "tests.test_spatial_execution",
         "tests.test_spatial_replay",
         "tests.test_spatial_snapshot",
         "tests.test_spatial_subgoal_bounds", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    (output / "tests.log").write_text(
        tests.stdout + tests.stderr, encoding="utf-8"
    )
    if tests.returncode:
        print(f"Testes falharam: veja {output / 'tests.log'}")
        return 1
    print("Testes: OK", flush=True)

    rows = []
    for label, run_name, plan_id in CASES:
        completed = subprocess.run(
            [sys.executable,
             str(ROOT / "estudos_e_analises/diagnosticar_planejamento_espacial.py"),
             str(ROOT / "datasets/spatial_mapping" / run_name),
             "--plan-id", str(plan_id),
             "--goal-acceptance-radius-m", str(args.goal_acceptance_radius_m)],
            cwd=ROOT, capture_output=True, text=True,
        )
        (output / f"{label}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        # The replay returns 2 for a rejected gate, not an execution error.
        if completed.returncode not in (0, 2):
            print(f"{label}: ERRO, veja {label}.stderr.log")
            return 1
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            print(f"{label}: saida JSON invalida")
            return 1
        (output / f"{label}.json").write_text(
            completed.stdout, encoding="utf-8"
        )
        plan = result["plan"]
        diag = plan.get("diagnostics") or {}
        row = {
            "case": label,
            "run": run_name,
            "plan_id": plan_id,
            "gate": result["gate"]["passed"],
            "success": plan["success"],
            "safe": plan["executable_path_is_safe"],
            "length_m": plan["executable_path_length_m"],
            "selected_goal": plan.get("selected_goal_ned_m"),
            "projection_m": diag.get("selected_goal_projection_m"),
            "remaining_goal_m": diag.get("remaining_global_goal_distance_m"),
            "reason": plan["reason"],
            "online_comparison": result.get("online_comparison"),
        }
        rows.append(row)
        print(
            f"{label}: gate={row['gate']} safe={row['safe']} "
            f"length={row['length_m']:.2f}m "
            f"reason={row['reason']} "
            f"projection={row['projection_m']} "
            f"remaining={row['remaining_goal_m']}",
            flush=True,
        )

        comparison = result.get("online_comparison") or {}
        print(
            "  alcancaveis online/replay="
            f"{comparison.get('saved_reachable_voxels')}/"
            f"{comparison.get('replayed_reachable_voxels')}; "
            "caminho salvo seguro no replay="
            f"{(comparison.get('saved_path_safety_in_replayed_map') or {}).get('safe')}",
            flush=True,
        )

    report = {
        "diagnostic_only": True,
        "closed_loop_validation": False,
        "goal_acceptance_radius_override_m": args.goal_acceptance_radius_m,
        "note": (
            "Replay de frames salvos, nao replica exatamente o mapa online "
            "nem comprova a conclusao de uma missao futura."
        ),
        "all_gates_passed": all(row["gate"] for row in rows),
        "cases": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Nao execute SITL/bateria com gate reprovado.")
    print("Codigo 2 = gate reprovado; codigo 1 = erro de execucao.")
    return 0 if report["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
