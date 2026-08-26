#!/usr/bin/env python3
"""Executa uma matriz controlada de replays espaciais e resume os gates."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import shlex
import subprocess
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_yaml_scalar(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
    replacement = rf"\g<1>{value}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"esperava uma ocorrencia de {key}, encontrei {count}")
    return updated


def generate_sitl_config(template, output, configuration):
    text = template.read_text(encoding="utf-8")
    replacements = {
        "spatial_depth_stride": configuration["runtime_depth_stride"],
        "spatial_depth_free_space_margin_m": configuration[
            "free_space_margin_m"
        ],
        "spatial_depth_occupied_uncertainty_m": configuration[
            "occupied_uncertainty_m"
        ],
        "spatial_free_observations_required": configuration[
            "free_observations_required"
        ],
        "spatial_occupied_observations_required": configuration.get(
            "occupied_observations_required", 1
        ),
        "spatial_occupied_support_radius_voxels": configuration.get(
            "occupied_support_radius_voxels", 0
        ),
        "spatial_pending_clear_free_observations_required": configuration.get(
            "pending_clear_free_observations_required", 3
        ),
        "spatial_occupied_evidence_window_frames": configuration.get(
            "occupied_evidence_window_frames", 6
        ),
        "spatial_obstacle_vertical_band_m": configuration[
            "obstacle_vertical_band_m"
        ],
    }
    for key, value in replacements.items():
        text = replace_yaml_scalar(text, key, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def evaluator_command(
    runtime_python,
    evaluator,
    dataset,
    model,
    configuration,
    report,
    cache_dir,
    max_frames,
):
    command = [
        str(runtime_python),
        str(evaluator),
        str(dataset),
        "--model",
        str(model),
        "--output",
        str(report),
        "--prediction-cache-dir",
        str(cache_dir),
        "--replay-stride",
        str(configuration["replay_stride"]),
        "--depth-output-scale",
        str(configuration.get("depth_output_scale", 1.0)),
        "--free-space-margin-m",
        str(configuration["free_space_margin_m"]),
        "--occupied-uncertainty-m",
        str(configuration["occupied_uncertainty_m"]),
        "--obstacle-vertical-band-m",
        str(configuration["obstacle_vertical_band_m"]),
        "--free-observations-required",
        str(configuration["free_observations_required"]),
        "--occupied-observations-required",
        str(configuration.get("occupied_observations_required", 1)),
        "--occupied-support-radius-voxels",
        str(configuration.get("occupied_support_radius_voxels", 0)),
        "--pending-clear-free-observations-required",
        str(configuration.get("pending_clear_free_observations_required", 3)),
        "--occupied-evidence-window-frames",
        str(configuration.get("occupied_evidence_window_frames", 6)),
        "--minimum-executable-path-m",
        str(configuration.get("minimum_executable_path_m", 1.5)),
        "--lock-path-altitude-to-goal",
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    return command


def execute_evaluation(command, report, log, force):
    command_file = report.with_suffix(".command.json")
    command_payload = {"command": command}
    command_matches = (
        command_file.is_file()
        and json.loads(command_file.read_text(encoding="utf-8"))
        == command_payload
    )
    if report.is_file() and command_matches and not force:
        return json.loads(report.read_text(encoding="utf-8")), 0.0, "cached"
    report.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(
        json.dumps(command_payload, indent=2) + "\n", encoding="utf-8"
    )
    started = perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"avaliador terminou com codigo {completed.returncode}; veja {log}"
        )
    return json.loads(report.read_text(encoding="utf-8")), elapsed, "executed"


def result_row(candidate_id, model_id, dataset_id, stage, report, elapsed, source):
    planning = report["planning"]
    occupancy = report["map"]
    successful = planning["successes"]
    collision_free = planning["collision_free_successes"]
    return {
        "candidate": candidate_id,
        "model": model_id,
        "dataset": dataset_id,
        "stage": stage,
        "source": source,
        "passed": report["qualification"]["passed"],
        "frames": report["frames_integrated"],
        "attempts": planning["attempts"],
        "successes": successful,
        "success_rate": planning["success_rate"],
        "collision_free_successes": collision_free,
        "collisions": max(0, successful - collision_free),
        "false_free_rate": occupancy["false_free_rate"],
        "precision": occupancy["precision"],
        "recall": occupancy["recall"],
        "pending_occupied_voxels": report["estimated_map"].get(
            "pending_occupied_voxels", 0
        ),
        "elapsed_s": elapsed,
        "report": "",
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shell_assignment(name, value):
    return f"{name}={shlex.quote(str(value))}"


def guarded_command(config, model, kind, short_waypoints):
    assignments = [
        shell_assignment("SPATIAL_CONFIG", config),
        shell_assignment("MONOCULAR_MODEL_PATH", model),
    ]
    if kind == "short":
        assignments.extend(
            [
                "GUARDED_SHORT_TEST=1",
                shell_assignment(
                    "SPATIAL_WAYPOINTS_RELATIVE_M",
                    json.dumps(short_waypoints),
                ),
            ]
        )
    else:
        assignments.append("GUARDED_ROUTE_TEST=1")
    return (
        " ".join(assignments)
        + " ./scripts/run_spatial_battery.sh monocular_topic 1"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=PROJECT_ROOT / "config/spatial_offline_sweep.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-cache-root", type=Path)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    matrix_path = resolve_project_path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        resolve_project_path(args.output)
        if args.output
        else PROJECT_ROOT / "logs/spatial_offline_sweep" / stamp
    )
    cache_root = (
        resolve_project_path(args.prediction_cache_root)
        if args.prediction_cache_root
        else output / "prediction_cache"
    )
    runtime_python = resolve_project_path(matrix["runtime_python"])
    evaluator = resolve_project_path(matrix["evaluator"])
    template = resolve_project_path(matrix["sitl_template"])
    for required in (runtime_python, evaluator, template):
        if not required.exists():
            raise FileNotFoundError(required)

    datasets = {
        name: {**item, "resolved": resolve_project_path(item["path"])}
        for name, item in matrix["datasets"].items()
    }
    models = {
        name: {**item, "resolved": resolve_project_path(item["path"])}
        for name, item in matrix["models"].items()
    }
    for name, item in datasets.items():
        for required_name in ("manifest.json", "events.jsonl"):
            required = item["resolved"] / required_name
            if not required.is_file():
                raise FileNotFoundError(f"{name}: {required}")
    for name, item in models.items():
        if not item["resolved"].is_file():
            raise FileNotFoundError(f"{name}: {item['resolved']}")
        item["sha256"] = model_digest(item["resolved"])

    candidates = []
    for configuration in matrix["configurations"]:
        for model_id in configuration["models"]:
            candidates.append(
                {
                    "id": f"{model_id}__{configuration['id']}",
                    "model_id": model_id,
                    "configuration": configuration,
                }
            )
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            parser.error("--max-candidates deve ser positivo")
        candidates = candidates[: args.max_candidates]

    print(f"Saida: {output}")
    print(f"Cache de previsoes: {cache_root}")
    print(f"Candidatos: {len(candidates)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrix.snapshot.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    candidate_results = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["id"]
        model_id = candidate["model_id"]
        model = models[model_id]
        configuration = candidate["configuration"]
        generated_config = output / "generated_configs" / f"{candidate_id}.yaml"
        generate_sitl_config(template, generated_config, configuration)
        dataset_ids = [model["primary_dataset"]]
        secondary_ids = list(model.get("secondary_datasets", []))
        passed_primary = False
        evaluated = []
        for stage, dataset_id in [
            ("primary", dataset_ids[0]),
            *[("secondary", item) for item in secondary_ids],
        ]:
            if stage == "secondary" and not passed_primary:
                break
            dataset = datasets[dataset_id]["resolved"]
            report = output / "reports" / candidate_id / f"{dataset_id}.json"
            log = output / "reports" / candidate_id / f"{dataset_id}.log"
            cache = (
                cache_root
                / model["sha256"]
                / dataset_id
            )
            command = evaluator_command(
                runtime_python,
                evaluator,
                dataset,
                model["resolved"],
                configuration,
                report,
                cache,
                args.max_frames,
            )
            print(
                f"[{index}/{len(candidates)}] {candidate_id} "
                f"{stage}:{dataset_id}"
            )
            if args.dry_run:
                print("  " + shlex.join(command))
                continue
            payload, elapsed, source = execute_evaluation(
                command, report, log, args.force
            )
            row = result_row(
                candidate_id,
                model_id,
                dataset_id,
                stage,
                payload,
                elapsed,
                source,
            )
            row["report"] = str(report)
            rows.append(row)
            evaluated.append(payload["qualification"]["passed"])
            if stage == "primary":
                passed_primary = payload["qualification"]["passed"]
        if args.dry_run:
            continue
        all_offline_passed = bool(evaluated) and all(evaluated)
        independent_ready = bool(model.get("independent_validation_ready"))
        eligible = all_offline_passed and independent_ready
        blocked_reason = ""
        if not all_offline_passed:
            blocked_reason = "gate offline reprovado"
        elif not independent_ready:
            blocked_reason = model.get(
                "blocked_reason", "falta referencia independente"
            )
        candidate_results.append(
            {
                "candidate": candidate_id,
                "model": model_id,
                "model_path": str(model["resolved"]),
                "model_sha256": model["sha256"],
                "generated_config": str(generated_config),
                "offline_passed": all_offline_passed,
                "sitl_eligible": eligible,
                "blocked_reason": blocked_reason,
                "configuration": configuration,
            }
        )

    if args.dry_run:
        print("Dry-run concluido; nenhum replay foi executado.")
        return

    row_fields = [
        "candidate",
        "model",
        "dataset",
        "stage",
        "source",
        "passed",
        "frames",
        "attempts",
        "successes",
        "success_rate",
        "collision_free_successes",
        "collisions",
        "false_free_rate",
        "precision",
        "recall",
        "pending_occupied_voxels",
        "elapsed_s",
        "report",
    ]
    write_csv(output / "summary.csv", rows, row_fields)
    primary_by_candidate = {
        row["candidate"]: row for row in rows if row["stage"] == "primary"
    }
    ranking = []
    for item in candidate_results:
        row = primary_by_candidate[item["candidate"]]
        score = (
            1000.0 * row["collisions"]
            + 100.0 * max(0.0, row["false_free_rate"] - 0.10)
            + 10.0 * max(0.0, 0.80 - row["success_rate"])
            + (0.0 if row["passed"] else 1.0)
        )
        ranking.append(
            {
                "rank": 0,
                "candidate": item["candidate"],
                "score": score,
                "primary_passed": row["passed"],
                "sitl_eligible": item["sitl_eligible"],
                "collisions": row["collisions"],
                "false_free_rate": row["false_free_rate"],
                "success_rate": row["success_rate"],
                "generated_config": item["generated_config"],
            }
        )
    ranking.sort(key=lambda item: item["score"])
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    write_csv(output / "ranking.csv", ranking, list(ranking[0]) if ranking else [])
    (output / "results.json").write_text(
        json.dumps(
            {
                "matrix": str(matrix_path),
                "output": str(output),
                "candidates": candidate_results,
                "ranking": ranking,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    short_lines = [
        "# Execute apenas um comando por vez e observe a run antes de continuar.",
        "./scripts/cleanup_spatial_sim.sh",
    ]
    route_lines = [
        "# Execute somente depois de o teste curto correspondente passar.",
        "./scripts/cleanup_spatial_sim.sh",
    ]
    eligible_items = [item for item in candidate_results if item["sitl_eligible"]]
    if eligible_items:
        for item in eligible_items:
            short_lines.append(
                guarded_command(
                    item["generated_config"],
                    item["model_path"],
                    "short",
                    matrix["short_test_waypoints_relative_m"],
                )
            )
            route_lines.append(
                guarded_command(
                    item["generated_config"],
                    item["model_path"],
                    "route",
                    matrix["short_test_waypoints_relative_m"],
                )
            )
    else:
        short_lines.append("# Nenhum candidato liberado pelo gate offline.")
        route_lines.append("# Nenhum candidato liberado pelo gate offline.")
    (output / "sitl_short_commands.txt").write_text(
        "\n".join(short_lines) + "\n", encoding="utf-8"
    )
    (output / "sitl_tree_route_commands.txt").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )
    print(f"Resumo: {output / 'summary.csv'}")
    print(f"Ranking: {output / 'ranking.csv'}")
    print(f"Finalistas SITL: {len(eligible_items)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1)
