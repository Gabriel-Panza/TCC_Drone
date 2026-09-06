#!/usr/bin/env python3
"""Seleciona hard frames proximos a regioes criticas sem contaminar validacao."""

import argparse
import hashlib
import json
from math import atan2, degrees
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_target(value):
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target deve ser x,y,z")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--target", type=parse_target, action="append", required=True)
    parser.add_argument("--exclude-run", type=Path, action="append", default=[])
    parser.add_argument("--radius-m", type=float, default=10.0)
    parser.add_argument("--per-run-target", type=int, default=3)
    parser.add_argument("--yaw-bin-deg", type=float, default=45.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-paths", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.training_metadata.read_text(encoding="utf-8"))
    excluded = {path.resolve() for path in args.exclude_run}
    source_runs = []
    for value in [*metadata.get("train_runs", []), *metadata.get("focus_runs", [])]:
        run = Path(value).resolve()
        if run not in source_runs and run not in excluded:
            source_runs.append(run)
    if not source_runs:
        raise RuntimeError("nenhuma run de treino disponivel")

    candidates = []
    for run in source_runs:
        for frame in sorted((run / "frames").glob("frame_*.npz")):
            try:
                with np.load(frame) as sample:
                    if (
                        sample["rgb_bgr"].size == 0
                        or sample["reference_depth_m"].size == 0
                    ):
                        continue
                    transform = np.asarray(sample["camera_to_ned"], dtype=float)
            except Exception:
                continue
            position = transform[:3, 3]
            optical_forward = transform[:3, 2]
            yaw_deg = degrees(atan2(optical_forward[1], optical_forward[0]))
            yaw_bin = int(np.floor((yaw_deg + 180.0) / args.yaw_bin_deg))
            for target_index, target in enumerate(args.target):
                target = np.asarray(target, dtype=float)
                horizontal_distance = float(
                    np.linalg.norm(position[:2] - target[:2])
                )
                if horizontal_distance <= args.radius_m:
                    candidates.append(
                        {
                            "frame": frame,
                            "run": run,
                            "target_index": target_index,
                            "horizontal_distance_m": horizontal_distance,
                            "position_ned_m": position.tolist(),
                            "yaw_deg": yaw_deg,
                            "yaw_bin": yaw_bin,
                        }
                    )

    selected = []
    for run in source_runs:
        for target_index in range(len(args.target)):
            group = [
                item
                for item in candidates
                if item["run"] == run and item["target_index"] == target_index
            ]
            group.sort(key=lambda item: item["horizontal_distance_m"])
            chosen = []
            used_bins = set()
            for item in group:
                if item["yaw_bin"] in used_bins:
                    continue
                chosen.append(item)
                used_bins.add(item["yaw_bin"])
                if len(chosen) >= args.per_run_target:
                    break
            if len(chosen) < args.per_run_target:
                for item in group:
                    if item in chosen:
                        continue
                    chosen.append(item)
                    if len(chosen) >= args.per_run_target:
                        break
            selected.extend(chosen)

    unique = {}
    for item in selected:
        unique[item["frame"].resolve()] = item
    selected = sorted(unique.values(), key=lambda item: str(item["frame"]))
    if not selected:
        raise RuntimeError("nenhum hard frame encontrado")
    contamination = [
        str(item["frame"])
        for item in selected
        if any(run in item["frame"].resolve().parents for run in excluded)
    ]
    if contamination:
        raise RuntimeError("contaminacao detectada: " + ", ".join(contamination))

    payload = {
        "training_metadata": str(args.training_metadata.resolve()),
        "training_metadata_sha256": sha256(args.training_metadata),
        "targets_ned_m": args.target,
        "radius_m": args.radius_m,
        "per_run_target": args.per_run_target,
        "yaw_bin_deg": args.yaw_bin_deg,
        "source_runs": [str(path) for path in source_runs],
        "excluded_runs": [str(path) for path in sorted(excluded)],
        "selected_count": len(selected),
        "selected": [
            {
                **{key: value for key, value in item.items() if key not in ("frame", "run")},
                "frame": str(item["frame"].resolve()),
                "run": str(item["run"]),
                "sha256": sha256(item["frame"]),
            }
            for item in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.print_paths:
        for item in selected:
            print(item["frame"].resolve())
    else:
        print(f"Hard frames: {len(selected)}")
        print(f"Manifesto: {args.output.resolve()}")


if __name__ == "__main__":
    main()
