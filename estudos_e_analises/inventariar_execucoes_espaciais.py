"""Classifica cada execucao espacial completa em uma categoria principal."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/spatial_mapping"
OUT = ROOT / "estudos_e_analises/logs/inventario_execucoes_espaciais.csv"

def names(values):
    return {Path(value).name for value in values}

def complete(run):
    events = run / "events.jsonl"
    if not (run / "manifest.json").is_file() or not events.is_file():
        return False
    parsed = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    states = {event.get("state") for event in parsed if event.get("event") == "mission_state"}
    return "mission_complete" in states and any(event.get("event") == "summary" for event in parsed)

def main():
    training = json.loads((ROOT / "config/depth_v19_final_training.json").read_text())
    gate = json.loads((ROOT / "config/spatial_v19_final_validation_sweep.json").read_text())
    train = names(training["train_runs"])
    calibration = {Path(training["calibration_run"]).name}
    final_battery = names(training["validation_runs"])
    historical = {Path(training["heldout_run"]).name, Path(training["independent_reserved_run"]).name}
    previous = {"run_20260820_082152_949956"}
    focus = names(training["focus_runs"])
    primary_gate = {Path(gate["datasets"]["reference_reserved_03"]["path"]).name}
    rows = []
    for run in sorted(DATA.glob("run_*")):
        if not complete(run): continue
        if run.name in train: category = "treino_v19"
        elif run.name in calibration: category = "calibracao_v19"
        elif run.name in final_battery: category = "bateria_referencia_final"
        elif run.name in historical: category = "reservado_historico"
        elif run.name in previous: category = "desenvolvimento_modelos_anteriores"
        else: category = "desenvolvimento_diagnostico"
        rows.append({"run": run.name, "categoria_principal": category,
                     "foco_v19": run.name in focus, "portao_primario_v19": run.name in primary_gate})
    counts = Counter(row["categoria_principal"] for row in rows)
    expected = {"desenvolvimento_diagnostico": 69, "treino_v19": 20,
                "bateria_referencia_final": 10, "reservado_historico": 2,
                "calibracao_v19": 1, "desenvolvimento_modelos_anteriores": 1}
    if len(rows) != 103 or dict(counts) != expected:
        raise RuntimeError(f"inventario divergente: total={len(rows)}, categorias={dict(counts)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(OUT)
    print(dict(counts))

if __name__ == "__main__": main()
