#!/usr/bin/env python3
"""Calcula IC95 por bootstrap de execuções para a bateria espacial final."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "spatial_final_analysis" / "final_20260901" / "reference_runs.csv"
OUTPUT = ROOT / "estudos_e_analises" / "logs" / "bateria_espacial_ic95.csv"
SEED = 42
RESAMPLES = 10000


def summarize(sample):
    attempts = sample["plan_attempts"].sum()
    return {
        "sucesso_planejamento": sample["plan_successes"].sum() / attempts,
        "falso_espaco_livre": sample["false_free_rate"].mean(),
        "precisao_ocupacao": sample["precision"].mean(),
        "revocacao_ocupacao": sample["recall"].mean(),
        "vetos_por_execucao": sample["active_path_vetoes"].mean(),
        "frenagens_por_execucao": sample["completed_brakes"].mean(),
    }


def main():
    data = pd.read_csv(SOURCE)
    rng = np.random.default_rng(SEED)
    estimates = summarize(data)
    boot = {metric: [] for metric in estimates}
    for _ in range(RESAMPLES):
        indices = rng.integers(0, len(data), len(data))
        current = summarize(data.iloc[indices])
        for metric, value in current.items():
            boot[metric].append(value)
    rows = []
    for metric, estimate in estimates.items():
        lower, upper = np.percentile(boot[metric], [2.5, 97.5])
        rows.append({
            "metrica": metric,
            "estimativa": estimate,
            "ic95_inferior": lower,
            "ic95_superior": upper,
            "unidade_amostral": "execucao",
            "execucoes": len(data),
            "reamostragens": RESAMPLES,
            "semente": SEED,
        })
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
