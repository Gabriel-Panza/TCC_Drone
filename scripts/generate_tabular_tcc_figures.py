#!/usr/bin/env python3
"""Regenera as figuras tabulares de resultados usadas no artigo e na monografia."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "estudos_e_analises" / "logs"
OUTPUT_DIRS = (
    ROOT / "Parte_Escrita" / "ModeloTCC_Artigo_CC_Latex" / "figuras",
    ROOT / "Parte_Escrita" / "ModeloTCC_Monografia_CC_Latex" / "figuras",
)
TARGET_LABELS = {
    "delta_depth_p10_m": "P10 (m)",
    "delta_depth_p50_m": "P50 (m)",
    "delta_depth_p90_m": "P90 (m)",
    "delta_depth_close_2m_pp": "<2 m (p.p.)",
    "delta_depth_close_5m_pp": "<5 m (p.p.)",
    "delta_depth_close_10m_pp": "<10 m (p.p.)",
}


def save_all(figure, filename):
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def learning_curve():
    results = pd.read_csv(LOGS / "comparacao_mlp_splits_fixos.csv")
    robust = pd.read_csv(LOGS / "robustez_multiplas_sementes_resumo.csv")
    results = results.query("split == 'teste'").copy()
    robust = robust.query("split == 'teste'").set_index("alvo_delta")
    targets = list(TARGET_LABELS)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, target in zip(axes.flat, targets):
        subset = results.query("alvo_delta == @target")
        for model, group in subset.groupby("modelo"):
            group = group.sort_values("marco_runs")
            axis.plot(group["marco_runs"], group["MAE"], marker="o", label=model)
        final = subset["marco_runs"].max()
        if target in robust.index:
            row = robust.loc[target]
            axis.errorbar(
                [final], [row["MAE_medio"]], yerr=[row["MAE_ic95"]],
                fmt="s", color="black", capsize=4, label="MLP: IC95 (5 sementes)",
            )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Execuções disponíveis")
        axis.set_ylabel("MAE")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("MAE no teste fixo; IC95 disponível no marco final")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_all(figure, "resultados_curva_aprendizado.png")


def loss_curves():
    curves = pd.read_csv(LOGS / "curvas_loss_mlp.csv")
    figure, axis = plt.subplots(figsize=(9, 5))
    for split, group in curves.groupby("split"):
        axis.plot(group["epoca"], group["mse_padronizado"], label=split)
    validation = curves.query("split == 'validacao'")
    best_epoch = int(validation.loc[validation["mse_padronizado"].idxmin(), "epoca"])
    axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.7)
    axis.text(best_epoch, axis.get_ylim()[1], f" melhor época: {best_epoch}", va="top")
    axis.set_xlabel("Época")
    axis.set_ylabel("MSE padronizado")
    axis.set_title("Curvas de perda da MLP no marco final")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_all(figure, "resultados_curvas_loss.png")


def attribution_and_ablation():
    importance = pd.read_csv(LOGS / "importancia_gradiente_erro_validacao.csv")
    ablation = pd.read_csv(LOGS / "ablacao_grupos_entradas_resumo.csv")
    importance_group = (
        importance.groupby("grupo", as_index=False)["importancia_pct"].mean()
        .sort_values("importancia_pct", ascending=False)
    )
    ablation_group = (
        ablation.groupby("grupo_removido", as_index=False)
        .agg(
            media=("delta_MAE_pct_medio", "mean"),
            desvio=("delta_MAE_pct_desvio", "mean"),
        )
        .fillna({"desvio": 0.0})
        .sort_values("media", ascending=False)
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(importance_group["grupo"], importance_group["importancia_pct"])
    axes[0].set_title("Sensibilidade média pelo gradiente")
    axes[0].set_ylabel("Importância (%)")
    axes[1].bar(
        ablation_group["grupo_removido"], ablation_group["media"],
        yerr=ablation_group["desvio"], capsize=4,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Ablação: média e dispersão entre sementes")
    axes[1].set_ylabel("Variação do MAE (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_all(figure, "resultados_explicabilidade_ablacao.png")


def main():
    learning_curve()
    loss_curves()
    attribution_and_ablation()


if __name__ == "__main__":
    main()
