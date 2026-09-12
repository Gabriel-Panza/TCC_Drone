#!/usr/bin/env python3
"""Executa extensoes estatisticas do notebook tabular sem gerar PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


EXTENSION = r"""
from scipy import stats

SEMENTES_EXTENDIDAS = (7, 21, 42, 84, 123)
diretorio_extendido = localizar_raiz_projeto_memmap() / "estudos_e_analises" / "logs"

linhas_crescimento = []
for marco_runs in marcos_comparacao(len(runs_ordenadas_mlp)):
    X_marco, y_marco, meta_marco = recortar_marco_runs(
        X_mlp_ciclico, y_mlp, meta_mlp, runs_ordenadas_mlp, marco_runs
    )
    split_marco = separar_intervalos(meta_marco)
    detalhado, _ = executar_multiplas_sementes(
        X_marco, y_marco, split_marco, criar_mlp_experimento, SEMENTES_EXTENDIDAS
    )
    linhas_crescimento.append(detalhado.assign(marco_runs=marco_runs))
crescimento_extendido = pd.concat(linhas_crescimento, ignore_index=True)
crescimento_resumo = (
    crescimento_extendido.groupby(["marco_runs", "split", "alvo_delta"], as_index=False)
    .agg(MAE_medio=("MAE", "mean"), MAE_desvio=("MAE", "std"), n=("MAE", "size"))
)
crescimento_resumo["IC95_MAE"] = stats.t.ppf(.975, crescimento_resumo["n"] - 1) * crescimento_resumo["MAE_desvio"] / np.sqrt(crescimento_resumo["n"])
crescimento_extendido.to_csv(diretorio_extendido / "curva_crescimento_cinco_sementes.csv", index=False)
crescimento_resumo.to_csv(diretorio_extendido / "curva_crescimento_cinco_sementes_resumo.csv", index=False)

linhas_arvores = []
for semente in SEMENTES_EXTENDIDAS:
    resultado, _ = executar_baselines_arvores(
        X_experimentos, y_experimentos, split_experimentos, random_state=semente
    )
    linhas_arvores.append(resultado.assign(semente=semente))
arvores_extendido = pd.concat(linhas_arvores, ignore_index=True)
arvores_resumo = (
    arvores_extendido.groupby(["modelo", "split", "alvo_delta"], as_index=False)
    .agg(MAE_medio=("MAE", "mean"), MAE_desvio=("MAE", "std"), n=("MAE", "size"))
)
arvores_resumo["IC95_MAE"] = stats.t.ppf(.975, arvores_resumo["n"] - 1) * arvores_resumo["MAE_desvio"] / np.sqrt(arvores_resumo["n"])
arvores_extendido.to_csv(diretorio_extendido / "baseline_arvores_cinco_sementes.csv", index=False)
arvores_resumo.to_csv(diretorio_extendido / "baseline_arvores_cinco_sementes_resumo.csv", index=False)

mlp_teste = resultados_sementes.query("split == 'teste'")[["semente", "alvo_delta", "MAE"]].rename(columns={"MAE": "MAE_MLP"})
comparacoes = []
for modelo in ("Random Forest", "Extra Trees", "Media do treino"):
    arvore = arvores_extendido.query("split == 'teste' and modelo == @modelo")[["semente", "alvo_delta", "MAE"]].rename(columns={"MAE": "MAE_comparador"})
    pares = mlp_teste.merge(arvore, on=["semente", "alvo_delta"], validate="one_to_one")
    for alvo, grupo in pares.groupby("alvo_delta"):
        diferenca = grupo["MAE_MLP"].to_numpy() - grupo["MAE_comparador"].to_numpy()
        t_stat, p_valor = stats.ttest_rel(grupo["MAE_MLP"], grupo["MAE_comparador"])
        desvio = diferenca.std(ddof=1)
        comparacoes.append({
            "comparador": modelo, "alvo_delta": alvo, "n_pares": len(grupo),
            "diferenca_MAE_media_MLP_menos_comparador": diferenca.mean(),
            "cohen_dz": diferenca.mean() / desvio if desvio > 0 else np.nan,
            "t_pareado": t_stat, "p_valor_bilateral": p_valor,
        })
pd.DataFrame(comparacoes).to_csv(diretorio_extendido / "comparacao_pareada_tamanho_efeito.csv", index=False)

ablacao_extendida, ablacao_comparacao_extendida = executar_ablacao_grupos(
    X_experimentos, y_experimentos, split_experimentos, criar_mlp_experimento,
    classificar_grupo_feature, sementes=SEMENTES_EXTENDIDAS,
)
ablacao_extendida.to_csv(diretorio_extendido / "ablacao_grupos_cinco_sementes.csv", index=False)
ablacao_comparacao_extendida.to_csv(diretorio_extendido / "ablacao_grupos_cinco_sementes_comparacao.csv", index=False)
ablacao_resumo_extendido = (
    ablacao_comparacao_extendida.query("split == 'teste' and grupo_removido != 'Nenhum'")
    .groupby(["alvo_delta", "grupo_removido"], as_index=False)
    .agg(delta_MAE_pct_medio=("delta_MAE_pct", "mean"), delta_MAE_pct_desvio=("delta_MAE_pct", "std"), n=("delta_MAE_pct", "size"))
)
ablacao_resumo_extendido["IC95_delta_MAE_pct"] = stats.t.ppf(.975, ablacao_resumo_extendido["n"] - 1) * ablacao_resumo_extendido["delta_MAE_pct_desvio"] / np.sqrt(ablacao_resumo_extendido["n"])
ablacao_resumo_extendido.to_csv(diretorio_extendido / "ablacao_grupos_cinco_sementes_resumo.csv", index=False)
print("EXTENSAO_TABULAR_CONCLUIDA")
"""

FIX_LEGACY_DATASET = r"""
# O eixo tabular publicado usa exclusivamente as 40 execucoes historicas.
_legacy_dir = localizar_raiz_projeto_memmap() / "datasets" / "depth_ground_truth" / "old"
list_analysis_run_dirs = lambda _project_root: sorted(
    path.parent for path in _legacy_dir.glob("run_*/manifest.json")
)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=Path("estudos_e_analises/estudo_das_metricas.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/estudo_das_metricas_extendido.ipynb"))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    notebook.cells.insert(14, nbformat.v4.new_code_cell(FIX_LEGACY_DATASET))
    notebook.cells.append(nbformat.v4.new_code_cell(EXTENSION))
    NotebookClient(
        notebook, timeout=args.timeout, kernel_name="python3",
        resources={"metadata": {"path": str(args.notebook.parent.resolve())}},
    ).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)


if __name__ == "__main__":
    main()
