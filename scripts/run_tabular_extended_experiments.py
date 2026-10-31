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
assert len(X_experimentos) == 1234 and meta_experimentos.run_id.nunique() == 40

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
for modelo in ("Random Forest", "Extra Trees", "Media treino"):
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

# Uma unica bateria final: o modelo completo da ablacao e a referencia de
# todos os contrastes e do marco 40. Evita misturar treinamentos distintos.
referencia_final = ablacao_extendida.query("grupo_removido == 'Nenhum'").copy()
referencia_final.to_csv(diretorio_extendido / "robustez_multiplas_sementes.csv", index=False)
resumo_final = referencia_final.groupby(["split", "alvo_delta"], as_index=False).agg(
    MAE_medio=("MAE", "mean"), MAE_desvio=("MAE", "std"),
    MAE_minimo=("MAE", "min"), MAE_maximo=("MAE", "max"),
    RMSE_medio=("RMSE", "mean"), RMSE_desvio=("RMSE", "std"),
    corr_media=("corr", "mean"), melhor_epoca_media=("melhor_epoca", "mean"),
)
resumo_final["MAE_ic95"] = stats.t.ppf(.975, 4) * resumo_final.MAE_desvio / np.sqrt(5)
resumo_final.to_csv(diretorio_extendido / "robustez_multiplas_sementes_resumo.csv", index=False)
crescimento_extendido = pd.concat([
    crescimento_extendido.query("marco_runs != 40"), referencia_final.assign(marco_runs=40)
], ignore_index=True)
crescimento_extendido.to_csv(diretorio_extendido / "curva_crescimento_cinco_sementes.csv", index=False)
crescimento_resumo = crescimento_extendido.groupby(["marco_runs", "split", "alvo_delta"], as_index=False).agg(
    MAE_medio=("MAE", "mean"), MAE_desvio=("MAE", "std"), n=("MAE", "size"),
)
crescimento_resumo["IC95_MAE"] = stats.t.ppf(.975, crescimento_resumo.n - 1) * crescimento_resumo.MAE_desvio / np.sqrt(crescimento_resumo.n)
crescimento_resumo.to_csv(diretorio_extendido / "curva_crescimento_cinco_sementes_resumo.csv", index=False)
mlp_teste = referencia_final.query("split == 'teste'")[["semente", "alvo_delta", "MAE"]].rename(columns={"MAE": "MAE_MLP"})
comparacoes = []
for nome_modelo in ("Random Forest", "Extra Trees", "Media treino"):
    outro = arvores_extendido.query("split == 'teste' and modelo == @nome_modelo")[["semente", "alvo_delta", "MAE"]].rename(columns={"MAE": "MAE_comparador"})
    pares = mlp_teste.merge(outro, on=["semente", "alvo_delta"], validate="one_to_one")
    assert len(pares) == 30
    for alvo, grupo in pares.groupby("alvo_delta"):
        diferenca = grupo.MAE_MLP - grupo.MAE_comparador
        teste = stats.ttest_rel(grupo.MAE_MLP, grupo.MAE_comparador)
        comparacoes.append(dict(comparador=nome_modelo, alvo_delta=alvo, n_pares=len(grupo),
            diferenca_MAE_media_MLP_menos_comparador=diferenca.mean(),
            cohen_dz=diferenca.mean() / diferenca.std(ddof=1),
            t_pareado=teste.statistic, p_valor_bilateral=teste.pvalue))
pd.DataFrame(comparacoes).to_csv(diretorio_extendido / "comparacao_pareada_tamanho_efeito.csv", index=False)

# Explicacao correspondente ao modelo final completo, semente 42.
x_final, y_final = arrays_validos_experimento(X_experimentos[list(X_experimentos.columns)], y_experimentos)
modelo_final = criar_mlp_experimento(42)
modelo_final.fit(x_final[split_experimentos["train"]], y_final[split_experimentos["train"]],
    x_final[split_experimentos["val"]], y_final[split_experimentos["val"]],
    x_final[split_experimentos["test"]], y_final[split_experimentos["test"]])
metricas_finais = pd.DataFrame(avaliar_delta(y_final[split_experimentos["test"]],
    modelo_final.predict(x_final[split_experimentos["test"]]), y_experimentos.columns, "MLP", "teste"))
verificacao = metricas_finais.merge(referencia_final.query("split == 'teste' and semente == 42"), on="alvo_delta")
assert np.allclose(verificacao.MAE_x, verificacao.MAE_y, rtol=1e-5, atol=1e-7)
gradiente_final, _, _ = calcular_importancia_gradiente_erro(modelo_final,
    X_experimentos.iloc[split_experimentos["val"]], y_experimentos.iloc[split_experimentos["val"]],
    list(X_experimentos.columns), list(y_experimentos.columns))
gradiente_final["representacao_angular"] = "seno/cosseno"
gradiente_final.to_csv(diretorio_extendido / "importancia_gradiente_erro_validacao.csv", index=False)
resultados_classificacao_evento.to_csv(diretorio_extendido / "eventos_classificacao_final_40.csv", index=False)
resultados_regressao_evento.to_csv(diretorio_extendido / "eventos_regressao_final_40.csv", index=False)
print("COORTE_HISTORICA_RECONCILIADA")
"""

FIX_LEGACY_DATASET = r"""
# O eixo tabular publicado usa exclusivamente as 40 execucoes historicas.
_legacy_dir = localizar_raiz_projeto_memmap() / "datasets" / "depth_ground_truth" / "old"
assert len(list(_legacy_dir.glob("run_*/manifest.json"))) == 40
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
