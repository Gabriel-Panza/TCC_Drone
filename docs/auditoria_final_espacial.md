# Auditoria final da etapa espacial

## Escopo e commits

A branch auditada e `mapeamento-3d-position-paper`. O commit `c929b8c`
altera apenas artigo, monografia e bibliografias. O commit experimental
`83a810e` concentra 95 arquivos e introduz as configuracoes v13--v19,
planejamento e recuperacao espacial, scripts de treino/validacao/relatorio e
os testes correspondentes. Nenhum resultado abaixo foi inferido pelo nome:
foram lidos JSONs de validacao, proveniencia, CSVs do sweep, relatorios de
replay e o TSV da bateria.

## Resultado consolidado

| Pipeline | Evidencia | Resultado | SITL |
|---|---|---|---|
| profundidade Gazebo, mapa e A* | bateria fixa | 10/10 completas; 30 objetivos; 147/156 planos; 0 caminhos adotados inseguros | executado e aceito |
| monocular v19, mapa e A* | teste por pixel e replay espacial primario | pixel aprovado; gate espacial reprovado | bloqueado |

A referencia teve taxa de sucesso de planejamento de 94,23%, duracao media de
77,37 s e 132 caminhos adotados. Cinquenta vetos e frenagens foram concluidos
sem recuperacao. A taxa de falso espaco livre do mapa de referencia, calculada
contra o proprio mapa acumulado final, foi 31,55%; ela nao deve ser comparada
diretamente com a qualificacao monocular sem explicar o protocolo temporal.

## Modelos v12--v19

| Modelo | MAE independente (m) | Gate pixel | Sucesso espacial | Colisoes | Falso livre | Decisao |
|---|---:|---|---:|---:|---:|---|
| v12 | 1,283 | sim | 27,78% | 0 | 36,71% | rejeitado |
| v13 | 1,242 | sim | 38,89% | 2 | 39,01% | rejeitado |
| v14 | 1,270 | sim | 47,06% | 1 | 43,11% | rejeitado |
| v15 | 1,726 | nao | 41,18% | 0 | 30,42% | rejeitado |
| v16 | 1,383 | sim | 52,94% | 1 | 29,10% | rejeitado |
| v17 | 1,498 | sim | 35,29% | 0 | 18,38% | rejeitado |
| v18 | 1,292 | sim | 52,94% | 1 | 26,26% | rejeitado |
| v19 | 1,286 | sim | 41,18% | 0 | 18,38% | rejeitado |

O v19 final usa o ONNX SHA-256
`b56cbb291b8fade37a0b5f298fdd4bbe5e45de05f92628a1956828b0fc902dc4`.
O melhor candidato do ranking primario foi
`v19_scale1p0_shift1p2`. Ele nao colidiu, mas falhou em numero de frames,
taxa de sucesso de planejamento e falso espaco livre. Nenhum candidato ficou
elegivel para SITL.

## Separacao dos dados

A proveniencia v19 registra 20 runs de treino, quatro runs de foco e quadros
dificeis repetidos. `reference_reserved_03` nao entrou nos gradientes, mas
foi usada como validacao durante o treino e como gate espacial primario. Ela
nao e um teste independente. `reference_reserved_06`,
`reference_reserved_10`, `independent_old` e `heldout_old` estavam
configuradas como secundarias. Como todos os candidatos falharam no primario,
essas etapas foram interrompidas pelo protocolo. Nao ha evidencia de aprovacao
nos conjuntos secundarios e ela nao deve ser alegada.

## Interpretacao

O pipeline geometrico funcionou com profundidade ideal do Gazebo. O v19
atingiu os limites por pixel, mas nao passou o gate espacial completo. A
ausencia de colisao em um replay nao compensa baixa disponibilidade de
caminhos ou falso espaco livre acima do limite. A bateria monocular foi
corretamente bloqueada e a etapa termina como resultado negativo controlado.

## Bibliografia e PDFs

Os dois arquivos BibTeX contem Hart--Nilsson--Raphael, Elfes, Hornung et al.,
Yamauchi, Amanatides--Woo e Depth Anything V2. Os DOI de Hart, Elfes e OctoMap
estavam corretos. Foram acrescentados `10.1109/CIRA.1997.613851` para
Yamauchi e `10.2312/egtp.19871000` para Amanatides--Woo. Depth Anything V2
esta registrado como NeurIPS 2024 com o identificador do arXiv. Nao existem
PDFs no repositorio; portanto, os seis PDFs ainda precisam ser anexados
manualmente na biblioteca JabRef, caso ela deva ser distribuida.

## Artefatos reproduziveis

O comando `python3 scripts/build_spatial_final_report.py` gera
`final_report.json`, `reference_runs.csv`,
`monocular_comparison.csv`, `decision_table.csv` e
`artifact_manifest.json`, incluindo hashes dos insumos congelados.
