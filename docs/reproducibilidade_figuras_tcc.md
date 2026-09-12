# Reprodutibilidade das figuras do TCC

## Ambiente confirmado

- Base do código: commit f7b5c29859c7fec74e82b02e7fa811a903056c86.
- Python 3.10.12.
- NumPy 1.26.4.
- pandas 2.3.3.
- Matplotlib 3.10.8.
- scikit-learn 1.5.2.
- PyTorch 2.12.0+cu130.
- OpenCV 4.11.0.

## Figuras tabulares de resultados

Comando único: python3 scripts/generate_tabular_tcc_figures.py

O script, com SHA-256
e0e883ff4cb2a4cbad693d6103de870477768b59438f4fa07f77064a39de1d68,
grava as mesmas três imagens nos diretórios de figuras do artigo e da
monografia. Os hashes dos CSVs de entrada e dos PNGs resultantes estão em
docs/manifesto_figuras_tcc.tsv.

A curva de crescimento apresenta intervalo de confiança de 95% apenas no
marco final, pois somente esse marco possui cinco repetições por semente.
Não se atribui incerteza aos marcos anteriores. A figura de ablação usa a
dispersão registrada nas três sementes pareadas.

## Limite de rastreabilidade

As quatro figuras metodológicas anteriores não possuem um comando autônomo
de regeneração. O manifesto preserva seus hashes e a origem conhecida, mas
não interpreta a data observada no sistema de arquivos como prova da data de
geração. Reproduzi-las integralmente requer executar e revisar as células
correspondentes de estudos_e_analises/estudo_das_metricas.ipynb com acesso
aos dados brutos do laboratório.
