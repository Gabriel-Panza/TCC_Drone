#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

echo "1/3 Limpando processos residuais de PX4/Gazebo..."
./scripts/cleanup_spatial_sim.sh

echo "2/3 Rodando regressao espacial..."
python3 -m unittest tests.test_spatial_mapping -v

echo "3/3 Comparando controle, filtro temporal e espaco-temporal..."
python3 scripts/run_spatial_offline_sweep.py \
  --matrix config/spatial_spatiotemporal_sweep.json \
  --prediction-cache-root logs/spatial_offline_sweep/manual_01/prediction_cache \
  --output logs/spatial_offline_sweep/manual_spatiotemporal_01

echo
echo "Concluido. Envie estes dois arquivos:"
echo "  logs/spatial_offline_sweep/manual_spatiotemporal_01/ranking.csv"
echo "  logs/spatial_offline_sweep/manual_spatiotemporal_01/summary.csv"
