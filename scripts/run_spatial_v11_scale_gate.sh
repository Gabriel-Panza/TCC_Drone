#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output="logs/spatial_offline_sweep/manual_v11_scale_01"

echo "1/4 Limpando processos residuais de PX4/Gazebo..."
./scripts/cleanup_spatial_sim.sh

echo "2/4 Rodando regressao espacial..."
python3 -m unittest tests.test_spatial_mapping -v

echo "3/4 Comparando escalas calibradas do v11..."
python3 scripts/run_spatial_offline_sweep.py \
  --matrix config/spatial_v11_scale_sweep.json \
  --prediction-cache-root logs/spatial_offline_sweep/manual_v11_validation_01/prediction_cache \
  --output "$output"

echo "4/4 Resumindo planos criticos 5--8..."
python3 scripts/summarize_spatial_blockage.py "$output"

echo
echo "Concluido. Envie:"
echo "  $output/ranking.csv"
echo "  $output/summary.csv"
echo "  $output/critical_plans_5_8.csv"
echo "Nao execute SITL antes da analise."
