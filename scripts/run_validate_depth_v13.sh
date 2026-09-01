#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
model="models/depth_anything_v2_metric_baylands_vits_v13_686x518_fp32.onnx"
output="logs/spatial_offline_sweep/manual_v13_validation_01"
calibration="datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260820_224244_854728"
heldout="datasets/spatial_mapping/run_20260819_191346_877996"
independent="datasets/spatial_mapping/run_20260820_225114_611267"

[[ -f "$model" ]] || { echo "Modelo v13 ausente; rode ./scripts/run_train_depth_v13.sh" >&2; exit 2; }

echo "1/6 Limpando apenas PX4/Gazebo; MicroXRCEAgent sera preservado..."
./scripts/cleanup_spatial_sim.sh
echo "2/6 Rodando regressoes..."
models/train_env/bin/python -m unittest tests.test_depth_training_loss -v
python3 -m unittest tests.test_spatial_mapping -v
echo "3/6 Avaliando profundidade na validacao antiga..."
models/runtime_env/bin/python estudos_e_analises/avaliar_profundidade_monocular.py \
  --calibration-run "$calibration" --validation-run "$heldout" \
  --model "$model" --frames 24 --output models/monocular_validation_v13_heldout.json
echo "4/6 Avaliando profundidade na run independente reservada..."
models/runtime_env/bin/python estudos_e_analises/avaliar_profundidade_monocular.py \
  --calibration-run "$calibration" --validation-run "$independent" \
  --model "$model" --frames 24 --output models/monocular_validation_v13_independent.json
echo "5/6 Executando replay integral e gate de colisao..."
python3 scripts/run_spatial_offline_sweep.py \
  --matrix config/spatial_v13_validation_sweep.json \
  --prediction-cache-root "$output/prediction_cache" --output "$output"
echo "6/6 Resumindo planos criticos 5--8..."
python3 scripts/summarize_spatial_blockage.py "$output"
echo
echo "Concluido. Envie:"
echo "  models/monocular_validation_v13_heldout.json"
echo "  models/monocular_validation_v13_independent.json"
echo "  $output/ranking.csv"
echo "  $output/summary.csv"
echo "  $output/critical_plans_5_8.csv"
echo "Este comando nao inicia PX4/Gazebo e preserva o MicroXRCEAgent."
echo "Nao execute SITL antes da analise."
