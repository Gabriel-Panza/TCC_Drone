#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
model="models/depth_anything_v2_metric_baylands_vits_v14_686x518_fp32.onnx"
output="logs/spatial_offline_sweep/manual_v14_validation_01"
calibration="datasets/spatial_mapping_quarantine/incomplete_20260831/run_20260820_224244_854728"
reserved03="datasets/spatial_mapping/run_20260831_082836_332335"
reserved06="datasets/spatial_mapping/run_20260831_083521_770086"
reserved10="datasets/spatial_mapping/run_20260831_084356_708163"
independent="datasets/spatial_mapping/run_20260820_225114_611267"

[[ -f "$model" ]] || { echo "Modelo v14 ausente; rode ./scripts/run_train_depth_v14.sh" >&2; exit 2; }

echo "1/7 Limpando apenas PX4/Gazebo; MicroXRCEAgent sera preservado..."
./scripts/cleanup_spatial_sim.sh
echo "2/7 Rodando regressoes..."
models/train_env/bin/python -m unittest tests.test_depth_training_loss -v
python3 -m unittest tests.test_spatial_mapping -v
evaluate_depth() {
    local run="$1" output_json="$2"
    models/runtime_env/bin/python estudos_e_analises/avaliar_profundidade_monocular.py \
      --calibration-run "$calibration" --validation-run "$run" \
      --model "$model" --frames 24 --output "$output_json"
}
echo "3/7 Avaliando profundidade nas runs reservadas 03 e 06..."
evaluate_depth "$reserved03" models/monocular_validation_v14_reserved03.json
evaluate_depth "$reserved06" models/monocular_validation_v14_reserved06.json
echo "4/7 Avaliando profundidade na run reservada 10..."
evaluate_depth "$reserved10" models/monocular_validation_v14_reserved10.json
echo "5/7 Avaliando profundidade na run independente antiga..."
evaluate_depth "$independent" models/monocular_validation_v14_independent.json
echo "6/7 Executando replay integral e gate de colisao..."
python3 scripts/run_spatial_offline_sweep.py \
  --matrix config/spatial_v14_validation_sweep.json \
  --prediction-cache-root "$output/prediction_cache" --output "$output"
echo "7/7 Resumindo planos criticos..."
python3 scripts/summarize_spatial_blockage.py "$output" --dataset reference_reserved_03
echo "Concluido. Nenhum PX4/Gazebo foi iniciado. Nao execute SITL antes da analise."
