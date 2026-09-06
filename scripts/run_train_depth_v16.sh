#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true
elif [[ -n "${1:-}" ]]; then echo "Uso: $0 [--dry-run]" >&2; exit 2
fi

source_dir="/home/prograf4080/Depth-Anything-V2"
metadata="config/depth_v16_training.json"
heldout="datasets/spatial_mapping/run_20260831_082836_332335"
independent="datasets/spatial_mapping/run_20260831_083521_770086"
selection="datasets/spatial_mapping/training_selections/v16_balanced_complete_frames.json"
checkpoint="models/depth_anything_v2_metric_baylands_vits_v12.pth"
output="models/depth_anything_v2_metric_baylands_vits_v16.pth"
onnx_output="models/depth_anything_v2_metric_baylands_vits_v16_686x518_fp32.onnx"
log="logs/training/depth_v16.log"

echo "1/6 Limpando apenas PX4/Gazebo; MicroXRCEAgent sera preservado..."
./scripts/cleanup_spatial_sim.sh
echo "2/6 Rodando regressao da perda e do pipeline espacial..."
models/train_env/bin/python -m unittest tests.test_depth_training_loss -v
python3 -m unittest tests.test_spatial_mapping -v
echo "3/6 Selecionando hard frames das arvores sem contaminar validacao..."
python3 scripts/select_spatial_hard_frames.py \
  --training-metadata "$metadata" \
  --target=-28.5,38.5,-1.5 \
  --target=-32.5,46.0,-1.5 \
  --target=-36.5,55.0,-1.5 \
  --target=-40.5,60.0,-1.5 \
  --target=-28.9,25.1,-1.5 \
  --target=-34.9,58.1,-1.5 \
  --exclude-run "datasets/spatial_mapping/run_20260831_082418_723471" \
  --exclude-run "datasets/spatial_mapping/run_20260831_082629_864845" \
  --exclude-run "datasets/spatial_mapping/run_20260831_082836_332335" \
  --exclude-run "datasets/spatial_mapping/run_20260831_083058_762402" \
  --exclude-run "datasets/spatial_mapping/run_20260831_083304_087161" \
  --exclude-run "datasets/spatial_mapping/run_20260831_083521_770086" \
  --exclude-run "datasets/spatial_mapping/run_20260831_083731_750398" \
  --exclude-run "datasets/spatial_mapping/run_20260831_083939_468039" \
  --exclude-run "datasets/spatial_mapping/run_20260831_084145_348727" \
  --exclude-run "datasets/spatial_mapping/run_20260831_084356_708163" \
  --exclude-run "datasets/spatial_mapping/run_20260819_191346_877996" \
  --exclude-run "datasets/spatial_mapping/run_20260820_225114_611267" \
  --exclude-run "datasets/spatial_mapping/run_20260827_165430_607620" \
  --radius-m 9 --per-run-target 2 --yaw-bin-deg 30 \
  --output "$selection"

mapfile -t train_runs < <(python3 -c 'import json; d=json.load(open("'"$metadata"'")); print(*d["train_runs"], sep="\n")')
mapfile -t focus_runs < <(python3 -c 'import json; d=json.load(open("'"$metadata"'")); print(*d["focus_runs"], sep="\n")')
mapfile -t hard_frames < <(python3 -c 'import json; d=json.load(open("'"$selection"'")); print(*(x["frame"] for x in d["selected"]), sep="\n")')

train_command=(
  models/train_env/bin/python scripts/finetune_depth_anything_baylands.py
  --source-dir "$source_dir" --checkpoint "$checkpoint"
  --validation-run "$heldout" --output "$output"
  --epochs 6 --encoder vits --freeze-encoder --batch-size 2 --head-lr 2e-6
  --focus-repeat 1 --focus-frame-repeat 2
  --edge-multiplier 3.5 --unsafe-tail-weight 0.08
  --unsafe-overestimate-weight 0.40
  --unsafe-overestimate-tail-weight 0.30
  --unsafe-overestimate-tail-fraction 0.05
  --edge-overestimate-weight 0.40
  --unsafe-underestimate-weight 0.30
  --unsafe-underestimate-tail-weight 0.20
  --unsafe-underestimate-tail-fraction 0.05
  --edge-underestimate-weight 0.30
  --edge-gradient-weight 0.10 --save-each-epoch
)
for run in "${train_runs[@]}"; do train_command+=(--train-run "$run"); done
for run in "${focus_runs[@]}"; do train_command+=(--focus-run "$run"); done
for frame in "${hard_frames[@]}"; do train_command+=(--focus-frame "$frame"); done

if [[ "$dry_run" == true ]]; then
  printf "Comando de treino (%d hard frames):\n" "${#hard_frames[@]}"
  printf " %q" "${train_command[@]}"; printf "\n"
  exit 0
fi

echo "4/6 Treinando Small v16 com perda balanceada em corpus ampliado..."
mkdir -p "$(dirname "$log")"
"${train_command[@]}" 2>&1 | tee "$log"
echo "5/6 Exportando ONNX FP32..."
models/export_env/bin/python scripts/export_depth_anything_v2_metric_onnx.py \
  --source-dir "$source_dir" --checkpoint "$output" --output "$onnx_output" \
  --encoder vits --input-width 686 --input-height 518
echo "6/6 Concluido."
echo "Modelo: $onnx_output"
echo "Historico: ${output%.pth}.history.json"
echo "Proveniencia: ${output%.pth}.training.json"
echo "Log: $log"
echo "Nao execute SITL; o proximo passo e o gate offline v16."
