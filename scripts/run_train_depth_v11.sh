#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ -n "${1:-}" ]]; then
  echo "Uso: $0 [--dry-run]" >&2
  exit 2
fi

source_dir="/home/prograf4080/Depth-Anything-V2"
metadata="models/depth_anything_v2_metric_baylands_vits_v8.training.json"
heldout="$project_dir/datasets/spatial_mapping/run_20260819_191346_877996"
independent="$project_dir/datasets/spatial_mapping/run_20260820_225114_611267"
selection="datasets/spatial_mapping/training_selections/v11_hard_frames.json"
checkpoint="models/depth_anything_v2_metric_baylands_vits_v8.pth"
output="models/depth_anything_v2_metric_baylands_vits_v11.pth"
onnx_output="models/depth_anything_v2_metric_baylands_vits_v11_686x518_fp32.onnx"
log="logs/training/depth_v11.log"

echo "1/5 Limpando processos residuais de PX4/Gazebo..."
./scripts/cleanup_spatial_sim.sh

echo "2/5 Selecionando hard frames sem contaminar validacao..."
python3 scripts/select_spatial_hard_frames.py \
  --training-metadata "$metadata" \
  --target=-28.875,25.125,-1.875 \
  --target=-34.875,58.125,-1.875 \
  --exclude-run "$heldout" \
  --exclude-run "$independent" \
  --radius-m 10 \
  --per-run-target 3 \
  --yaw-bin-deg 45 \
  --output "$selection"

mapfile -t train_runs < <(
  python3 -c 'import json; from pathlib import Path; d=json.loads(Path("'"$metadata"'").read_text()); print(*d["train_runs"], sep="\n")'
)
mapfile -t focus_runs < <(
  python3 -c 'import json; from pathlib import Path; d=json.loads(Path("'"$metadata"'").read_text()); print(*d["focus_runs"], sep="\n")'
)
mapfile -t hard_frames < <(
  python3 -c 'import json; from pathlib import Path; d=json.loads(Path("'"$selection"'").read_text()); print(*(x["frame"] for x in d["selected"]), sep="\n")'
)

train_command=(
  models/train_env/bin/python scripts/finetune_depth_anything_baylands.py
  --source-dir "$source_dir"
  --checkpoint "$checkpoint"
  --validation-run "$heldout"
  --output "$output"
  --epochs 4
  --encoder vits
  --freeze-encoder
  --batch-size 2
  --head-lr 2e-6
  --focus-repeat 1
  --focus-frame-repeat 6
  --edge-multiplier 4.0
  --unsafe-tail-weight 0.10
  --unsafe-overestimate-weight 0.40
  --unsafe-overestimate-tail-weight 0.30
  --unsafe-overestimate-tail-fraction 0.05
  --edge-overestimate-weight 0.40
  --edge-gradient-weight 0.10
  --save-each-epoch
)
for run in "${train_runs[@]}"; do train_command+=(--train-run "$run"); done
for run in "${focus_runs[@]}"; do train_command+=(--focus-run "$run"); done
for frame in "${hard_frames[@]}"; do train_command+=(--focus-frame "$frame"); done

if [[ "$dry_run" == true ]]; then
  printf "Comando de treino (%d hard frames):\n" "${#hard_frames[@]}"
  printf " %q" "${train_command[@]}"
  printf "\n"
  exit 0
fi

echo "3/5 Treinando Small v11 com ${#hard_frames[@]} hard frames..."
mkdir -p "$(dirname "$log")"
"${train_command[@]}" 2>&1 | tee "$log"

echo "4/5 Exportando ONNX FP32..."
models/export_env/bin/python scripts/export_depth_anything_v2_metric_onnx.py \
  --source-dir "$source_dir" --checkpoint "$output" --output "$onnx_output" \
  --encoder vits --input-width 686 --input-height 518

echo "5/5 Concluido."
echo "Modelo: $onnx_output"
echo "Historico: ${output%.pth}.history.json"
echo "Proveniencia: ${output%.pth}.training.json"
echo "Proximo comando: ./scripts/run_validate_depth_v11.sh"
