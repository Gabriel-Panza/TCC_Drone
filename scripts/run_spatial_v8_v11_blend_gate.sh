#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output="logs/spatial_offline_sweep/manual_v8_v11_blend_01"
base="models/depth_anything_v2_metric_baylands_vits_v8.pth"
adapted="models/depth_anything_v2_metric_baylands_vits_v11.pth"

[[ -f "$base" ]] || { echo "Checkpoint v8 ausente: $base" >&2; exit 2; }
[[ -f "$adapted" ]] || { echo "Checkpoint v11 ausente: $adapted" >&2; exit 2; }

echo "1/5 Limpando processos residuais de PX4/Gazebo..."
./scripts/cleanup_spatial_sim.sh
echo "2/5 Rodando regressao espacial..."
python3 -m unittest tests.test_spatial_mapping -v
echo "3/5 Gerando tres checkpoints intermediarios v8--v11..."
for item in "0.25:a025" "0.50:a050" "0.75:a075"; do
  alpha="${item%%:*}"
  tag="${item##*:}"
  checkpoint="models/depth_anything_v2_metric_baylands_vits_v8_v11_${tag}.pth"
  onnx="models/depth_anything_v2_metric_baylands_vits_v8_v11_${tag}_686x518_fp32.onnx"
  models/train_env/bin/python scripts/blend_depth_checkpoints.py --base "$base" --adapted "$adapted" --alpha "$alpha" --output "$checkpoint"
  models/export_env/bin/python scripts/export_depth_anything_v2_metric_onnx.py --source-dir /home/prograf4080/Depth-Anything-V2 --checkpoint "$checkpoint" --output "$onnx" --encoder vits --input-width 686 --input-height 518
done
echo "4/5 Executando gate integral no percurso antigo e na run independente..."
python3 scripts/run_spatial_offline_sweep.py --matrix config/spatial_v8_v11_blend_sweep.json --prediction-cache-root "$output/prediction_cache" --output "$output"
echo "5/5 Resumindo planos criticos 5--8..."
python3 scripts/summarize_spatial_blockage.py "$output"
echo
echo "Concluido. Envie:"
echo "  $output/ranking.csv"
echo "  $output/summary.csv"
echo "  $output/critical_plans_5_8.csv"
echo "Este comando nao inicia PX4 nem Gazebo. Nao execute SITL antes da analise."
