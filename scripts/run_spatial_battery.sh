#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Uso: $0 {ground_truth_debug|monocular_topic} [runs] [--dry-run]"
    echo "Monocular exige MONOCULAR_MODEL_PATH=/caminho/modelo.onnx"
}

mode="${1:-}"
runs="${2:-10}"
dry_run="${3:-}"
if [[ "$mode" != "ground_truth_debug" && "$mode" != "monocular_topic" ]]; then
    usage
    exit 2
fi
if ! [[ "$runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "runs deve ser inteiro positivo" >&2
    exit 2
fi
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]]; then
    usage
    exit 2
fi

project_dir="${PROJECT_DIR:-/home/prograf4080/TCC_Drone}"
px4_dir="${PX4_DIR:-/home/prograf4080/PX4-Autopilot}"
ros_overlay="${ROS_OVERLAY:-$project_dir/ws_ros2/install/setup.bash}"
mission_timeout_s="${MISSION_TIMEOUT_S:-180}"
startup_timeout_s="${STARTUP_TIMEOUT_S:-45}"
startup_attempts="${STARTUP_ATTEMPTS:-3}"
px4_parameter_delay_s="${PX4_PARAMETER_DELAY_S:-8}"
monocular_model="${MONOCULAR_MODEL_PATH:-$project_dir/models/depth_anything_v2_metric_baylands_vits_v12_686x518_fp32.onnx}"
monocular_python="${MONOCULAR_PYTHON:-$project_dir/models/runtime_env/bin/python}"
monocular_validation_report="${MONOCULAR_VALIDATION_REPORT:-$project_dir/models/monocular_validation_v12_independent.json}"
monocular_mapping_report="${MONOCULAR_MAPPING_REPORT:-$project_dir/models/monocular_mapping_validation_v12.json}"
spatial_waypoints="${SPATIAL_WAYPOINTS_RELATIVE_M:-}"
guarded_short_test="${GUARDED_SHORT_TEST:-0}"
guarded_route_test="${GUARDED_ROUTE_TEST:-0}"
continue_safe_aborts="${BATTERY_CONTINUE_SAFE_ABORTS:-1}"

if [[ "$continue_safe_aborts" != "0" && "$continue_safe_aborts" != "1" ]]; then
    echo "Preflight falhou: BATTERY_CONTINUE_SAFE_ABORTS deve ser 0 ou 1." >&2
    exit 3
fi

if [[ "$guarded_short_test" == "1" && "$guarded_route_test" == "1" ]]; then
    echo "Preflight falhou: use somente um modo guardado por vez." >&2
    exit 3
fi

for required in     /opt/ros/humble/setup.bash     "$ros_overlay"     "$project_dir/main.py"     "$px4_dir/Makefile"; do
    if [[ ! -e "$required" ]]; then
        echo "Preflight falhou: ausente $required" >&2
        exit 3
    fi
done
if [[ "$mode" == "monocular_topic" ]]; then
    if [[ -z "$monocular_model" || ! -f "$monocular_model" ]]; then
        echo "Preflight falhou: defina MONOCULAR_MODEL_PATH para um .onnx existente." >&2
        exit 3
    fi
    if [[ ! -x "$monocular_python" ]]; then
        echo "Preflight falhou: runtime monocular ausente em $monocular_python." >&2
        exit 3
    fi
    if [[ "$guarded_short_test" == "1" ]]; then
        if [[ "$runs" != "1" || -z "$spatial_waypoints" ]]; then
            echo "Preflight falhou: GUARDED_SHORT_TEST exige uma run e waypoints curtos." >&2
            exit 3
        fi
        echo "AVISO: teste curto guardado; mapa GT pode apenas vetar caminhos." >&2
    elif [[ "$guarded_route_test" == "1" ]]; then
        if [[ "$runs" != "1" ]]; then
            echo "Preflight falhou: GUARDED_ROUTE_TEST exige exatamente uma run." >&2
            exit 3
        fi
        echo "AVISO: teste guardado da rota antiga; mapa GT pode apenas vetar caminhos." >&2
    else
        if [[ ! -f "$monocular_validation_report" ]]; then
            echo "Preflight falhou: relatorio offline ausente em $monocular_validation_report." >&2
            exit 3
        fi
        if [[ ! -f "$monocular_mapping_report" ]]; then
            echo "Preflight falhou: relatorio de mapeamento offline ausente." >&2
            exit 3
        fi
        if ! python3 - "$monocular_model" "$monocular_validation_report" "$monocular_mapping_report" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

digest = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
reports = [
    json.loads(Path(path).read_text(encoding="utf-8"))
    for path in sys.argv[2:]
]
valid = all(
    report.get("qualification", {}).get("passed") is True
    and report.get("model_sha256") == digest
    for report in reports
)
raise SystemExit(0 if valid else 1)
PY
        then
            echo "Preflight falhou: profundidade/mapa nao qualificados ou SHA divergente." >&2
            exit 3
        fi
    fi
fi
if ! "$project_dir/scripts/cleanup_spatial_sim.sh" --check; then
    echo "Preflight falhou: limpe a simulacao antes da bateria com:" >&2
    echo "  ./scripts/cleanup_spatial_sim.sh" >&2
    exit 3
fi

config="${SPATIAL_CONFIG:-$project_dir/config/spatial_debug.yaml}"
if [[ -z "${SPATIAL_CONFIG:-}" && "$mode" == "monocular_topic" ]]; then
    config="$project_dir/config/spatial_monocular.yaml"
fi
if [[ ! -f "$config" ]]; then
    echo "Preflight falhou: configuracao ausente em $config" >&2
    exit 3
fi

stamp="$(date +%Y%m%d_%H%M%S)"
battery_dir="$project_dir/logs/spatial_battery/${stamp}_${mode}"
mkdir -p "$battery_dir"
summary="$battery_dir/summary.tsv"
printf 'run_index\tmode\texit_code\tmission_complete\toutcome\tabort_reason\tdataset\n' > "$summary"

echo "Preflight OK"
echo "  mode=$mode runs=$runs timeout=${mission_timeout_s}s"
echo "  config=$config"
echo "  output=$battery_dir"
echo "  continue_safe_aborts=$continue_safe_aborts"
if [[ -n "$spatial_waypoints" ]]; then
    echo "  waypoints_override=$spatial_waypoints"
fi
if [[ "$mode" == "monocular_topic" ]]; then
    echo "  model=$monocular_model"
    echo "  model_sha256=$(sha256sum "$monocular_model" | cut -d' ' -f1)"
    echo "  python=$monocular_python"
    echo "  validation_report=$monocular_validation_report"
    echo "  mapping_report=$monocular_mapping_report"
fi
if [[ "$dry_run" == "--dry-run" ]]; then
    exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source "$ros_overlay"
set -u
cd "$project_dir"

px4_pid=""
mono_pid=""
controller_pid=""

stop_group() {
    local pid="${1:-}"
    if [[ -z "$pid" ]]; then
        return
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT -- "-$pid" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.25
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.25
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

stop_residual_gazebo() {
    local pattern
    for pattern in 'gz sim' 'gzserver'; do
        if ! pgrep -f "$pattern" >/dev/null; then
            continue
        fi
        pkill -INT -f "$pattern" 2>/dev/null || true
        for _ in {1..20}; do
            pgrep -f "$pattern" >/dev/null || break
            sleep 0.25
        done
        if pgrep -f "$pattern" >/dev/null; then
            pkill -TERM -f "$pattern" 2>/dev/null || true
            for _ in {1..20}; do
                pgrep -f "$pattern" >/dev/null || break
                sleep 0.25
            done
        fi
        if pgrep -f "$pattern" >/dev/null; then
            pkill -KILL -f "$pattern" 2>/dev/null || true
            for _ in {1..20}; do
                pgrep -f "$pattern" >/dev/null || break
                sleep 0.25
            done
        fi
    done
}

cleanup_run() {
    stop_group "$controller_pid"
    controller_pid=""
    stop_group "$mono_pid"
    mono_pid=""
    stop_group "$px4_pid"
    px4_pid=""
    stop_residual_gazebo
    "$project_dir/scripts/cleanup_spatial_sim.sh" || true
}

on_abort() {
    echo "Bateria interrompida; encerrando processos da run ativa." >&2
    cleanup_run
    exit 130
}
trap on_abort INT TERM
trap cleanup_run EXIT

latest_dataset() {
    find "$project_dir/datasets/spatial_mapping"         -maxdepth 1 -type d -name 'run_*' -printf '%T@ %p\n'         | sort -n | tail -1 | cut -d' ' -f2-
}

for ((run_index=1; run_index<=runs; run_index++)); do
    echo "=== Run $run_index/$runs ($mode) ==="
    run_dir="$battery_dir/run_$(printf '%02d' "$run_index")"
    mkdir -p "$run_dir"
    before_dataset="$(latest_dataset)"

    ready=false
    : > "$run_dir/px4_gazebo.log"
    for ((startup_attempt=1; startup_attempt<=startup_attempts; startup_attempt++)); do
        echo "=== PX4/Gazebo startup attempt $startup_attempt/$startup_attempts ===" \
            >>"$run_dir/px4_gazebo.log"
        # O airframe gz_x500 redefine NAV_DLL_ACT=2 quando o armazenamento de
        # parametros e recriado. Isso faz o SITL exigir uma GCS e impede o
        # marcador "Ready for takeoff!" usado por este runner ROS 2. Enviamos
        # os overrides conhecidos pela shell PX4 em toda inicializacao para a
        # execucao nao depender de residuos persistentes entre baterias.
        setsid bash -lc \
            "cd '$px4_dir'; { sleep '$px4_parameter_delay_s'; printf '%s\\n' 'param set EKF2_MAG_CHK_STR 0.25' 'param set NAV_DLL_ACT 0' 'param save' 'echo SPATIAL_PX4_PARAMETERS_APPLIED'; sleep infinity; } | exec env PX4_GZ_WORLD=baylands stdbuf -oL -eL make px4_sitl gz_x500_mono_cam" \
            >>"$run_dir/px4_gazebo.log" 2>&1 &
        px4_pid=$!

        for ((second=0; second<startup_timeout_s; second++)); do
            if grep -q 'Ready for takeoff!' "$run_dir/px4_gazebo.log" \
                && grep -q 'SPATIAL_PX4_PARAMETERS_APPLIED' "$run_dir/px4_gazebo.log"; then
                ready=true
                break
            fi
            if ! kill -0 "$px4_pid" 2>/dev/null; then
                break
            fi
            if ((second > 0 && second % 15 == 0)); then
                if grep -q 'heading estimate not stable' "$run_dir/px4_gazebo.log"; then
                    echo "Aguardando heading do PX4 estabilizar: ${second}s/${startup_timeout_s}s (tentativa ${startup_attempt}/${startup_attempts})." >&2
                else
                    echo "Aguardando prontidao PX4/Gazebo: ${second}s/${startup_timeout_s}s (tentativa ${startup_attempt}/${startup_attempts})." >&2
                fi
            fi
            sleep 1
        done
        if [[ "$ready" == true ]]; then
            break
        fi
        echo "PX4/Gazebo nao ficou pronto na tentativa $startup_attempt." >&2
        stop_group "$px4_pid"
        px4_pid=""
        stop_residual_gazebo
        "$project_dir/scripts/cleanup_spatial_sim.sh" || true
        if ((startup_attempt < startup_attempts)); then
            echo "Reiniciando PX4/Gazebo apos 3s; MicroXRCEAgent preservado." >&2
            sleep 3
        fi
    done
    if [[ "$ready" != true ]]; then
        echo "PX4/Gazebo nao ficou pronto apos $startup_attempts tentativas; abortando." >&2
        cleanup_run
        exit 4
    fi

    if [[ "$mode" == "monocular_topic" ]]; then
        setsid env PYTHONNOUSERSITE=1 "$monocular_python" monocular_depth_node.py             --ros-args             --params-file config/monocular_depth_onnx.yaml             -p "model_path:=$monocular_model"             >"$run_dir/monocular.log" 2>&1 &
        mono_pid=$!
        depth_ready=false
        for ((second=0; second<startup_timeout_s; second++)); do
            if grep -q 'Depth publicado:' "$run_dir/monocular.log"; then
                depth_ready=true
                break
            fi
            if ! kill -0 "$mono_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if [[ "$depth_ready" != true ]]; then
            echo "Profundidade monocular não publicou frame válido; abortando antes de armar." >&2
            cleanup_run
            exit 5
        fi
    fi

    controller_args=(
        --ros-args
        --params-file "$config"
        -p spatial_execute_path:=true
    )
    if [[ -n "$spatial_waypoints" ]]; then
        controller_args+=(
            -p "spatial_waypoints_relative_m:=$spatial_waypoints"
        )
    fi
    if [[ "$guarded_short_test" == "1" || "$guarded_route_test" == "1" ]]; then
        controller_args+=(
            -p spatial_reference_safety_veto:=true
        )
    fi
    set +e
    setsid timeout --signal=TERM --kill-after=15s "$mission_timeout_s"         env PYTHONNOUSERSITE=1 python3 main.py "${controller_args[@]}"         >"$run_dir/controller.log" 2>&1 &
    controller_pid=$!
    wait "$controller_pid"
    controller_exit=$?
    controller_pid=""
    set -e

    dataset="$(latest_dataset)"
    mission_complete=false
    outcome="missing_dataset"
    abort_reason="-"
    if [[ -n "$dataset" && "$dataset" != "$before_dataset" ]]; then
        IFS=$'	' read -r outcome abort_reason < <(
            python3 - "$dataset/events.jsonl" <<'PY'
import json
import sys
from pathlib import Path
events = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
states = [event for event in events if event.get("event") == "mission_state"]
if any(event.get("state") == "mission_complete" for event in states):
    print("completed	-")
else:
    aborted = next((event for event in reversed(states) if event.get("state") == "mission_aborted"), None)
    if aborted is None:
        print("incomplete_without_abort	-")
    else:
        reason = str(aborted.get("reason") or "unspecified_abort")
        safe_reasons = {
            "recovery_observation_timeout",
            "recovery_no_progress_timeout",
            "recovery_brake_no_safe_path",
        }
        outcome = "safe_abort" if reason in safe_reasons else "unsafe_abort"
        print(f"{outcome}	{reason}")
PY
        )
        if [[ "$outcome" == "completed" ]]; then
            mission_complete=true
        fi
        python3 estudos_e_analises/analisar_mapeamento_espacial.py "$dataset"             >"$run_dir/analysis.json" 2>"$run_dir/analysis.err" || true
    fi

    printf '%s	%s	%s	%s	%s	%s	%s
'         "$run_index" "$mode" "$controller_exit" "$mission_complete"         "$outcome" "$abort_reason" "$dataset" >> "$summary"

    cleanup_run
    if ! "$project_dir/scripts/cleanup_spatial_sim.sh" --check; then
        echo "Processo residual detectado apos a run $run_index; abortando." >&2
        exit 6
    fi
    if [[ "$mission_complete" == true && "$controller_exit" -eq 124 ]]; then
        echo "Run $run_index concluida antes do limite; timeout encerrou somente a espera de pouso."
    elif [[ "$controller_exit" -ne 0 ]]; then
        echo "Run $run_index falhou: controlador terminou com codigo $controller_exit." >&2
        exit 7
    fi
    if [[ "$mission_complete" != true ]]; then
        if [[ "$outcome" == "safe_abort" && "$continue_safe_aborts" == "1" ]]; then
            echo "Run $run_index terminou em aborto seguro ($abort_reason); registrando e continuando." >&2
        else
            echo "Run $run_index falhou ($outcome); bateria interrompida antes da proxima decolagem." >&2
            exit 7
        fi
    fi
done

trap - EXIT
echo "Bateria concluída: $summary"
