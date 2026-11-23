#!/usr/bin/env bash
set -euo pipefail

attempts="${1:-1}"
dry_run=()
if [[ "$attempts" == "--dry-run" ]]; then
    attempts=1
    dry_run=(--dry-run)
fi
if (( $# > 1 )) || ! [[ "$attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "Uso: bash $0 [tentativas|--dry-run]" >&2
    exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
for ((attempt = 1; attempt <= attempts; attempt++)); do
    echo "=== Rota publicada com v21: tentativa $attempt de $attempts ==="
    if env -u HEADLESS -u SPATIAL_CONFIG -u SPATIAL_WAYPOINTS_RELATIVE_M \
        -u MONOCULAR_MODEL_PATH -u MONOCULAR_PARAMS_FILE -u MONOCULAR_BUNDLE_REPORT \
        PROJECT_DIR="$project_dir" PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}" \
        GUARDED_ROUTE_TEST=1 GUARDED_SHORT_TEST=0 \
        MISSION_TIMEOUT_S="${MISSION_TIMEOUT_S:-600}" BATTERY_CONTINUE_SAFE_ABORTS=0 \
        "$project_dir/scripts/run_spatial_battery.sh" monocular_topic 1 "${dry_run[@]}"; then
        exit 0
    else
        status=$?
        if [[ "$status" != 7 ]]; then
            exit "$status"
        fi
    fi
done
echo "A rota publicada não foi completada nas $attempts tentativas." >&2
exit 7
