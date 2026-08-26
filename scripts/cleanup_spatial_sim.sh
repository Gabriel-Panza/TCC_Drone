#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cleanup}"
if [[ "$mode" != "cleanup" && "$mode" != "--check" ]]; then
    echo "Uso: $0 [--check]" >&2
    exit 2
fi

patterns=(
    '(^|/)gz[[:space:]]+sim([[:space:]]|$)'
    '(^|/)gzserver([[:space:]]|$)'
    '(^|/)px4([[:space:]]|$)'
)

processes_for_pattern() {
    local pattern="$1"
    local pid
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
        printf '%s\n' "$pid"
    done < <(pgrep -u "$(id -u)" -f "$pattern" 2>/dev/null || true)
}

all_processes() {
    local pattern
    for pattern in "${patterns[@]}"; do
        processes_for_pattern "$pattern"
    done | sort -nu
}

mapfile -t initial < <(all_processes)
if [[ "$mode" == "--check" ]]; then
    if ((${#initial[@]} == 0)); then
        echo "Simulacao limpa: nenhum Gazebo/PX4 residual. MicroXRCEAgent preservado."
        exit 0
    fi
    echo "Processos residuais da simulacao:" >&2
    ps -o pid=,stat=,args= -p "$(IFS=,; echo "${initial[*]}")" >&2 || true
    exit 1
fi

if ((${#initial[@]} == 0)); then
    echo "Nenhum processo residual da simulacao."
    exit 0
fi

echo "Encerrando processos da simulacao: ${initial[*]}"
kill -INT "${initial[@]}" 2>/dev/null || true
for _ in {1..20}; do
    mapfile -t remaining < <(all_processes)
    ((${#remaining[@]} == 0)) && break
    sleep 0.25
done

mapfile -t remaining < <(all_processes)
if ((${#remaining[@]} > 0)); then
    kill -TERM "${remaining[@]}" 2>/dev/null || true
    for _ in {1..20}; do
        mapfile -t remaining < <(all_processes)
        ((${#remaining[@]} == 0)) && break
        sleep 0.25
    done
fi

mapfile -t remaining < <(all_processes)
if ((${#remaining[@]} > 0)); then
    kill -KILL "${remaining[@]}" 2>/dev/null || true
    sleep 0.5
fi

mapfile -t remaining < <(all_processes)
if ((${#remaining[@]} > 0)); then
    echo "Falha ao limpar processos: ${remaining[*]}" >&2
    exit 1
fi
echo "Gazebo e PX4 encerrados. MicroXRCEAgent preservado."
