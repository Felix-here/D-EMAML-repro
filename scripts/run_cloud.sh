#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAGE="${1:-help}"
CSV="${CSV:-/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv}"
FREE_GB="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')"
if (( FREE_GB < 15 )); then
  echo "Refusing to run: only ${FREE_GB} GiB free on data disk." >&2
  exit 2
fi

mkdir -p artifacts/logs
STAMP="$(date +%Y%m%d-%H%M%S)"
log_run() {
  local name="$1"
  shift
  local log="artifacts/logs/${STAMP}-${name}.log"
  echo "Free data disk: ${FREE_GB} GiB"
  nvidia-smi || true
  echo "Logging to ${log}"
  "$@" 2>&1 | tee "$log"
}

case "$STAGE" in
  audit)
    log_run audit demaml audit-data --csv "$CSV" --output artifacts/logs/data-audit.json
    ;;
  smoke)
    log_run smoke demaml run --config configs/smoke.yaml
    ;;
  prepare)
    log_run prepare demaml prepare --config configs/amlworld_demgnn.yaml
    ;;
  gate)
    log_run gate demaml run --config configs/amlworld_gate.yaml
    ;;
  gat)
    log_run gat demaml run --config configs/amlworld_gat.yaml
    ;;
  demgnn)
    log_run demgnn demaml run --config configs/amlworld_demgnn.yaml
    ;;
  demgnn-many)
    log_run demgnn-many demaml run-many --config configs/amlworld_demgnn.yaml --seeds 1 2 3 4 5
    ;;
  summarize)
    shopt -s nullglob
    files=(artifacts/runs/demgnn/seed-*/metrics.json)
    if (( ${#files[@]} == 0 )); then
      echo "No D-EMGNN metrics found under artifacts/runs/demgnn/" >&2
      exit 1
    fi
    log_run summarize demaml summarize "${files[@]}"
    ;;
  help|*)
    cat <<'EOF'
Usage: bash scripts/run_cloud.sh <stage>
  audit         verify HI-Small bytes/rows/positives
  smoke         cheap 2-epoch run with checkpointing
  prepare       full motif generation, no training
  gate          original-graph GAT reference
  gat           motif-graph GAT baseline
  demgnn        D-EMGNN single seed
  demgnn-many   D-EMGNN seeds 1-5
  summarize     mean/std versus paper Table 2
Stop if smoke fails, motif counts look impossible, or a single epoch OOMs.
If cache exceeds 35 GiB, inspect artifacts/cache-full/prepare.json before training.
EOF
    ;;
esac
