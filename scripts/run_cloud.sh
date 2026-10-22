#!/usr/bin/env bash
# Thin wrapper around configs/experiment.yaml. Edit that file, not this script.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/experiment.yaml}"
STAGE="${1:-plan}"
FREE_GB="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')"
if (( FREE_GB < 15 )); then
  echo "数据盘只剩 ${FREE_GB} GiB。" >&2
  exit 2
fi

case "$STAGE" in
  plan)
    demaml plan --config "$CONFIG"
    ;;
  prepare)
    demaml prepare --config "$CONFIG"
    ;;
  compare)
    demaml compare --config "$CONFIG"
    ;;
  run)
    demaml run --config "$CONFIG" --model "${2:?请写模型标签，例如 gat}"
    ;;
  *)
    echo "用法: bash scripts/run_cloud.sh plan|prepare|compare|run <模型标签>" >&2
    echo "参数都在 configs/experiment.yaml" >&2
    exit 1
    ;;
esac
