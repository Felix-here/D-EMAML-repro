#!/usr/bin/env bash
# Show the models selected in configs/experiment.yaml. This does not train.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "框架: docs/FRAMEWORK.cn.md"
echo "参数: configs/experiment.yaml"
echo
if python -c "import demaml" >/dev/null 2>&1; then
  demaml plan --config configs/experiment.yaml
else
  echo "环境还没装。有卡后再执行 bash scripts/bootstrap_cloud.sh"
fi
