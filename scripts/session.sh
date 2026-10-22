#!/usr/bin/env bash
# One cloud session, one stage. Run this same command every time the GPU is on.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p artifacts/status artifacts/logs

CSV="/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv"
say() { printf '\n%s\n\n' "$1"; }

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  say "当前没有 GPU。无卡开机不用安装、不用训练，直接关机。下次有卡时再执行：bash scripts/session.sh"
  exit 0
fi

FREE_GB="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')"
if (( FREE_GB < 15 )); then
  say "数据盘只剩 ${FREE_GB} GiB。先不要继续，把这条提示发我。"
  exit 2
fi

stamp() { date -Is > "artifacts/status/$1"; }

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c "import demaml, torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  say "这一步只装环境，不读全量数据，也不训练。装完请关机。"
  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo || true
  fi
  bash scripts/bootstrap_cloud.sh
  stamp bootstrap.ok
  say "环境装好了。请关机。下次有卡开机后，还是只执行：bash scripts/session.sh"
  exit 0
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f artifacts/status/audit.ok ]]; then
  say "这一步只核对已有 CSV，不训练。"
  bash scripts/run_cloud.sh audit
  python - <<'PY'
import json
import sys
report = json.load(open("artifacts/logs/data-audit.json", encoding="utf-8"))
flags = report["matches_known_hi_small"]
print(flags)
if not all(flags.values()):
    sys.exit(1)
PY
  stamp audit.ok
  say "数据核对通过。请关机。下次还是：bash scripts/session.sh"
  exit 0
fi

if [[ ! -f artifacts/status/smoke.ok ]]; then
  say "这一步只做小规模冒烟，确认程序能跑完并保存断点。不是论文结果。"
  bash scripts/run_cloud.sh smoke
  test -f artifacts/smoke/runs/demgnn/seed-1/metrics.json
  stamp smoke.ok
  say "冒烟结束。日志在 artifacts/logs/。没有 Error 或 OOM 就关机；下次同一条命令才会开始全量预处理。"
  exit 0
fi

if [[ ! -f artifacts/status/prepare.ok ]]; then
  say "这一步只生成全量基序和图，不训练。时间会比较长。"
  bash scripts/run_cloud.sh prepare
  python - <<'PY'
import json
import sys
report = json.load(open("artifacts/cache-full/prepare.json", encoding="utf-8"))
print("motifs:", report.get("motifs", {}).get("count"))
print("cache_gb:", report.get("cache_gb"))
if report.get("stop_recommended") or float(report.get("cache_gb") or 0) >= 35:
    print(report.get("stop_recommended") or "cache >= 35 GiB")
    sys.exit(1)
PY
  stamp prepare.ok
  say "全量数据已备好。请关机。下次同一条命令才训练第一个种子。"
  exit 0
fi

if [[ ! -f artifacts/status/demgnn.ok ]]; then
  say "这一步训练 D-EMGNN 的一个种子。GAT 基线先不跑。"
  bash scripts/run_cloud.sh demgnn
  python - <<'PY'
import json
metrics = json.load(open("artifacts/runs/demgnn/seed-1/metrics.json", encoding="utf-8"))
test = metrics["test"]
print("seed 1  F1", round(test["f1"], 4), " AUPRC", round(test["auprc"], 4))
PY
  stamp demgnn.ok
  say "第一个种子跑完。这还不是论文里的五次平均。请关机；下次同一条命令才会跑五个种子。"
  exit 0
fi

if [[ ! -f artifacts/status/five.ok ]]; then
  say "这一步连续跑五个种子并汇总。可以离开，结束后再看终端。"
  bash scripts/run_cloud.sh demgnn-many
  bash scripts/run_cloud.sh summarize
  stamp five.ok
  say "五次运行已汇总到 artifacts/runs/demgnn/summary.json。把这个文件发我，然后关机。"
  exit 0
fi

say "约定的步骤已经走完。结果文件是 artifacts/runs/demgnn/summary.json。不需要再执行本命令。"
