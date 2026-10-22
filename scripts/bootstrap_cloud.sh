#!/usr/bin/env bash
# Install into the image Python. Do not create a uv environment: that can replace
# the CUDA build of torch shipped with the AutoDL image.
set -euo pipefail

cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if ! [[ "$OMP_NUM_THREADS" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=1
fi
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

python - <<'PY'
import torch
assert torch.__version__.startswith("2.3.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available()
print("system torch:", torch.__version__, torch.cuda.get_device_name(0))
PY

python -m pip install -r requirements-cloud.txt -i "$INDEX"
python -m pip install -e . -i "$INDEX"

python - <<'PY'
import torch
import torch_geometric
import demaml
print("torch:", torch.__version__)
print("pyg:", torch_geometric.__version__)
print("package:", demaml.__version__)
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("cuda smoke:", float(x.mean()))
PY

echo "Installed into the system Python. Use demaml directly; do not activate .venv."
