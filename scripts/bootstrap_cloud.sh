#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import torch
assert torch.__version__.startswith("2.3.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available()
print("system torch:", torch.__version__, torch.cuda.get_device_name(0))
PY

python -m pip install -q uv
uv venv --python "$(command -v python)" --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements-cloud.txt
uv pip install --python .venv/bin/python --editable .

# NeighborLoader needs one of the accelerated PyG sampling backends.
uv pip install --python .venv/bin/python \
  pyg_lib torch_scatter torch_sparse \
  --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html

.venv/bin/python - <<'PY'
import torch
import torch_geometric
import pyg_lib
import demaml
print("torch:", torch.__version__)
print("pyg:", torch_geometric.__version__)
print("package:", demaml.__version__)
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("cuda smoke:", float(x.mean()))
PY

echo "Activate with: source .venv/bin/activate"
