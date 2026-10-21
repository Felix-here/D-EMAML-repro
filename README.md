# D-EMAML independent reproduction

中文说明：[README.cn.md](README.cn.md)

This is an independent, auditable reimplementation of Han et al.,
“Beyond Single Transactions: D-EMAML” (AAAI 2026). The authors' public
repository does not contain the model or training entrypoint, so this project
must not be represented as the official implementation.

## Local checks

```powershell
cd D:\GDUF_Recording\AI反洗钱paper\D-EMAML-repro
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
```

Do not preprocess the 5,078,345-row CSV on the laptop.

## Sync with GitHub

Push this repository, then on AutoDL:

```bash
cd /root/autodl-tmp
git clone https://github.com/Felix-here/D-EMAML-repro.git
cd D-EMAML-repro
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
bash scripts/run_cloud.sh audit
```

Later updates:

```bash
cd /root/autodl-tmp/D-EMAML-repro
git pull
```

Do not commit the CSV, `.venv`, `artifacts/`, or checkpoints.
`scripts/upload.ps1` remains only as an SCP fallback.

## Cloud quick start

The raw file is expected at:

`/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv`

After cloning, follow [docs/CLOUD_RUNBOOK.md](docs/CLOUD_RUNBOOK.md) or the
Chinese copy [docs/CLOUD_RUNBOOK.cn.md](docs/CLOUD_RUNBOOK.cn.md):

```bash
bash scripts/run_cloud.sh smoke
bash scripts/run_cloud.sh prepare
bash scripts/run_cloud.sh gat
bash scripts/run_cloud.sh demgnn
bash scripts/run_cloud.sh demgnn-many
bash scripts/run_cloud.sh summarize
```

All generated files go below `artifacts/` on AutoDL's persistent data disk.
Read `docs/ASSUMPTIONS.md` (or `docs/ASSUMPTIONS.cn.md`) before interpreting
numbers.

## Workflow

1. Audit the exact source CSV.
2. Preprocess and chronologically split edges 60/10/30.
3. Enumerate temporally valid dual-edge motifs without a global quadratic
   adjacency tensor.
4. Build the motif graph through an edge-to-motif inverted index.
5. Train motif baselines or D-EMGNN and map predictions back with MELP.
6. Select a threshold on validation data once, then evaluate the test split.

This code records the configuration, file hashes, data statistics, runtime,
and package versions in each run directory.
