# D-EMAML reproduction report

中文版：[REPRODUCTION_REPORT.cn.md](REPRODUCTION_REPORT.cn.md)

## Status

- Implementation: independent reimplementation, not official author code
- Dataset: AMLWorld HI-Small
- Target: paper Table 2, AMLWorld D-EMGNN
- Target F1: `26.4 ± 1.55%`
- Target AUPRC: `15.2 ± 1.96%`
- Local package: source, unit tests and cloud scripts are in `D-EMAML-repro`
- Cloud execution: clone this repository on AutoDL, then run the staged
  commands. Five-seed Table 2 numbers cannot be invented locally.

## Environment

The intended instance has one RTX 4090 24GB, 16 CPU cores, 120GB RAM and a
50GB persistent data disk. The image provides PyTorch 2.3.0+cu121 and
PyTorch Geometric 2.8.1. Each run writes an `environment.json`.

Local verification used CPython 3.12, `torch==2.3.0` (CPU),
`torch-geometric==2.8.1` and `numpy==1.26.4` in `D-EMAML-repro/.venv`.
`pytest` : 18 passed. `ruff check src tests` : clean.

## Data provenance

The previously audited CSV has:

- 475,664,283 bytes
- 5,078,345 transactions
- 5,177 positive transactions

Runtime `data-audit.json` records SHA-256 and verifies these counts again.
See `DATA_AUDIT.md`.

## Configuration

Formal settings:

- `configs/smoke.yaml`: 2-epoch cheap path, all positives kept
- `configs/amlworld_gate.yaml`: reconstructed original-graph GAT
- `configs/amlworld_gat.yaml`: motif-graph GAT baseline
- `configs/amlworld_demgnn.yaml`: hidden=256, 2 layers, neighbors `[100,100]`,
  50 epochs, `xi=0`, `eta=6`, batch=16

Every run copies its resolved settings to `config.json`.

## Deviations forced by missing public information

See `ASSUMPTIONS.md`. The consequential ones are:

1. Reconstructed edge features instead of the unexplained 1048-d vector.
2. Unknown `dmax`; default 256 with evenly spaced degree pruning.
3. Mini-batch occurrence BCE instead of global train-time MELP averaging.
4. Same-split motif pairs only.
5. Hour-based `floor(log2(ceil(hours)))` because Figure 3 maps 1/3/5 h to
   δ=0/1/2.
6. Motif-graph cap `max_motifs_per_edge=128`.
7. Self-loop transactions are dropped from motif enumeration.

## Results

Populate after cloud execution. Values below stay pending until
`bash scripts/run_cloud.sh demgnn-many` finishes on AutoDL.

| Seed | Best epoch | F1 | AUPRC | Precision | Recall | Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | pending | pending | pending | pending | pending | pending |
| 2 | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending |
| 4 | pending | pending | pending | pending | pending | pending |
| 5 | pending | pending | pending | pending | pending | pending |

Statistical reproduction is predeclared as: five-run mean F1 within
`26.4 ± 3.10` percentage points (two paper standard deviations) and mean
AUPRC within `15.2 ± 3.92`. If that fails, the claim remains “independent
reimplementation that did not reproduce Table 2,” and debugging follows
feature encoding, time units, motif distribution, `dmax`, MELP, neighbor
sampling, then threshold selection.

## Local dry-run of the five-seed protocol

`pytest` exercises audit, chronological split, six motif types, Figure 3
deltas, four-class MELP, shared-edge graphs, SAIRR/MGCN shapes, checkpoint
resume and a tiny end-to-end D-EMGNN run. That is not an AMLWorld result.
