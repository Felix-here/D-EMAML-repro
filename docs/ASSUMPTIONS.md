# Reproduction assumptions and known gaps

中文版：[ASSUMPTIONS.cn.md](ASSUMPTIONS.cn.md)

## Confirmed by the paper

- AMLWorld has 518,573 accounts, 5,078,345 transactions and roughly 0.15%
  anomalous transactions in the paper's table.
- Edges are split chronologically 60%/10%/30%.
- A dual-edge motif is an ordered pair of transactions that shares at least
  one endpoint.
- Motif labels use `Y = y1 + 2*y2`.
- The motif graph connects motif instances that share an original edge.
- AMLWorld uses `xi=0`, `eta=6`; the reported architecture has two layers,
  hidden size 256, two-hop sampling with at most 100 neighbors per hop, and
  up to 50 epochs.
- The primary metric is AUPRC; five independent runs are reported.
- Figure 3 names six structures: `b1` repeated transfer, `b2` reciprocal
  transfer, `t1` fan-in, `t2` reverse-time chain, `t3` forward chain, `t4`
  fan-out.

## Necessary implementation assumptions

1. **Time unit.** Figure 3 maps gaps of 1, 3 and 5 hours to delta 0, 1 and 2.
   We therefore calculate
   `delta=floor(log2(max(1, ceil((t2-t1)/3600))))`.
2. **Six type IDs.** Classification priority is `b1`, `b2`, then remaining
   ternary predicates: `t1` same destination, `t2` first source equals second
   destination, `t3` first destination equals second source, `t4` same source.
3. **Self-loops.** Transactions with identical source and destination are
   excluded from motif enumeration.
4. **Equal timestamps.** Definition 2 allows `t1 <= t2`; Algorithm 1 requires
   `t2 > t1`. Ties keep a deterministic original-row order.
5. **Split isolation.** Motifs are generated only from two transactions in the
   same chronological split, which prevents future-label leakage.
6. **Degree pruning.** The paper gives no AMLWorld value for `dmax`. This
   implementation exposes it and uses deterministic, evenly spaced retention
   when an account has too many incident transactions. Default `dmax=256`.
7. **Features.** The README mentions `input_dims=1048`, but neither the paper
   nor public code defines those 1048 values. The default reproducible feature
   vector concatenates the two encoded transaction vectors, a scaled delta,
   and a six-way motif type indicator. Its actual dimension is logged.
8. **Mini-batch loss.** Equation (16) averages every motif prediction
   containing an edge before BCE. Exact global aggregation is incompatible
   with neighbor mini-batches. Training uses the equivalent edge-occurrence
   marginal loss on seed motifs; validation/test aggregate every seed motif
   exactly before computing metrics.
9. **Motif-graph pruning.** The paper does not state a cap for the number of
   motifs sharing one transaction. Default `max_motifs_per_edge=128`.
10. **Uncovered edges.** Transactions that never appear in a retained motif
    are excluded from F1/AUPRC; coverage is reported instead of inventing a
    fallback classifier.
11. **Original-graph GAT.** `gate` is a reconstructed reference, not IBM
    Multi-GNN GATe. It is trained on the same chronological split.

## Public-code defects intentionally not copied

- `Amount Received` appears twice in the source column list.
- Payment-format one-hot columns are generated from currency values.
- `process_data.py` always starts DGraph processing after AMLWorld.
- Equation (13) prints a softmax range of `0..2^|E|-1`; the model has four
  motif classes, so this implementation uses four logits.
- Fast-Motif-Gen's dense `AdjInd` tensor is `O(|E|^2)` and is not materialized.

Results from this project are an independent reimplementation, not verification
of the unavailable author implementation.
