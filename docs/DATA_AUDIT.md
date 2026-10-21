# HI-Small data audit

中文版：[DATA_AUDIT.cn.md](DATA_AUDIT.cn.md)

The official D-EMAML repository never shipped training code. This project only
uses the already verified AutoDL copy of IBM AMLWorld HI-Small.

## Expected file

- Path on AutoDL: `/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv`
- Bytes: `475664283`
- Transactions: `5078345` (plus one header row, `wc -l` reports `5078346`)
- Positive labels: `5177`
- Schema:

```
Timestamp, From Bank, Account, To Bank, Account.1,
Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering
```

The paper's AMLWorld table lists 518,573 accounts, 5,078,345 transactions and
about 0.15% anomalies. `5177 / 5078345 ≈ 0.102%` is the raw HI-Small positive
rate; the paper's 0.15% figure is treated as a rounded table value, not a
reason to change the file.

## Local policy

Do not copy the 454 MB CSV onto the laptop. `demaml.audit_csv` re-checks
schema, byte size, row count, positives, timestamp range and SHA-256 on the
cloud machine and writes `data-audit.json` into the run directory.

`matches_known_hi_small` must be all true before motif generation.
