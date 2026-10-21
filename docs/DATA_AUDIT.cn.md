# HI-Small 数据核验

官方 D-EMAML 仓库从未提供训练代码。本项目只用已经核验过的 AutoDL 上那份 IBM AMLWorld HI-Small。

## 期望文件

- AutoDL 路径：`/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv`
- 字节数：`475664283`
- 交易笔数：`5078345`（加表头后 `wc -l` 为 `5078346`）
- 正例：`5177`
- 字段：

```
Timestamp, From Bank, Account, To Bank, Account.1,
Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering
```

论文 AMLWorld 表写了 518,573 个账户、5,078,345 笔交易、大约 0.15% 异常。原始 HI-Small 正例率是 `5177 / 5078345 ≈ 0.102%`。论文的 0.15% 按表格四舍五入理解，不作为改文件的理由。

## 本机原则

不要把约 454 MB 的 CSV 拷到笔记本。`demaml.audit_csv` 会在云端再查 schema、字节数、行数、正例、时间范围和 SHA-256，并写成 run 目录里的 `data-audit.json`。

进入基序生成前，`matches_known_hi_small` 必须全部为 true。
