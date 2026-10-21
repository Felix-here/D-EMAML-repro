# D-EMAML 独立复现（中文说明）

这是 Han 等人 AAAI 2026 论文 *Beyond Single Transactions: D-EMAML* 的**独立重实现**。作者公开仓库里没有模型和训练入口，所以这个项目**不能**当成官方代码。

英文版：[README.md](README.md)

## 本机检查

```powershell
cd D:\GDUF_Recording\AI反洗钱paper\D-EMAML-repro
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
```

不要在笔记本上处理那 5,078,345 行 CSV。

## 用 GitHub 同步到 AutoDL

1. 本机把代码推到 GitHub（仓库创建后会得到一个地址）。
2. 云端只 `git clone` / `git pull` 代码。
3. 数据仍留在 AutoDL 数据盘，不要推进 GitHub。

云端首次：

```bash
cd /root/autodl-tmp
git clone https://github.com/<你的用户名>/D-EMAML-repro.git
cd D-EMAML-repro
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
bash scripts/run_cloud.sh audit
```

之后每次改完代码，本机 `git push`，云端：

```bash
cd /root/autodl-tmp/D-EMAML-repro
git pull
```

原始交易表在：

`/root/autodl-tmp/D-EMAML/data/AMLWorld/raw/HI-Small_Trans.csv`

冒烟通过后再按 [docs/CLOUD_RUNBOOK.cn.md](docs/CLOUD_RUNBOOK.cn.md) 往下跑。看结果前先读 [docs/ASSUMPTIONS.cn.md](docs/ASSUMPTIONS.cn.md)。

## 流程

1. 核对 CSV。
2. 按时间 60%/10%/30% 切分交易。
3. 按账户局部生成双边基序，不构造全局平方张量。
4. 用“原交易 → motif”倒排索引建 motif 图。
5. 训练基线或 D-EMGNN，再用 MELP 映回原边。
6. 只在验证集上选一次阈值，再评测试集。
