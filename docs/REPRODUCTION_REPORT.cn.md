# D-EMAML 复现报告

## 状态

- 实现：独立重实现，不是官方代码
- 数据：AMLWorld HI-Small
- 目标：论文表 2，AMLWorld 上的 D-EMGNN
- 目标 F1：`26.4 ± 1.55%`
- 目标 AUPRC：`15.2 ± 1.96%`
- 本机：源码、单元测试和云端脚本都在 `D-EMAML-repro`
- 云端五次运行：等 AutoDL 上 `git clone` 后按阶段执行。表 2 数字不能在笔记本上编造。

## 环境

计划使用一块 RTX 4090 24GB、16 核 CPU、120GB 内存、50GB 数据盘。镜像提供 PyTorch 2.3.0+cu121 和 PyTorch Geometric 2.8.1。每次运行会写 `environment.json`。

本机核验使用 CPython 3.12、`torch==2.3.0`（CPU）、`torch-geometric==2.8.1`、`numpy==1.26.4`。`pytest` 18 项通过，`ruff check src tests` 通过。

## 数据来源

此前核验过的 CSV：

- 475,664,283 字节
- 5,078,345 笔交易
- 5,177 笔正例

运行时的 `data-audit.json` 会再记 SHA-256 并核对上述计数。见 `DATA_AUDIT.cn.md`。

## 配置

- `configs/experiment.yaml`：共用数据和训练设置；各模型在 `models` 下分别开关和调参

每次运行都会把解析后的配置复制到 `config.json`。

## 因公开信息缺失造成的偏差

见 `ASSUMPTIONS.cn.md`。影响最大的是：

1. 用重构造的边特征，而不是未说明的 1048 维向量。
2. `dmax` 未知；默认 256，均匀间隔裁剪度数。
3. 训练时用小批量 occurrence BCE，而不是全局 MELP 平均。
4. 只在同一划分内组成基序。
5. 用小时计算 `floor(log2(ceil(hours)))`，因为图 3 把 1/3/5 小时标成 δ=0/1/2。
6. motif 图上限 `max_motifs_per_edge=128`。
7. 自环交易不进入基序枚举。

## 结果

云端跑完后再填。在 AutoDL 上执行完 `bash scripts/run_cloud.sh demgnn-many` 之前，下表保持 pending。

| Seed | 最佳 epoch | F1 | AUPRC | Precision | Recall | Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | pending | pending | pending | pending | pending | pending |
| 2 | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending |
| 4 | pending | pending | pending | pending | pending | pending |
| 5 | pending | pending | pending | pending | pending | pending |

事先约定的统计复现标准：五次平均 F1 落在 `26.4 ± 3.10` 个百分点（论文标准差的两倍）以内，平均 AUPRC 落在 `15.2 ± 3.92` 以内。达不到就报告为“独立重实现，未复现表 2”，并按特征编码、时间单位、基序分布、`dmax`、MELP、邻居采样、阈值选择的顺序排查。

## 本机对五次运行协议的干跑

`pytest` 覆盖了数据核验、时间切分、六类基序、图 3 的 δ、四类 MELP、共享边连图、SAIRR/MGCN 形状、断点续训，以及一个微型端到端 D-EMGNN。那不是 AMLWorld 结果。
