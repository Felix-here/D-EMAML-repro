# 实验框架

参数只改 [configs/experiment.yaml](../configs/experiment.yaml)。一次对比里，所有启用的模型共用同一份数据和同一套训练设置；模型之间的差异写在各自的块里。

## 数据怎么变成指标

```text
HI-Small CSV
    data.py            按时间切成 60% / 10% / 30%，只在训练集上拟合特征
    motifs.py          每个账户局部配对，得到 b1 b2 t1 t2 t3 t4
    motif_graph.py     共享同一笔原交易的基序连边
    models.py          按配置里的 name 选择模型
    melp.py            四个基序类别折回原交易的概率
    evaluate.py        验证集选一次阈值，测试集算 F1 和 AUPRC
```

`gate` 停在原交易图，不经过基序。其余模型走完整条链。

## 模型块里的 name

| 标签（结果目录） | name | 代码 |
|---|---|---|
| `demgnn` | `demgnn` | SAIRR + 带类型偏置的 MGCN |
| `demgnn_no_sairr` | `demgnn` | 关掉 SAIRR 的消融 |
| `demgnn_no_type_bias` | `demgnn` | 关掉类型对偏置的消融 |
| `gat` / `gin` / `gcn` | 同名 | 基序图上的对应基线 |
| `gate` | `gate` | 原交易图 GAT |

标签可以改，用来区分同一种架构的不同参数。`name` 决定用哪段模型代码。

## 云上你自己决定跑哪一步

环境只装一次：

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/bootstrap_cloud.sh
```

装到镜像自带的 Python 里，不要使用 `uv`，也不要 `source .venv/bin/activate`。

先看这次会跑哪些模型，不消耗训练：

```bash
demaml plan --config configs/experiment.yaml
```

只做数据处理：

```bash
demaml prepare --config configs/experiment.yaml
```

跑所有 `enabled: true` 的模型和 `train.seeds`：

```bash
demaml compare --config configs/experiment.yaml
```

只跑其中一个标签：

```bash
demaml run --config configs/experiment.yaml --model gat
```

结果在 `artifacts/runs/<标签>/seed-<数字>/metrics.json`。对比汇总在 `artifacts/runs/comparison.json`。这些文件不进 GitHub。

试跑时把 `data.smoke` 改成 `true`，并把 `epochs` 改小。试跑结果会写到 `artifacts/smoke/`，不会覆盖正式结果。
