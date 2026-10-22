# D-EMAML 独立复现

这是论文的独立重实现，不是作者官方代码。

先看框架：[docs/FRAMEWORK.cn.md](docs/FRAMEWORK.cn.md)

对比实验只改一个文件：[configs/experiment.yaml](configs/experiment.yaml)。上面的 `data`、`motif`、`train` 是共用的；每个模型一块，用 `enabled` 决定这次是否加入对比。

云上装好环境后，常用这三条：

```bash
demaml plan --config configs/experiment.yaml
demaml prepare --config configs/experiment.yaml
demaml compare --config configs/experiment.yaml
```

`plan` 只打印将要运行的模型和参数。`prepare` 只处理数据。`compare` 才训练当前启用的模型。
