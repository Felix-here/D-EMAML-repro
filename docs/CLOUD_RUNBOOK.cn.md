# 云上操作

框架和命令见 [FRAMEWORK.cn.md](FRAMEWORK.cn.md)。参数见 [configs/experiment.yaml](../configs/experiment.yaml)。

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/bootstrap_cloud.sh
demaml plan --config configs/experiment.yaml
```

`plan` 的清单符合预期后，再自己决定要不要 `prepare` 或 `compare`。不要同时维护多份模型配置；旧的单模型 yaml 已不再使用。
