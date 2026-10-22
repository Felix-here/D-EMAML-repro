# 云上怎么做

只看 [README.cn.md](../README.cn.md)。有卡时反复执行：

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/session.sh
```

每次只前进一步，然后关机。`artifacts/` 里的日志、缓存和 checkpoint 都是自动生成的，不用整理，也不用提交。

下面这些配置已经接在这条命令里，出现报错前不用单独运行：

1. 安装环境
2. 核对 HI-Small
3. 小规模冒烟，结果单独放在 `artifacts/smoke/`
4. 全量预处理，缓存达到 35 GiB 会自己停下
5. D-EMGNN 一个种子
6. 五个种子并写出 `artifacts/runs/demgnn/summary.json`

`scripts/run_cloud.sh` 和 `configs/` 是这条命令的内部步骤。GAT、原图基线不在默认路径里。
