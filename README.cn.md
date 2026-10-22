# D-EMAML 独立复现

这是论文的独立重实现，不是作者官方代码。算法代码在 `src/demaml/`，平时不用打开。

## 云上只记这一条

有卡开机后：

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/session.sh
```

它每次只做一件事：装环境、核对数据、小规模试跑、全量预处理、训练一个种子，或最后跑五个种子。做完会让你关机。下次开机还是这一条，不要自己改配置，也不要接着跑别的脚本。

无卡开机不用执行它。代码已经在数据盘上，直接关机。

产物都在 `artifacts/`，不进 GitHub。`configs/` 里的文件由这条命令选用，不用手改。GAT 基线先不跑，避免和正式结果混在一起。

终端报错时，把最后 30 行发我，不要换命令重试。

## 本机

代码改完后，在 Cursor 里提交并同步到 GitHub。云端先执行一次：

```bash
cd /root/autodl-tmp/D-EMAML-repro
git pull
```

然后再运行上面的 `session.sh`。不要把 CSV 拷进仓库。
