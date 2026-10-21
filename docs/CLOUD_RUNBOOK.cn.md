# AutoDL 操作说明

AMLWorld 的五次独立运行要在现有 RTX 4090 上做。代码用 GitHub 同步，数据留在数据盘，不要上传 CSV。

## 第一次：从 GitHub 拉代码

实例开机后，在 JupyterLab Terminal：

```bash
cd /root/autodl-tmp
git clone https://github.com/<你的用户名>/D-EMAML-repro.git
cd D-EMAML-repro
ls
```

应看到 `src`、`configs`、`scripts`、`docs`。

如果仓库是私有的，先在 GitHub 建一个有 `repo` 权限的 Personal Access Token，然后：

```bash
git clone https://<TOKEN>@github.com/<你的用户名>/D-EMAML-repro.git
```

不要把 token 写进会提交的文件里。

## 之后改代码

本机：

```powershell
cd D:\GDUF_Recording\AI反洗钱paper\D-EMAML-repro
git add -A
git commit -m "说明这次改了什么"
git push
```

云端：

```bash
cd /root/autodl-tmp/D-EMAML-repro
git pull
```

不要把 `artifacts/`、`*.pt`、CSV 推进 GitHub。它们已在 `.gitignore` 里。

## 云端运行顺序

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
bash scripts/run_cloud.sh audit
bash scripts/run_cloud.sh smoke
# 关机，检查 artifacts/cache-smoke 和 smoke 的 checkpoint
bash scripts/run_cloud.sh prepare
# 看基序数量、内存和缓存大小；缓存 >= 35 GiB 先停
bash scripts/run_cloud.sh gate
bash scripts/run_cloud.sh gat
bash scripts/run_cloud.sh demgnn
# 单 seed 量级合理后再跑
bash scripts/run_cloud.sh demgnn-many
bash scripts/run_cloud.sh summarize
```

阶段之间把实例关掉。Smoke 失败、数据分布不对、或一个 epoch 就 OOM，都不要进入下一阶段。如果论文配置 OOM，先确认没有构造全局 `|E|^2` 张量，再考虑换 48GB 卡。不要悄悄改小 batch size 还声称是同一设置复现。
