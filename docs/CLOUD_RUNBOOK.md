# AutoDL runbook

中文版：[CLOUD_RUNBOOK.cn.md](CLOUD_RUNBOOK.cn.md)

The 5-seed AMLWorld job needs the existing RTX 4090 instance. Sync code with
GitHub. Leave `HI-Small_Trans.csv` on the data disk; never push it.

## First clone

```bash
cd /root/autodl-tmp
git clone https://github.com/Felix-here/D-EMAML-repro.git
cd D-EMAML-repro
```

If the repository is private, clone with a GitHub token that has `repo`
scope. Do not write that token into a committed file.

## Later updates

On the laptop: `git add`, `git commit`, `git push`.

On AutoDL:

```bash
cd /root/autodl-tmp/D-EMAML-repro
git pull
```

`scripts/upload.ps1` is an SCP fallback only.

## Cloud order

```bash
cd /root/autodl-tmp/D-EMAML-repro
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
bash scripts/run_cloud.sh audit
bash scripts/run_cloud.sh smoke
# shut down, inspect artifacts/cache-smoke and the smoke checkpoint
bash scripts/run_cloud.sh prepare
# inspect motif counts, RAM and cache size; stop if cache >= 35 GiB
bash scripts/run_cloud.sh gate
bash scripts/run_cloud.sh gat
bash scripts/run_cloud.sh demgnn
# only if the single-seed scale looks right
bash scripts/run_cloud.sh demgnn-many
bash scripts/run_cloud.sh summarize
```

Keep the instance off between stages. If smoke fails, the data distribution
looks impossible, or one epoch OOMs, do not continue. If the paper
configuration OOMs, first confirm the code is not building a global
`|E|^2` tensor; only then move to a 48 GB GPU. Do not silently lower the
batch size and still call the result a same-setting reproduction.
