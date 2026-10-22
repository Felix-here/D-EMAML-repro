from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    csv: str
    work_dir: str = "artifacts"
    smoke: bool = False
    smoke_negatives: int = 50_000


@dataclass
class MotifConfig:
    xi: int = 0
    eta: int = 6
    dmax: int = 256
    max_motifs_per_edge: int = 128


@dataclass
class ModelConfig:
    name: str = "demgnn"
    tag: str = ""
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    use_sairr: bool = True
    use_type_bias: bool = True


@dataclass
class TrainConfig:
    seed: int = 1
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    neighbors: list[int] = field(default_factory=lambda: [100, 100])
    checkpoint_every: int = 1
    resume: bool = True


@dataclass
class RunConfig:
    data: DataConfig
    motif: MotifConfig = field(default_factory=MotifConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_label(config: RunConfig) -> str:
    return config.model.tag or config.model.name


def _read_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return raw


def _isolate_smoke(config: RunConfig) -> RunConfig:
    work = config.data.work_dir.rstrip("/\\")
    if config.data.smoke and not work.endswith("smoke"):
        config.data.work_dir = str(Path(work) / "smoke")
    return config


def _build_run(
    data: dict[str, Any],
    motif: dict[str, Any],
    model: dict[str, Any],
    train: dict[str, Any],
) -> RunConfig:
    train = {key: value for key, value in train.items() if key != "seeds"}
    model = {key: value for key, value in model.items() if key != "enabled"}
    return _isolate_smoke(
        RunConfig(
            data=DataConfig(**data),
            motif=MotifConfig(**motif),
            model=ModelConfig(**model),
            train=TrainConfig(**train),
        )
    )


def load_config(path: str | Path, model: str | None = None) -> RunConfig:
    runs, _seeds = load_runs(path, only=model, enabled_only=model is None)
    if len(runs) != 1:
        labels = ", ".join(run_label(run) for run in runs) or "无"
        raise ValueError(
            f"{path} 解析出 {len(runs)} 个模型（{labels}）。"
            "对比全部启用项用 compare；只跑一个时传入 --model 标签。"
        )
    return runs[0]


def load_runs(
    path: str | Path,
    only: str | None = None,
    enabled_only: bool = True,
) -> tuple[list[RunConfig], list[int]]:
    raw = _read_yaml(path)
    if "models" not in raw:
        config = _build_run(
            raw["data"],
            raw.get("motif", {}),
            raw.get("model", {}),
            raw.get("train", {}),
        )
        if only and run_label(config) != only and config.model.name != only:
            raise ValueError(f"{path} 里没有模型 {only}")
        return [config], [config.train.seed]

    train = dict(raw.get("train", {}))
    seeds = list(train.get("seeds") or [train.get("seed", 1)])
    runs: list[RunConfig] = []
    for tag, spec in raw["models"].items():
        if not isinstance(spec, dict):
            raise TypeError(f"模型 {tag} 必须是一组参数")
        enabled = bool(spec.get("enabled", True))
        selected = only is not None and tag == only
        if only is not None and not selected:
            continue
        if enabled_only and not enabled and not selected:
            continue
        model = dict(spec)
        model.setdefault("tag", tag)
        runs.append(_build_run(raw["data"], raw.get("motif", {}), model, train))
    if only is not None and not runs:
        raise ValueError(f"{path} 里没有模型标签 {only}")
    return runs, [int(seed) for seed in seeds]
