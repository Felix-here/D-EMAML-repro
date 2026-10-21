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


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig(
        data=DataConfig(**raw["data"]),
        motif=MotifConfig(**raw.get("motif", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
