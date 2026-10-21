from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load(path: str | Path, map_location: Any = "cpu") -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
