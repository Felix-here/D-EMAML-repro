from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .io import torch_load
from .motifs import MotifData


@dataclass
class MotifGraph:
    edge_index: torch.Tensor
    num_nodes: int

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "MotifGraph":
        return cls(**torch_load(path, map_location="cpu"))


def build_motif_graph(
    motifs: MotifData,
    output_path: str | Path,
    *,
    max_motifs_per_edge: int = 128,
) -> MotifGraph:
    pairs = motifs.original_edges.numpy()
    motif_ids = np.arange(len(pairs), dtype=np.int64)
    original_edge = np.concatenate([pairs[:, 0], pairs[:, 1]])
    incident_motif = np.concatenate([motif_ids, motif_ids])
    order = np.lexsort((incident_motif, original_edge))
    original_edge = original_edge[order]
    incident_motif = incident_motif[order]
    boundaries = np.flatnonzero(
        np.r_[True, original_edge[1:] != original_edge[:-1], True]
    )

    graph_parts: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = np.unique(incident_motif[start:stop])
        if len(local) > max_motifs_per_edge:
            positions = np.linspace(
                0, len(local) - 1, max_motifs_per_edge, dtype=np.int64
            )
            local = local[positions]
        if len(local) < 2:
            continue
        left, right = np.triu_indices(len(local), 1)
        graph_parts.append(np.stack([local[left], local[right]], axis=0))

    if graph_parts:
        forward = np.concatenate(graph_parts, axis=1)
        edge_index = np.concatenate([forward, forward[::-1]], axis=1)
        edge_index = torch.from_numpy(edge_index).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    graph = MotifGraph(edge_index=edge_index, num_nodes=len(pairs))
    graph.save(output_path)
    return graph
