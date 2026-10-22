from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .io import torch_load
from .motifs import MotifData

MAX_EDGE_INDEX_BYTES = 8 * 1024**3


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


def _selected(local: np.ndarray, max_motifs_per_edge: int) -> np.ndarray:
    if len(local) <= max_motifs_per_edge:
        return local
    positions = np.linspace(0, len(local) - 1, max_motifs_per_edge, dtype=np.int64)
    return local[positions]


def build_motif_graph(
    motifs: MotifData,
    output_path: str | Path,
    *,
    max_motifs_per_edge: int = 8,
) -> MotifGraph:
    pairs = motifs.original_edges.numpy()
    motif_ids = np.arange(len(pairs), dtype=np.int64)
    original_edge = np.concatenate([pairs[:, 0], pairs[:, 1]])
    incident_motif = np.concatenate([motif_ids, motif_ids])
    order = np.lexsort((incident_motif, original_edge))
    original_edge = original_edge[order]
    incident_motif = incident_motif[order]
    boundaries = np.flatnonzero(np.diff(original_edge)) + 1
    boundaries = np.r_[0, boundaries, len(original_edge)]

    directed = 0
    kept_groups = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        kept = min(int(stop - start), max_motifs_per_edge)
        if kept >= 2:
            directed += kept * (kept - 1)
            kept_groups += 1
    edge_bytes = directed * 16
    print(
        f"motif graph groups={kept_groups} directed_edges={directed} "
        f"estimated_gib={edge_bytes / 1024**3:.2f}",
        flush=True,
    )
    if edge_bytes > MAX_EDGE_INDEX_BYTES:
        raise MemoryError(
            "Motif graph is too large for this instance. "
            f"Lower motif.max_motifs_per_edge below {max_motifs_per_edge} and rerun. "
            "Existing processed.pt and motifs files will be reused."
        )

    edge_index = np.empty((2, directed), dtype=np.int64)
    cursor = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = _selected(incident_motif[start:stop], max_motifs_per_edge)
        count = len(local)
        if count < 2:
            continue
        left, right = np.triu_indices(count, 1)
        width = len(left)
        block = edge_index[:, cursor : cursor + 2 * width]
        block[0, :width] = local[left]
        block[1, :width] = local[right]
        block[0, width:] = local[right]
        block[1, width:] = local[left]
        cursor += 2 * width

    graph = MotifGraph(
        edge_index=torch.from_numpy(edge_index[:, :cursor]).long(),
        num_nodes=len(pairs),
    )
    print(f"writing motif graph to {output_path}", flush=True)
    graph.save(output_path)
    return graph
