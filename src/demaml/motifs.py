from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import ProcessedData
from .io import torch_load

MOTIF_NAMES = ("b1", "b2", "t1", "t2", "t3", "t4")


@dataclass
class MotifData:
    original_edges: torch.Tensor
    x: torch.Tensor
    motif_type: torch.Tensor
    label4: torch.Tensor
    split: torch.Tensor
    delta: torch.Tensor
    original_num_edges: int

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "MotifData":
        return cls(**torch_load(path, map_location="cpu"))


def temporal_span(seconds: np.ndarray | int) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=np.int64)
    elapsed_hours = np.maximum(1, (seconds + 3599) // 3600)
    return np.floor(np.log2(elapsed_hours)).astype(np.int16)


def classify_pair(
    src1: np.ndarray,
    dst1: np.ndarray,
    src2: np.ndarray,
    dst2: np.ndarray,
) -> np.ndarray:
    """IDs follow Figure 3: b1, b2, then t1 fan-in, t2 reverse-time, t3 forward, t4 fan-out."""
    result = np.full(np.broadcast_shapes(src1.shape, src2.shape), -1, dtype=np.int8)
    same_direction = (src1 == src2) & (dst1 == dst2)
    reverse = (src1 == dst2) & (dst1 == src2) & ~same_direction
    result[same_direction] = 0
    result[reverse] = 1
    remaining = result < 0
    result[remaining & (dst1 == dst2)] = 2  # t1 fan-in
    result[remaining & (src1 == dst2)] = 3  # t2 reverse-time chain
    result[remaining & (dst1 == src2)] = 4  # t3 forward chain
    result[remaining & (src1 == src2)] = 5  # t4 fan-out
    return result


def _retained_positions(length: int, dmax: int) -> np.ndarray:
    if dmax <= 0 or length <= dmax:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, dmax, dtype=np.int64))


def build_motifs(
    data: ProcessedData,
    output_path: str | Path,
    *,
    xi: int = 0,
    eta: int = 6,
    dmax: int = 256,
) -> MotifData:
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    times = data.time.numpy()
    splits = data.split.numpy()
    labels = data.y.numpy()
    edge_ids = np.arange(len(times), dtype=np.int64)
    usable = src != dst
    incidence_nodes = np.concatenate([src[usable], dst[usable]])
    incidence_edges = np.concatenate([edge_ids[usable], edge_ids[usable]])
    order = np.lexsort((times[incidence_edges], incidence_nodes))
    incidence_nodes = incidence_nodes[order]
    incidence_edges = incidence_edges[order]
    boundaries = np.flatnonzero(
        np.r_[True, incidence_nodes[1:] != incidence_nodes[:-1], True]
    )

    pair_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = incidence_edges[start:stop]
        local = local[_retained_positions(len(local), dmax)]
        if len(local) < 2:
            continue
        left, right = np.triu_indices(len(local), 1)
        e1, e2 = local[left], local[right]
        swap = (times[e2] < times[e1]) | ((times[e2] == times[e1]) & (e2 < e1))
        e1, e2 = np.where(swap, e2, e1), np.where(swap, e1, e2)
        delta = temporal_span(times[e2] - times[e1])
        motif_type = classify_pair(src[e1], dst[e1], src[e2], dst[e2])
        valid = (delta >= xi) & (delta <= eta) & (motif_type >= 0)
        valid &= splits[e1] == splits[e2]
        valid &= e1 != e2
        node = incidence_nodes[start]
        binary = motif_type <= 1
        owner = np.minimum(src[e1], dst[e1])
        valid &= ~binary | (node == owner)
        if not np.any(valid):
            continue
        pair_parts.append(np.stack([e1[valid], e2[valid]], axis=1))
        type_parts.append(motif_type[valid])
        delta_parts.append(delta[valid])
        split_parts.append(splits[e1[valid]])

    if not pair_parts:
        raise RuntimeError("No motifs generated; check time units and xi/eta")
    pairs = np.concatenate(pair_parts)
    types = np.concatenate(type_parts)
    deltas = np.concatenate(delta_parts)
    motif_split = np.concatenate(split_parts)

    attrs = data.edge_attr.numpy()
    type_one_hot = np.eye(6, dtype=np.float32)[types]
    delta_feature = (deltas.astype(np.float32) / max(1, eta))[:, None]
    x = np.concatenate([attrs[pairs[:, 0]], attrs[pairs[:, 1]], delta_feature, type_one_hot], 1)
    label4 = labels[pairs[:, 0]] + 2 * labels[pairs[:, 1]]
    motifs = MotifData(
        original_edges=torch.from_numpy(pairs).long(),
        x=torch.from_numpy(x),
        motif_type=torch.from_numpy(types).long(),
        label4=torch.from_numpy(label4).long(),
        split=torch.from_numpy(motif_split).long(),
        delta=torch.from_numpy(deltas.astype(np.int64)),
        original_num_edges=len(times),
    )
    motifs.save(output_path)
    return motifs


def motif_statistics(motifs: MotifData) -> dict:
    result: dict[str, object] = {
        "count": len(motifs.x),
        "feature_dim": int(motifs.x.shape[1]),
        "original_num_edges": motifs.original_num_edges,
    }
    result["types"] = {
        MOTIF_NAMES[i]: int((motifs.motif_type == i).sum()) for i in range(6)
    }
    result["delta"] = {
        str(int(v)): int(c)
        for v, c in zip(*torch.unique(motifs.delta, return_counts=True))
    }
    result["labels"] = {str(i): int((motifs.label4 == i).sum()) for i in range(4)}
    result["splits"] = {
        name: int((motifs.split == i).sum())
        for i, name in enumerate(("train", "valid", "test"))
    }
    covered = torch.zeros(motifs.original_num_edges, dtype=torch.bool)
    covered[motifs.original_edges.reshape(-1)] = True
    result["edge_coverage"] = float(covered.float().mean())
    return result
