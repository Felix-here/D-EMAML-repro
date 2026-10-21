from __future__ import annotations

import torch
from torch.nn import functional as F


def motif_edge_probabilities(logits4: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits4, dim=-1)
    first = probabilities[:, 1] + probabilities[:, 3]
    second = probabilities[:, 2] + probabilities[:, 3]
    return torch.stack([first, second], dim=1)


def aggregate_edge_probabilities(
    logits4: torch.Tensor,
    motif_edges: torch.Tensor,
    num_edges: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    occurrence_prob = motif_edge_probabilities(logits4).reshape(-1)
    edge_ids = motif_edges.reshape(-1)
    sums = torch.zeros(num_edges, dtype=logits4.dtype, device=logits4.device)
    counts = torch.zeros(num_edges, dtype=logits4.dtype, device=logits4.device)
    sums.index_add_(0, edge_ids, occurrence_prob)
    counts.index_add_(0, edge_ids, torch.ones_like(occurrence_prob))
    observed = counts > 0
    result = torch.full_like(sums, torch.nan)
    result[observed] = sums[observed] / counts[observed]
    return result, observed


def occurrence_bce_loss(
    logits4: torch.Tensor,
    motif_edges: torch.Tensor,
    edge_labels: torch.Tensor,
) -> torch.Tensor:
    """Mini-batch approximation documented in docs/ASSUMPTIONS.md."""
    probabilities = motif_edge_probabilities(logits4).clamp(1e-7, 1 - 1e-7)
    targets = edge_labels[motif_edges].to(probabilities.dtype)
    positives = targets.sum()
    negatives = targets.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(max=10_000)
    weights = torch.where(targets > 0, positive_weight, 1.0)
    return F.binary_cross_entropy(probabilities, targets, weight=weights)
