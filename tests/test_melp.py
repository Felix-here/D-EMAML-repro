import torch

from demaml.melp import (
    aggregate_edge_probabilities,
    motif_edge_probabilities,
    occurrence_bce_loss,
)


def test_four_class_marginalization():
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    logits = probabilities.log()
    result = motif_edge_probabilities(logits)
    assert torch.allclose(result, torch.tensor([[0.6, 0.7]]), atol=1e-6)


def test_one_hot_classes_map_to_roles():
    eye = torch.eye(4)
    mapped = motif_edge_probabilities(eye.log().clamp(min=-50))
    # After softmax the rows are not exactly one-hot, so use exact class probabilities.
    mapped = motif_edge_probabilities(torch.log(eye + 1e-12))
    assert torch.allclose(mapped[0], torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(mapped[1], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(mapped[2], torch.tensor([0.0, 1.0]), atol=1e-5)
    assert torch.allclose(mapped[3], torch.tensor([1.0, 1.0]), atol=1e-5)


def test_shared_edge_is_averaged():
    probabilities = torch.tensor(
        [
            [0.1, 0.7, 0.1, 0.1],  # first edge p=0.8
            [0.1, 0.1, 0.3, 0.5],  # second edge p=0.8
        ]
    )
    motif_edges = torch.tensor([[0, 1], [2, 0]])
    scores, observed = aggregate_edge_probabilities(
        probabilities.log(), motif_edges, num_edges=3
    )
    assert observed.tolist() == [True, True, True]
    assert torch.allclose(scores[0], torch.tensor(0.8), atol=1e-6)


def test_uncovered_edge_is_nan():
    probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    scores, observed = aggregate_edge_probabilities(
        probabilities.log(), torch.tensor([[0, 1]]), num_edges=3
    )
    assert observed.tolist() == [True, True, False]
    assert torch.isnan(scores[2])


def test_occurrence_loss_can_overfit():
    logits = torch.nn.Parameter(torch.zeros(2, 4))
    edges = torch.tensor([[0, 1], [1, 2]])
    labels = torch.tensor([1, 0, 1])
    optimizer = torch.optim.Adam([logits], lr=0.1)
    first = None
    last = None
    for _ in range(100):
        optimizer.zero_grad()
        loss = occurrence_bce_loss(logits, edges, labels)
        if first is None:
            first = float(loss)
        loss.backward()
        optimizer.step()
        last = float(loss)
    assert last < first * 0.2
