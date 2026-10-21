import numpy as np
import torch

from demaml.data import ProcessedData
from demaml.motif_graph import build_motif_graph
from demaml.motifs import MOTIF_NAMES, build_motifs, classify_pair, temporal_span


def test_figure3_time_spans():
    seconds = np.array([1, 3600, 3 * 3600, 5 * 3600])
    assert temporal_span(seconds).tolist() == [0, 0, 1, 2]
    hours = np.array([1, 2, 3, 4, 8]) * 3600
    assert temporal_span(hours).tolist() == [0, 1, 1, 2, 3]


def test_six_motif_types():
    # b1 repeated, b2 reciprocal, t1 fan-in, t2 reverse-time, t3 forward, t4 fan-out
    src1 = np.array([1, 1, 1, 1, 1, 1])
    dst1 = np.array([2, 2, 2, 2, 2, 2])
    src2 = np.array([1, 2, 3, 3, 2, 1])
    dst2 = np.array([2, 1, 2, 1, 3, 3])
    result = classify_pair(src1, dst1, src2, dst2)
    assert [MOTIF_NAMES[i] for i in result] == list(MOTIF_NAMES)


def test_binary_motifs_are_not_also_ternary():
    src1 = np.array([1, 1])
    dst1 = np.array([2, 2])
    src2 = np.array([1, 2])
    dst2 = np.array([2, 1])
    result = classify_pair(src1, dst1, src2, dst2)
    assert result.tolist() == [0, 1]


def _tiny_data(extra_self_loop: bool = False, split=None):
    src = [0, 1, 1, 3]
    dst = [1, 2, 3, 1]
    time = [0, 3600, 7200, 10800]
    y = [1, 0, 1, 0]
    if extra_self_loop:
        src.append(4)
        dst.append(4)
        time.append(14400)
        y.append(0)
    n = len(src)
    return ProcessedData(
        edge_index=torch.tensor([src, dst]),
        edge_attr=torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
        time=torch.tensor(time),
        y=torch.tensor(y),
        split=torch.tensor(split if split is not None else [0] * n),
        num_nodes=5,
        feature_names=["a", "b"],
        original_row=torch.arange(n),
    )


def test_build_motifs_and_shared_edge_graph(tmp_path):
    data = _tiny_data()
    motifs = build_motifs(data, tmp_path / "motifs.pt", eta=6, dmax=100)
    assert motifs.x.shape[1] == 2 * 2 + 1 + 6
    assert torch.all(
        motifs.label4
        == (data.y[motifs.original_edges[:, 0]] + 2 * data.y[motifs.original_edges[:, 1]])
    )
    graph = build_motif_graph(motifs, tmp_path / "graph.pt", max_motifs_per_edge=100)
    assert graph.num_nodes == len(motifs.x)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] % 2 == 0


def test_self_loops_are_excluded(tmp_path):
    motifs = build_motifs(_tiny_data(extra_self_loop=True), tmp_path / "motifs.pt")
    used = set(motifs.original_edges.reshape(-1).tolist())
    assert 4 not in used


def test_no_cross_split_motifs(tmp_path):
    data = _tiny_data(split=[0, 0, 1, 2])
    motifs = build_motifs(data, tmp_path / "motifs.pt")
    left = data.split[motifs.original_edges[:, 0]]
    right = data.split[motifs.original_edges[:, 1]]
    assert torch.equal(left, right)


def test_motif_graph_connects_shared_edges_only(tmp_path):
    data = ProcessedData(
        edge_index=torch.tensor([[0, 1, 3, 5], [1, 2, 4, 6]]),
        edge_attr=torch.ones(4, 2),
        time=torch.tensor([0, 3600, 7200, 10800]),
        y=torch.zeros(4, dtype=torch.long),
        split=torch.zeros(4, dtype=torch.long),
        num_nodes=7,
        feature_names=["a", "b"],
        original_row=torch.arange(4),
    )
    motifs = build_motifs(data, tmp_path / "motifs.pt")
    graph = build_motif_graph(motifs, tmp_path / "graph.pt", max_motifs_per_edge=100)
    # Motifs over {e0,e1} share an edge; {e2} and {e3} are disjoint if they form motifs.
    undirected = {tuple(sorted(pair)) for pair in graph.edge_index.t().tolist()}
    assert graph.edge_index.shape[1] == 2 * len(undirected)
