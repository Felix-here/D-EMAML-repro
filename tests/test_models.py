import torch

from demaml.models import DEMGNN, SAIRR, MGCNLayer, MotifBaseline


def _graph():
    x = torch.randn(6, 11)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
    )
    motif_type = torch.arange(6)
    return x, edge_index, motif_type


def test_sairr_and_mgcn_shapes():
    _, edge_index, motif_type = _graph()
    h = torch.randn(6, 16)
    module = SAIRR(16)
    reduced = module(h, motif_type)
    assert reduced.shape == h.shape
    typed = torch.stack([projection(h) for projection in module.projections], dim=1)
    row = torch.arange(len(h))
    own = typed[row, motif_type]
    query = torch.einsum("nh,nhd->nd", own, module.attention[motif_type])
    keys = torch.einsum("nth,thd->ntd", typed, module.attention)
    scores = torch.einsum("nd,ntd->nt", query, keys)
    scores[row, motif_type] = -torch.inf
    weights = torch.softmax(scores, dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-5)
    output = MGCNLayer(16)(reduced, edge_index, motif_type)
    assert output.shape == h.shape
    lonely = MGCNLayer(16)(
        h[:1], torch.empty((2, 0), dtype=torch.long), motif_type[:1]
    )
    assert lonely.shape == (1, 16)
    assert torch.isfinite(lonely).all()


def test_demgnn_forward_and_ablation():
    x, edge_index, motif_type = _graph()
    for use_sairr in (True, False):
        for use_bias in (True, False):
            model = DEMGNN(
                11,
                hidden=16,
                layers=2,
                use_sairr=use_sairr,
                use_type_bias=use_bias,
            )
            assert model(x, edge_index, motif_type).shape == (6, 4)


def test_motif_baselines():
    x, edge_index, motif_type = _graph()
    for kind in ("gcn", "gin", "gat"):
        model = MotifBaseline(11, hidden=16, layers=2, kind=kind)
        assert model(x, edge_index, motif_type).shape == (6, 4)
