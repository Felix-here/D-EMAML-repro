from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.utils import add_self_loops, degree


class SAIRR(nn.Module):
    """Attention-driven cross-type redundancy subtraction (paper eq. 4-9)."""

    def __init__(self, hidden: int, num_types: int = 6):
        super().__init__()
        self.num_types = num_types
        self.projections = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_types)]
        )
        self.attention = nn.Parameter(torch.empty(num_types, hidden, hidden))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, h: torch.Tensor, motif_type: torch.Tensor) -> torch.Tensor:
        typed = torch.stack([projection(h) for projection in self.projections], dim=1)
        row = torch.arange(len(h), device=h.device)
        own = typed[row, motif_type]
        query = torch.einsum("nh,nhd->nd", own, self.attention[motif_type])
        keys = torch.einsum("nth,thd->ntd", typed, self.attention)
        scores = torch.einsum("nd,ntd->nt", query, keys) / math.sqrt(self.num_types)
        scores[row, motif_type] = -torch.inf
        weights = torch.softmax(scores, dim=1)
        redundant = torch.einsum("nt,nth->nh", weights, typed)
        beta = torch.sigmoid((own * redundant).sum(dim=1, keepdim=True))
        return F.relu(beta * (own - redundant))


class MGCNLayer(nn.Module):
    """Degree-normalized convolution with ordered motif-type pair biases."""

    def __init__(self, hidden: int, num_types: int = 6, use_type_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self.use_type_bias = use_type_bias
        self.type_bias = nn.Parameter(torch.zeros(num_types, num_types, hidden))

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, motif_type: torch.Tensor
    ) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(h))
        src, dst = edge_index
        deg = degree(dst, len(h), dtype=h.dtype).clamp_min_(1)
        norm = deg[src].rsqrt() * deg[dst].rsqrt()
        message = self.linear(h[src]) * norm[:, None]
        if self.use_type_bias:
            message = message + self.type_bias[motif_type[dst], motif_type[src]]
        out = torch.zeros_like(h)
        out.index_add_(0, dst, message)
        return F.relu(out)


class DEMGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        use_sairr: bool = True,
        use_type_bias: bool = True,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.sairr = nn.ModuleList(
            [SAIRR(hidden) if use_sairr else nn.Identity() for _ in range(layers)]
        )
        self.convs = nn.ModuleList(
            [MGCNLayer(hidden, use_type_bias=use_type_bias) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 4),
        )
        self.use_sairr = use_sairr

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, motif_type: torch.Tensor
    ) -> torch.Tensor:
        h = F.relu(self.input(x))
        for reduction, conv, norm in zip(self.sairr, self.convs, self.norms):
            reduced = reduction(h, motif_type) if self.use_sairr else reduction(h)
            update = norm(conv(reduced, edge_index, motif_type))
            h = (h + F.dropout(update, self.dropout, self.training)) / 2
        return self.output(h)


class MotifBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        layers: int = 2,
        kind: str = "gat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if kind == "gat":
                conv = GATConv(hidden, hidden // 4, heads=4, concat=True)
            elif kind == "gin":
                conv = GINConv(
                    nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                )
            elif kind == "gcn":
                conv = GCNConv(hidden, hidden)
            else:
                raise ValueError(f"Unknown motif baseline: {kind}")
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden))
        self.output = nn.Linear(hidden, 4)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, motif_type: torch.Tensor
    ) -> torch.Tensor:
        del motif_type
        h = F.relu(self.input(x))
        for conv, norm in zip(self.convs, self.norms):
            h = (h + F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training)) / 2
        return self.output(h)


class OriginalGAT(nn.Module):
    """Edge classifier used as an original-transaction-graph reference."""

    def __init__(self, edge_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.node_embedding = nn.Linear(1, hidden)
        self.edge_embedding = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList(
            [GATConv(hidden, hidden // 4, heads=4, edge_dim=hidden) for _ in range(layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, num_nodes: int, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        x = torch.ones((num_nodes, 1), device=edge_index.device)
        h = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)
        for conv in self.convs:
            h = (h + F.relu(conv(h, edge_index, e))) / 2
        src, dst = edge_index
        return self.classifier(torch.cat([h[src], h[dst], e], dim=1)).squeeze(1)
