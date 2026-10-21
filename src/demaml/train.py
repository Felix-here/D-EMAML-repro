from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from .config import RunConfig
from .data import ProcessedData
from .evaluate import binary_metrics, save_metrics, select_f1_threshold
from .io import torch_load
from .melp import aggregate_edge_probabilities, occurrence_bce_loss
from .models import DEMGNN, MotifBaseline, OriginalGAT
from .motif_graph import MotifGraph
from .motifs import MotifData
from .resources import snapshot

FULL_GRAPH_NODE_LIMIT = 8192


def _flatnonzero(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(config: RunConfig, input_dim: int) -> torch.nn.Module:
    kwargs = {
        "input_dim": input_dim,
        "hidden": config.model.hidden,
        "layers": config.model.layers,
        "dropout": config.model.dropout,
    }
    if config.model.name == "demgnn":
        return DEMGNN(
            **kwargs,
            use_sairr=config.model.use_sairr,
            use_type_bias=config.model.use_type_bias,
        )
    return MotifBaseline(**kwargs, kind=config.model.name)


def _seed_batches(nodes: torch.Tensor, batch_size: int, shuffle: bool):
    if shuffle:
        nodes = nodes[torch.randperm(len(nodes))]
    for start in range(0, len(nodes), batch_size):
        yield nodes[start : start + batch_size]


def _try_neighbor_loader(graph_data: Data, input_nodes: torch.Tensor, config: RunConfig, shuffle: bool):
    if graph_data.num_nodes <= FULL_GRAPH_NODE_LIMIT:
        return None
    try:
        from torch_geometric.loader import NeighborLoader

        return NeighborLoader(
            graph_data,
            input_nodes=input_nodes,
            num_neighbors=config.train.neighbors,
            batch_size=config.train.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
    except Exception:
        return None


@torch.no_grad()
def infer_split(
    model: torch.nn.Module,
    graph_data: Data,
    motifs: MotifData,
    split_id: int,
    config: RunConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _flatnonzero(motifs.split == split_id)
    model.eval()
    loader = _try_neighbor_loader(graph_data, indices, config, shuffle=False)
    if loader is None:
        output = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device),
            graph_data.motif_type.to(device),
        )
        return output[indices].cpu(), indices
    logits = torch.empty((len(indices), 4), dtype=torch.float32)
    motif_ids = torch.empty(len(indices), dtype=torch.long)
    cursor = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index, batch.motif_type)
        size = int(batch.batch_size)
        logits[cursor : cursor + size] = output[:size].cpu()
        motif_ids[cursor : cursor + size] = batch.n_id[:size].cpu()
        cursor += size
    return logits[:cursor], motif_ids[:cursor]


def _edge_metrics(
    logits: torch.Tensor,
    motif_ids: torch.Tensor,
    motifs: MotifData,
    original: ProcessedData,
    split_id: int,
    threshold: float | None,
) -> dict:
    scores, observed = aggregate_edge_probabilities(
        logits, motifs.original_edges[motif_ids], motifs.original_num_edges
    )
    mask = observed & (original.split == split_id)
    if int(mask.sum()) == 0:
        return {
            "threshold": 0.5 if threshold is None else threshold,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auprc": 0.0,
            "positives": 0,
            "examples": 0,
            "coverage": 0.0,
        }
    truth = original.y[mask].numpy()
    values = scores[mask].numpy()
    if threshold is None:
        threshold = select_f1_threshold(truth, values)
    result = binary_metrics(truth, values, threshold)
    result["coverage"] = float(mask.sum() / max(int((original.split == split_id).sum()), 1))
    return result


def _maybe_load_checkpoint(path: Path, model, optimizer, config: RunConfig, device):
    if not (config.train.resume and path.exists()):
        return 0, -1.0
    state = torch_load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1, float(state["best_f1"])


def train_motif_model(
    config: RunConfig,
    original: ProcessedData,
    motifs: MotifData,
    graph: MotifGraph,
    run_dir: str | Path,
) -> dict:
    set_seed(config.train.seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = Data(
        x=motifs.x,
        edge_index=graph.edge_index,
        motif_type=motifs.motif_type,
        num_nodes=graph.num_nodes,
    )
    train_nodes = _flatnonzero(motifs.split == 0)
    model = _model(config, motifs.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    checkpoint = run_dir / "last.pt"
    start_epoch, best_f1 = _maybe_load_checkpoint(checkpoint, model, optimizer, config, device)
    history: list[dict] = []
    edge_labels = original.y.to(device)
    peak = snapshot(run_dir)

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        losses = []
        loader = _try_neighbor_loader(graph_data, train_nodes, config, shuffle=True)
        if loader is None:
            full_x = graph_data.x.to(device)
            full_index = graph_data.edge_index.to(device)
            full_type = graph_data.motif_type.to(device)
            for seeds in tqdm(
                list(_seed_batches(train_nodes, config.train.batch_size, True)),
                desc=f"epoch {epoch + 1}",
                leave=False,
            ):
                optimizer.zero_grad(set_to_none=True)
                output = model(full_x, full_index, full_type)
                seed_edges = motifs.original_edges[seeds].to(device)
                loss = occurrence_bce_loss(output[seeds.to(device)], seed_edges, edge_labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        else:
            for batch in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch.x, batch.edge_index, batch.motif_type)
                size = int(batch.batch_size)
                seed_ids = batch.n_id[:size].cpu()
                seed_edges = motifs.original_edges[seed_ids].to(device)
                loss = occurrence_bce_loss(output[:size], seed_edges, edge_labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

        val_logits, val_ids = infer_split(model, graph_data, motifs, 1, config, device)
        val_metrics = _edge_metrics(val_logits, val_ids, motifs, original, 1, None)
        epoch_result = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "validation": val_metrics,
            "resources": snapshot(run_dir),
        }
        history.append(epoch_result)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
        }
        if (epoch + 1) % config.train.checkpoint_every == 0:
            torch.save(state, checkpoint)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=str), encoding="utf-8"
        )

    if not (run_dir / "best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": max(start_epoch - 1, 0)}, run_dir / "best.pt")
    best = torch_load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    val_logits, val_ids = infer_split(model, graph_data, motifs, 1, config, device)
    val = _edge_metrics(val_logits, val_ids, motifs, original, 1, None)
    test_logits, test_ids = infer_split(model, graph_data, motifs, 2, config, device)
    test = _edge_metrics(test_logits, test_ids, motifs, original, 2, val["threshold"])
    result = {
        "validation": val,
        "test": test,
        "best_epoch": best["epoch"],
        "peak_resources": peak,
        "final_resources": snapshot(run_dir),
        "sampler": "neighbor" if graph.num_nodes > FULL_GRAPH_NODE_LIMIT else "full",
    }
    save_metrics(result, run_dir / "metrics.json")
    return result


def train_original_model(
    config: RunConfig,
    original: ProcessedData,
    run_dir: str | Path,
) -> dict:
    set_seed(config.train.seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OriginalGAT(
        original.edge_attr.shape[1], hidden=config.model.hidden, layers=config.model.layers
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    checkpoint = run_dir / "last.pt"
    start_epoch, best_f1 = _maybe_load_checkpoint(checkpoint, model, optimizer, config, device)
    train_idx = _flatnonzero(original.split == 0).to(device)
    history: list[dict] = []
    labels = original.y.float().to(device)
    edge_index = original.edge_index.to(device)
    edge_attr = original.edge_attr.to(device)
    positives = labels[train_idx].sum()
    negatives = train_idx.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(max=10_000)

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(original.num_nodes, edge_index, edge_attr)
        batch_labels = labels[train_idx]
        weight = torch.where(batch_labels > 0, pos_weight, torch.ones_like(batch_labels))
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], batch_labels, weight=weight)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scores = torch.sigmoid(model(original.num_nodes, edge_index, edge_attr)).cpu()
        val_mask = original.split == 1
        val = binary_metrics(
            original.y[val_mask].numpy(),
            scores[val_mask].numpy(),
            select_f1_threshold(original.y[val_mask].numpy(), scores[val_mask].numpy()),
        )
        history.append({"epoch": epoch, "loss": float(loss.detach()), "validation": val})
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        if (epoch + 1) % config.train.checkpoint_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                },
                checkpoint,
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=str), encoding="utf-8"
        )

    if not (run_dir / "best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": max(start_epoch - 1, 0)}, run_dir / "best.pt")
    best = torch_load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    with torch.no_grad():
        scores = torch.sigmoid(model(original.num_nodes, edge_index, edge_attr)).cpu()
    val_mask = original.split == 1
    test_mask = original.split == 2
    val = binary_metrics(
        original.y[val_mask].numpy(),
        scores[val_mask].numpy(),
        select_f1_threshold(original.y[val_mask].numpy(), scores[val_mask].numpy()),
    )
    test = binary_metrics(original.y[test_mask].numpy(), scores[test_mask].numpy(), val["threshold"])
    test["coverage"] = 1.0
    val["coverage"] = 1.0
    result = {"validation": val, "test": test, "best_epoch": best["epoch"]}
    save_metrics(result, run_dir / "metrics.json")
    return result
