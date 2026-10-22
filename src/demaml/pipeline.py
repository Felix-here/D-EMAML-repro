from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from .config import RunConfig, run_label
from .data import ProcessedData, audit_csv, preprocess_csv
from .motif_graph import MotifGraph, build_motif_graph
from .motifs import MotifData, build_motifs, motif_statistics
from .resources import snapshot
from .train import train_motif_model, train_original_model

ORIGINAL_GRAPH_MODELS = {"gate", "original_gat"}


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _file_gb(path: Path) -> float:
    return round(path.stat().st_size / 1024**3, 4) if path.exists() else 0.0


def prepare_artifacts(config: RunConfig, run_dir: Path | None = None) -> dict:
    root = Path(config.data.work_dir).resolve()
    cache = root / ("cache-smoke" if config.data.smoke else "cache-full")
    cache.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    resources: dict[str, object] = {"start": snapshot(cache)}

    started = time.perf_counter()
    audit_path = (run_dir or cache) / "data-audit.json"
    audit = audit_csv(config.data.csv, audit_path)
    timings["audit"] = time.perf_counter() - started
    resources["after_audit"] = snapshot(cache)

    processed_path = cache / "processed.pt"
    started = time.perf_counter()
    if processed_path.exists():
        original = ProcessedData.load(processed_path)
    else:
        original = preprocess_csv(
            config.data.csv,
            processed_path,
            smoke=config.data.smoke,
            smoke_negatives=config.data.smoke_negatives,
            seed=config.train.seed,
        )
    timings["preprocess"] = time.perf_counter() - started
    resources["after_preprocess"] = snapshot(cache)

    result = {
        "audit": audit,
        "processed_edges": int(original.y.numel()),
        "processed_positives": int(original.y.sum()),
        "num_nodes": original.num_nodes,
        "feature_dim_edge": int(original.edge_attr.shape[1]),
        "feature_names": original.feature_names,
        "self_loops": int((original.edge_index[0] == original.edge_index[1]).sum()),
        "cache": str(cache),
        "processed_gb": _file_gb(processed_path),
        "timings_seconds": timings,
        "resources": resources,
    }

    if config.model.name in ORIGINAL_GRAPH_MODELS:
        return result

    motif_path = cache / (
        f"motifs-x{config.motif.xi}-e{config.motif.eta}-d{config.motif.dmax}.pt"
    )
    started = time.perf_counter()
    if motif_path.exists():
        print(f"loading existing motifs from {motif_path}", flush=True)
        motifs = MotifData.load(motif_path)
    else:
        print(f"building motifs into {motif_path}", flush=True)
        motifs = build_motifs(
            original,
            motif_path,
            xi=config.motif.xi,
            eta=config.motif.eta,
            dmax=config.motif.dmax,
        )
    timings["motifs"] = time.perf_counter() - started
    statistics = motif_statistics(motifs)
    resources["after_motifs"] = snapshot(cache)

    graph_path = cache / f"motif-graph-cap{config.motif.max_motifs_per_edge}.pt"
    started = time.perf_counter()
    if graph_path.exists():
        print(f"loading existing motif graph from {graph_path}", flush=True)
        graph = MotifGraph.load(graph_path)
    else:
        print(f"building motif graph into {graph_path}", flush=True)
        graph = build_motif_graph(
            motifs,
            graph_path,
            max_motifs_per_edge=config.motif.max_motifs_per_edge,
        )
    timings["motif_graph"] = time.perf_counter() - started
    resources["after_motif_graph"] = snapshot(cache)

    result.update(
        {
            "motifs": statistics,
            "motif_graph_edges": int(graph.edge_index.shape[1]),
            "motif_gb": _file_gb(motif_path),
            "motif_graph_gb": _file_gb(graph_path),
            "cache_gb": round(sum(_file_gb(path) for path in cache.glob("*")), 4),
        }
    )
    if run_dir is not None:
        (run_dir / "prepare.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        (run_dir / "motif-statistics.json").write_text(
            json.dumps(statistics, indent=2, default=str), encoding="utf-8"
        )
    else:
        (cache / "prepare.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    if result.get("cache_gb", 0) >= 35:
        result["stop_recommended"] = (
            "Cache exceeded 35 GiB. Inspect motif counts before training."
        )
    return result


def run_pipeline(config: RunConfig) -> dict:
    root = Path(config.data.work_dir).resolve()
    run_dir = root / "runs" / run_label(config) / f"seed-{config.train.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8"
    )
    prepared = prepare_artifacts(config, run_dir)
    cache = Path(prepared["cache"])
    original = ProcessedData.load(cache / "processed.pt")

    started = time.perf_counter()
    if config.model.name in ORIGINAL_GRAPH_MODELS:
        result = train_original_model(config, original, run_dir)
    else:
        motif_path = cache / (
            f"motifs-x{config.motif.xi}-e{config.motif.eta}-d{config.motif.dmax}.pt"
        )
        graph_path = cache / f"motif-graph-cap{config.motif.max_motifs_per_edge}.pt"
        motifs = MotifData.load(motif_path)
        graph = MotifGraph.load(graph_path)
        result = train_motif_model(config, original, motifs, graph, run_dir)
    prepared.setdefault("timings_seconds", {})["train_evaluate"] = time.perf_counter() - started
    result.update(prepared)
    result["model_tag"] = run_label(config)
    result["architecture"] = config.model.name
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
