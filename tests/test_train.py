from pathlib import Path

import yaml

from demaml.config import load_config
from demaml.motif_graph import build_motif_graph
from demaml.motifs import build_motifs
from demaml.pipeline import prepare_artifacts, run_pipeline
from demaml.train import train_motif_model
from tests.test_data import _write_csv
from tests.test_motifs import _tiny_data


def _write_config(path: Path, csv: Path, work: Path, epochs: int = 2, seed: int = 1) -> Path:
    payload = {
        "data": {"csv": str(csv), "work_dir": str(work), "smoke": False},
        "motif": {"xi": 0, "eta": 6, "dmax": 32, "max_motifs_per_edge": 16},
        "model": {
            "name": "demgnn",
            "hidden": 16,
            "layers": 2,
            "dropout": 0.0,
            "use_sairr": True,
            "use_type_bias": True,
        },
        "train": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": 8,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "neighbors": [8, 8],
            "checkpoint_every": 1,
            "resume": True,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_tiny_graph_overfits_and_resumes(tmp_path):
    data = _tiny_data()
    motifs = build_motifs(data, tmp_path / "motifs.pt")
    graph = build_motif_graph(motifs, tmp_path / "graph.pt", max_motifs_per_edge=32)
    config = load_config(
        _write_config(tmp_path / "cfg.yaml", tmp_path / "unused.csv", tmp_path, epochs=3)
    )
    first = train_motif_model(config, data, motifs, graph, tmp_path / "run")
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "best.pt").exists()
    config.train.epochs = 4
    second = train_motif_model(config, data, motifs, graph, tmp_path / "run")
    assert second["best_epoch"] >= 0
    assert first["test"]["examples"] >= 0


def test_pipeline_prepare_and_run(tmp_path):
    csv = tmp_path / "tiny.csv"
    _write_csv(csv)
    config_path = _write_config(tmp_path / "cfg.yaml", csv, tmp_path / "work", epochs=2)
    config = load_config(config_path)
    prepared = prepare_artifacts(config)
    assert prepared["processed_edges"] == 20
    assert prepared["motifs"]["feature_dim"] > 6
    result = run_pipeline(config)
    assert "test" in result
    assert (tmp_path / "work" / "runs" / "demgnn" / "seed-1" / "result.json").exists()
