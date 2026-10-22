from demaml.config import load_config, load_runs, run_label


def test_experiment_file_keeps_shared_settings_and_model_overrides(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
data:
  csv: data.csv
  work_dir: artifacts
  smoke: true
motif:
  dmax: 32
train:
  seeds: [1, 2]
  epochs: 3
  batch_size: 8
models:
  demgnn:
    enabled: true
    name: demgnn
    hidden: 16
    use_sairr: true
  small_gat:
    enabled: false
    name: gat
    hidden: 8
  demgnn_no_sairr:
    enabled: true
    name: demgnn
    hidden: 16
    use_sairr: false
""",
        encoding="utf-8",
    )
    enabled, seeds = load_runs(path)
    assert seeds == [1, 2]
    assert [run_label(run) for run in enabled] == ["demgnn", "demgnn_no_sairr"]
    assert enabled[0].motif.dmax == 32
    assert enabled[0].train.epochs == 3
    assert enabled[0].data.work_dir.endswith("smoke")
    assert enabled[1].model.use_sairr is False
    assert enabled[0].model.name == enabled[1].model.name == "demgnn"

    gat = load_config(path, model="small_gat")
    assert gat.model.name == "gat"
    assert gat.model.hidden == 8
