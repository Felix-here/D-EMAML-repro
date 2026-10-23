from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config, load_runs, run_label
from .data import audit_csv
from .pipeline import prepare_artifacts, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent D-EMAML reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data")
    audit.add_argument("--csv", required=True)
    audit.add_argument("--output")
    plan = commands.add_parser("plan")
    plan.add_argument("--config", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--model", help="experiment.yaml 里的模型标签")
    many = commands.add_parser("run-many")
    many.add_argument("--config", required=True)
    many.add_argument("--model", help="experiment.yaml 里的模型标签")
    many.add_argument("--seeds", nargs="+", type=int)
    compare = commands.add_parser("compare")
    compare.add_argument("--config", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("metrics", nargs="+")
    return parser


def _summarize(paths: list[str]) -> dict:
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    return _summarize_results(rows)


def _summarize_results(rows: list[dict]) -> dict:
    result: dict[str, object] = {"runs": len(rows), "claim": "independent-reimplementation"}
    paper = {"f1": 0.264, "auprc": 0.152, "f1_std": 0.0155, "auprc_std": 0.0196}
    for key in ("f1", "auprc", "precision", "recall", "coverage"):
        values = np.asarray([row["test"][key] for row in rows], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    f1_mean = result["f1"]["mean"]
    auprc_mean = result["auprc"]["mean"]
    result["paper"] = paper
    result["within_two_std"] = {
        "f1": abs(f1_mean - paper["f1"]) <= 2 * paper["f1_std"],
        "auprc": abs(auprc_mean - paper["auprc"]) <= 2 * paper["auprc_std"],
    }
    return result


def _plan(path: str) -> dict:
    enabled, seeds = load_runs(path, enabled_only=True)
    available, _ignored = load_runs(path, enabled_only=False)
    return {
        "config": str(path),
        "seeds": seeds,
        "enabled": [_describe(run) for run in enabled],
        "disabled": [
            _describe(run)
            for run in available
            if run_label(run) not in {run_label(item) for item in enabled}
        ],
    }


def _describe(config) -> dict:
    model = config.model
    described = {
        "tag": run_label(config),
        "architecture": model.name,
        "hidden": model.hidden,
        "layers": model.layers,
        "dropout": model.dropout,
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.learning_rate,
        "neighbors": config.train.neighbors,
        "smoke": config.data.smoke,
        "work_dir": config.data.work_dir,
    }
    if model.name == "demgnn":
        described["use_sairr"] = model.use_sairr
        described["use_type_bias"] = model.use_type_bias
    return described


def _seeded(config, seed: int):
    return replace(config, train=replace(config.train, seed=seed))


def _run_seeds(config, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        results.append(run_pipeline(_seeded(config, seed)))
    summary = _summarize_results(results)
    summary["tag"] = run_label(config)
    summary["architecture"] = config.model.name
    work = Path(config.data.work_dir) / "runs" / run_label(config)
    work.mkdir(parents=True, exist_ok=True)
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def write_comparison_report(
    work_dir: Path,
    tags: list[str],
    seeds: list[int],
    total_seconds: float | None = None,
) -> dict:
    """Refresh the comparison table. Finished models stay visible if a later run stops."""
    rows = []
    lines = [
        f"{'model':<8} {'seed':>4} {'best_epoch':>10} {'hours':>8} {'F1':>8} "
        f"{'AUPRC':>8} {'Precision':>9} {'Recall':>8} {'coverage':>8} status"
    ]
    for tag in tags:
        for seed in seeds:
            path = work_dir / "runs" / tag / f"seed-{seed}" / "metrics.json"
            if not path.exists():
                rows.append({"model": tag, "seed": seed, "status": "pending"})
                lines.append(
                    f"{tag:<8} {seed:>4} {'':>10} {'':>8} {'':>8} {'':>8} {'':>9} {'':>8} {'':>8} pending"
                )
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            test = saved["test"]
            best_epoch = int(saved["best_epoch"]) + 1
            elapsed = saved.get("elapsed_seconds")
            hours = None if elapsed is None else float(elapsed) / 3600
            row = {
                "model": tag,
                "seed": seed,
                "status": "done",
                "best_epoch": best_epoch,
                "elapsed_seconds": elapsed,
                "elapsed_hours": hours,
                "f1": test["f1"],
                "auprc": test["auprc"],
                "precision": test["precision"],
                "recall": test["recall"],
                "coverage": test.get("coverage"),
                "threshold": test.get("threshold"),
                "metrics_file": str(path),
            }
            rows.append(row)
            hours_text = "" if hours is None else f"{hours:.2f}"
            lines.append(
                f"{tag:<8} {seed:>4} {best_epoch:>10} {hours_text:>8} {test['f1']:>8.4f} "
                f"{test['auprc']:>8.4f} {test['precision']:>9.4f} {test['recall']:>8.4f} "
                f"{test.get('coverage', 0):>8.4f} done"
            )
    report = {"models": rows, "table": "\n".join(lines)}
    if total_seconds is not None:
        report["total_seconds"] = total_seconds
        report["total_hours"] = total_seconds / 3600
        lines.append(f"total_runtime_hours {total_seconds / 3600:.3f}")
        lines.append(f"total_runtime_seconds {total_seconds:.1f}")
        report["table"] = "\n".join(lines)
    runs = work_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (runs / "comparison.txt").write_text(report["table"] + "\n", encoding="utf-8")
    print("\n" + report["table"] + "\n", flush=True)
    print(f"comparison table: {runs / 'comparison.txt'}", flush=True)
    return report


def _compare(path: str) -> dict:
    runs, seeds = load_runs(path, enabled_only=True)
    if not runs:
        raise ValueError(f"{path} 没有 enabled: true 的模型")
    tags = [run_label(config) for config in runs]
    work_dir = Path(runs[0].data.work_dir)
    started = time.perf_counter()
    report = write_comparison_report(work_dir, tags, seeds, 0.0)
    for config in runs:
        _run_seeds(config, seeds)
        report = write_comparison_report(work_dir, tags, seeds, time.perf_counter() - started)
    total = time.perf_counter() - started
    print(f"total runtime: {total / 3600:.3f} hours ({total:.1f} seconds)", flush=True)
    return report


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit-data":
        print(json.dumps(audit_csv(args.csv, args.output), indent=2))
    elif args.command == "plan":
        print(json.dumps(_plan(args.config), indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        runs, _seeds = load_runs(args.config, enabled_only=True)
        motif_runs = [run for run in runs if run.model.name not in {"gate", "original_gat"}]
        chosen = motif_runs[0] if motif_runs else runs[0]
        print(json.dumps(prepare_artifacts(chosen), indent=2, default=str))
    elif args.command == "run":
        config = load_config(args.config, model=args.model)
        print(json.dumps(run_pipeline(config), indent=2, default=str))
    elif args.command == "run-many":
        config = load_config(args.config, model=args.model)
        _seeds, file_seeds = load_runs(args.config, only=run_label(config), enabled_only=False)
        del _seeds
        seeds = args.seeds or file_seeds
        _run_seeds(config, seeds)
        summary = Path(config.data.work_dir) / "runs" / run_label(config) / "summary.json"
        print(summary.read_text(encoding="utf-8"))
    elif args.command == "compare":
        print(json.dumps(_compare(args.config), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(_summarize(args.metrics), indent=2))


if __name__ == "__main__":
    main()
