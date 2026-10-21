from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data import audit_csv
from .pipeline import prepare_artifacts, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent D-EMAML reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data")
    audit.add_argument("--csv", required=True)
    audit.add_argument("--output")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    many = commands.add_parser("run-many")
    many.add_argument("--config", required=True)
    many.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
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


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit-data":
        print(json.dumps(audit_csv(args.csv, args.output), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare_artifacts(load_config(args.config)), indent=2))
    elif args.command == "run":
        print(json.dumps(run_pipeline(load_config(args.config)), indent=2))
    elif args.command == "run-many":
        config = load_config(args.config)
        results = []
        for seed in args.seeds:
            config.train.seed = seed
            results.append(run_pipeline(config))
        summary = _summarize_results(results)
        work = Path(config.data.work_dir) / "runs" / config.model.name
        work.mkdir(parents=True, exist_ok=True)
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(_summarize(args.metrics), indent=2))


if __name__ == "__main__":
    main()
