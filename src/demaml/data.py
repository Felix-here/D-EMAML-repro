from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .io import torch_load

EXPECTED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


@dataclass
class ProcessedData:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    time: torch.Tensor
    y: torch.Tensor
    split: torch.Tensor
    num_nodes: int
    feature_names: list[str]
    original_row: torch.Tensor

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProcessedData":
        return cls(**torch_load(path, map_location="cpu"))


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path: str | Path, output: str | Path | None = None) -> dict:
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    if list(header.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {list(header.columns)}")

    rows = positives = 0
    min_time = max_time = None
    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp", "Is Laundering"],
        chunksize=500_000,
        dtype={"Timestamp": "string", "Is Laundering": "int8"},
    ):
        parsed = pd.to_datetime(chunk["Timestamp"], errors="raise")
        rows += len(chunk)
        positives += int(chunk["Is Laundering"].sum())
        cmin, cmax = parsed.min(), parsed.max()
        min_time = cmin if min_time is None else min(min_time, cmin)
        max_time = cmax if max_time is None else max(max_time, cmax)

    report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": rows,
        "positives": positives,
        "positive_rate": positives / rows,
        "min_timestamp": min_time.isoformat(),
        "max_timestamp": max_time.isoformat(),
        "columns": EXPECTED_COLUMNS,
        "matches_known_hi_small": {
            "bytes": path.stat().st_size == 475_664_283,
            "rows": rows == 5_078_345,
            "positives": positives == 5_177,
        },
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _split_ids(n: int) -> np.ndarray:
    train_end = int(n * 0.6)
    valid_end = train_end + int(n * 0.1)
    split = np.full(n, 2, dtype=np.int8)
    split[:train_end] = 0
    split[train_end:valid_end] = 1
    return split


def _smoke_selection(split: np.ndarray, y: np.ndarray, negatives: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = y == 1
    for sid, fraction in ((0, 0.6), (1, 0.1), (2, 0.3)):
        candidates = np.flatnonzero((split == sid) & (y == 0))
        count = min(len(candidates), max(1, int(negatives * fraction)))
        keep[rng.choice(candidates, count, replace=False)] = True
    return np.flatnonzero(keep)


def preprocess_csv(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    smoke: bool = False,
    smoke_negatives: int = 50_000,
    seed: int = 1,
) -> ProcessedData:
    frame = pd.read_csv(
        csv_path,
        dtype={
            "From Bank": "int64",
            "Account": "string",
            "To Bank": "int64",
            "Account.1": "string",
            "Receiving Currency": "category",
            "Payment Currency": "category",
            "Payment Format": "category",
            "Is Laundering": "int8",
        },
    )
    frame["_row"] = np.arange(len(frame), dtype=np.int64)
    frame["_time"] = pd.to_datetime(frame["Timestamp"], errors="raise").astype("int64") // 10**9
    frame.sort_values(["_time", "_row"], kind="stable", inplace=True, ignore_index=True)
    split = _split_ids(len(frame))

    if smoke:
        selected = _smoke_selection(
            split, frame["Is Laundering"].to_numpy(), smoke_negatives, seed
        )
        frame = frame.iloc[selected].copy()
        split = split[selected]

    sender = frame["From Bank"].astype("string") + ":" + frame["Account"]
    receiver = frame["To Bank"].astype("string") + ":" + frame["Account.1"]
    all_accounts = pd.concat([sender, receiver], ignore_index=True)
    codes, uniques = pd.factorize(all_accounts, sort=True)
    n = len(frame)
    edge_index = np.stack([codes[:n], codes[n:]], axis=0)

    train = split == 0
    numeric_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for column in ("Amount Received", "Amount Paid"):
        values = np.log1p(frame[column].to_numpy(dtype=np.float64))
        mean = values[train].mean()
        std = values[train].std()
        numeric_parts.append(((values - mean) / max(std, 1e-8)).astype(np.float32)[:, None])
        feature_names.append(f"log_z_{column}")
    numeric_parts.append(
        (frame["From Bank"].to_numpy() == frame["To Bank"].to_numpy())
        .astype(np.float32)[:, None]
    )
    feature_names.append("same_bank")

    categorical_parts: list[np.ndarray] = []
    for column in ("Receiving Currency", "Payment Currency", "Payment Format"):
        train_categories = sorted(frame.loc[train, column].astype("string").dropna().unique())
        values = frame[column].astype("string")
        for category in train_categories:
            categorical_parts.append((values == category).to_numpy(np.float32)[:, None])
            feature_names.append(f"{column}={category}")

    features = np.concatenate(numeric_parts + categorical_parts, axis=1)
    data = ProcessedData(
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(features),
        time=torch.from_numpy(frame["_time"].to_numpy(np.int64)),
        y=torch.from_numpy(frame["Is Laundering"].to_numpy(np.int64)),
        split=torch.from_numpy(split),
        num_nodes=len(uniques),
        feature_names=[str(name) for name in feature_names],
        original_row=torch.from_numpy(frame["_row"].to_numpy(np.int64)),
    )
    data.save(output_path)
    return data
