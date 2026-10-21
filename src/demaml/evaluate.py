from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def select_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or int(np.sum(y_true)) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    prediction = score >= threshold
    truth = y_true.astype(bool)
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    if int(truth.sum()) == 0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_true, score))
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "positives": int(truth.sum()),
        "examples": len(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
