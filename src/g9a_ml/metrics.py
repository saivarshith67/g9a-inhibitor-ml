"""Classification metrics helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Compute binary classification metrics.

    ``roc_auc`` / ``pr_auc`` use ``y_score`` (probabilities or decision scores)
    when provided; otherwise they fall back to hard predictions.
    """
    scores = y_pred if y_score is None else y_score
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def save_metrics_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return path


def metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def save_metrics(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def confusion_and_report(y_true, y_pred) -> tuple[list[list[int]], str]:
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["other (0)", "active-inhibitor (1)"],
    )
    return cm, report
