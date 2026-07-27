"""Evaluation helpers for trait classification outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def to_json_safe(value: Any) -> Any:
    """Convert numpy and sklearn values into JSON-serializable objects."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def evaluate_classification(y_true, y_pred) -> dict[str, Any]:
    """Compute standard classification metrics."""

    labels = sorted(set(y_true) | set(y_pred))
    return to_json_safe(
        {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": {
                "labels": labels,
                "matrix": confusion_matrix(y_true, y_pred, labels=labels),
            },
        }
    )


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Write metrics to JSON with stable formatting."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(metrics), indent=2) + "\n", encoding="utf-8")
