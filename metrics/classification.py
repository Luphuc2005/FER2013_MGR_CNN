from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Union

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], labels: Iterable[str]) -> Dict[str, object]:
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)
    label_list = list(labels)
    ids = list(range(len(label_list)))
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class = np.nan_to_num(per_class, nan=0.0).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=ids, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=ids, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": [float(x) for x in per_class],
        "per_class_recall": [float(x) for x in per_class],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=ids,
            target_names=label_list,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics(metrics: Dict[str, object], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
