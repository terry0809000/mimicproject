from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["macro_auroc"] = roc_auc_score(y_true, y_prob, average="macro")
        except ValueError:
            metrics["macro_auroc"] = float("nan")
    return metrics
