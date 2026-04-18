from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    out = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    return out


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f, "support": s})


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, n_iter: int = 200, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {"macro_f1_mean": float(np.mean(vals)), "macro_f1_ci_low": float(np.percentile(vals, 2.5)), "macro_f1_ci_high": float(np.percentile(vals, 97.5))}
