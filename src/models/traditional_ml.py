from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class ModelResult:
    model_name: str
    model: object
    val_macro_f1: float
    best_params: Dict[str, object]


def _class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def train_logreg(x_train, y_train, x_val, y_val) -> ModelResult:
    class_weights = _class_weights(y_train.ravel())
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight=class_weights,
            solver="liblinear",
        )
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    val_macro = f1_score(y_val, preds, average="macro", zero_division=0)
    return ModelResult("logreg", clf, val_macro, {"class_weight": class_weights})


def train_rf(x_train, y_train, x_val, y_val) -> ModelResult:
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    val_macro = f1_score(y_val, preds, average="macro", zero_division=0)
    return ModelResult("rf", clf, val_macro, {"n_estimators": 200})


def train_xgb(x_train, y_train, x_val, y_val) -> ModelResult:
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed.")
    clf = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    val_macro = f1_score(y_val, preds, average="macro", zero_division=0)
    return ModelResult("xgb", clf, val_macro, {"n_estimators": 300})


def train_models(
    x_train,
    y_train,
    x_val,
    y_val,
    models: List[str],
) -> Dict[str, ModelResult]:
    results: Dict[str, ModelResult] = {}
    for model_name in models:
        if model_name == "logreg":
            results[model_name] = train_logreg(x_train, y_train, x_val, y_val)
        elif model_name == "rf":
            results[model_name] = train_rf(x_train, y_train, x_val, y_val)
        elif model_name == "xgb":
            results[model_name] = train_xgb(x_train, y_train, x_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return results
