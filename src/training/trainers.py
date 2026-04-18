from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score

from src.models.sklearn_models import build_model, model_search_space


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def train_sklearn_with_val(
    model_name: str,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    model_cfg: dict,
    seed: int,
) -> tuple[Any, dict, dict]:
    best = None
    for params in model_search_space(model_name, model_cfg):
        m = build_model(model_name, seed=seed, base_cfg=model_cfg)
        m.set_params(**params)
        t0 = time.time()
        mem0 = _memory_mb()
        m.fit(X_train, y_train)
        train_s = time.time() - t0
        mem1 = _memory_mb()
        pred = m.predict(X_val)
        score = f1_score(y_val, pred, average="macro")
        if best is None or score > best["score"]:
            best = {"model": m, "params": params, "score": score, "train_seconds": train_s, "peak_mem_mb": max(mem0, mem1)}
    assert best is not None
    return best["model"], best["params"], {"val_macro_f1": best["score"], "train_seconds": best["train_seconds"], "peak_mem_mb": best["peak_mem_mb"]}


def persist_model(model: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def cost_record(model_name: str, task: str, split: str, details: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"model": model_name, "task": task, "split": split, **details}])
