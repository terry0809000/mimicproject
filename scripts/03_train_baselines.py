from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PATHS, OUTPUT_DIR, TRAINING, TASKS
from src.evaluation.metrics import compute_multilabel_metrics
from src.features.tfidf import build_tfidf, transform_tfidf
from src.models.traditional_ml import train_models
from src.utils.logging import setup_logger, write_json
from src.utils.seeding import set_global_seed


def load_split_ids(split_path: Path) -> list[int]:
    with split_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["subject_ids"]


def main() -> None:
    logger = setup_logger("train_baselines")
    set_global_seed(TRAINING.random_seed)

    df = pd.read_csv(PATHS.dataset_path)
    label_cols = TASKS["multi_label"]["labels"]

    train_ids = load_split_ids(OUTPUT_DIR / "split_ids" / "train_ids.json")
    val_ids = load_split_ids(OUTPUT_DIR / "split_ids" / "val_ids.json")
    test_ids = load_split_ids(OUTPUT_DIR / "split_ids" / "test_ids.json")

    train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_ids)].reset_index(drop=True)

    tfidf_bundle = build_tfidf(
        train_df["text"],
        max_features=TRAINING.tfidf_config["max_features"],
        ngram_range=TRAINING.tfidf_config["ngram_range"],
        min_df=TRAINING.tfidf_config["min_df"],
    )
    x_train = tfidf_bundle.features
    x_val = transform_tfidf(tfidf_bundle.vectorizer, val_df["text"])
    x_test = transform_tfidf(tfidf_bundle.vectorizer, test_df["text"])

    y_train = train_df[label_cols].values
    y_val = val_df[label_cols].values
    y_test = test_df[label_cols].values

    results = train_models(x_train, y_train, x_val, y_val, TRAINING.baseline_models)

    metrics_output = {}
    models_dir = OUTPUT_DIR / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        model_path = models_dir / f"{model_name}_tfidf.joblib"
        joblib.dump({"model": result.model, "vectorizer": tfidf_bundle.vectorizer}, model_path)
        logger.info("Saved %s model to %s", model_name, model_path)

        y_pred = result.model.predict(x_test)
        y_prob = None
        if hasattr(result.model, "predict_proba"):
            try:
                probs = result.model.predict_proba(x_test)
                y_prob = np.vstack([p[:, 1] for p in probs]).T
            except Exception:  # pragma: no cover - fallback for unsupported outputs
                y_prob = None

        metrics = compute_multilabel_metrics(y_test, y_pred, y_prob)
        metrics_output[model_name] = {
            "val_macro_f1": result.val_macro_f1,
            "test_metrics": metrics,
            "best_params": result.best_params,
        }

    metrics_path = OUTPUT_DIR / "metrics" / "baseline_metrics.json"
    write_json(metrics_path, metrics_output)
    logger.info("Saved baseline metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
