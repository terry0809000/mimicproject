from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.98),
        max_features=cfg.get("max_features", 50000),
        sublinear_tf=True,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, p)
