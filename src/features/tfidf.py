from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfBundle:
    vectorizer: TfidfVectorizer
    features: object


def build_tfidf(texts: pd.Series, max_features: int, ngram_range: Tuple[int, int], min_df: int) -> TfidfBundle:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    features = vectorizer.fit_transform(texts)
    return TfidfBundle(vectorizer=vectorizer, features=features)


def transform_tfidf(vectorizer: TfidfVectorizer, texts: pd.Series) -> object:
    return vectorizer.transform(texts)
