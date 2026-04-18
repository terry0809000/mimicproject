from __future__ import annotations

import pandas as pd


ERROR_PATTERNS = {
    "negation": ["no ", "denies", "without"],
    "attribution": ["family reports", "per wife", "per daughter"],
    "temporality": ["history of", "previously", "former"],
    "misspelling": ["smkoer", "alchol", "drnks"],
}


def categorize_error(text: str) -> str:
    t = (text or "").lower()
    for cat, pats in ERROR_PATTERNS.items():
        if any(p in t for p in pats):
            return cat
    return "other"


def extract_fp_fn(df: pd.DataFrame, y_true_col: str, y_pred_col: str, text_col: str, n_examples: int = 50) -> pd.DataFrame:
    mask = df[y_true_col] != df[y_pred_col]
    errs = df.loc[mask, [text_col, y_true_col, y_pred_col]].copy()
    errs["error_type"] = errs[text_col].map(categorize_error)
    return errs.head(n_examples)
