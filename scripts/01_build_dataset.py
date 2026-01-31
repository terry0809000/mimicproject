from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PATHS, DATA_SOURCE, OUTPUT_DIR, SDOH_CATEGORIES
from src.data_access.loaders import load_dataset
from src.preprocessing.text import clean_text
from src.utils.logging import setup_logger, write_json
from src.utils.seeding import set_global_seed


def summarize_dataset(df: pd.DataFrame, label_cols: list[str]) -> dict:
    summary = {
        "num_rows": len(df),
        "num_patients": df["subject_id"].nunique(),
        "num_notes": df["note_id"].nunique(),
        "label_prevalence": {},
        "note_length": {
            "mean": float(df["text"].str.len().mean()),
            "median": float(df["text"].str.len().median()),
        },
    }
    for label in label_cols:
        summary["label_prevalence"][label] = float(df[label].mean())
    return summary


def main() -> None:
    logger = setup_logger("build_dataset")
    set_global_seed(42)

    bundle = load_dataset(DATA_SOURCE, PATHS.mimic_notes_path, PATHS.mimic_sbdh_path)
    df = bundle.data.copy()

    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["subject_id", "note_id", "text"]).reset_index(drop=True)

    df.to_csv(PATHS.dataset_path, index=False)
    summary = summarize_dataset(df, bundle.label_columns)
    write_json(PATHS.dataset_summary_path, summary)

    logger.info("Saved dataset to %s", PATHS.dataset_path)
    logger.info("Saved summary to %s", PATHS.dataset_summary_path)


if __name__ == "__main__":
    main()
