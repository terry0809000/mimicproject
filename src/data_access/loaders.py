from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Dict, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd

from src.config import SDOH_CATEGORIES


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    label_columns: list[str]


def load_dummy_dataset(n_patients: int = 200, notes_per_patient: int = 3) -> DatasetBundle:
    rng = np.random.default_rng(42)
    total_notes = n_patients * notes_per_patient
    subject_ids = np.repeat(np.arange(1, n_patients + 1), notes_per_patient)
    note_ids = np.arange(1, total_notes + 1)
    texts = [
        f"Patient {pid} note {nid}: reports housing instability and limited support."
        for pid, nid in zip(subject_ids, note_ids)
    ]
    labels = rng.binomial(1, 0.2, size=(total_notes, len(SDOH_CATEGORIES)))
    df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "note_id": note_ids,
            "text": texts,
        }
    )
    for idx, label in enumerate(SDOH_CATEGORIES):
        df[label] = labels[:, idx]
    return DatasetBundle(data=df, label_columns=SDOH_CATEGORIES)


def _read_noteevents(notes_path: Path) -> pd.DataFrame:
    if notes_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(notes_path) as zf:
            candidates = [
                name
                for name in zf.namelist()
                if name.lower().endswith("noteevents.csv")
                or name.lower().endswith("noteevents.csv.gz")
            ]
            if not candidates:
                raise FileNotFoundError(
                    "NOTEEVENTS.csv or NOTEEVENTS.csv.gz not found inside zip."
                )
            noteevents_name = candidates[0]
            with zf.open(noteevents_name) as note_file:
                if noteevents_name.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=note_file) as gz_file:
                        return pd.read_csv(gz_file)
                return pd.read_csv(note_file)
    if notes_path.suffix.lower() == ".gz":
        return pd.read_csv(notes_path, compression="gzip")
    return pd.read_csv(notes_path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.strip().lower() for col in df.columns})


def _extract_subject_note_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename_map = {}
    if "subject_id" not in df.columns:
        for candidate in ("subjectid", "subject_id"):
            if candidate in df.columns:
                rename_map[candidate] = "subject_id"
                break
    if "note_id" not in df.columns:
        for candidate in ("note_id", "row_id", "rowid"):
            if candidate in df.columns:
                rename_map[candidate] = "note_id"
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    if "subject_id" not in df.columns or "note_id" not in df.columns:
        raise ValueError("Both subject_id and note_id columns are required.")
    return df


def _build_label_matrix(
    annotations_df: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    df = _extract_subject_note_ids(annotations_df)
    df = _normalize_columns(df)

    direct_label_map = {
        col: col
        for col in label_columns
        if col in df.columns
    }
    if direct_label_map:
        label_df = df[["subject_id", "note_id", *direct_label_map.keys()]].copy()
        label_df = label_df.rename(columns=direct_label_map)
        return label_df

    category_candidates = [
        "sdoh_category",
        "category",
        "label",
        "label_name",
        "sdoh_label",
    ]
    value_candidates = ["value", "present", "label_value", "annotation", "flag"]
    category_col = next(
        (col for col in category_candidates if col in df.columns),
        None,
    )
    value_col = next(
        (col for col in value_candidates if col in df.columns),
        None,
    )
    if category_col is None or value_col is None:
        raise ValueError(
            "Annotations must include either explicit SDOH columns or "
            "category/value columns to build labels."
        )

    def normalize_category(category: str) -> Optional[str]:
        key = (
            str(category)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        mapping = {
            "housing_instability": "housing_insecurity",
            "housing_insecurity": "housing_insecurity",
            "food_insecurity": "food_insecurity",
            "transportation": "transportation",
            "transportation_insecurity": "transportation",
            "financial_strain": "financial_strain",
            "financial": "financial_strain",
            "social_support": "social_support",
            "social_supports": "social_support",
            "support": "social_support",
        }
        return mapping.get(key)

    df["normalized_category"] = df[category_col].map(normalize_category)
    df = df[df["normalized_category"].notna()].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0).astype(int)
    pivot = (
        df.pivot_table(
            index=["subject_id", "note_id"],
            columns="normalized_category",
            values=value_col,
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    for label in label_columns:
        if label not in pivot.columns:
            pivot[label] = 0
    return pivot[["subject_id", "note_id", *label_columns]]


def load_real_dataset(mimic_notes_path: Path, mimic_sbdh_path: Path) -> DatasetBundle:
    """
    Load NOTEEVENTS from MIMIC-III and SBDH annotations, then join on subject_id/note_id.
    """
    notes_df = _read_noteevents(mimic_notes_path)
    notes_df = _normalize_columns(notes_df)
    if "text" not in notes_df.columns:
        raise ValueError("NOTEEVENTS must include a TEXT column.")
    notes_df = _extract_subject_note_ids(notes_df)
    notes_df = notes_df.rename(columns={"text": "text"})
    notes_df = notes_df[["subject_id", "note_id", "text"]]

    annotations_df = pd.read_csv(mimic_sbdh_path)
    label_df = _build_label_matrix(annotations_df, SDOH_CATEGORIES)

    merged = notes_df.merge(label_df, on=["subject_id", "note_id"], how="inner")
    return DatasetBundle(data=merged, label_columns=SDOH_CATEGORIES)


def load_dataset(
    data_source: str,
    mimic_notes_path: Path,
    mimic_sbdh_path: Path,
) -> DatasetBundle:
    if data_source == "dummy":
        return load_dummy_dataset()
    if data_source == "real":
        return load_real_dataset(mimic_notes_path, mimic_sbdh_path)
    raise ValueError(f"Unknown data_source: {data_source}")
