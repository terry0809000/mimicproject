from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_NOTEEVENTS_COLUMNS = [
    "ROW_ID",
    "SUBJECT_ID",
    "HADM_ID",
    "CATEGORY",
    "TEXT",
]


@dataclass(frozen=True)
class RealDatasetConfig:
    mimic_root: Path
    sbdh_path: Path
    category_filter: Optional[str] = "Discharge summary"
    chunksize: int = 50_000
    usecols: Optional[list[str]] = None


def find_noteevents_file(mimic_root: Path) -> Path:
    candidates = [
        mimic_root / "NOTEEVENTS.csv.gz",
        mimic_root / "NOTEEVENTS.csv",
        mimic_root / "mimic-iii-clinical-database-1.4" / "NOTEEVENTS.csv.gz",
        mimic_root / "mimic-iii-clinical-database-1.4" / "NOTEEVENTS.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(mimic_root.rglob("NOTEEVENTS.csv.gz"))
    if not matches:
        matches = list(mimic_root.rglob("NOTEEVENTS.csv"))
    if not matches:
        raise FileNotFoundError(
            "Could not find NOTEEVENTS.csv(.gz) under "
            f"{mimic_root}. Verify MIMIC-III extraction."
        )
    return matches[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.strip().lower() for col in df.columns})


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {context}: {missing}. "
            f"Detected columns: {sorted(df.columns)}"
        )


def _load_sbdh_labels(sbdh_path: Path) -> pd.DataFrame:
    if not sbdh_path.exists():
        raise FileNotFoundError(f"SBDH labels file not found: {sbdh_path}")
    labels = pd.read_csv(sbdh_path)
    labels = _normalize_columns(labels)
    return labels


def _resolve_merge_keys(labels: pd.DataFrame) -> tuple[list[str], list[str]]:
    label_columns = [
        col
        for col in labels.columns
        if col not in {"subject_id", "hadm_id", "note_id", "row_id"}
    ]
    if "subject_id" in labels.columns and "hadm_id" in labels.columns:
        return ["subject_id", "hadm_id"], label_columns
    if "subject_id" in labels.columns and "note_id" in labels.columns:
        return ["subject_id", "note_id"], label_columns
    if "subject_id" in labels.columns and "row_id" in labels.columns:
        labels = labels.rename(columns={"row_id": "note_id"})
        return ["subject_id", "note_id"], label_columns
    raise ValueError(
        "Could not resolve merge keys for SBDH labels. "
        "Expected SUBJECT_ID + HADM_ID (preferred) or SUBJECT_ID + NOTE_ID. "
        f"Detected columns: {sorted(labels.columns)}"
    )


def _chunk_noteevents(
    note_path: Path,
    category_filter: Optional[str],
    usecols: list[str],
    chunksize: int,
) -> pd.DataFrame:
    LOGGER.info("Reading NOTEEVENTS from %s", note_path)
    chunks = []
    for chunk in pd.read_csv(
        note_path,
        usecols=usecols,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
    ):
        if category_filter is not None and "CATEGORY" in chunk.columns:
            chunk = chunk.loc[chunk["CATEGORY"] == category_filter]
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def load_mimic_sbdh_dataset(config: RealDatasetConfig) -> pd.DataFrame:
    note_path = find_noteevents_file(config.mimic_root)
    usecols = config.usecols or DEFAULT_NOTEEVENTS_COLUMNS
    notes = _chunk_noteevents(
        note_path,
        category_filter=config.category_filter,
        usecols=usecols,
        chunksize=config.chunksize,
    )
    if notes.empty:
        raise ValueError(
            "No notes returned after filtering. "
            "Check CATEGORY filter or input file."
        )
    notes = notes.rename(
        columns={
            "ROW_ID": "note_id",
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "TEXT": "text",
        }
    )
    notes = _normalize_columns(notes)
    _ensure_columns(notes, ["subject_id", "text"], "NOTEEVENTS")

    labels = _load_sbdh_labels(config.sbdh_path)
    merge_keys, label_columns = _resolve_merge_keys(labels)
    labels = labels.rename(columns={col: col.lower() for col in labels.columns})
    merged = notes.merge(labels, on=merge_keys, how="inner")
    if merged.empty:
        raise ValueError(
            "Merge resulted in zero rows. Verify merge keys and label file "
            f"columns: {sorted(labels.columns)}"
        )
    keep_cols = ["subject_id", "hadm_id", "note_id", "text"]
    keep_cols = [col for col in keep_cols if col in merged.columns]
    return merged[keep_cols + label_columns]
