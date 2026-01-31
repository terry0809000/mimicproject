from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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


def load_real_dataset(mimic_notes_path: Path, mimic_sbdh_path: Path) -> DatasetBundle:
    """
    TODO: Implement real data loading using MIMIC-III notes and MIMIC-SBDH labels.

    Expected workflow:
    1) Load NOTEEVENTS from MIMIC-III.
    2) Load expert labels from MIMIC-SBDH.
    3) Join on subject_id / note_id.
    4) Build multi-label columns for SDOH_CATEGORIES.
    """
    raise NotImplementedError(
        "Real data loader not implemented. Set DATA_SOURCE='dummy' or implement loader."
    )


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
