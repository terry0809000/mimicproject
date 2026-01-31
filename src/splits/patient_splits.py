from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.seeding import get_random_state


@dataclass
class SplitResult:
    split_ids: Dict[str, list[int]]


def make_patient_splits(
    df: pd.DataFrame,
    patient_col: str,
    split_ratios: Dict[str, float],
    seed: int,
) -> SplitResult:
    if not np.isclose(sum(split_ratios.values()), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    patients = df[patient_col].dropna().unique()
    rng = get_random_state(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])

    train_ids = patients[:n_train].tolist()
    val_ids = patients[n_train : n_train + n_val].tolist()
    test_ids = patients[n_train + n_val :].tolist()

    return SplitResult(split_ids={"train": train_ids, "val": val_ids, "test": test_ids})


def assert_no_leakage(split_ids: Dict[str, list[int]]) -> None:
    train = set(split_ids["train"])
    val = set(split_ids["val"])
    test = set(split_ids["test"])
    if train & val or train & test or val & test:
        raise AssertionError("Patient leakage detected across splits.")
