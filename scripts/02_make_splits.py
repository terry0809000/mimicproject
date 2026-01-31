from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PATHS, SPLIT_DIR, TRAINING
from src.splits.patient_splits import assert_no_leakage, make_patient_splits
from src.utils.logging import setup_logger, write_json
from src.utils.seeding import set_global_seed


def main() -> None:
    logger = setup_logger("make_splits")
    set_global_seed(TRAINING.random_seed)

    df = pd.read_csv(PATHS.dataset_path)
    split_result = make_patient_splits(
        df,
        patient_col="subject_id",
        split_ratios=TRAINING.split_ratios,
        seed=TRAINING.random_seed,
    )
    assert_no_leakage(split_result.split_ids)

    for split_name, ids in split_result.split_ids.items():
        path = SPLIT_DIR / f"{split_name}_ids.json"
        write_json(path, {"subject_ids": ids})
        logger.info("Saved %s split IDs to %s", split_name, path)


if __name__ == "__main__":
    main()
