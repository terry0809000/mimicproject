from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42
DATA_SOURCE = "real"  # "dummy" or "real"

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

SPLIT_DIR = OUTPUT_DIR / "split_ids"

SDOH_CATEGORIES = [
    "housing_insecurity",
    "food_insecurity",
    "transportation", 
    "financial_strain",
    "social_support",
]

TASKS = {
    "multi_label": {
        "description": "Multi-label classification over SDoH categories.",
        "labels": SDOH_CATEGORIES,
    },
}

TFIDF_CONFIG = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

BASELINE_MODELS = ["logreg", "rf", "xgb"]

@dataclass
class PathsConfig:
    mimic_notes_path: Path = Path(
        "D:\\Social Determinants Research\\MIMIC DATASETS\\mimic-iii-clinical-database-1.4.zip"
    )
    mimic_sbdh_path: Path = Path(
        "D:\\Social Determinants Research\\MIMIC DATASETS\\MIMIC-SBDH.csv"
    )
    dataset_path: Path = OUTPUT_DIR / "dataset.csv"
    dataset_summary_path: Path = OUTPUT_DIR / "dataset_summary.json"


@dataclass
class TrainingConfig:
    random_seed: int = RANDOM_SEED
    split_ratios: Dict[str, float] = field(default_factory=lambda: SPLIT_RATIOS.copy())
    tfidf_config: Dict[str, object] = field(default_factory=lambda: TFIDF_CONFIG.copy())
    baseline_models: List[str] = field(default_factory=lambda: BASELINE_MODELS.copy())


PATHS = PathsConfig()
TRAINING = TrainingConfig()
