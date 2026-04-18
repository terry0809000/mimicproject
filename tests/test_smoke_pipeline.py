from pathlib import Path

import pandas as pd

from src.pipelines.benchmark_pipeline import make_splits, prepare_dataset
from src.utils.config import load_all_configs


def test_smoke_prepare_and_split(tmp_path: Path):
    cfg = load_all_configs("config")
    cfg["paths"]["notes_path"] = "tests/fixtures/synthetic_notes.csv"
    cfg["paths"]["annotations_path"] = "tests/fixtures/synthetic_annotations.csv"
    cfg["paths"]["output_dir"] = str(tmp_path / "outputs")
    cfg["paths"]["processed_path"] = str(tmp_path / "data.csv")
    cfg["paths"]["split_dir"] = str(tmp_path / "splits")
    cfg["models"]["tfidf"]["min_df"] = 1
    df = prepare_dataset(cfg)
    out = make_splits(cfg, df)
    assert "split" in out.columns
    assert Path(cfg["paths"]["processed_path"]).exists()
