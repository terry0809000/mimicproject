#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
from src.pipelines.benchmark_pipeline import make_splits
from src.utils.config import load_all_configs

if __name__ == "__main__":
    cfg = load_all_configs("config")
    df = pd.read_csv(cfg["paths"]["processed_path"])
    out = make_splits(cfg, df)
    out.to_csv(cfg["paths"]["processed_path"], index=False)
