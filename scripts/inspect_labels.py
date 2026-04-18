#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from src.data.loaders import label_prevalence_table
from src.reporting.export_tables import export_table
from src.utils.config import load_all_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    args = ap.parse_args()
    cfg = load_all_configs("config")

    df = pd.read_csv(cfg["paths"]["processed_path"])
    prev = label_prevalence_table(df, cfg["data"]["annotation_task_columns"])
    export_table(prev, Path(cfg["paths"]["output_dir"]) / "tables" / "label_prevalence")
    print(prev)


if __name__ == "__main__":
    main()
