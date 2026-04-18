#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.benchmark_pipeline import train_rule_based_task
from src.reporting.export_tables import export_table
from src.utils.config import load_all_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    args = ap.parse_args()
    cfg = load_all_configs("config")
    df = pd.read_csv(cfg["paths"]["processed_path"])
    tasks = cfg["data"]["annotation_task_columns"] if args.task == "all" else [args.task]

    rows = []
    for task in tasks:
        m = train_rule_based_task(cfg, df, task)
        rows.append({"model": "rule_based", "task": task, **m})
    out = pd.DataFrame(rows)
    export_table(out, Path(cfg["paths"]["output_dir"]) / "metrics" / "rule_based_metrics")


if __name__ == "__main__":
    main()
