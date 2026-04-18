#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import pandas as pd

from src.reporting.export_tables import export_table
from src.visualization.plots import plot_perf_vs_cost
from src.utils.config import load_all_configs


def main() -> None:
    cfg = load_all_configs("config")
    out = Path(cfg["paths"]["output_dir"])
    metric_files = list((out / "metrics").glob("*_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError("No metrics found. Run training scripts first.")
    frames = [pd.read_csv(p) for p in metric_files]
    summary = pd.concat(frames, ignore_index=True)
    export_table(summary, out / "tables" / "benchmark_summary")
    if "train_seconds" in summary.columns and "macro_f1" in summary.columns:
        plot_perf_vs_cost(summary.fillna(0), "train_seconds", "macro_f1", "model", out / "figures" / "cost_vs_performance.png")


if __name__ == "__main__":
    main()
