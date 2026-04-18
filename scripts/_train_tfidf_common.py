from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import pandas as pd

from src.pipelines.benchmark_pipeline import train_tfidf_model_task
from src.reporting.export_tables import export_table
from src.utils.config import load_all_configs


def run(model_name: str, task: str) -> None:
    cfg = load_all_configs("config")
    df = pd.read_csv(cfg["paths"]["processed_path"])
    tasks = cfg["data"]["annotation_task_columns"] if task == "all" else [task]
    rows = []
    per_class_all = []
    for t in tasks:
        m, per_class = train_tfidf_model_task(cfg, df, t, model_name)
        rows.append({"model": model_name, "task": t, **m})
        per_class["task"] = t
        per_class_all.append(per_class)

    root = Path(cfg["paths"]["output_dir"])
    export_table(pd.DataFrame(rows), root / "metrics" / f"{model_name}_metrics")
    export_table(pd.concat(per_class_all, ignore_index=True), root / "metrics" / f"{model_name}_per_class")
