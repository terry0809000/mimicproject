#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import time

import pandas as pd

from src.evaluation.metrics import classification_report_dict
from src.models.transformer_models import train_transformer_binary
from src.reporting.export_tables import export_table
from src.utils.config import load_all_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--task", default="all")
    args = ap.parse_args()
    cfg = load_all_configs("config")
    df = pd.read_csv(cfg["paths"]["processed_path"])
    tasks = cfg["data"]["annotation_task_columns"] if args.task == "all" else [args.task]

    rows = []
    for task in tasks:
        tr = df[df["split"] == "train"]
        va = df[df["split"] == "val"]
        te = df[df["split"] == "test"]
        t0 = time.time()
        art = train_transformer_binary(
            args.model,
            tr["analysis_text"].tolist(),
            tr[task].astype(int).tolist(),
            va["analysis_text"].tolist(),
            va[task].astype(int).tolist(),
            output_dir=str(Path(cfg["paths"]["output_dir"]) / "models" / "transformers" / task),
            seed=cfg["project"]["seed"],
            max_length=cfg["models"]["transformer"]["max_length"],
            lr=cfg["models"]["transformer"]["learning_rate"],
            epochs=cfg["models"]["transformer"]["epochs"],
            train_batch_size=cfg["models"]["transformer"]["train_batch_size"],
            eval_batch_size=cfg["models"]["transformer"]["eval_batch_size"],
        )
        pred = art.trainer.predict(art.trainer.eval_dataset)
        y_pred = pred.predictions.argmax(axis=1)
        y_true = pred.label_ids
        m = classification_report_dict(y_true, y_pred)
        m["train_seconds"] = time.time() - t0
        rows.append({"model": args.model, "task": task, **m})
    export_table(pd.DataFrame(rows), Path(cfg["paths"]["output_dir"]) / "metrics" / "transformer_metrics")


if __name__ == "__main__":
    main()
