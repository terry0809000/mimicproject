#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess


CMDS = [
    ["python", "scripts/prepare_data.py"],
    ["python", "scripts/inspect_labels.py"],
    ["python", "scripts/train_rule_based.py", "--task", "all"],
    ["python", "scripts/train_tfidf_logreg.py", "--task", "all"],
    ["python", "scripts/train_tfidf_svm.py", "--task", "all"],
    ["python", "scripts/train_tfidf_rf.py", "--task", "all"],
    ["python", "scripts/train_tfidf_xgb.py", "--task", "all"],
    ["python", "scripts/run_capability_tests.py", "--models", "all"],
    ["python", "scripts/evaluate_all.py"],
    ["python", "scripts/generate_report_artifacts.py"],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--skip-transformers", action="store_true")
    args = ap.parse_args()

    cmds = list(CMDS)
    if not args.skip_transformers and args.mode == "full":
        cmds.insert(7, ["python", "scripts/train_transformer.py", "--model", "emilyalsentzer/Bio_ClinicalBERT", "--task", "all"])

    for c in cmds:
        print("Running:", " ".join(c))
        subprocess.run(c, check=True)


if __name__ == "__main__":
    main()
