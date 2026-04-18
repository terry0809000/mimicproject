#!/usr/bin/env python
from __future__ import annotations
import argparse
from scripts._train_tfidf_common import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    args = ap.parse_args()
    run("logreg", args.task)
