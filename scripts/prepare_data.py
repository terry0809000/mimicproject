#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.pipelines.benchmark_pipeline import prepare_dataset, make_splits
from src.utils.config import load_all_configs, load_yaml, merge_dicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    args = ap.parse_args()

    cfg = merge_dicts(load_all_configs("config"), load_yaml(args.config)) if args.config != "config/base.yaml" else load_all_configs("config")
    df = prepare_dataset(cfg)
    make_splits(cfg, df).to_csv(cfg["paths"]["processed_path"], index=False)


if __name__ == "__main__":
    main()
