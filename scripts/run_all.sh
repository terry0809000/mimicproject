#!/usr/bin/env bash
set -euo pipefail

python scripts/01_build_dataset.py
python scripts/02_make_splits.py
python scripts/03_train_baselines.py
