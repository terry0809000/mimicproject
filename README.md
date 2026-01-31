# MIMIC SDoH Benchmarking Pipeline

This repository provides a fully reproducible benchmarking pipeline for extracting Social/Behavioral Determinants of Health (SDoH/SBDH) from ICU clinical notes using **MIMIC-III** notes and **MIMIC-SBDH** expert labels. The current scaffold runs end-to-end with **dummy data adapters** so you can validate the pipeline without protected datasets. Once you have PhysioNet access, switch the data loader flag to use the real adapters.

## Project Goals (Summary)
- Define SDoH extraction tasks aligned to established taxonomies.
- Compare traditional ML, LSTM, and transformer baselines under identical splits and metrics.
- Perform error analysis (negation, temporality, indirect mentions).
- Quantify computational costs and produce publication-ready tables/figures.

## Quickstart (Dummy Data)
```bash
# Create environment
conda env create -f environment.yml
conda activate mimic-sdoh

# Run the pipeline (dummy adapters)
bash scripts/run_all.sh
```

## Data Access (MIMIC-III + MIMIC-SBDH)
1. Obtain PhysioNet credentials and download datasets.
2. Place files under a local directory (e.g., `data/`) and set paths in `src/config.py`.
3. Switch `DATA_SOURCE` to `"real"`.

**Important:** Do not hardcode credentials. Use environment variables or local config overrides.

## Repo Layout
```
src/
  config.py
  data_access/
  preprocessing/
  splits/
  features/
  models/
  evaluation/
  utils/
scripts/
outputs/
```

## Running Specific Stages
```bash
python scripts/01_build_dataset.py
python scripts/02_make_splits.py
python scripts/03_train_baselines.py
```

## Expected Outputs
- `outputs/dataset_summary.json`
- `outputs/split_ids/train_ids.json` (and val/test)
- `outputs/metrics/*.json`

## Reproducibility
- Fixed random seeds in `src/config.py`.
- Deterministic settings for numpy/sklearn.
- Patient-level splits with leakage checks.

## Notes
- The real MIMIC loaders are stubbed with `TODO` markers.
- Sentence-level tasks and transformer baselines will be added in later steps.
