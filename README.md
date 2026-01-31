# MIMIC SDoH Benchmarking Pipeline

This repository provides a reproducible benchmarking workflow for extracting Social/Behavioral Determinants of Health (SDoH/SBDH) from ICU clinical notes using **MIMIC-III** notes and **MIMIC-SBDH** expert labels. The workflow is designed to run end-to-end in a single notebook for controlled experimental benchmarking.

## Project Goals (Summary)
- Benchmark traditional ML baselines vs. clinical transformers for SDoH extraction.
- Report macro-F1 (primary), micro-F1, AUROC, and per-class metrics.
- Conduct error analysis focused on negation, temporality, and implicit mentions.
- Quantify computational costs and produce publication-ready tables/figures.

## How to Run (Notebook)
### Option A: Google Colab (preferred)
1. Upload this repo to a Colab workspace or open it from GitHub.
2. Open `notebooks/01_end_to_end_benchmark.ipynb`.
3. Run the setup cell to install dependencies.
4. (Optional) Mount Google Drive and update paths in the config cell.
5. Execute all cells top-to-bottom.

### Option B: Local Jupyter (conda/venv)
```bash
# Create environment
conda env create -f environment.yml
conda activate mimic-sdoh

# OR use pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/01_end_to_end_benchmark.ipynb
```

## Data Access (MIMIC-III + MIMIC-SBDH)
1. Obtain PhysioNet credentials and download datasets.
2. Update the notebook configuration cell with your local paths (defaults are Windows paths).
3. Ensure you have `NOTEEVENTS.csv(.gz)` under the MIMIC-III root.

**Important:** Do not hardcode credentials. Use environment variables or a local config file that is excluded from git.

## Outputs Produced
The notebook writes deterministic artifacts under `outputs/`:
- `outputs/dataset_summary.json`
- `outputs/splits/{train,val,test}.csv`
- `outputs/metrics/metrics_table.csv`
- `outputs/figures/*.png`
- `outputs/cost/cost_table.csv`

## Repo Layout
```
notebooks/
  01_end_to_end_benchmark.ipynb
src/
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

## Reproducibility
- Fixed random seeds in the notebook setup section.
- Patient-level splits with leakage checks.
- Version pinning in `requirements.txt` and `environment.yml`.
