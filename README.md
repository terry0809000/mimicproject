# Applying NLP Methods to Extract Social Determinants of Health from MIMIC-III Clinical Notes

A publication-grade, reproducible benchmark pipeline for SBDH extraction using MIMIC-SBDH annotations linked to MIMIC-III notes.

## Benchmark scope
- Controlled benchmarking across model families with **identical patient-level splits**.
- Primary analytical text: **social history section** of discharge summaries.
- Families:
  1. Rule-based baseline with NegEx-style negation handling
  2. Sparse-feature ML baselines (TF-IDF + LR/SVM/RF/XGBoost)
  3. Transformer fine-tuning (BERT/BioBERT/ClinicalBERT)
- Standardized metrics and capability-oriented tests.

## Data assumptions
Input requires two files:
1. Note table with note ID, patient ID, note text, and category columns.
2. Annotation table with note ID and binary task labels:
   - `sdoh_community_present`
   - `sdoh_community_absent`
   - `sdoh_education`
   - `sdoh_economics`
   - `sdoh_environment`
   - `behavior_alcohol`
   - `behavior_tobacco`
   - `behavior_drug`

All column names are configurable in `config/data.yaml`.

## Secure use and governance
- Run only in approved secure environments.
- Raw note text is never sent to external APIs.
- Preserve de-identification tokens.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
1. Copy `.env.example` to `.env` (optional for environment-driven overrides).
2. Update paths in `config/base.yaml` and schema in `config/data.yaml`.

## Reproduce benchmark
Quick smoke benchmark:
```bash
python scripts/run_all_experiments.py --mode quick --skip-transformers
```

Full benchmark:
```bash
python scripts/run_all_experiments.py --mode full
```

Granular commands:
```bash
python scripts/prepare_data.py --config config/base.yaml
python scripts/inspect_labels.py --config config/base.yaml
python scripts/train_rule_based.py --task all
python scripts/train_tfidf_logreg.py --task all
python scripts/train_tfidf_svm.py --task all
python scripts/train_tfidf_rf.py --task all
python scripts/train_tfidf_xgb.py --task all
python scripts/train_transformer.py --model emilyalsentzer/Bio_ClinicalBERT --task all
python scripts/run_capability_tests.py --models all
python scripts/evaluate_all.py
python scripts/generate_report_artifacts.py
```

## Workflow
1. **Prepare data**: schema validation, linkage checks, section extraction, prevalence reports.
2. **Split**: leakage-safe patient-level train/val/test with diagnostics.
3. **Train**: model-family-specific scripts save models and metrics.
4. **Evaluate**: standardized metrics, confusion matrices, per-class summaries.
5. **Capability tests**: negation/attribution/temporality/misspelling challenge set.
6. **Reporting**: CSV/JSON/LaTeX tables and figure artifacts.

## Expected repository structure
```text
config/
scripts/
src/
  data/
  preprocessing/
  features/
  models/
  training/
  evaluation/
  error_analysis/
  capability_tests/
  reporting/
  visualization/
  pipelines/
notebooks/
tests/
data/
outputs/
```

## Common failure points
- Missing columns: check `config/data.yaml` mappings.
- Sparse-label tasks: macro-F1 may be unstable; inspect per-class tables and bootstrap intervals.
- Transformer OOM: reduce `max_length` and batch size in `config/models.yaml`.
- Empty social history extraction: verify section header conventions and patterns.

## Determinism and reproducibility
- Centralized seed in `config/base.yaml`.
- Split files persisted in `data/splits/`.
- Artifacts persisted: metrics, predictions, figures, logs, costs, error analyses, capability outputs.
