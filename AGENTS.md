# AGENTS.md

Repository-wide operating rules for coding agents.

## Scope
This file applies to the entire repository.

## Mission
Maintain a FAIR, leakage-safe, publication-grade benchmarking pipeline for SBDH extraction from MIMIC clinical notes.

## Code style
- Python 3.10+ with type hints on public functions.
- Keep business logic in `src/`; scripts in `scripts/` should stay thin.
- Prefer pure functions with explicit inputs/outputs.
- Use dataclasses/pydantic for structured configs.
- No hidden constants; all tunables belong in `config/*.yaml`.
- Never put `try/except` around imports.

## Testing rules
- Every substantial change must update or add tests.
- Keep tests deterministic with fixed seeds.
- Run after meaningful changes:
  1. `pytest -q`
  2. `python scripts/run_all_experiments.py --mode quick --skip-transformers`
- Add integration tests for end-to-end smoke behavior on fixtures.

## Reproducibility rules
- Seed all RNGs (python, numpy, torch, sklearn-compatible random_state).
- Persist split files and ensure zero patient overlap diagnostics.
- Save model configs, vectorizers/tokenizers, metrics JSON/CSV, and runtime cost logs.
- Do not overwrite artifacts silently unless `--force` is set.

## Data governance and privacy
- Assume all note text is sensitive and must remain in approved secure environments.
- Never send raw clinical text to external APIs/services.
- De-identification placeholders (e.g., `[**Name**]`) must be preserved during cleaning.

## Naming conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Config keys: `snake_case`
- Output files include model, task, and split when relevant.

## Expected outputs
Ensure scripts generate outputs under:
- `outputs/metrics`
- `outputs/predictions`
- `outputs/figures`
- `outputs/tables`
- `outputs/logs`
- `outputs/error_analysis`
- `outputs/capability_results`
