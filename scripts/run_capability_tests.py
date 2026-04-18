#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.capability_tests.builder import build_capability_set
from src.capability_tests.runner import evaluate_capability
from src.models.rule_based import NegationAwareRuleModel, RuleBasedConfig
from src.reporting.export_tables import export_table
from src.utils.config import load_all_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rule_based")
    args = ap.parse_args()

    cfg = load_all_configs("config")
    cap_cfg = cfg["capability_tests"]
    cap = build_capability_set(seed=cap_cfg["seed"], n_cases_per_type=cap_cfg["n_cases_per_type"])

    model_names = ["rule_based"] if args.models == "all" else args.models.split(",")
    rows = []
    for m in model_names:
        if m == "rule_based":
            lex = yaml.safe_load(Path(cfg["models"]["rule_based"]["lexicon_path"]).read_text())
            rb = NegationAwareRuleModel(RuleBasedConfig(lexicons=lex, negation_cues=lex.get("negation_cues", [])), task="behavior_tobacco")
            s = evaluate_capability(rb, cap)
            s["model"] = m
            rows.append(s)

    result = pd.concat(rows, ignore_index=True)
    export_table(result, Path(cfg["paths"]["output_dir"]) / "capability_results" / "capability_summary")


if __name__ == "__main__":
    main()
