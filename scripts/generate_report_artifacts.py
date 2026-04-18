#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

from src.utils.config import load_all_configs


def main() -> None:
    cfg = load_all_configs("config")
    out = Path(cfg["paths"]["output_dir"])
    captions = out / "tables" / "figure_captions.txt"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(
        "Figure 1: Label prevalence across SBDH tasks.\n"
        "Figure 2: Cost versus macro-F1 comparison across model families.\n"
        "Figure 3: Confusion matrices by task/model.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
