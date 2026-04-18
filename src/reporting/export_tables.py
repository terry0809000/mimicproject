from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table(df: pd.DataFrame, base_path: str | Path) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    df.to_json(base.with_suffix(".json"), orient="records", indent=2)
    df.to_latex(base.with_suffix(".tex"), index=False, float_format="%.4f")
