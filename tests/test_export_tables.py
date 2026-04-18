from pathlib import Path

import pandas as pd

from src.reporting.export_tables import export_table


def test_export_table(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2]})
    export_table(df, tmp_path / "tbl")
    assert (tmp_path / "tbl.csv").exists()
    assert (tmp_path / "tbl.tex").exists()
