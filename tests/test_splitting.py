import pandas as pd

from src.data.splitting import patient_level_split, split_diagnostics


def test_patient_split_has_no_overlap():
    df = pd.DataFrame({"subject_id": [1, 1, 2, 2, 3, 4, 5, 6], "x": range(8)})
    out = patient_level_split(df, "subject_id", seed=42)
    d = split_diagnostics(out, "subject_id")
    assert d["overlap_train_val"] == 0
    assert d["overlap_train_test"] == 0
    assert d["overlap_val_test"] == 0
