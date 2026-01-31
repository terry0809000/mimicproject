import os
from pathlib import Path
import pandas as pd

# -----------------------------
# 1) Define the real MIMIC root
# -----------------------------
REAL_MIMIC_ROOT = Path(r"C:\Users\Terry Yu\Documents\mimic-iii-clinical-database-1.4")

# -----------------------------
# 2) Validate that it exists
# -----------------------------
if not REAL_MIMIC_ROOT.exists():
    raise FileNotFoundError(
        f"MIMIC root folder does not exist:\n  {REAL_MIMIC_ROOT}\n\n"
        "Fix: check spelling, permissions, and that you extracted the dataset correctly."
    )
if not REAL_MIMIC_ROOT.is_dir():
    raise NotADirectoryError(f"Path is not a directory:\n  {REAL_MIMIC_ROOT}")

print(f"✅ Found MIMIC root: {REAL_MIMIC_ROOT}")

# ------------------------------------------
# 3) Locate NOTEEVENTS.csv or NOTEEVENTS.csv.gz
# ------------------------------------------
candidates = [
    REAL_MIMIC_ROOT / "NOTEEVENTS.csv.gz",
    REAL_MIMIC_ROOT / "NOTEEVENTS.csv",
    # sometimes files are inside a nested subfolder
    REAL_MIMIC_ROOT / "mimic-iii-clinical-database-1.4" / "NOTEEVENTS.csv.gz",
    REAL_MIMIC_ROOT / "mimic-iii-clinical-database-1.4" / "NOTEEVENTS.csv",
]

note_file = next((p for p in candidates if p.exists()), None)

if note_file is None:
    # recursive search fallback
    matches = list(REAL_MIMIC_ROOT.rglob("NOTEEVENTS.csv")) + list(REAL_MIMIC_ROOT.rglob("NOTEEVENTS.csv.gz"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find NOTEEVENTS.csv(.gz) under:\n  {REAL_MIMIC_ROOT}\n\n"
            "Check that the MIMIC-III files are extracted correctly."
        )
    note_file = matches[0]

print(f"✅ Using NOTEEVENTS file: {note_file}")

# ------------------------------------------
# 4) Read NOTEEVENTS efficiently
# ------------------------------------------
# Columns typically present in MIMIC-III NOTEEVENTS
usecols = [
    "ROW_ID", "SUBJECT_ID", "HADM_ID",
    "CHARTDATE", "CHARTTIME", "STORETIME",
    "CATEGORY", "DESCRIPTION", "TEXT"
]

# pandas can read .csv.gz directly via compression="infer"
notes = pd.read_csv(note_file, usecols=usecols, compression="infer", low_memory=False)

print(f"✅ Loaded NOTEEVENTS rows: {len(notes):,}")

# ------------------------------------------
# 5) Optional: filter to Discharge summaries
# ------------------------------------------
notes_ds = notes.loc[notes["CATEGORY"] == "Discharge summary"].copy()

# Rename into a modelling-friendly schema
notes_ds.rename(columns={"ROW_ID": "note_id", "TEXT": "text"}, inplace=True)

# Keep a clean set of columns
keep = ["note_id", "SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "CHARTDATE", "text"]
notes_ds = notes_ds[keep]

print(f"✅ Discharge summaries retained: {len(notes_ds):,}")

# ------------------------------------------
# 6) Write output
# ------------------------------------------
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "real_mimic_notes_discharge.csv"
notes_ds.to_csv(out_path, index=False)

print(f"✅ Wrote dataset to: {out_path.resolve()}")

# ------------------------------------------
# 7) Sanity checks
# ------------------------------------------
avg_chars = notes_ds["text"].astype(str).str.len().mean()
print("\n--- Sanity checks ---")
print(f"n_notes    : {len(notes_ds):,}")
print(f"n_subjects : {notes_ds['SUBJECT_ID'].nunique():,}")
print(f"n_hadm     : {notes_ds['HADM_ID'].nunique():,}")
print(f"avg_chars  : {avg_chars:,.1f}")
