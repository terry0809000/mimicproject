from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class CapabilityCase:
    text: str
    expected: int
    category: str


def build_capability_set(seed: int = 42, n_cases_per_type: int = 40) -> pd.DataFrame:
    random.seed(seed)
    bases = {
        "negation": ("Patient denies tobacco use.", 0, "Patient uses tobacco daily.", 1),
        "attribution": ("Family reports patient drinks alcohol.", 0, "Patient drinks alcohol nightly.", 1),
        "temporality": ("Former smoker quit 20 years ago.", 0, "Current smoker one pack per day.", 1),
        "misspelling": ("Pt is a smkoer.", 1, "Pt dnies smkoing.", 0),
    }
    rows = []
    for cat, (neg_text, neg_y, pos_text, pos_y) in bases.items():
        for _ in range(n_cases_per_type // 2):
            rows.append(CapabilityCase(text=neg_text, expected=neg_y, category=cat).__dict__)
            rows.append(CapabilityCase(text=pos_text, expected=pos_y, category=cat).__dict__)
    random.shuffle(rows)
    return pd.DataFrame(rows)
