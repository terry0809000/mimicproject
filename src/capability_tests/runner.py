from __future__ import annotations

from typing import Protocol

import pandas as pd


class PredictProtocol(Protocol):
    def predict(self, texts: list[str]) -> list[int]: ...


def evaluate_capability(model: PredictProtocol, capability_df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(capability_df["text"].tolist())
    out = capability_df.copy()
    out["pred"] = preds
    out["correct"] = (out["pred"] == out["expected"]).astype(int)
    summary = out.groupby("category", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
    summary["failure_rate"] = 1 - summary["accuracy"]
    return summary
