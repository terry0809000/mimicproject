from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prevalence(prevalence_df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(prevalence_df["task"], prevalence_df["prevalence"])
    ax.set_ylabel("Prevalence")
    ax.set_xticklabels(prevalence_df["task"], rotation=45, ha="right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_perf_vs_cost(df: pd.DataFrame, x: str, y: str, label_col: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x], df[y])
    for _, r in df.iterrows():
        ax.annotate(str(r[label_col]), (r[x], r[y]))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
