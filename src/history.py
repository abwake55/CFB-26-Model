"""Append-only CSV history (timestamped projections). Never drops rows; if
new columns appear the file is rewritten with the union of columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def append(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    old_cols = list(pd.read_csv(path, nrows=0).columns)
    if set(df.columns) <= set(old_cols):
        df.reindex(columns=old_cols).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path, low_memory=False)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
