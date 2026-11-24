"""Evaluate spread-lab rules: dev (2019-2023) chooses the threshold, 2024-25 is holdout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_lab import grade, DEV, HOLD, OUT  # noqa: E402


def evaluate(d, edge_fn, at="open", thresholds=(2, 3, 4, 5, 6), min_dev_n=100,
             mask_fn=None, label="", dev=DEV, hold=HOLD):
    """edge_fn(d)->Series (pred margin minus market margin at the decision line).
    Returns (grid rows, chosen row dict)."""
    rows = []
    for period, seasons in (("dev", dev), ("holdout", hold)):
        x = d[d.season.isin(seasons)]
        e = edge_fn(x)
        line = x.open_line if at == "open" else x.close_line
        mask = mask_fn(x) if mask_fn else None
        for t in thresholds:
            r = grade(x, e, line, t, mask)
            rc = grade(x, e, x.close_line, t, mask & line.notna() if mask is not None else line.notna())
            rows.append({"rule": label, "period": period, "thr": t, "bet_at": at, **r,
                         "close_win%": rc["win%"], "close_roi%": rc["roi%"]})
    g = pd.DataFrame(rows)
    dv = g[(g.period == "dev") & (g.n >= min_dev_n)]
    if dv.empty:
        return g, None
    t = dv.sort_values("roi%", ascending=False).iloc[0].thr
    return g, t


def show(g, t, label):
    sel = g[g.thr == t] if t is not None else g.iloc[0:0]
    cols = ["rule", "period", "thr", "bet_at", "n", "w", "l", "p", "win%", "roi%", "clv", "clv+%",
            "close_win%", "close_roi%"]
    print(f"\n== {label} (thr chosen on dev: {t})")
    print(sel[cols].to_string(index=False))
    return sel[cols]
