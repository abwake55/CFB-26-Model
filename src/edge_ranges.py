"""Uncertainty ranges for every pick (models/edge_ranges.json, fitted by
scripts/research/fit_edge_ranges.py).

For each spread signal adds, in points for the side the signal bets:
  {sig}_edge_exp   expected cover margin given the stated edge (backtest)
  {sig}_edge_lo80  80% range, low end
  {sig}_edge_hi80  80% range, high end
and for the main model's projected home margin:
  pred_margin_lo80 / pred_margin_hi80  (10th-90th percentile game outcome)

The stated edge is what the model says; the expected range is what edges of
that size actually delivered in 2019-2025, including season-to-season drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "models" / "edge_ranges.json"
Z80 = 1.2816
# signal -> edge column (home points, > 0 = bet home)
EDGE_COLS = {"spread": "spread_edge", "opener_gap": "opener_gap_edge", "pt": "pt_edge"}


def load() -> dict | None:
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else None


def expected(p: dict, e) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.abs(np.asarray(e, dtype=float))
    c = np.array(p["cov"])
    mu = p["a"] + p["b"] * e
    se = np.sqrt(np.maximum(c[0, 0] + 2 * e * c[0, 1] + e * e * c[1, 1], 0))
    return mu, mu - Z80 * se, mu + Z80 * se


def attach(out: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or load()
    if not params:
        return out
    out = out.copy()
    sig = params["signals"]
    for name, col in EDGE_COLS.items():
        if name not in sig or col not in out.columns:
            continue
        e = pd.to_numeric(out[col], errors="coerce")
        mu, lo, hi = expected(sig[name], e)
        for k, v in (("exp", mu), ("lo80", lo), ("hi80", hi)):
            out[f"{name}_edge_{k}"] = np.round(np.where(e.notna(), v, np.nan), 2)
    if "spread" in sig and "pred_spread" in out.columns:
        m = pd.to_numeric(out["pred_spread"], errors="coerce")
        out["pred_margin_lo80"] = (m + sig["spread"]["resid_q10"]).round(1)
        out["pred_margin_hi80"] = (m + sig["spread"]["resid_q90"]).round(1)
    return out


def describe(row: dict, name: str) -> str:
    """'edge 7.5 -> expect +3.0 (80%: +1.2 to +4.8)' for messages/docs."""
    e = row.get(EDGE_COLS.get(name, ""))
    lo, mu, hi = (row.get(f"{name}_edge_{k}") for k in ("lo80", "exp", "hi80"))
    if any(v is None or pd.isna(v) for v in (e, lo, mu, hi)):
        return ""
    return f"edge {abs(float(e)):.1f} -> expect {mu:+.1f} (80%: {lo:+.1f} to {hi:+.1f})"
