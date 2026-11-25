"""High-total under signal (adopted 2026-11-24).

Bet the under when the first total we see is at least THRESHOLD. The decision
line is the captured opener if we have one (the week's line-snapshot `first`
total, else CFBD `over_under_open`), otherwise the current total:

    high_total_under_line = snap_open_total or over_under_open or over_under
    flag_high_total_under = high_total_under_line >= THRESHOLD

THRESHOLD = 60 was chosen on 2019-2023 only (docs/high_total_under_2026-11-24.md).
On the 2024-25 holdout, unders at a close >= 60 went 118-96-2 (55.14%, +5.27%
ROI). The same rule at the opener (open >= 60) went 118-90-1 (56.73%, +8.30%,
CLV +0.70). Not proven live: size at 0.25u and grade ATS / ROI / CLV in the
ledger until the promotion rule is met.

No Odds API calls. Snapshot totals are read from files already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "lines_snapshots"
THRESHOLD = 60.0
UNITS = 0.25
VERSION = "high_total_under_v1"


def snapshot_first_totals(season, week) -> dict:
    """{game_id: first captured total} from the week's line snapshot, if any."""
    try:
        season, week = int(season), int(week)
    except (TypeError, ValueError):
        return {}
    path = SNAP_DIR / f"lines_{season}_W{week:02d}.json"
    if not path.exists():
        return {}
    try:
        games = json.loads(path.read_text()).get("games") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for gid, rec in games.items():
        ou = (rec.get("first") or {}).get("over_under")
        if ou is None:
            continue
        try:
            out[int(gid)] = float(ou)
        except (TypeError, ValueError):
            continue
    return out


def attach_snapshot_first(df: pd.DataFrame) -> pd.DataFrame:
    """Fill snap_open_total from the on-disk week snapshot (no API calls)."""
    out = df.copy()
    if "snap_open_total" not in out.columns:
        out["snap_open_total"] = np.nan
    if "season" not in out.columns or "week" not in out.columns or "game_id" not in out.columns:
        return out
    for (season, week), grp in out.groupby(["season", "week"], dropna=False):
        first = snapshot_first_totals(season, week)
        if not first:
            continue
        gids = pd.to_numeric(grp["game_id"], errors="coerce")
        mapped = gids.map(first)
        out.loc[grp.index, "snap_open_total"] = out.loc[grp.index, "snap_open_total"].where(
            pd.to_numeric(out.loc[grp.index, "snap_open_total"], errors="coerce").notna(),
            mapped,
        )
    return out


def decision_line(df: pd.DataFrame) -> pd.Series:
    """First total we see: snapshot opener, else CFBD opener, else current."""
    snap = pd.to_numeric(df["snap_open_total"], errors="coerce") if "snap_open_total" in df.columns \
        else pd.Series(np.nan, index=df.index)
    opened = pd.to_numeric(df["over_under_open"], errors="coerce") if "over_under_open" in df.columns \
        else pd.Series(np.nan, index=df.index)
    current = pd.to_numeric(df["over_under"], errors="coerce") if "over_under" in df.columns \
        else pd.Series(np.nan, index=df.index)
    line = snap.copy()
    line = line.where(line.notna(), opened)
    line = line.where(line.notna(), current)
    return line


def attach(out: pd.DataFrame) -> pd.DataFrame:
    """Add high_total_under_line and flag_high_total_under."""
    out = attach_snapshot_first(out)
    out = out.copy()
    out["high_total_under_line"] = decision_line(out)
    out["flag_high_total_under"] = out["high_total_under_line"] >= THRESHOLD
    # NaN >= 60 is False; keep explicit False on missing lines
    out.loc[out["high_total_under_line"].isna(), "flag_high_total_under"] = False
    return out
