"""
Feature audit: which are raw vs. opponent-adjusted, and what's available
to add from existing data pulls without a new API call.
"""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROC = ROOT / "data" / "processed"

# ── 1. PPA games — what columns exist? ───────────────────────────────────────
ppa = pd.read_csv(PROC / "master_ppa_games.csv")
print("=== master_ppa_games.csv ===")
print("Shape:", ppa.shape)
print("Columns:", list(ppa.columns))
print("Seasons:", sorted(ppa["season"].unique()) if "season" in ppa.columns else "?")
print()

# ── 2. Advanced stats (havoc) — what columns exist now? ───────────────────────
havoc = pd.read_csv(PROC / "master_havoc.csv")
print("=== master_havoc.csv ===")
print("Shape:", havoc.shape)
print("Columns:", list(havoc.columns))
print()

# ── 3. WEPA — already opponent-adjusted ──────────────────────────────────────
wepa = pd.read_csv(PROC / "master_wepa.csv")
print("=== master_wepa.csv (already opponent-adjusted) ===")
print("Shape:", wepa.shape)
print("Columns:", list(wepa.columns))
print()

# ── 4. Can we compute opponent-adjusted rolling EPA from existing data? ───────
# Strategy: for each game, adjust offensive EPA by opponent's season-avg def EPA
print("=== Opponent-adjustment feasibility ===")
if "off_epa" in ppa.columns and "def_epa" in ppa.columns and "opponent" in ppa.columns:
    # Compute each team's season-average defensive EPA allowed
    def_avg = (ppa.groupby(["season", "team"])["def_epa"]
                  .mean()
                  .reset_index()
                  .rename(columns={"team": "opponent", "def_epa": "opp_avg_def_epa"}))
    ppa2 = ppa.merge(def_avg, on=["season", "opponent"], how="left")
    coverage = ppa2["opp_avg_def_epa"].notna().mean()
    print(f"Opponent-adjusted EPA feasible: coverage = {coverage:.1%}")
    # Adjusted EPA = raw EPA - opponent's average defensive EPA allowed
    # (controls for playing a strong vs weak defense)
    ppa2["adj_off_epa"] = ppa2["off_epa"] - ppa2["opp_avg_def_epa"].fillna(0)
    print(f"adj_off_epa sample: mean={ppa2['adj_off_epa'].mean():.3f}, "
          f"std={ppa2['adj_off_epa'].std():.3f}")
    print(f"raw off_epa sample: mean={ppa2['off_epa'].mean():.3f}, "
          f"std={ppa2['off_epa'].std():.3f}")
else:
    print("Columns available in PPA:", list(ppa.columns))

# ── 5. What run-game metrics are in havoc but not yet used? ───────────────────
print()
print("=== Run-game metrics in havoc (not yet in feature lists) ===")
unused = [c for c in havoc.columns
          if c not in ("season", "team", "havoc_total", "havoc_front_seven",
                       "havoc_db", "rush_success_rate", "pass_success_rate")]
print("Available but unused:", unused)
if unused:
    print(havoc[["season", "team"] + unused].head(5).to_string(index=False))
