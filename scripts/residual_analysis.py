"""
Deep residual analysis on walk-forward OOS predictions.
Finds systematic patterns in where the model is wrong.

Run: /opt/homebrew/bin/python3 scripts/residual_analysis.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
wf = pd.read_csv(ROOT / "outputs" / "predictions" / "walk_forward_results.csv")

# Load feature matrix for additional context
fm = pd.read_csv(ROOT / "data" / "processed" / "feature_matrix.csv")
fm_cols = ["game_id", "season", "week", "home_team", "away_team",
           "neutral_site", "conference_game", "home_conference", "away_conference"]
fm = fm[[c for c in fm_cols if c in fm.columns]].copy()

wf = wf.merge(fm, on=["season", "week", "home_team", "away_team"], how="left")

# Core columns
wf["season"]         = pd.to_numeric(wf["season"], errors="coerce")
wf["week"]           = pd.to_numeric(wf["week"], errors="coerce")
wf["spread"]         = pd.to_numeric(wf["spread"], errors="coerce")
wf["point_diff"]     = pd.to_numeric(wf["point_diff"], errors="coerce")
wf["pred_spread"]    = pd.to_numeric(wf["pred_spread"], errors="coerce")
wf["spread_edge"]    = pd.to_numeric(wf["spread_edge"], errors="coerce")
wf["covered_spread"] = pd.to_numeric(wf["covered_spread"], errors="coerce")

# Error = predicted margin minus actual margin (positive = over-predicted home)
wf["error"]     = wf["pred_spread"] - wf["point_diff"]
wf["error_abs"] = wf["error"].abs()
wf["vegas_err"] = (-wf["spread"]) - wf["point_diff"]  # vegas error for comparison

print(f"Walk-forward OOS games: {len(wf):,}")
print(f"Model MAE:  {wf['error_abs'].mean():.3f} pts")
print(f"Vegas MAE:  {wf['vegas_err'].abs().mean():.3f} pts")
print(f"Model bias: {wf['error'].mean():+.3f} pts (+ = over-predicts home)")
print()

# ── 1. Error by week of season ─────────────────────────────────────────────────
print("=== Error by week of season ===")
print(f"{'Week':>6} {'Games':>6} {'MAE':>7} {'Bias':>8} {'ATS%':>7}")
print("-" * 40)
for wk in range(1, 17):
    sub = wf[wf["week"] == wk].dropna(subset=["error", "covered_spread"])
    if len(sub) < 20:
        continue
    ats = sub["covered_spread"].mean() * 100
    flag = " ⚠️" if abs(sub["error"].mean()) > 2 else ""
    print(f"  Wk {wk:2d}  {len(sub):>5}  {sub['error_abs'].mean():>6.2f}  "
          f"{sub['error'].mean():>+7.2f}  {ats:>6.1f}%{flag}")

# ── 2. Error by favorite/underdog magnitude ────────────────────────────────────
print("\n=== Error by game type (spread magnitude) ===")
bins = [(-50,-14,"Big underdog (<-14)"), (-14,-7,"Moderate underdog (-7 to -14)"),
        (-7,-3,"Slight underdog (-3 to -7)"), (-3,3,"Toss-up (0-3)"),
        (3,7,"Slight fav (3-7)"), (7,14,"Moderate fav (7-14)"),
        (14,50,"Big favorite (14+)")]
print(f"{'Type':<35} {'N':>5} {'MAE':>7} {'Bias':>8} {'ATS%':>7}")
print("-" * 65)
for lo, hi, label in bins:
    sub = wf[(wf["spread"] >= lo) & (wf["spread"] < hi)].dropna(subset=["error","covered_spread"])
    if len(sub) < 20: continue
    ats = sub["covered_spread"].mean() * 100
    flag = " ⚠️" if abs(sub["error"].mean()) > 3 else ""
    print(f"  {label:<33} {len(sub):>5}  {sub['error_abs'].mean():>6.2f}  "
          f"{sub['error'].mean():>+7.2f}  {ats:>6.1f}%{flag}")

# ── 3. Error by conference ─────────────────────────────────────────────────────
P4 = {"Big Ten", "SEC", "Big 12", "ACC", "Pac-12"}
if "home_conference" in wf.columns:
    print("\n=== Error by home team conference ===")
    print(f"{'Conference':<30} {'N':>5} {'MAE':>7} {'Bias':>8} {'ATS%':>7}")
    print("-" * 60)
    for conf in sorted(wf["home_conference"].dropna().unique()):
        sub = wf[wf["home_conference"] == conf].dropna(subset=["error","covered_spread"])
        if len(sub) < 40: continue
        ats = sub["covered_spread"].mean() * 100
        flag = " ⚠️" if abs(sub["error"].mean()) > 2 or abs(ats-50) > 4 else ""
        marker = " [P4]" if conf in P4 else ""
        print(f"  {conf+marker:<28} {len(sub):>5}  {sub['error_abs'].mean():>6.2f}  "
              f"{sub['error'].mean():>+7.2f}  {ats:>6.1f}%{flag}")

# ── 4. Error by season ────────────────────────────────────────────────────────
print("\n=== Error by season ===")
print(f"{'Season':>8} {'N':>5} {'MAE':>7} {'Bias':>8} {'ATS%':>7} {'VegasMAE':>9}")
print("-" * 55)
for szn in sorted(wf["season"].dropna().unique()):
    sub = wf[wf["season"] == szn].dropna(subset=["error","covered_spread"])
    if len(sub) < 10: continue
    ats = sub["covered_spread"].mean() * 100
    vmae = sub["vegas_err"].abs().mean()
    print(f"  {int(szn):>6}  {len(sub):>5}  {sub['error_abs'].mean():>6.2f}  "
          f"{sub['error'].mean():>+7.2f}  {ats:>6.1f}%  {vmae:>8.2f}")

# ── 5. Neutral site games ─────────────────────────────────────────────────────
if "neutral_site" in wf.columns:
    print("\n=== Neutral site vs. home games ===")
    for ns, label in [(0,"Home games"), (1,"Neutral site")]:
        sub = wf[wf["neutral_site"] == ns].dropna(subset=["error","covered_spread"])
        if len(sub) < 20: continue
        ats = sub["covered_spread"].mean() * 100
        print(f"  {label:<20} n={len(sub):4d}  MAE={sub['error_abs'].mean():.2f}  "
              f"bias={sub['error'].mean():+.2f}  ATS={ats:.1f}%")

# ── 6. Conference games vs non-conference ────────────────────────────────────
if "conference_game" in wf.columns:
    print("\n=== Conference vs non-conference games ===")
    for cg, label in [(0,"Non-conference"), (1,"Conference game")]:
        sub = wf[wf["conference_game"] == cg].dropna(subset=["error","covered_spread"])
        if len(sub) < 20: continue
        ats = sub["covered_spread"].mean() * 100
        print(f"  {label:<22} n={len(sub):4d}  MAE={sub['error_abs'].mean():.2f}  "
              f"bias={sub['error'].mean():+.2f}  ATS={ats:.1f}%")

# ── 7. Edge size vs outcome ───────────────────────────────────────────────────
print("\n=== Does larger model edge predict better? ===")
print(f"{'Edge range':<25} {'N':>5} {'ATS%':>7} {'Dir acc':>8}")
print("-" * 45)
for lo, hi in [(0,2),(2,4),(4,6),(6,9),(9,99)]:
    sub = wf[wf["spread_edge"].abs().between(lo, hi)].dropna(subset=["covered_spread","spread_edge"])
    if len(sub) < 20: continue
    home_bet = sub["spread_edge"] > 0
    correct = ((home_bet & (sub["covered_spread"]==1)) |
               (~home_bet & (sub["covered_spread"]==0)))
    ats_home = sub["covered_spread"].mean()*100
    dir_acc = correct.mean()*100
    print(f"  Edge {lo:.0f}-{hi:.0f} pts          {len(sub):>5}  {ats_home:>6.1f}%  {dir_acc:>7.1f}%")

# ── 8. Top systematic mispredictions ─────────────────────────────────────────
print("\n=== Games with largest systematic bias (model vs Vegas) ===")
wf["model_vs_vegas"] = wf["error"] - wf["vegas_err"]
worst = wf.nlargest(10, "error_abs")[
    ["season","week","home_team","away_team","spread","pred_spread","point_diff","error"]]
print("Biggest model errors (top 10):")
print(worst.to_string(index=False))

print("\nDone. Key findings above will guide next feature engineering.")
