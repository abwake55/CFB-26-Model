"""
Home Bias Diagnostic
====================
Quantifies the systematic home-team bias in the spread model using
the full walk-forward out-of-sample results.

What this script answers:
  1. Does the model systematically over-predict home team margins?
  2. Which features drive the bias?
  3. How large does the correction need to be?
  4. What does ATS look like after applying a simple bias correction?

Run:
    python3 scripts/diagnose_home_bias.py

No external data needed — reads outputs/predictions/walk_forward_results.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
WF_PATH  = ROOT / "outputs" / "predictions" / "walk_forward_results.csv"
FEAT_PATH = ROOT / "models" / "feature_lists.json"

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

df = pd.read_csv(WF_PATH)
print(f"Walk-forward games loaded: {len(df):,}  |  seasons: {sorted(df['season'].unique())}\n")

# Core columns
df = df.dropna(subset=["pred_spread", "point_diff", "spread_edge"])
df["residual"] = df["pred_spread"] - df["point_diff"]   # positive = over-predicted home
df["abs_residual"] = df["residual"].abs()

# ─── 1. OVERALL DIRECTIONAL BIAS ─────────────────────────────────────────────

print("=" * 60)
print("1. OVERALL BIAS")
print("=" * 60)
mean_res = df["residual"].mean()
med_res  = df["residual"].median()
print(f"  Mean residual (pred - actual):   {mean_res:+.3f} pts")
print(f"  Median residual (pred - actual): {med_res:+.3f} pts")
print(f"  (Positive = model over-predicts home team margin)")

if abs(mean_res) >= 1.0:
    print(f"\n  ⚠️  Significant bias detected: {mean_res:+.1f} pts per game")
else:
    print(f"\n  ✓  Small overall bias: {mean_res:+.1f} pts — look for conditional bias below")

# ─── 2. BIAS BY SEASON ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. BIAS BY SEASON")
print("=" * 60)
season_bias = df.groupby("season")["residual"].agg(["mean", "median", "count"]).round(3)
season_bias.columns = ["mean_resid", "med_resid", "n_games"]
print(season_bias.to_string())

# ─── 3. BIAS BY SPREAD MAGNITUDE ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. BIAS BY VEGAS SPREAD MAGNITUDE")
print("=" * 60)
df_lined = df.dropna(subset=["spread"])
df_lined["spread_abs"] = df_lined["spread"].abs()
bins   = [0, 3, 7, 14, 21, 100]
labels = ["0-3", "3-7", "7-14", "14-21", "21+"]
df_lined["spread_bucket"] = pd.cut(df_lined["spread_abs"], bins=bins, labels=labels)
spread_bias = df_lined.groupby("spread_bucket", observed=True)["residual"].agg(["mean","count"]).round(3)
spread_bias.columns = ["mean_resid", "n_games"]
print(spread_bias.to_string())
print("  (Heavy favorites are most likely to show over-prediction if HFA is double-counted)")

# ─── 4. BIAS BY PREDICTED SPREAD ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. BIAS BY MODEL'S OWN PREDICTED SPREAD")
print("=" * 60)
df["pred_spread_bucket"] = pd.cut(df["pred_spread"],
    bins=[-100, -21, -14, -7, -3, 3, 7, 14, 21, 100],
    labels=["<-21","-21 to -14","-14 to -7","-7 to -3","-3 to 3","3 to 7","7 to 14","14 to 21",">21"])
pred_bias = df.groupby("pred_spread_bucket", observed=True)["residual"].agg(["mean","count"]).round(3)
pred_bias.columns = ["mean_resid", "n_games"]
print(pred_bias.to_string())

# ─── 5. ATS BY EDGE DIRECTION ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. ATS PERFORMANCE BY EDGE DIRECTION & SIZE")
print("=" * 60)
df_ats = df.dropna(subset=["covered_spread"])
bins_e = [-np.inf, -10, -5, -3, -1, 1, 3, 5, 10, np.inf]
lbls_e = ["<-10","-10:-5","-5:-3","-3:-1","-1:1","1:3","3:5","5:10",">10"]
df_ats["edge_bucket"] = pd.cut(df_ats["spread_edge"], bins=bins_e, labels=lbls_e)
ats_tbl = df_ats.groupby("edge_bucket", observed=True).agg(
    n=("covered_spread","count"),
    ats=("covered_spread","mean"),
    mean_edge=("spread_edge","mean"),
).round(3)
print(ats_tbl.to_string())
print("\n  Breakeven ATS = 0.524.  Buckets below 0.50 when model 'likes home' confirm home bias.")

# ─── 6. SIMPLE BIAS CORRECTION ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. BIAS CORRECTION — WHAT OFFSET MAXIMISES ATS?")
print("=" * 60)

df_ats2 = df.dropna(subset=["covered_spread", "spread"]).copy()

# The Vegas spread is from home team's perspective (negative = home favored)
# covered_spread = 1 if home team covered
# To bet: if our (corrected) pred > vegas line → bet home
# We sweep a correction offset applied to pred_spread

best_ats, best_offset = 0, 0
results = []
MIN_EDGE = 3.0

for offset in np.arange(-7, 7.1, 0.5):
    corrected_edge = df_ats2["pred_spread"] - offset - df_ats2["vegas_home_margin"]
    bets = df_ats2[corrected_edge.abs() >= MIN_EDGE].copy()
    bets["corrected_edge"] = corrected_edge[bets.index]
    if len(bets) < 200:
        continue
    ats = bets["covered_spread"].mean()
    results.append({"offset": offset, "n_bets": len(bets), "ats": round(ats, 4)})
    if ats > best_ats:
        best_ats, best_offset = ats, offset

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print(f"\n  Best offset: {best_offset:+.1f} pts  →  ATS = {best_ats:.4f} (edge >= {MIN_EDGE})")
print(f"  Interpretation: subtract {best_offset:+.1f} from pred_spread before computing edge")

# ─── 7. FEATURE CORRELATION WITH RESIDUAL ───────────────────────────────────

print("\n" + "=" * 60)
print("7. WHICH FEATURES CORRELATE MOST WITH THE RESIDUAL?")
print("   (High positive correlation → feature inflates home predictions)")
print("=" * 60)

# Candidate home-advantage-related features available in the results CSV
candidate_cols = [c for c in df.columns if any(k in c for k in [
    "home_", "hfa", "elo", "sp_", "fpi", "srs", "recruiting", "portal",
    "neutral_site", "week", "rest", "conference", "line_movement"
]) and c in df.columns and df[c].dtype in [np.float64, np.int64, float, int]]

if candidate_cols:
    corr = df[candidate_cols + ["residual"]].corr()["residual"].drop("residual")
    corr = corr.dropna().abs().sort_values(ascending=False)
    top = corr.head(20)
    print("\n  Top 20 features by |correlation with residual|:")
    for feat, val in top.items():
        direction = "+" if df[candidate_cols].corrwith(df["residual"]).get(feat, 0) > 0 else "-"
        print(f"    {direction}{val:.4f}  {feat}")
else:
    print("  (Feature columns not present in walk-forward CSV — run with full feature matrix)")

# ─── 8. CORRECTED ATS SIMULATION ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. CORRECTED VS. UNCORRECTED ATS COMPARISON")
print("=" * 60)

df_sim = df.dropna(subset=["covered_spread", "vegas_home_margin"]).copy()
df_sim["raw_edge"]       = df_sim["pred_spread"] - df_sim["vegas_home_margin"]
df_sim["corrected_edge"] = df_sim["pred_spread"] - best_offset - df_sim["vegas_home_margin"]

for min_edge in [2, 3, 5]:
    raw_bets  = df_sim[df_sim["raw_edge"].abs() >= min_edge]
    corr_bets = df_sim[df_sim["corrected_edge"].abs() >= min_edge]
    raw_ats   = raw_bets["covered_spread"].mean()
    corr_ats  = corr_bets["covered_spread"].mean()
    print(f"  Edge >= {min_edge}:  Raw {len(raw_bets):4d} bets @ {raw_ats:.4f} ATS  →  "
          f"Corrected {len(corr_bets):4d} bets @ {corr_ats:.4f} ATS  "
          f"({'▲' if corr_ats > raw_ats else '▼'} {abs(corr_ats - raw_ats):.4f})")

# ─── 9. CONFERENCE GAME VS NON-CONFERENCE ────────────────────────────────────

if "conference_game" in df.columns:
    print("\n" + "=" * 60)
    print("9. BIAS: CONFERENCE VS NON-CONFERENCE GAMES")
    print("=" * 60)
    conf_bias = df.groupby("conference_game")["residual"].agg(["mean","count"]).round(3)
    conf_bias.index = ["Non-conference", "Conference"]
    conf_bias.columns = ["mean_resid", "n_games"]
    print(conf_bias.to_string())

# ─── 10. WEEK-BY-WEEK BIAS (EARLY VS LATE SEASON) ───────────────────────────

if "week" in df.columns:
    print("\n" + "=" * 60)
    print("10. BIAS BY WEEK (EARLY VS LATE SEASON)")
    print("=" * 60)
    df["week_group"] = pd.cut(df["week"],
        bins=[0, 3, 8, 14, 100],
        labels=["Weeks 1-3", "Weeks 4-8", "Weeks 9-14", "Bowl/Playoff"])
    week_bias = df.groupby("week_group", observed=True)["residual"].agg(["mean","count"]).round(3)
    week_bias.columns = ["mean_resid", "n_games"]
    print(week_bias.to_string())
    print("  (Early season bias is common — less data on teams, HFA/travel harder to estimate)")

print("\n" + "=" * 60)
print("SUMMARY & NEXT STEPS")
print("=" * 60)
print(f"  1. Overall mean residual: {mean_res:+.3f} pts (positive = home over-predicted)")
print(f"  2. Best simple correction: subtract {best_offset:+.1f} pts from pred_spread")
print(f"  3. Apply in retrain.sh or src/model.py post-prediction step")
print(f"  4. Re-run walk_forward.py after retraining with corrected HFA weight")
print(f"  5. Target: ATS > 0.500 at edge >= 3 before adding further complexity")
print()
