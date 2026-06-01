"""
Find exactly which features have extreme values for the 2023 outlier games.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "feature_matrix.csv")

# Focus on the worst-predicted 2023 games
bad_games = [
    ("Colorado", "Oregon State"),
    ("Old Dominion", "Coastal Carolina"),
    ("Florida State", "Georgia"),
    ("Kentucky", "Alabama"),
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
skip = ["game_id","season","week","home_points","away_points",
        "point_diff","total_points","home_win","covered_spread","went_over"]
feat_cols = [c for c in numeric_cols if c not in skip]

# Global feature stats for context
global_means = df[feat_cols].mean()
global_stds  = df[feat_cols].std().replace(0, 1)

print("=== Extreme feature values in 2023 outlier games ===\n")
for home, away in bad_games:
    game = df[(df["home_team"] == home) & (df["away_team"] == away) &
              (df["season"] == 2023)]
    if game.empty:
        print(f"  {home} vs {away}: NOT FOUND in feature matrix")
        continue

    row = game.iloc[0]
    print(f"\n{home} vs {away}  (wk {int(row.get('week',0))})")

    # Find features with extreme z-scores
    extreme = []
    for col in feat_cols:
        val = row[col]
        if pd.isna(val):
            continue
        z = abs((val - global_means[col]) / global_stds[col])
        if z > 3:
            extreme.append((col, val, global_means[col], z))

    if extreme:
        extreme.sort(key=lambda x: -x[3])
        print(f"  Features with |z| > 3 (extreme values):")
        for col, val, mean, z in extreme[:15]:
            print(f"    {col:<40} val={val:>12.3f}  mean={mean:>8.3f}  z={z:>6.1f}σ")
    else:
        print("  No extreme features found")

# ── Check line_movement and spread_open_val specifically ──────────────────────
print("\n\n=== line_movement and spread_open_val in 2023 ===")
df_2023 = df[df["season"] == 2023]
for col in ["line_movement", "spread_open_val", "total_movement", "sharp_move_home"]:
    if col not in df.columns:
        continue
    vals = df_2023[col].dropna()
    if vals.empty:
        continue
    print(f"\n{col}:")
    print(f"  Coverage: {len(vals)}/{len(df_2023)} ({len(vals)/len(df_2023):.0%})")
    print(f"  Range: {vals.min():.2f} to {vals.max():.2f}")
    print(f"  Mean: {vals.mean():.2f}  Std: {vals.std():.2f}")
    # Check for outliers
    big = vals[vals.abs() > 20]
    if not big.empty:
        print(f"  ⚠️  {len(big)} values with |val| > 20:")
        idx = big.abs().nlargest(5).index
        for i in idx:
            row = df_2023.loc[i]
            print(f"    {row.get('home_team','?')} vs {row.get('away_team','?')}: {big[i]:.2f}")

# ── Coverage comparison: training seasons vs 2023 ─────────────────────────────
print("\n\n=== Coverage: 2017-2022 training vs 2023 test ===")
train = df[df["season"].isin([2017,2018,2019,2021,2022])]
test  = df[df["season"] == 2023]
print(f"{'Feature':<40} {'Train %':>8} {'Test %':>8} {'Gap':>8}")
print("-" * 65)
for col in feat_cols[:50]:
    tr_cov = train[col].notna().mean() * 100
    te_cov = test[col].notna().mean() * 100
    gap = te_cov - tr_cov
    if abs(gap) > 15:
        flag = " ⚠️" if abs(gap) > 30 else ""
        print(f"{col:<40} {tr_cov:>7.0f}% {te_cov:>7.0f}% {gap:>+7.0f}%{flag}")
