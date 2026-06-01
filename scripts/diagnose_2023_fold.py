"""
Diagnose why the 2023 walk-forward fold produced MAE 17.16 (vs ~13 in all other years).
Run: /opt/homebrew/bin/python3 scripts/diagnose_2023_fold.py
"""
import sys, os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

df = pd.read_csv(ROOT / "data" / "processed" / "feature_matrix.csv")
print(f"Feature matrix: {len(df):,} rows  seasons {int(df['season'].min())}–{int(df['season'].max())}")
print(f"Columns: {len(df.columns)}")

seasons = sorted(df["season"].unique())

# ── 1. Feature coverage by season ─────────────────────────────────────────────
print("\n=== Feature coverage by season (% non-null) ===")
check_cols = [
    "spread", "over_under", "vegas_home_margin",
    "home_sp_rating", "away_sp_rating",
    "home_off_epa_roll3", "away_off_epa_roll3",
    "home_havoc_total", "away_havoc_total",
    "home_turnover_margin", "away_turnover_margin",
    "home_plays_per_drive", "away_plays_per_drive",
    "wind_speed", "is_dome",
    "home_wepa_offense", "away_wepa_offense",
    "home_talent", "away_talent",
    "home_fpi", "away_fpi",
    "home_srs", "away_srs",
    "home_portal_net_rating", "away_portal_net_rating",
    "line_movement", "spread_open_val",
]
check_cols = [c for c in check_cols if c in df.columns]

hdr = f"{'Feature':<35}" + "".join(f"{int(s):>6}" for s in seasons)
print(hdr)
print("-" * len(hdr))

suspicious = []
for col in check_cols:
    row_str = f"{col:<35}"
    covs = []
    for s in seasons:
        sub = df[df["season"] == s]
        cov = sub[col].notna().mean() * 100
        covs.append(cov)
        row_str += f"{cov:>5.0f}%"
    print(row_str)

    # Flag if 2022 or 2023 coverage is very different from the median
    med_cov = np.median(covs)
    idx_2022 = list(seasons).index(2022) if 2022 in seasons else None
    idx_2023 = list(seasons).index(2023) if 2023 in seasons else None
    if idx_2022 is not None and abs(covs[idx_2022] - med_cov) > 20:
        suspicious.append((col, 2022, covs[idx_2022], med_cov))
    if idx_2023 is not None and abs(covs[idx_2023] - med_cov) > 20:
        suspicious.append((col, 2023, covs[idx_2023], med_cov))

# ── 2. Flag suspicious coverage gaps ──────────────────────────────────────────
if suspicious:
    print("\n=== ⚠️  Suspicious coverage gaps (>20pp from median) ===")
    for col, season, cov, med in suspicious:
        print(f"  {col:<35} season={season}  coverage={cov:.0f}%  median={med:.0f}%")
else:
    print("\n✅ No suspicious coverage gaps found")

# ── 3. Feature value distributions for 2022 vs 2023 ──────────────────────────
print("\n=== Feature value distributions: 2022 (train) vs 2023 (test) ===")
train_2022 = df[df["season"] == 2022]
test_2023  = df[df["season"] == 2023]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in
                ["game_id", "season", "week", "home_points", "away_points",
                 "point_diff", "total_points", "home_win", "covered_spread", "went_over"]]

print(f"\n{'Feature':<35} {'2022 mean':>10} {'2023 mean':>10} {'shift':>10} {'flag':>6}")
print("-" * 75)

big_shifts = []
for col in numeric_cols[:60]:  # check first 60 numeric features
    m22 = train_2022[col].mean()
    m23 = test_2023[col].mean()
    if pd.isna(m22) or pd.isna(m23):
        continue
    std = df[col].std()
    if std < 0.001:
        continue
    shift = abs(m23 - m22) / std  # standardised shift
    flag = "⚠️" if shift > 1.5 else ""
    if shift > 1.5:
        big_shifts.append((col, m22, m23, shift))
    if shift > 0.8:
        print(f"{col:<35} {m22:>10.3f} {m23:>10.3f} {shift:>10.2f}σ {flag:>6}")

# ── 4. Check the actual walk-forward predictions for 2023 ─────────────────────
wf_path = ROOT / "outputs" / "predictions" / "walk_forward_results.csv"
if wf_path.exists():
    wf = pd.read_csv(wf_path)
    fold_2023 = wf[wf["season"] == 2023].copy()
    if not fold_2023.empty:
        print(f"\n=== Walk-forward 2023 fold predictions ===")
        print(f"  Games: {len(fold_2023)}")
        print(f"  pred_spread: mean={fold_2023['pred_spread'].mean():.2f}  "
              f"std={fold_2023['pred_spread'].std():.2f}  "
              f"min={fold_2023['pred_spread'].min():.1f}  "
              f"max={fold_2023['pred_spread'].max():.1f}")
        print(f"  actual diff: mean={fold_2023['point_diff'].mean():.2f}  "
              f"std={fold_2023['point_diff'].std():.2f}")
        print(f"  vegas margin: mean={fold_2023['vegas_home_margin'].mean():.2f}  "
              f"std={fold_2023['vegas_home_margin'].std():.2f}")

        # Distribution of errors
        fold_2023["error"] = fold_2023["pred_spread"] - fold_2023["point_diff"]
        print(f"  error: mean={fold_2023['error'].mean():.2f}  "
              f"std={fold_2023['error'].std():.2f}  "
              f"MAE={fold_2023['error'].abs().mean():.2f}")

        # Biggest mispredictions
        print(f"\n  Worst 10 predictions:")
        worst = fold_2023.nlargest(10, "error")[
            ["week","home_team","away_team","pred_spread","point_diff","vegas_home_margin","spread_edge"]]
        print(worst.to_string(index=False))

        print(f"\n  Other folds for comparison:")
        for s in [2019, 2020, 2021, 2022, 2024, 2025]:
            sub = wf[wf["season"] == s]
            if sub.empty: continue
            err = (sub["pred_spread"] - sub["point_diff"]).abs().mean()
            std = sub["pred_spread"].std()
            print(f"    {s}: MAE={err:.2f}  pred_std={std:.2f}")
        s23 = fold_2023
        print(f"    2023: MAE={(s23['pred_spread'] - s23['point_diff']).abs().mean():.2f}"
              f"  pred_std={s23['pred_spread'].std():.2f}")

print("\nDone.")
