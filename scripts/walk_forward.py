"""
Walk-Forward Backtester
========================
Generates fully out-of-sample predictions for every season from 2019–2025.

For each test season:
  - Train on ALL prior seasons (2020 always excluded — COVID distortion)
  - Tune ensemble blend weights on the most recent prior season (val)
  - Predict on the test season, store results

Result: ~5,500 out-of-sample games in walk_forward_results.csv — enough
for statistically meaningful backtesting in the Streamlit Backtester tab.

Run:
    /opt/homebrew/bin/python3 scripts/walk_forward.py

Expected time: ~4-7 minutes (one full model train per season fold).
"""

import sys, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Load API key ───────────────────────────────────────────────────────────────
if not os.getenv("CFB_API_KEY"):
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            import toml as tomllib
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        os.environ["CFB_API_KEY"] = secrets.get("CFB_API_KEY", "")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from model import (
    load_data,
    EnsembleRegressor, EnsembleClassifier,
    make_linear, make_gbm_regressor, make_gbm_classifier, make_logistic,
    SPREAD_FEATURES, TOTALS_FEATURES, WIN_PROB_FEATURES,
    evaluate_spread, evaluate_totals,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

COVID_SEASON = 2020          # always excluded from training (attendance anomaly)
FIRST_TEST   = 2019          # earliest season we predict (needs 2017+2018 to train)
LAST_TEST    = 2025
OUT_PATH     = ROOT / "outputs" / "predictions" / "walk_forward_results.csv"

BLEND_CANDIDATES = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]


# ─── SINGLE FOLD ──────────────────────────────────────────────────────────────

def run_fold(df: pd.DataFrame, test_season: int) -> pd.DataFrame:
    """
    Train on all prior non-COVID seasons, tune on most recent prior season,
    predict on test_season. Returns a DataFrame of per-game predictions.
    """
    # Training pool: everything before test_season except COVID year
    train_seasons = sorted(s for s in range(2017, test_season) if s != COVID_SEASON)

    if len(train_seasons) < 2:
        print(f"  ⚠️  {test_season}: need ≥2 training seasons — skipping")
        return pd.DataFrame()

    # Val = most recent training season (for blend-weight tuning only)
    val_season = train_seasons[-1]

    train = df[df["season"].isin(train_seasons)].copy()
    val   = df[df["season"] == val_season].copy()
    test  = df[df["season"] == test_season].copy()

    if test.empty:
        print(f"  ⚠️  {test_season}: no test data in feature matrix — skipping")
        return pd.DataFrame()

    n_train = len(train)
    n_test  = len(test)
    print(f"\n{'='*62}")
    print(f"  Fold {test_season}  |  train: {train_seasons}  |  val: {val_season}")
    print(f"  Train games: {n_train:,}   Val: {len(val):,}   Test: {n_test:,}")

    # ── Feature filtering ──────────────────────────────────────────────────────
    spread_feats = [f for f in SPREAD_FEATURES if f in df.columns]
    totals_feats = [f for f in TOTALS_FEATURES if f in df.columns]
    win_feats    = [f for f in WIN_PROB_FEATURES if f in df.columns]

    X_tr_sp,  y_tr_sp  = train[spread_feats], train["point_diff"]
    X_val_sp, y_val_sp = val[spread_feats],   val["point_diff"]
    X_te_sp            = test[spread_feats]

    X_tr_tot,  y_tr_tot  = train[totals_feats], train["total_points"]
    X_val_tot, y_val_tot = val[totals_feats],   val["total_points"]
    X_te_tot              = test[totals_feats]

    X_tr_win,  y_tr_win  = train[win_feats], train["home_win"]
    X_val_win, y_val_win = val[win_feats],   val["home_win"]
    X_te_win              = test[win_feats]

    # ── Spread model ──────────────────────────────────────────────────────────
    ridge_sp = make_linear(alpha=10.0); ridge_sp.fit(X_tr_sp, y_tr_sp)
    gbm_sp   = make_gbm_regressor();   gbm_sp.fit(X_tr_sp, y_tr_sp)

    best_sp_rmse, best_sp_w1 = 999.0, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        rmse = np.sqrt(np.mean(
            (EnsembleRegressor(ridge_sp, gbm_sp, w1, w2).predict(X_val_sp) - y_val_sp) ** 2))
        if rmse < best_sp_rmse:
            best_sp_rmse, best_sp_w1 = rmse, w1
    sp_w2     = round(1 - best_sp_w1, 1)
    ens_sp    = EnsembleRegressor(ridge_sp, gbm_sp, best_sp_w1, sp_w2)

    # ── Totals model ──────────────────────────────────────────────────────────
    ridge_tot = make_linear(alpha=10.0); ridge_tot.fit(X_tr_tot, y_tr_tot)
    gbm_tot   = make_gbm_regressor();   gbm_tot.fit(X_tr_tot, y_tr_tot)

    best_tot_rmse, best_tot_w1 = 999.0, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        rmse = np.sqrt(np.mean(
            (EnsembleRegressor(ridge_tot, gbm_tot, w1, w2).predict(X_val_tot) - y_val_tot) ** 2))
        if rmse < best_tot_rmse:
            best_tot_rmse, best_tot_w1 = rmse, w1
    tot_w2    = round(1 - best_tot_w1, 1)
    ens_tot   = EnsembleRegressor(ridge_tot, gbm_tot, best_tot_w1, tot_w2)

    # ── Win-probability model ─────────────────────────────────────────────────
    gbm_win_base  = make_gbm_classifier(); gbm_win_base.fit(X_tr_win, y_tr_win)
    logit_win     = make_logistic(C=0.3);  logit_win.fit(X_tr_win,  y_tr_win)

    # Calibrate with isotonic regression (5-fold CV on training data)
    gbm_win_cal = CalibratedClassifierCV(make_gbm_classifier(), method="isotonic", cv=5)
    gbm_win_cal.fit(X_tr_win, y_tr_win)

    best_brier, best_w_w1 = 999.0, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        ens = EnsembleClassifier(gbm_win_cal, logit_win, w1, w2)
        b   = brier_score_loss(y_val_win, ens.predict_proba(X_val_win)[:, 1])
        if b < best_brier:
            best_brier, best_w_w1 = b, w1
    win_w2  = round(1 - best_w_w1, 1)
    ens_win = EnsembleClassifier(gbm_win_cal, logit_win, best_w_w1, win_w2)

    # ── Assemble predictions ──────────────────────────────────────────────────
    base_cols = ["game_id", "season", "week", "home_team", "away_team",
                 "home_points", "away_points", "point_diff", "total_points",
                 "spread", "over_under", "vegas_home_margin",
                 "home_win", "covered_spread", "went_over"]
    ml_cols = [c for c in ["home_moneyline", "away_moneyline"] if c in test.columns]
    out = test[base_cols + ml_cols].copy()

    out["pred_spread"]     = ens_sp.predict(X_te_sp)
    out["pred_total"]      = ens_tot.predict(X_te_tot)
    out["pred_home_win_p"] = ens_win.predict_proba(X_te_win)[:, 1]
    out["spread_edge"]     = out["pred_spread"] - out["vegas_home_margin"]
    out["totals_edge"]     = (out["pred_total"]
                              - pd.to_numeric(out["over_under"], errors="coerce"))
    out["training_cutoff"] = val_season  # last season in training window

    # ── Per-fold metrics ──────────────────────────────────────────────────────
    sp_ev  = evaluate_spread(test["point_diff"], out["pred_spread"])
    tot_ev = evaluate_totals(test["total_points"], out["pred_total"])
    brier  = brier_score_loss(test["home_win"], out["pred_home_win_p"])
    veg_sp = evaluate_spread(test["point_diff"], test["vegas_home_margin"], "Vegas")
    veg_tt = evaluate_totals(test["total_points"],
                              pd.to_numeric(test["over_under"], errors="coerce"), "Vegas")

    print(f"  Spread  — MAE {sp_ev['MAE']:5.2f}  R² {sp_ev['R2']:+.3f}"
          f"  Dir {sp_ev['Direction_Acc']:.1%}"
          f"  (Vegas MAE {veg_sp['MAE']:.2f}  R² {veg_sp['R2']:+.3f})")
    print(f"  Totals  — MAE {tot_ev['MAE']:5.2f}  R² {tot_ev['R2']:+.3f}"
          f"  (Vegas MAE {veg_tt['MAE']:.2f}  R² {veg_tt['R2']:+.3f})")
    print(f"  WinProb — Brier {brier:.4f}"
          f"  blend Spread {best_sp_w1:.0%}/GBM {sp_w2:.0%}"
          f"  Tot {best_tot_w1:.0%}/GBM {tot_w2:.0%}")

    return out


# ─── BACKTESTING SUMMARY ──────────────────────────────────────────────────────

def print_backtest_summary(df: pd.DataFrame):
    """Quick ATS & O/U win rates across all thresholds."""
    df = df.dropna(subset=["covered_spread", "went_over", "spread_edge"]).copy()
    df["covered_spread"] = df["covered_spread"].astype(int)
    df["went_over"]      = df["went_over"].astype(int)
    df["spread_edge"]    = pd.to_numeric(df["spread_edge"], errors="coerce")
    df["totals_edge"]    = pd.to_numeric(df.get("totals_edge",
                                                  pd.Series(dtype=float)), errors="coerce")

    print("\n" + "="*62)
    print("WALK-FORWARD BACKTEST SUMMARY  (flat -110 betting)")
    print("="*62)
    print(f"  {'Edge ≥':>8}  {'#Bets':>6}  {'Win%':>6}  {'P&L':>8}  {'ROI':>7}")
    print(f"  {'-'*50}")

    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        bets = []
        for _, r in df.iterrows():
            sp_e  = r["spread_edge"]
            tot_e = r.get("totals_edge", float("nan"))
            if pd.notna(sp_e) and abs(sp_e) >= thresh:
                home = sp_e > 0
                won  = (r["covered_spread"] == 1) if home else (r["covered_spread"] == 0)
                bets.append({"won": won, "pnl": 1.0 if won else -1.1, "type": "SP"})
            if pd.notna(tot_e) and abs(tot_e) >= thresh:
                over = tot_e > 0
                won  = (r["went_over"] == 1) if over else (r["went_over"] == 0)
                bets.append({"won": won, "pnl": 1.0 if won else -1.1, "type": "TOT"})
        if not bets:
            continue
        bdf  = pd.DataFrame(bets)
        n    = len(bdf)
        wp   = bdf["won"].mean() * 100
        fp   = bdf["pnl"].sum()
        roi  = fp / (n * 1.1) * 100
        mark = "  ◀" if thresh == 3.0 else ""
        print(f"  {thresh:>6.1f}pt  {n:>6,}  {wp:>5.1f}%  {fp:>+7.1f}u  {roi:>+6.1f}%{mark}")

    # Bet-type breakdown at 3pt
    print(f"\n  Bet-type breakdown at 3.0pt threshold:")
    for bet_type, label in [("SP","Spread"),("TOT","Total")]:
        sub = [b for _, r in df.iterrows()
               for edge, col, bt in [
                   (r["spread_edge"], "covered_spread", "SP"),
                   (r.get("totals_edge", float("nan")), "went_over", "TOT"),
               ]
               if bt == bet_type and pd.notna(edge) and abs(edge) >= 3.0
               for won in [(r[col]==1) if (edge>0) else (r[col]==0)]
               for b in [{"won": won, "pnl": 1.0 if won else -1.1}]]
        if not sub:
            continue
        bdf  = pd.DataFrame(sub)
        n    = len(bdf); wp = bdf["won"].mean()*100; fp = bdf["pnl"].sum()
        print(f"    {label:7s}: {n:4d} bets  {wp:.1f}%  {fp:+.1f}u")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          CFB Walk-Forward Out-of-Sample Backtester           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Test seasons: {FIRST_TEST}–{LAST_TEST}  (one model trained per fold)       ║")
    print(f"║  2020 always excluded from training (COVID season)           ║")
    print(f"║  Expected runtime: 4–7 minutes                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df = load_data()
    print(f"\nLoaded {len(df):,} FBS-vs-FBS games "
          f"({int(df['season'].min())}–{int(df['season'].max())})")

    all_folds = []
    for test_season in range(FIRST_TEST, LAST_TEST + 1):
        fold = run_fold(df, test_season)
        if not fold.empty:
            all_folds.append(fold)

    if not all_folds:
        print("\n❌ No predictions generated — check that feature_matrix.csv exists.")
        sys.exit(1)

    master = pd.concat(all_folds, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n\n✅ Saved {len(master):,} walk-forward predictions → {OUT_PATH.name}")
    print(f"   Seasons: {sorted(master['season'].unique())}")
    print(f"   Runtime: {elapsed/60:.1f} minutes")

    print_backtest_summary(master)
    print(f"\n→ Open the Streamlit app → Backtester tab to explore interactively.")
