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

from preseason import apply_preseason_shrinkage, DEFAULT_SPREAD_SIGMA
from model import (
    load_data,
    EnsembleRegressor, EnsembleClassifier,
    MarketAnchoredEnsemble, tune_anchor_weights,
    make_linear, make_gbm_regressor, make_gbm_classifier, make_logistic,
    SPREAD_FEATURES, TOTALS_FEATURES, WIN_PROB_FEATURES,
    evaluate_spread, evaluate_totals, make_sample_weights, tune_gbm_params,
)

from validation import split_seasons, temporal_calibrator
from inference import adjust_predictions, tune_probability_blend
from betting import settle
import lightgbm as lgb

# ─── CONFIG ───────────────────────────────────────────────────────────────────

COVID_SEASON = 2020          # always excluded from training (attendance anomaly)
FIRST_TEST   = 2019          # earliest season we predict (needs 2017+2018 to train)
LAST_TEST    = 2025
OUT_PATH     = ROOT / "outputs" / "predictions" / "walk_forward_results.csv"

BLEND_CANDIDATES = [(w / 10, 1 - w / 10) for w in range(2, 9)]


# ─── SINGLE FOLD ──────────────────────────────────────────────────────────────

def run_fold(df: pd.DataFrame, test_season: int, n_trials: int = 50) -> pd.DataFrame:
    """
    Train on all prior non-COVID seasons, tune on most recent prior season,
    predict on test_season. Returns a DataFrame of per-game predictions.
    """
    try:
        train, val, test = split_seasons(df, test_season)
    except ValueError as exc:
        print(f"  {test_season}: {exc} — skipping")
        return pd.DataFrame()
    train_seasons = sorted(train["season"].unique())
    val_season = int(val["season"].max())

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

    # Spread target: residual vs Vegas line (same reframe as model.py).
    # pred_spread = vegas_home_margin + predicted_residual.
    vm_tr  = pd.to_numeric(train["vegas_home_margin"], errors="coerce")
    vm_val = pd.to_numeric(val["vegas_home_margin"],   errors="coerce")
    vm_te  = pd.to_numeric(test["vegas_home_margin"],  errors="coerce")

    X_tr_sp,  y_tr_sp  = train[spread_feats], train["point_diff"] - vm_tr
    X_val_sp, y_val_sp = val[spread_feats],   val["point_diff"]   - vm_val
    X_te_sp            = test[spread_feats]

    X_tr_tot  = train[totals_feats]
    X_val_tot = val[totals_feats]
    X_te_tot  = test[totals_feats]

    # Totals target: deviation from line (same reframe as model.py)
    ou_tr  = pd.to_numeric(train["over_under"], errors="coerce")
    ou_val = pd.to_numeric(val["over_under"],   errors="coerce")
    ou_te  = pd.to_numeric(test["over_under"],  errors="coerce")
    y_tr_tot  = train["total_points"] - ou_tr   # deviation: train
    y_val_tot = val["total_points"]   - ou_val  # deviation: val (blend tuning)
    # y_test_tot = test["total_points"]          # actual: final eval (below)

    X_tr_win,  y_tr_win  = train[win_feats], train["home_win"]
    X_val_win, y_val_win = val[win_feats],   val["home_win"]
    X_te_win              = test[win_feats]

    # ── Spread model ──────────────────────────────────────────────────────────
    ridge_sp = make_linear(alpha=10.0); ridge_sp.fit(X_tr_sp, y_tr_sp)
    sw = make_sample_weights(train["season"], decay=0.3)
    sp_params = tune_gbm_params(X_tr_sp, y_tr_sp, X_val_sp, y_val_sp,
                                sample_weight=sw, n_trials=n_trials)
    gbm_sp = lgb.LGBMRegressor(**sp_params)
    gbm_sp.fit(X_tr_sp, y_tr_sp, sample_weight=sw)

    best_sp_rmse, best_sp_w1 = 999.0, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        rmse = np.sqrt(np.mean(
            (EnsembleRegressor(ridge_sp, gbm_sp, w1, w2).predict(X_val_sp) - y_val_sp) ** 2))
        if rmse < best_sp_rmse:
            best_sp_rmse, best_sp_w1 = rmse, w1
    sp_w2    = round(1 - best_sp_w1, 1)
    # Residual target is anchored to the market by construction — no
    # MarketAnchoredEnsemble wrapper needed.
    ens_sp   = EnsembleRegressor(ridge_sp, gbm_sp, best_sp_w1, sp_w2)

    # ── Totals model ──────────────────────────────────────────────────────────
    ridge_tot = make_linear(alpha=10.0); ridge_tot.fit(X_tr_tot, y_tr_tot)
    tot_params = tune_gbm_params(X_tr_tot, y_tr_tot, X_val_tot, y_val_tot,
                                 n_trials=n_trials)
    gbm_tot = lgb.LGBMRegressor(**tot_params)
    gbm_tot.fit(X_tr_tot, y_tr_tot)

    best_tot_rmse, best_tot_w1 = 999.0, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        rmse = np.sqrt(np.mean(
            (EnsembleRegressor(ridge_tot, gbm_tot, w1, w2).predict(X_val_tot) - y_val_tot) ** 2))
        if rmse < best_tot_rmse:
            best_tot_rmse, best_tot_w1 = rmse, w1
    tot_w2    = round(1 - best_tot_w1, 1)
    ens_tot   = EnsembleRegressor(ridge_tot, gbm_tot, best_tot_w1, tot_w2)

    # ── Win-probability model ─────────────────────────────────────────────────
    win_params = tune_gbm_params(X_tr_win, y_tr_win, X_val_win, y_val_win,
                                 sample_weight=sw, n_trials=n_trials, task="classification")
    logit_win = make_logistic(C=0.3); logit_win.fit(X_tr_win, y_tr_win)
    gbm_win_cal = temporal_calibrator(lgb.LGBMClassifier(**win_params), train["season"])
    gbm_win_cal.fit(X_tr_win, y_tr_win, sample_weight=sw)

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

    out["pred_spread"]     = vm_te.values + ens_sp.predict(X_te_sp)  # residual → margin
    out["pred_total"]      = ou_te.values + ens_tot.predict(X_te_tot)  # deviation → actual
    out["pred_home_win_p"] = ens_win.predict_proba(X_te_win)[:, 1]

    calibration = tune_probability_blend(
        vm_val.values + ens_sp.predict(X_val_sp), val["point_diff"].values,
        ens_win.predict_proba(X_val_win)[:, 1], y_val_win.values)
    out = adjust_predictions(out, calibration, probability_col="pred_home_win_p")
    out["training_cutoff"] = int(train["season"].max())
    out["validation_season"] = val_season
    out["evaluation_version"] = "season_holdout_v2"
    out["optimizer"] = "optuna" if __import__("model").OPTUNA_AVAILABLE else "fixed_fallback"

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
    """Price-aware flat-stake summary, treating pushes as returned stakes."""
    print("\nWALK-FORWARD SUMMARY (-110; pushes excluded from win rate)")
    for threshold in [2., 3., 4., 5., 6.]:
        for kind, edge_col, actual_col, line_col in [
            ("SP", "spread_edge", "point_diff", "vegas_home_margin"),
            ("TOT", "totals_edge", "total_points", "over_under")]:
            selected = df[df[edge_col].abs() >= threshold]
            settled = [settle(r[actual_col], r[line_col], 1 if r[edge_col] > 0 else -1)
                       for _, r in selected.iterrows()]
            if not settled:
                continue
            wins = sum(r[0] == "win" for r in settled)
            losses = sum(r[0] == "loss" for r in settled)
            pushes = sum(r[0] == "push" for r in settled)
            profit = sum(r[1] for r in settled)
            risk = sum(r[2] for r in settled)
            hit = wins / (wins + losses) if wins + losses else float("nan")
            print(f"  {kind} edge>={threshold:.1f}: {wins}W-{losses}L-{pushes}P "
                  f"hit={hit:.1%} profit={profit:+.1f}u ROI={profit/risk:+.1%}")


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

    # Regenerate the CORE portfolio history so the app's landing-page
    # equity curve always reflects the latest walk-forward run.
    try:
        import subprocess
        subprocess.run([sys.executable,
                        str(Path(__file__).parent / "build_core_history.py")],
                       check=True)
    except Exception as e:
        print(f"[WARN] core_history rebuild failed: {e} — "
              f"run scripts/build_core_history.py manually")

    print(f"\n→ Open the Streamlit app → Backtester tab to explore interactively.")
