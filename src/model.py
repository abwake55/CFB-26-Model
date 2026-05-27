"""
CFB Betting Model — Model Training
=====================================
Trains three models:
  1. Spread model   → predicts home team point differential
  2. Totals model   → predicts combined score
  3. Win prob model → predicts home team win probability

Uses walk-forward validation:
  Train: 2019–2022  |  Test: 2023–2024

Run:
    python3 src/model.py

Outputs:
    outputs/predictions/model_results.csv   — per-game predictions on test set
    outputs/charts/feature_importance.csv   — feature weights
    models/ (saved model files)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# ─── ENSEMBLE MODEL ──────────────────────────────────────────────────────────

class EnsembleRegressor:
    """
    Weighted blend of two sklearn-compatible regressors.

    Blending Ridge + GBM almost always outperforms either model alone:
    - Ridge is strong on linear signals (SP+, Elo differentials)
    - GBM captures non-linear interactions (QB portal changes × weak opponent, etc.)
    - 50/50 blend balances both and reduces overfitting vs. picking one

    This class is joblib-serializable and drop-in compatible with sklearn's predict().
    """
    def __init__(self, m1, m2, w1: float = 0.5, w2: float = 0.5):
        self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2

    def predict(self, X):
        return self.w1 * np.array(self.m1.predict(X)) + \
               self.w2 * np.array(self.m2.predict(X))

    def fit(self, X, y):
        """Not used directly — models are pre-fit. Kept for sklearn API compatibility."""
        return self


class EnsembleClassifier:
    """
    Weighted blend of two sklearn-compatible classifiers' probabilities.
    Both must implement predict_proba().
    """
    def __init__(self, m1, m2, w1: float = 0.5, w2: float = 0.5):
        self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2

    def predict_proba(self, X):
        p1 = np.array(self.m1.predict_proba(X))
        p2 = np.array(self.m2.predict_proba(X))
        return self.w1 * p1 + self.w2 * p2

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def fit(self, X, y):
        return self


DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR     = Path(__file__).parent.parent / "outputs" / "predictions"
CHART_DIR   = Path(__file__).parent.parent / "outputs" / "charts"
MODELS_DIR  = Path(__file__).parent.parent / "models"

for d in [OUT_DIR, CHART_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [2017, 2018, 2019, 2021, 2022]  # exclude 2020 (COVID distortion)
VAL_SEASONS   = [2023]        # held out for hyperparameter / weight tuning only
TEST_SEASONS  = [2024, 2025]  # final holdout — never used for any tuning decision

# ─── 1. LOAD & PREPARE DATA ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "feature_matrix.csv")

    # Filter to FBS-vs-FBS only
    df = df.dropna(subset=["home_sp_rating", "away_sp_rating"]).copy()

    # Add Elo differential (pre-game Elo from CFBD API)
    df["elo_diff"] = df["home_pregame_elo"] - df["away_pregame_elo"]

    # Numeric coercion
    for col in ["spread", "over_under", "home_pregame_elo", "away_pregame_elo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} FBS-vs-FBS games "
          f"({df['season'].min()}–{df['season'].max()})")
    return df


# ─── 2. FEATURE DEFINITIONS ──────────────────────────────────────────────────

# Features used to predict SPREAD (point differential, home - away)
SPREAD_FEATURES = [
    # SP+ ratings (season-long efficiency)
    "sp_diff",           # overall SP+ gap
    "sp_off_diff",       # offensive SP+ gap
    "sp_def_diff",       # defensive SP+ gap
    "home_sp_rating", "away_sp_rating",
    "home_sp_offense", "away_sp_offense",
    "home_sp_defense", "away_sp_defense",

    # Pre-game Elo ratings
    "elo_diff",
    "home_pregame_elo", "away_pregame_elo",

    # Rolling EPA (last 3 games — most recent form)
    "home_off_epa_roll3", "away_off_epa_roll3",
    "home_def_epa_roll3", "away_def_epa_roll3",
    "home_off_epa_pass_roll3", "away_off_epa_pass_roll3",
    "home_off_epa_rush_roll3", "away_off_epa_rush_roll3",

    # Rolling EPA (last 5 games — slightly longer window)
    "home_off_epa_roll5", "away_off_epa_roll5",
    "home_def_epa_roll5", "away_def_epa_roll5",

    # Season-to-date EPA
    "home_off_epa_ytd", "away_off_epa_ytd",
    "home_def_epa_ytd", "away_def_epa_ytd",

    # Derived differentials
    "epa_off_diff_roll3", "epa_def_diff_roll3",

    # Recruiting (4-year rolling composite)
    "recruiting_diff",
    "home_recruiting_4yr", "away_recruiting_4yr",

    # Game context
    "neutral_site", "conference_game",

    # Home field advantage (team-specific, computed from historical margins)
    "home_hfa", "away_hfa", "hfa_diff",

    # Schedule rest (days since last game)
    "home_rest_days", "away_rest_days", "rest_diff",

    # Line movement (sharp money signal)
    "line_movement", "line_moved_home", "line_moved_away",

    # ESPN FPI (independent composite rating)
    "fpi_diff", "home_fpi", "away_fpi",

    # SRS — adjusted point differential per game
    "srs_diff", "home_srs", "away_srs",

    # ── Transfer Portal (biggest unmodeled factor in modern CFB) ──────────
    # Net talent change via portal (positive = net gain vs opponent)
    "portal_net_rating_diff",
    "home_portal_net_rating", "away_portal_net_rating",
    # Absolute talent flows (team's own incoming/outgoing)
    "home_portal_talent_in",  "away_portal_talent_in",
    "home_portal_talent_out", "away_portal_talent_out",
    # Roster turnover volume
    "home_portal_net_count",  "away_portal_net_count",
    # QB-specific changes — single biggest position impact
    "home_portal_qb_in",  "away_portal_qb_in",
    "home_portal_qb_out", "away_portal_qb_out",
    # Recruiting quality of incoming transfers
    "home_portal_stars_in_avg", "away_portal_stars_in_avg",

    # ── WEPA (opponent-adjusted EPA) — better than raw EPA vs tough schedules ──
    "wepa_off_diff",           # home off WEPA minus away off WEPA
    "wepa_def_diff",           # home def WEPA minus away def WEPA
    "home_wepa_offense", "away_wepa_offense",
    "home_wepa_defense", "away_wepa_defense",
    # Success rates & explosiveness (opponent-adjusted)
    "wepa_success_off_diff", "wepa_success_def_diff",
    "wepa_explosiveness_diff",
    "home_wepa_success_off", "away_wepa_success_off",
    "home_wepa_success_def", "away_wepa_success_def",
    "home_wepa_explosiveness", "away_wepa_explosiveness",
    "home_wepa_explosiveness_def", "away_wepa_explosiveness_def",

    # ── Talent composite (247Sports roster ratings) ────────────────────────
    "talent_diff",             # home talent minus away talent
    "home_talent", "away_talent",

    # ── Havoc rate (defensive disruption) ─────────────────────────────────
    "havoc_diff",              # home havoc rate minus away (positive = home D more disruptive)
    "home_havoc_total", "away_havoc_total",
    "home_havoc_front_seven", "away_havoc_front_seven",
    # Offensive success rates (past season — predictive of future efficiency)
    "rush_sr_diff",
    "home_rush_success_rate", "away_rush_success_rate",
    "home_pass_success_rate", "away_pass_success_rate",
]

# Totals model uses both teams' offense AND defense independently
# (not just differentials — a game between two great offenses scores more)
TOTALS_FEATURES = [
    "home_sp_offense", "away_sp_offense",
    "home_sp_defense", "away_sp_defense",
    "home_off_epa_roll3", "away_off_epa_roll3",
    "home_def_epa_roll3", "away_def_epa_roll3",
    "home_off_epa_pass_roll3", "away_off_epa_pass_roll3",
    "home_off_epa_rush_roll3", "away_off_epa_rush_roll3",
    "home_off_epa_roll5", "away_off_epa_roll5",
    "home_def_epa_roll5", "away_def_epa_roll5",
    "home_off_epa_ytd", "away_off_epa_ytd",
    "home_def_epa_ytd", "away_def_epa_ytd",
    "home_sp_rating", "away_sp_rating",
    "home_hfa", "away_hfa",
    "neutral_site", "conference_game",

    # Weather (outdoor games only — big effect on totals)
    "wind_speed", "temp_avg", "precipitation", "is_dome",

    # Schedule rest
    "rest_diff",

    # Line movement
    "line_movement",

    # Additional ratings
    "fpi_diff", "home_fpi", "away_fpi",
    "srs_diff", "home_srs", "away_srs",

    # WEPA (total scoring context — both sides' adjusted efficiency matters for totals)
    "home_wepa_offense", "away_wepa_offense",
    "home_wepa_defense", "away_wepa_defense",
    "home_wepa_success_off", "away_wepa_success_off",
    "home_wepa_explosiveness", "away_wepa_explosiveness",
    "home_wepa_explosiveness_def", "away_wepa_explosiveness_def",

    # Talent (high-talent games tend to be lower variance)
    "home_talent", "away_talent", "talent_diff",

    # Havoc + success rates (both affect total scoring pace)
    "home_havoc_total", "away_havoc_total",
    "home_rush_success_rate", "away_rush_success_rate",
    "home_pass_success_rate", "away_pass_success_rate",
]

WIN_PROB_FEATURES = SPREAD_FEATURES  # same features, different target


# ─── 3. BUILD MODELS ─────────────────────────────────────────────────────────

def make_linear(alpha: float = 1.0):
    """Ridge regression pipeline with imputation + scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=alpha)),
    ])


def make_gbm_regressor():
    # HistGradientBoosting is scikit-learn's fast gradient booster —
    # handles missing values natively (no separate imputer needed),
    # comparable performance to XGBoost, zero extra dependencies.
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=2.0,
        random_state=42,
    )


def make_gbm_classifier():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=2.0,
        random_state=42,
    )


def make_logistic(C: float = 0.3):
    """
    Logistic regression pipeline — linear complement to GBM for win probability.

    C=0.3 is less aggressive than the previous 0.1 default.
    Lower C = more regularization = predictions pushed toward 50%.
    Too much regularization explains why models underpredict home win rate.
    """
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(C=C, max_iter=1000, random_state=42)),
    ])


def print_calibration_summary(y_true, probs, label: str, n_bins: int = 8):
    """
    Print a text calibration curve: for each predicted probability bucket,
    show what the actual win rate was. Well-calibrated = diagonal line.

    ECE (Expected Calibration Error) summarises the gap in one number —
    lower is better; 0 = perfect calibration.
    """
    try:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins,
                                                 strategy="quantile")
    except ValueError:
        return
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    print(f"\n  {label} calibration  (ECE = {ece:.4f})")
    print(f"  {'Predicted':>10}  {'Actual':>8}  {'Gap':>7}  Bar")
    for pred, actual in zip(mean_pred, frac_pos):
        gap  = actual - pred
        bar  = "▓" * int(round(actual * 30))
        flag = " ← over" if gap > 0.04 else (" ← under" if gap < -0.04 else "")
        print(f"  {pred:>9.1%}  {actual:>7.1%}  {gap:>+6.1%}  {bar}{flag}")


# ─── 4. EVALUATE ─────────────────────────────────────────────────────────────

def evaluate_spread(y_true, y_pred, label="Model"):
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # How often does the model pick the correct winner?
    acc  = ((y_true > 0) == (y_pred > 0)).mean()
    return {"label": label, "MAE": round(mae, 3),
            "R2": round(r2, 3), "Direction_Acc": round(acc, 3)}


def evaluate_totals(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return {"label": label, "MAE": round(mae, 3), "R2": round(r2, 3)}


def vegas_spread_baseline(df_test):
    """Vegas expected to be very good — use as the gold standard comparison."""
    return evaluate_spread(
        df_test["point_diff"], df_test["vegas_home_margin"], label="Vegas (baseline)"
    )


def vegas_totals_baseline(df_test):
    return evaluate_totals(
        df_test["total_points"], df_test["over_under"].astype(float), label="Vegas (baseline)"
    )


# ─── 5. FEATURE IMPORTANCE ───────────────────────────────────────────────────

def extract_importance(pipeline, feature_names, label=""):
    # Handle both raw estimators and sklearn Pipelines
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps["model"]
    else:
        model = pipeline
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    return pd.DataFrame({
        "feature": feature_names,
        "importance": imps,
        "model": label,
    }).sort_values("importance", ascending=False)


# ─── 6. MAIN TRAINING PIPELINE ───────────────────────────────────────────────

def train_and_evaluate():
    df = load_data()

    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val   = df[df["season"].isin(VAL_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"\nTrain: {len(train):,} games  ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
    print(f"Val:   {len(val):,}  games  ({VAL_SEASONS[0]}–{VAL_SEASONS[-1]})  ← weight tuning only")
    print(f"Test:  {len(test):,}  games  ({TEST_SEASONS[0]}–{TEST_SEASONS[-1]})  ← final holdout")

    # Only keep features that actually exist in the dataframe
    spread_feats = [f for f in SPREAD_FEATURES if f in df.columns]
    totals_feats = [f for f in TOTALS_FEATURES if f in df.columns]
    win_feats    = [f for f in WIN_PROB_FEATURES if f in df.columns]

    X_train_sp = train[spread_feats]
    y_train_sp = train["point_diff"]
    X_test_sp  = test[spread_feats]
    y_test_sp  = test["point_diff"]

    X_train_tot = train[totals_feats]
    y_train_tot = train["total_points"]
    X_test_tot  = test[totals_feats]
    y_test_tot  = test["total_points"]

    X_val_sp    = val[spread_feats]
    y_val_sp    = val["point_diff"]
    X_val_tot   = val[totals_feats]
    y_val_tot   = val["total_points"]

    X_train_win = train[win_feats]
    y_train_win = train["home_win"]
    X_val_win   = val[win_feats]
    y_val_win   = val["home_win"]
    X_test_win  = test[win_feats]
    y_test_win  = test["home_win"]

    # ── Train spread models ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("SPREAD MODEL  (predicting home team point differential)")
    print("="*55)

    ridge_sp = make_linear(alpha=10.0)
    ridge_sp.fit(X_train_sp, y_train_sp)

    gbm_sp = make_gbm_regressor()
    gbm_sp.fit(X_train_sp, y_train_sp)

    results_sp = []
    for pipe, label in [(ridge_sp, "Ridge"), (gbm_sp, "GradientBoost")]:
        preds = pipe.predict(X_test_sp)
        results_sp.append(evaluate_spread(y_test_sp, preds, label))
    results_sp.append(vegas_spread_baseline(test))

    print(pd.DataFrame(results_sp).to_string(index=False))

    # ── Train totals models ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("TOTALS MODEL  (predicting combined score)")
    print("="*55)

    ridge_tot = make_linear(alpha=10.0)
    ridge_tot.fit(X_train_tot, y_train_tot)

    gbm_tot = make_gbm_regressor()
    gbm_tot.fit(X_train_tot, y_train_tot)

    results_tot = []
    for pipe, label in [(ridge_tot, "Ridge"), (gbm_tot, "GradientBoost")]:
        preds = pipe.predict(X_test_tot)
        results_tot.append(evaluate_totals(y_test_tot, preds, label))
    results_tot.append(vegas_totals_baseline(test))

    print(pd.DataFrame(results_tot).to_string(index=False))

    # ── Train win probability model ────────────────────────────────────────
    print("\n" + "="*55)
    print("WIN PROBABILITY MODEL  (predicting home win %)")
    print("="*55)

    # Base classifiers
    gbm_win_base  = make_gbm_classifier()
    logit_win     = make_logistic(C=0.3)
    gbm_win_base.fit(X_train_win,  y_train_win)
    logit_win.fit(X_train_win,     y_train_win)

    # Calibrate GBM with isotonic regression using 5-fold CV on training data.
    # Isotonic calibration learns a monotone mapping from raw scores → calibrated probs,
    # fixing the systematic underestimation of high-confidence predictions.
    # cv=5 ensures we never calibrate on the same data used to train the base model.
    print("  Calibrating GBM with isotonic regression (5-fold CV)...")
    gbm_win_calib = CalibratedClassifierCV(make_gbm_classifier(),
                                           method="isotonic", cv=5)
    gbm_win_calib.fit(X_train_win, y_train_win)

    # Auto-tune ensemble weights on the VALIDATION set (2023) only.
    # The test set (2024-2025) is never used for any tuning decision — it is the
    # final, clean holdout for honest evaluation of the chosen model.
    best_brier, best_w1, best_ensemble = 999, 0.5, None
    weight_candidates = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
                         (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    for w1, w2 in weight_candidates:
        ens = EnsembleClassifier(gbm_win_calib, logit_win, w1=w1, w2=w2)
        b = brier_score_loss(y_val_win, ens.predict_proba(X_val_win)[:, 1])
        if b < best_brier:
            best_brier, best_w1, best_ensemble = b, w1, ens

    actual_win_rate = y_test_win.mean()
    best_w2 = round(1 - best_w1, 1)

    print(f"\n  {'Model':25s}  {'Brier':>7}  {'Avg pred':>9}  {'vs actual':>10}")
    print(f"  {'-'*60}")
    for label, mdl in [
        ("GBM (uncalibrated)",      gbm_win_base),
        ("GBM (calibrated)",        gbm_win_calib),
        ("Logistic (C=0.3)",        logit_win),
        (f"Ensemble {best_w1:.0%}/{best_w2:.0%} (best)", best_ensemble),
    ]:
        probs  = mdl.predict_proba(X_test_win)[:, 1]
        brier  = brier_score_loss(y_test_win, probs)
        avg    = probs.mean()
        gap    = avg - actual_win_rate
        marker = " ◀ SAVED" if mdl is best_ensemble else ""
        print(f"  {label:25s}  {brier:.4f}  {avg:>9.1%}  {gap:>+9.1%}{marker}")
    print(f"  {'Actual home win %':25s}           {actual_win_rate:>9.1%}")

    # Print calibration curves for uncalibrated vs calibrated ensemble
    print_calibration_summary(y_test_win,
                               gbm_win_base.predict_proba(X_test_win)[:, 1],
                               "GBM uncalibrated")
    print_calibration_summary(y_test_win,
                               best_ensemble.predict_proba(X_test_win)[:, 1],
                               f"Ensemble {best_w1:.0%}/{best_w2:.0%} calibrated")

    gbm_win = best_ensemble  # use best calibrated ensemble going forward

    # ── Auto-tune ensemble blend weights on validation set (2023) ─────────
    # Same discipline as win-prob: pick the blend on val, evaluate on test.
    # RMSE on val is minimised — lower = better calibrated predictions.
    weight_candidates = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
                         (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

    # Spread blend
    best_sp_rmse, best_sp_w1 = 999.0, 0.5
    for w1, w2 in weight_candidates:
        ens = EnsembleRegressor(ridge_sp, gbm_sp, w1=w1, w2=w2)
        val_rmse = np.sqrt(np.mean((ens.predict(X_val_sp) - y_val_sp) ** 2))
        if val_rmse < best_sp_rmse:
            best_sp_rmse, best_sp_w1 = val_rmse, w1
    best_sp_w2 = round(1 - best_sp_w1, 1)
    ensemble_sp = EnsembleRegressor(ridge_sp, gbm_sp, w1=best_sp_w1, w2=best_sp_w2)

    # ── Cross-calibrate win probability with spread-implied probability ────────
    # Spread model and win prob classifier are trained independently and can give
    # inconsistent signals (e.g. spread says coin-flip but classifier says 70/30).
    # Fix: convert spread prediction → implied win prob via N(0,σ), then blend.
    #   P(home wins | spread) = Φ(pred_spread / σ)
    # where σ = std of spread residuals on the val set.
    # Blend weight α is tuned on the val set to minimise Brier score.
    print("\n  Cross-calibrating spread model with win probability model...")
    from math import erf as _erf, sqrt as _msqrt
    def _norm_cdf(x): return 0.5 * (1 + _erf(float(x) / _msqrt(2)))

    sp_val_preds       = ensemble_sp.predict(X_val_sp)
    spread_sigma       = float(np.std(sp_val_preds - y_val_sp.values))
    spread_implied_val = np.array([_norm_cdf(p / spread_sigma) for p in sp_val_preds])
    classifier_val     = gbm_win.predict_proba(X_val_win)[:, 1]
    y_val_win_arr      = y_val_win.values

    best_cal_brier, best_alpha = 999.0, 0.5
    for a in [i / 10 for i in range(0, 11)]:
        blended = np.clip(a * spread_implied_val + (1 - a) * classifier_val, 1e-6, 1 - 1e-6)
        b = brier_score_loss(y_val_win_arr, blended)
        if b < best_cal_brier:
            best_cal_brier, best_alpha = b, a

    print(f"  Calibration: spread σ={spread_sigma:.2f} pts  "
          f"blend α={best_alpha:.1f} (spread) / {1-best_alpha:.1f} (classifier)  "
          f"val Brier={best_cal_brier:.4f}")

    import json as _json
    with open(MODELS_DIR / "win_prob_calibration.json", "w") as _f:
        _json.dump({"spread_sigma": spread_sigma, "blend_alpha": best_alpha}, _f, indent=2)

    print(f"\n  Spread ensemble: best val blend = Ridge {best_sp_w1:.0%} / GBM {best_sp_w2:.0%}"
          f"  (val RMSE {best_sp_rmse:.3f})")

    # Totals blend
    best_tot_rmse, best_tot_w1 = 999.0, 0.5
    for w1, w2 in weight_candidates:
        ens = EnsembleRegressor(ridge_tot, gbm_tot, w1=w1, w2=w2)
        val_rmse = np.sqrt(np.mean((ens.predict(X_val_tot) - y_val_tot) ** 2))
        if val_rmse < best_tot_rmse:
            best_tot_rmse, best_tot_w1 = val_rmse, w1
    best_tot_w2 = round(1 - best_tot_w1, 1)
    ensemble_tot = EnsembleRegressor(ridge_tot, gbm_tot, w1=best_tot_w1, w2=best_tot_w2)
    print(f"  Totals ensemble: best val blend = Ridge {best_tot_w1:.0%} / GBM {best_tot_w2:.0%}"
          f"  (val RMSE {best_tot_rmse:.3f})")

    # Evaluate best ensembles on test set
    ens_sp_preds  = ensemble_sp.predict(X_test_sp)
    ens_tot_preds = ensemble_tot.predict(X_test_tot)
    ens_sp_label  = f"Ensemble ({best_sp_w1:.0%}/{best_sp_w2:.0%})"
    ens_tot_label = f"Ensemble ({best_tot_w1:.0%}/{best_tot_w2:.0%})"
    ens_sp_result  = evaluate_spread(y_test_sp,  ens_sp_preds,  ens_sp_label)
    ens_tot_result = evaluate_totals(y_test_tot, ens_tot_preds, ens_tot_label)
    results_sp.append(ens_sp_result)
    results_tot.append(ens_tot_result)

    print("\nUpdated spread results (with ensemble):")
    print(pd.DataFrame(results_sp).to_string(index=False))
    print("\nUpdated totals results (with ensemble):")
    print(pd.DataFrame(results_tot).to_string(index=False))

    best_sp_pipe  = ensemble_sp
    best_tot_pipe = ensemble_tot
    print(f"\nSaving ensemble model:"
          f" Spread Ridge {best_sp_w1:.0%}/GBM {best_sp_w2:.0%},"
          f" Totals Ridge {best_tot_w1:.0%}/GBM {best_tot_w2:.0%}.")

    # ── Feature importance (from GBM component of ensemble) ───────────────
    print("\n" + "="*55)
    print("TOP 15 SPREAD FEATURES  (GBM component importance)")
    print("="*55)
    # EnsembleRegressor doesn't expose feature_importances_ directly —
    # extract from the GBM sub-model (best_sp_pipe.m2 = gbm_sp)
    imp_df = extract_importance(best_sp_pipe.m2, spread_feats, "GBM component")
    if not imp_df.empty:
        print(imp_df.head(15)[["feature", "importance"]].to_string(index=False))
        imp_df.to_csv(CHART_DIR / "feature_importance_spread.csv", index=False)

    # ── Save per-game predictions on test set ─────────────────────────────
    # Base columns — include moneylines if present (needed for moneyline backtesting)
    base_cols = ["game_id", "season", "week", "home_team", "away_team",
                 "home_points", "away_points", "point_diff", "total_points",
                 "spread", "over_under", "vegas_home_margin",
                 "home_win", "covered_spread", "went_over"]
    ml_cols = [c for c in ["home_moneyline", "away_moneyline"] if c in test.columns]
    results_df = test[base_cols + ml_cols].copy()

    results_df["pred_spread"]      = best_sp_pipe.predict(X_test_sp)
    results_df["pred_total"]       = best_tot_pipe.predict(X_test_tot)
    results_df["pred_home_win_p"]  = gbm_win.predict_proba(X_test_win)[:, 1]

    # Model edge vs Vegas line.
    # Convention: pred_spread and vegas_home_margin are both expressed as
    # "home team margin" (positive = home wins). A positive spread_edge means
    # the model is MORE bullish on the home team than Vegas → bet home to cover.
    # A negative edge means model favors away → bet away to cover.
    # This matches the convention used in predict.py and app.py.
    results_df["spread_edge"] = (
        results_df["pred_spread"] - results_df["vegas_home_margin"]
    )

    results_df.to_csv(OUT_DIR / "model_results.csv", index=False)
    print(f"\n✅ Saved {len(results_df):,} game predictions → outputs/predictions/model_results.csv")

    # ── Save models ────────────────────────────────────────────────────────
    joblib.dump(best_sp_pipe,  MODELS_DIR / "spread_model.pkl")
    joblib.dump(best_tot_pipe, MODELS_DIR / "totals_model.pkl")
    joblib.dump(gbm_win,       MODELS_DIR / "win_prob_model.pkl")

    # Also save the feature lists so we know what to feed the models later
    import json
    with open(MODELS_DIR / "feature_lists.json", "w") as f:
        json.dump({
            "spread":   spread_feats,
            "totals":   totals_feats,
            "win_prob": win_feats,
        }, f, indent=2)

    print(f"✅ Saved models → models/")
    print(f"\nNext step: run python3 src/backtester.py to simulate historical betting performance.")

    return results_df


# ─── 7. QUICK SANITY CHECK ───────────────────────────────────────────────────

def show_sample_predictions(results_df: pd.DataFrame, n: int = 15):
    """Show a sample of predictions vs Vegas to build intuition."""
    print("\n" + "="*90)
    print("SAMPLE PREDICTIONS vs VEGAS (2023–2024 test set)")
    print("="*90)

    sample = results_df.sample(n, random_state=42).sort_values(["season", "week"])
    for _, row in sample.iterrows():
        model_line = row["pred_spread"]
        vegas_line = row["vegas_home_margin"]
        actual     = row["point_diff"]
        edge       = row["spread_edge"]
        flag       = "◀ BET?" if abs(edge) >= 3 else ""
        print(
            f"  {row['season']} Wk{int(row['week']):02d}  "
            f"{row['home_team']:18s} vs {row['away_team']:18s}  |  "
            f"Model: {model_line:+5.1f}  Vegas: {vegas_line:+5.1f}  "
            f"Edge: {edge:+5.1f}  Actual: {actual:+5.1f}  {flag}"
        )


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = train_and_evaluate()
    show_sample_predictions(results)
