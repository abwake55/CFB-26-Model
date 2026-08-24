#!/usr/bin/env python3
"""
Head-to-head: does the DEPLOYED totals config actually validate better?
======================================================================
The published CORE number is produced by walk_forward.py, which builds the
totals GBM with FIXED hyperparameters and NO sample weights. Production
(src/model.py) ships a different model: Optuna-tuned hyperparameters plus
exponential time-decay sample weights. So the advertised hit rate describes a
configuration that is not the one serving picks.

This script walk-forwards BOTH configurations over identical folds and scores
them on the metric that actually matters — the CORE unders portfolio — so the
choice is made on evidence instead of preference.

Only the totals model is rebuilt: CORE is a totals-only strategy, so the
spread and win-probability models cannot affect its hit rate. Everything else
(folds, features, target, ridge, blend search) is held constant; the ONLY
things that vary are GBM hyperparameters and sample weights.

    python3 scripts/config_headtohead.py            # both configs
    python3 scripts/config_headtohead.py --trials 25  # faster, coarser tune

Writes outputs/predictions/config_headtohead.csv (per-game, both configs).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import (  # noqa: E402
    load_data, EnsembleRegressor, make_linear, make_gbm_regressor,
    make_sample_weights, tune_gbm_params, TOTALS_FEATURES,
)
import lightgbm as lgb  # noqa: E402

COVID_SEASON = 2020
FIRST_TEST, LAST_TEST = 2019, 2025
BLEND_CANDIDATES = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
               "Big East", "FBS Independents"}


def _fit_totals(X_tr, y_tr, X_val, y_val, train_seasons_col, config, n_trials):
    """Return a fitted ridge+GBM ensemble for one fold under `config`."""
    ridge = make_linear(alpha=10.0)
    ridge.fit(X_tr, y_tr)

    # 2x2 decomposition: {fixed params, Optuna} x {no weights, time-decay}
    use_optuna = config in ("production", "optuna_only")
    use_weights = config in ("production", "weights_only")
    sw = make_sample_weights(train_seasons_col, decay=0.3) if use_weights else None
    if use_optuna:
        params = tune_gbm_params(X_tr, y_tr, X_val, y_val,
                                 sample_weight=sw, n_trials=n_trials)
        gbm = lgb.LGBMRegressor(**params)
    else:
        gbm = make_gbm_regressor()
    gbm.fit(X_tr, y_tr, sample_weight=sw)

    # Identical blend search for both configs.
    best_rmse, best_w1 = 9e9, 0.5
    for w1, w2 in BLEND_CANDIDATES:
        rmse = np.sqrt(np.mean(
            (EnsembleRegressor(ridge, gbm, w1, w2).predict(X_val) - y_val) ** 2))
        if rmse < best_rmse:
            best_rmse, best_w1 = rmse, w1
    return EnsembleRegressor(ridge, gbm, best_w1, round(1 - best_w1, 1))


def run(df: pd.DataFrame, config: str, n_trials: int) -> pd.DataFrame:
    feats = [f for f in TOTALS_FEATURES if f in df.columns]
    out = []
    for test_season in range(FIRST_TEST, LAST_TEST + 1):
        # Val must be a HOLDOUT, disjoint from train — mirroring production
        # (TRAIN=[2017-19,2021-23], VAL=[2024], TEST=[2025]). walk_forward.py
        # instead sets val = train_seasons[-1], i.e. val is INSIDE train; tuning
        # hyperparameters against that drives Optuna to maximally overfit
        # (observed: val MAE 0.0012, lr=0.099, leaves=96), which would make the
        # production config look far worse than it is.
        prior = sorted(s for s in range(2017, test_season) if s != COVID_SEASON)
        if len(prior) < 3:
            continue
        val_season = prior[-1]
        train_seasons = prior[:-1]
        train = df[df["season"].isin(train_seasons)].copy()
        val = df[df["season"] == val_season].copy()
        test = df[df["season"] == test_season].copy()
        if test.empty:
            continue

        ou_tr = pd.to_numeric(train["over_under"], errors="coerce")
        ou_val = pd.to_numeric(val["over_under"], errors="coerce")
        ou_te = pd.to_numeric(test["over_under"], errors="coerce")

        t0 = time.time()
        ens = _fit_totals(train[feats], train["total_points"] - ou_tr,
                          val[feats], val["total_points"] - ou_val,
                          train["season"], config, n_trials)
        pred_total = ou_te + ens.predict(test[feats])
        print(f"    {config:10s} fold {test_season}  train={train_seasons} "
              f"val={val_season} n={len(test):4d} ({time.time()-t0:.0f}s)")

        out.append(pd.DataFrame({
            "game_id": test["game_id"].values,
            "season": test["season"].values,
            "over_under": ou_te.values,
            "total_points": test["total_points"].values,
            "pred_total": pred_total.values,
            "home_conference": test.get("home_conference"),
            "away_conference": test.get("away_conference"),
            "wind_speed": test.get("wind_speed"),
            "is_dome": test.get("is_dome"),
            "config": config,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def core_stats(d: pd.DataFrame) -> dict:
    """CORE gate: under, edge 2-7, power-conf, wind<15, total>=48."""
    d = d.dropna(subset=["over_under", "pred_total", "total_points"]).copy()
    d = d[d["total_points"] != d["over_under"]]
    edge = d["pred_total"] - d["over_under"]
    power = (d["home_conference"].isin(POWER_CONFS)
             | d["away_conference"].isin(POWER_CONFS))
    windy = (~d["is_dome"].fillna(0).astype(bool)) & (d["wind_speed"].fillna(0) >= 15)
    m = (edge <= -2) & (edge >= -7) & power & ~windy & (d["over_under"] >= 48)
    sel = d[m]
    win = sel["total_points"] < sel["over_under"]
    n = len(sel)
    if n == 0:
        return {"n": 0}
    h = win.mean()
    per = sel.assign(_w=win).groupby("season")["_w"].mean()
    return {
        "n": n, "hit": h, "roi": h * 0.909 - (1 - h),
        "z": (h - 0.5) * 2 * np.sqrt(n),
        "seasons_profitable": int((per > 0.5238).sum()),
        "seasons": len(per),
        "mae": float((sel["pred_total"] - sel["total_points"]).abs().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50,
                    help="Optuna trials per fold for the production config")
    args = ap.parse_args()

    df = load_data()
    print(f"Feature matrix: {len(df):,} games\n")

    frames = []
    for config in ("fixed", "weights_only", "optuna_only", "production"):
        print(f"  === {config} ===")
        frames.append(run(df, config, args.trials))
    allf = pd.concat(frames, ignore_index=True)
    dest = ROOT / "outputs" / "predictions" / "config_headtohead.csv"
    allf.to_csv(dest, index=False)

    print(f"\n{'='*74}")
    print("CORE unders portfolio — deployed config vs validated config")
    print(f"{'='*74}")
    print(f"{'config':12s} {'n':>5s} {'hit':>7s} {'ROI':>8s} {'z':>7s} "
          f"{'seasons':>9s} {'totalsMAE':>10s}")
    for config in ("fixed", "weights_only", "optuna_only", "production"):
        s = core_stats(allf[allf["config"] == config])
        if not s.get("n"):
            print(f"{config:12s}  (no bets)"); continue
        print(f"{config:12s} {s['n']:5d} {s['hit']:6.3f} {s['roi']:+7.1%} "
              f"{s['z']:+7.2f} {s['seasons_profitable']:5d}/{s['seasons']} "
              f"{s['mae']:10.2f}")
    print(f"\nwrote {dest.name}")


if __name__ == "__main__":
    main()
