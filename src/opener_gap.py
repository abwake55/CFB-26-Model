"""Opener-gap signal (adopted 2026-11-24 after scripts/research walk-forward).

A market-free home-margin model (main SPREAD_FEATURES + point-in-time
in-season ratings) compared with the OPENING spread:

    opener_gap_edge = pred_indep_margin + spread_open      (home perspective)
    flag_opener_gap = |opener_gap_edge| >= THRESHOLD

The research (docs/spread_research_2026-11-24.md) chose the in-season rating
settings and THRESHOLD on 2019-2023 only. On the 2024-25 holdout, bets at the
opener went 167-155-8 (51.9%, -1.0% ROI) with mean CLV +0.42 pts. The main
model at its dev-chosen threshold went 26-33 (-15.9% ROI, CLV -0.08). This
is a better opener-stage signal than the main model, but it is NOT proven
profitable: treat flags as paper/small-stake until live CLV and ATS confirm.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "opener_gap_model.pkl"
META_PATH = ROOT / "models" / "opener_gap_meta.json"
RATING_PARAMS = dict(lam_ppa=2.0, lam_margin=1.5, prior_regress=0.8, half_life_days=None)
THRESHOLD = 5.0
VERSION = "opener_gap_v1"


def inseason_features(extra_games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time in-season ratings for every game in master_games plus any
    `extra_games` (unplayed schedule rows: game_id, season, start_date,
    home_team, away_team, neutral_site)."""
    import inseason_ratings as ir
    proc = ROOT / "data" / "processed"
    games = pd.read_csv(proc / "master_games.csv", low_memory=False).drop_duplicates("game_id")
    if extra_games is not None and len(extra_games):
        x = extra_games[["game_id", "season", "start_date", "home_team", "away_team",
                         "neutral_site"]].copy()
        x = x[~x["game_id"].isin(games["game_id"])]
        x["point_diff"] = np.nan
        games = pd.concat([games, x], ignore_index=True)
    ppa = pd.read_csv(proc / "master_ppa_games.csv")
    r = ir.build_inseason_ratings(games, ppa, **RATING_PARAMS)
    return r.drop(columns=["inseason_version"], errors="ignore")


def feature_names(spread_features) -> list:
    import inseason_ratings as ir
    return list(spread_features) + [f for f in ir.INSEASON_SPREAD_FEATURES
                                    if f not in spread_features]


def train(df: pd.DataFrame, spread_features, sample_weights_fn, make_linear, gbm_params) -> dict:
    """Fit on all completed non-2020 seasons; blend weight picked on the latest season."""
    import lightgbm as lgb
    feats = feature_names(spread_features)
    d = df.merge(inseason_features(), on="game_id", how="left")
    d = d[d["season"] != 2020].dropna(subset=["point_diff"])
    feats = [f for f in feats if f in d.columns]
    last = int(d["season"].max())
    tr, va = d[d["season"] < last], d[d["season"] == last]

    def fit(part):
        r = make_linear(alpha=10.0).fit(part[feats], part["point_diff"])
        g = lgb.LGBMRegressor(**gbm_params).fit(
            part[feats], part["point_diff"], sample_weight=sample_weights_fn(part["season"], decay=0.3))
        return r, g

    r, g = fit(tr)
    best, bw = 1e9, 0.5
    for w in [i / 10 for i in range(2, 9)]:
        rmse = float(np.sqrt(np.mean((w * r.predict(va[feats]) + (1 - w) * g.predict(va[feats])
                                      - va["point_diff"]) ** 2)))
        if rmse < best:
            best, bw = rmse, w
    r, g = fit(d)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"ridge": r, "gbm": g, "w_ridge": bw, "features": feats}, f)
    meta = {"version": VERSION, "threshold": THRESHOLD, "rating_params": RATING_PARAMS,
            "w_ridge": bw, "val_season": last, "val_rmse_pre_refit": round(best, 4),
            "n_train": int(len(d)), "features": feats,
            "data_asof": str(pd.to_datetime(d["start_date"], utc=True, errors="coerce",
                                            format="mixed").max())}
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"  opener-gap model: {len(d):,} games, ridge weight {bw}, val {last} RMSE {best:.3f}")
    return meta


def predict(df: pd.DataFrame) -> pd.Series:
    """Market-free home-margin prediction; df must carry the model features
    (missing ones are imputed by the pipeline)."""
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    X = pd.DataFrame({c: (df[c] if c in df.columns else np.nan) for c in m["features"]}, index=df.index)
    return pd.Series(m["w_ridge"] * m["ridge"].predict(X)
                     + (1 - m["w_ridge"]) * m["gbm"].predict(X), index=df.index)


def attach(out: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Add pred_indep_margin, opener_gap_line, opener_gap_edge, flag_opener_gap.
    The bet line is spread_open when present (the decision is made at the
    opener), else the current spread."""
    if not MODEL_PATH.exists():
        return out
    ff = feature_frame.merge(inseason_features(feature_frame), on="game_id", how="left") \
        if "is_margin_diff" not in feature_frame.columns else feature_frame
    ff = ff.drop_duplicates("game_id").set_index("game_id")
    pred = predict(ff)
    out = out.copy()
    out["pred_indep_margin"] = out["game_id"].map(pred)
    so = pd.to_numeric(out.get("spread_open"), errors="coerce")
    sp = pd.to_numeric(out.get("spread"), errors="coerce")
    out["opener_gap_line"] = so.where(so.notna(), sp)
    out["opener_gap_edge"] = out["pred_indep_margin"] + out["opener_gap_line"]
    out["flag_opener_gap"] = out["opener_gap_edge"].abs() >= THRESHOLD
    return out
