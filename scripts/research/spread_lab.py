"""Spread research lab: walk-forward variants of the spread model.

Spread-only replica of scripts/walk_forward.run_fold (Ridge a=10 + LightGBM
defaults, val-RMSE blend, 0.3 season decay). Trains only on seasons before
the test season (2020 never trained on). All features are pre-kickoff.

Outputs per-game predictions to outputs/research/lab_<name>.csv with:
  pred_resid_close  model residual vs closing line (main model target)
  pred_resid_open   residual vs opener (opener-anchored target), if fitted
  pred_indep        market-free margin prediction, if fitted
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import model as M  # noqa: E402
from validation import split_seasons  # noqa: E402

OUT = ROOT / "outputs/research"
BLEND = [(w / 10, 1 - w / 10) for w in range(2, 9)]
TEST_SEASONS = list(range(2019, 2026))


def load_base() -> pd.DataFrame:
    df = M.load_data()
    df["open_line"] = pd.to_numeric(df.get("spread_open"), errors="coerce")
    df["open_src"] = np.where(df["open_line"].notna(), "consensus", None)
    prov = ROOT / "data/research/cfbd_lines_by_provider.csv"
    if prov.exists():
        p = pd.read_csv(prov, low_memory=False)
        p = p.dropna(subset=["spread_open"])
        med = p.groupby("game_id")["spread_open"].median().rename("prov_open")
        df = df.merge(med, left_on="game_id", right_index=True, how="left")
        fill = df["open_line"].isna() & df["prov_open"].notna()
        df.loc[fill, "open_line"] = df.loc[fill, "prov_open"]
        df.loc[fill, "open_src"] = "provider_median"
        df = df.drop(columns="prov_open")
    df["close_line"] = pd.to_numeric(df["spread"], errors="coerce")
    return df


def _fit(train, val, test, feats, ytr, yval):
    mtr, mval = ytr.notna(), yval.notna()
    Xtr, Xval = train.loc[mtr, feats], val.loc[mval, feats]
    ridge = M.make_linear(alpha=10.0).fit(Xtr, ytr[mtr])
    sw = M.make_sample_weights(train.loc[mtr, "season"], decay=0.3)
    params = M.tune_gbm_params(Xtr, ytr[mtr], Xval, yval[mval], sample_weight=sw, n_trials=0)
    gbm = lgb.LGBMRegressor(**params).fit(Xtr, ytr[mtr], sample_weight=sw)
    best, bw = 1e9, 0.5
    pr_v, pg_v = ridge.predict(Xval), gbm.predict(Xval)
    for w1, w2 in BLEND:
        rmse = np.sqrt(np.mean((w1 * pr_v + w2 * pg_v - yval[mval]) ** 2))
        if rmse < best:
            best, bw = rmse, w1
    Xte = test[feats]
    return bw * ridge.predict(Xte) + (1 - bw) * gbm.predict(Xte)


def run(df, feats, targets=("close",), name="variant", seasons=TEST_SEASONS):
    feats = [f for f in feats if f in df.columns]
    outs = []
    for s in seasons:
        train, val, test = split_seasons(df, s)
        o = test[["game_id", "season", "week", "home_team", "away_team",
                  "home_conference", "away_conference", "conference_game", "neutral_site",
                  "point_diff", "close_line", "open_line", "open_src"]].copy()
        if "close" in targets:
            o["pred_resid_close"] = _fit(train, val, test, feats,
                                         train.point_diff + train.close_line,
                                         val.point_diff + val.close_line)
        if "open" in targets and train.open_line.notna().sum() >= 300 \
                and val.open_line.notna().sum() >= 300:
            o["pred_resid_open"] = _fit(train, val, test, feats,
                                        train.point_diff + train.open_line,
                                        val.point_diff + val.open_line)
        elif "open" in targets:
            o["pred_resid_open"] = np.nan
        if "indep" in targets:
            o["pred_indep"] = _fit(train, val, test, feats, train.point_diff, val.point_diff)
        outs.append(o)
        print(f"  {name} {s}: test {len(test)} train {len(train)} "
              f"(open in train {int(train.open_line.notna().sum())})", flush=True)
    out = pd.concat(outs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"lab_{name}.csv", index=False)
    return out


# ── grading ──────────────────────────────────────────────────────────────────
def grade(d: pd.DataFrame, edge: pd.Series, bet_line: pd.Series, thr: float,
          extra_mask=None) -> dict:
    """Flat -110 bets on sign(edge) when |edge| >= thr, graded at bet_line
    (home spread convention). CLV vs close in points when bet_line != close."""
    m = edge.abs().ge(thr) & bet_line.notna() & d["point_diff"].notna()
    if extra_mask is not None:
        m &= extra_mask.fillna(False)
    side = np.sign(edge[m])
    res = side * (d.loc[m, "point_diff"] + bet_line[m])
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    units = w - 1.1 * l
    clv = side * (bet_line[m] - d.loc[m, "close_line"])
    return {"n": int(m.sum()), "w": w, "l": l, "p": p,
            "win%": round(100 * w / (w + l), 2) if w + l else np.nan,
            "units": round(units, 1),
            "roi%": round(100 * units / (1.1 * (w + l)), 2) if w + l else np.nan,
            "clv": round(float(clv.mean()), 3) if len(clv) else np.nan,
            "clv+%": round(100 * float((clv > 0).mean()), 1) if len(clv) else np.nan}


DEV = [2019, 2020, 2021, 2022, 2023]
HOLD = [2024, 2025]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--targets", default="close")
    a = ap.parse_args()
    df = load_base()
    run(df, list(M.SPREAD_FEATURES), tuple(a.targets.split(",")), a.name)
