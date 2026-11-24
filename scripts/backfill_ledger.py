"""Backfill ledger weeks the scheduled job never recorded, labelled NOT live.

    python scripts/backfill_ledger.py --season 2026 --weeks 7 10 11 12

For each week the models are refit on games that kicked off BEFORE that
week's first game (prior seasons + earlier 2026 weeks; validation = last
prior season), then predict that week's lined FBS games from the
point-in-time feature matrix. Lines are CFBD's (closing-ish), so these rows
are research only: provenance=backfill_not_live, CLV left blank. Weeks with
no feature rows yet (results not ingested) are skipped; rerun after the
weekly refresh ingests them.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gates  # noqa: E402
import model  # noqa: E402
import walk_forward as wf  # noqa: E402
import prediction_ledger as pl  # noqa: E402


def predict_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    df = df.copy()
    df["_kick"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce", format="mixed")
    test = df[(df["season"] == season) & (df["week"] == week)
              & (df.get("season_type", "regular") == "regular")]
    if test.empty:
        return pd.DataFrame()
    cutoff = test["_kick"].min()
    prior = sorted(s for s in df["season"].unique() if 2017 <= s < season and s != 2020)
    val = df[df["season"] == prior[-1]]
    train = df[(df["season"].isin(prior[:-1]))
               | ((df["season"] == season) & (df["_kick"] < cutoff))]
    orig = wf.split_seasons
    wf.split_seasons = lambda _df, _s: (train.drop(columns="_kick"), val.drop(columns="_kick"),
                                        test.drop(columns="_kick"))
    try:
        out = wf.run_fold(df.drop(columns="_kick"), season)
    finally:
        wf.split_seasons = orig
    out["training_cutoff_utc"] = cutoff.isoformat()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    args = ap.parse_args()

    df = model.load_data()
    fm = pd.read_csv(ROOT / "data/processed/feature_matrix.csv",
                     usecols=["game_id", "home_conference", "away_conference", "neutral_site",
                              "start_date", "spread_open", "wind_speed", "is_dome"]
                     ).drop_duplicates("game_id")
    ledger = pd.read_csv(pl.LEDGER)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = []
    for week in args.weeks:
        have = ledger[(ledger["season"] == args.season) & (ledger["week"] == week)]
        if len(have):
            print(f"week {week}: already in ledger ({len(have)} rows) - skipping")
            continue
        preds = predict_week(df, args.season, week)
        if preds.empty:
            print(f"week {week}: no feature rows yet - skipping (rerun after refresh)")
            continue
        p = preds.merge(fm, on="game_id", how="left", suffixes=("", "_fm"))
        for _, r in p.iterrows():
            row = r.to_dict()
            done = pd.notna(r.get("home_points")) and pd.notna(r.get("away_points"))
            new_rows.append({
                "game_id": r.game_id, "season": args.season, "week": week,
                "home_team": r.home_team, "away_team": r.away_team,
                "home_conference": row.get("home_conference"),
                "away_conference": row.get("away_conference"),
                "neutral_site": row.get("neutral_site"), "start_date": row.get("start_date"),
                "recorded_at": now,
                "pred_spread": r.pred_spread, "pred_total": r.pred_total,
                "pred_win_p": r.pred_home_win_p,
                "spread": r.spread, "over_under": r.over_under,
                "spread_open": row.get("spread_open"),
                "home_moneyline": row.get("home_moneyline"),
                "away_moneyline": row.get("away_moneyline"),
                "spread_edge": r.spread_edge, "totals_edge": r.totals_edge,
                # retrospective research flags (no live approval existed)
                "flag_core": bool(gates.core_candidate(row)),
                "flag_spread": bool(pd.notna(r.spread_edge) and abs(r.spread_edge) >= 3),
                "flag_ml": False,
                "closing_spread": np.nan, "closing_over_under": np.nan,
                "home_score": r.home_points if done else np.nan,
                "away_score": r.away_points if done else np.nan,
                "completed": bool(done),
                "home_win": float(r.home_points > r.away_points) if done else np.nan,
                "actual_margin": r.home_points - r.away_points if done else np.nan,
                "actual_total": r.home_points + r.away_points if done else np.nan,
                "provenance": "backfill_not_live",
            })
        print(f"week {week}: backfilled {int((p.season == args.season).sum())} games "
              f"(train cutoff {preds['training_cutoff_utc'].iloc[0]})")
    if not new_rows:
        return
    ledger["provenance"] = pl._provenance(ledger)
    out = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["season", "week", "game_id"])
    out.to_csv(pl.LEDGER, index=False)
    c = out["completed"]
    out["completed"] = c.notna() & c.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    pl._build_summary(out)


if __name__ == "__main__":
    main()
