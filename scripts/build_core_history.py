#!/usr/bin/env python3
"""
Build the CORE totals portfolio history from walk-forward results.

Applies the validated CORE gate (under, edge 2-7 pts, power-conf involved,
wind < 15 mph, market total >= 48) to every out-of-sample walk-forward
prediction 2019-25 and writes one row per bet with P&L at -110 to:

    outputs/predictions/core_history.csv

The app plots this as the track-record equity curve; regenerate after each
walk-forward rerun (walk_forward.py) so the curve always matches the model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
               "Big East", "FBS Independents"}


def main() -> None:
    wf = pd.read_csv(ROOT / "outputs/predictions/walk_forward_results.csv")
    fm = pd.read_csv(ROOT / "data/processed/feature_matrix.csv",
                     usecols=["game_id", "home_conference", "away_conference",
                              "wind_speed", "is_dome", "start_date"]
                     if "start_date" in pd.read_csv(
                         ROOT / "data/processed/feature_matrix.csv", nrows=0).columns
                     else ["game_id", "home_conference", "away_conference",
                           "wind_speed", "is_dome"])
    df = wf.merge(fm, on="game_id", how="left")

    t = df.dropna(subset=["over_under", "pred_total", "total_points"]).copy()
    t["totals_edge"] = t["pred_total"] - t["over_under"]
    t["power"] = (t["home_conference"].isin(POWER_CONFS)
                  | t["away_conference"].isin(POWER_CONFS))
    t["windy"] = (~t["is_dome"].fillna(0).astype(bool)
                  & (t["wind_speed"].fillna(0) >= 15))

    core = t[(t["totals_edge"] <= -2) & (t["totals_edge"] >= -7)
             & t["power"] & ~t["windy"]
             & (t["over_under"] >= 48)].copy()

    push = core["total_points"] == core["over_under"]
    win = core["total_points"] < core["over_under"]
    core["result"] = np.select([push, win], ["push", "win"], "loss")
    core["pnl"] = np.select([push, win], [0.0, 1.0], -1.1)

    out = core[["season", "week", "game_id", "home_team", "away_team",
                "over_under", "totals_edge", "total_points",
                "result", "pnl"]].sort_values(["season", "week", "game_id"])
    out["cum_units"] = out["pnl"].cumsum().round(1)

    dest = ROOT / "outputs/predictions/core_history.csv"
    out.to_csv(dest, index=False)

    graded = out[out["result"] != "push"]
    hit = (graded["result"] == "win").mean()
    print(f"CORE history: {len(out)} bets, hit {hit:.1%}, "
          f"{out['pnl'].sum():+.1f}u -> {dest.name}")


if __name__ == "__main__":
    main()
