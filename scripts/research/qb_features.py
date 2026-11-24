"""Starting-QB features from CFBD per-game passing (data/research/cfbd_passing_by_game.csv).

Starter of a team-game = player with the most pass attempts in that game.
Two feature sets, per team, then home/away:
  PRE (knowable before any lineup news, i.e. at the opener):
    qb_prev_changed   previous game's starter differs from the one before it
    qb_prev_share     share of the team's prior season attempts thrown by the
                      previous game's starter (low = unsettled QB room)
    qb_prev_starts    prior starts this season by the previous game's starter
  LINEUP (needs the confirmed starter, i.e. only near kickoff; diagnostic):
    qb_new_starter    this game's starter != previous game's starter
    qb_starter_starts prior starts this season by this game's starter
    qb_starter_ypa_gap starter's prior-season-to-date YPA minus team's
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRE = ["qb_prev_changed", "qb_prev_share", "qb_prev_starts"]
LINEUP = ["qb_new_starter", "qb_starter_starts", "qb_starter_ypa_gap"]


def team_game_features() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/research/cfbd_passing_by_game.csv")
    g = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False,
                    usecols=["game_id", "start_date"]).drop_duplicates("game_id")
    p = p.merge(g, on="game_id", how="left")
    p["t"] = pd.to_datetime(p["start_date"], utc=True, errors="coerce", format="mixed")
    p = p.dropna(subset=["t"])
    st = (p.sort_values("attempts", ascending=False)
           .drop_duplicates(["game_id", "team"])[["game_id", "season", "team", "t", "player_id"]]
           .rename(columns={"player_id": "starter"}))
    tm = p.groupby(["game_id", "team"]).agg(team_att=("attempts", "sum"),
                                             team_yds=("yards", "sum")).reset_index()
    rows = []
    for (season, team), grp in st.merge(tm, on=["game_id", "team"]).sort_values("t").groupby(["season", "team"]):
        pp = p[(p.season == season) & (p.team == team)]
        hist = []
        for r in grp.itertuples():
            prior = pp[pp.t < r.t]
            prev = hist[-1] if hist else None
            prev2 = hist[-2] if len(hist) > 1 else None
            starts = pd.Series([h for h in hist])
            att = prior.groupby("player_id").attempts.sum()
            yds = prior.groupby("player_id").yards.sum()
            tot_att = att.sum()
            team_ypa = (yds.sum() / tot_att) if tot_att else np.nan
            s_ypa = (yds.get(r.starter, np.nan) / att.get(r.starter)) if att.get(r.starter, 0) else np.nan
            rows.append({
                "game_id": r.game_id, "team": team,
                "qb_prev_changed": float(prev is not None and prev2 is not None and prev != prev2),
                "qb_prev_share": (att.get(prev, 0) / tot_att) if (prev is not None and tot_att) else np.nan,
                "qb_prev_starts": float((starts == prev).sum()) if prev is not None else 0.0,
                "qb_new_starter": float(prev is not None and r.starter != prev),
                "qb_starter_starts": float((starts == r.starter).sum()),
                "qb_starter_ypa_gap": s_ypa - team_ypa,
            })
            hist.append(r.starter)
    return pd.DataFrame(rows)


def game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach home_/away_ features (+ diffs) to a frame with game_id/home_team/away_team."""
    tg = team_game_features()
    out = df[["game_id", "home_team", "away_team"]].copy()
    for side in ("home", "away"):
        x = tg.rename(columns={c: f"{side}_{c}" for c in PRE + LINEUP})
        out = out.merge(x, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"],
                        how="left").drop(columns="team")
    for c in PRE + LINEUP:
        out[f"{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out.drop(columns=["home_team", "away_team"])


if __name__ == "__main__":
    tg = team_game_features()
    print(tg.describe().T)
