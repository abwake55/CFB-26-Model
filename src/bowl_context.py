"""Bowl-game context features (CFBD research pulls; see
scripts/research/pull_cfbd_research.py --what bowl,passing).

Per team per bowl (all knowable by game day, except the draft proxy):
  qb_change        bowl starter is not the team's regular-season primary
                   starter (most starts). Needs the confirmed / likely starter.
  qb_starter_share bowl starter's share of the team's regular-season attempts
  portal_out       players from this team in the transfer portal before kickoff
  portal_out_q     ... rated 3+ stars (or rating >= 0.85)
  portal_qb_out    ... at QB
  interim_coach    the bowl is coached by an interim (coach with a 1-game season)
  coach_change     more than one head coach listed for the season
  draft_r1_3       players drafted in rounds 1-3 the next spring. POST-HOC
                   opt-out proxy: research only, never used live.
Game frame columns are home_/away_ and *_diff (home minus away).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "research"
LIVE_FEATS = ["qb_change", "qb_starter_share", "portal_out", "portal_out_q",
              "portal_qb_out", "interim_coach", "coach_change"]
RESEARCH_ONLY = ["draft_r1_3"]
FEATS = LIVE_FEATS + RESEARCH_ONLY


def _t(s):
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def bowl_games(fbs_only: bool = True) -> pd.DataFrame:
    g = pd.read_csv(RES / "cfbd_postseason_games.csv")
    if fbs_only:
        g = g[(g.home_classification == "fbs") & (g.away_classification == "fbs")]
    g = g.copy()
    g["t"] = _t(g.start_date)
    g["cfp"] = g.notes.fillna("").str.contains("Playoff|National Championship", case=False)
    return g


def starters(passing: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the starter = most pass attempts."""
    return (passing.sort_values("attempts", ascending=False)
            .drop_duplicates(["game_id", "team"]))


def qb_features(team_games: pd.DataFrame, passing: pd.DataFrame | None = None) -> pd.DataFrame:
    """team_games: game_id, season, team, optional starter_id (the confirmed /
    likely starter, used live). Without starter_id the bowl box score is used,
    but only to ask "did the regular-season primary QB throw a pass?" A QB
    benched mid-game still threw passes, so in-game benchings (which follow
    bad play and would leak the result) never count as a change.

      qb_change        1 if the primary starter is out (live: the named
                       starter is not the primary)
      qb_starter_share regular-season attempt share of the QB expected to
                       play (the primary when he plays)"""
    p = passing if passing is not None else pd.read_csv(RES / "cfbd_passing_by_game.csv")
    reg = p[p.season_type == "regular"]
    st = starters(reg)
    starts = st.groupby(["season", "team", "player_id"]).size().rename("starts").reset_index()
    prim = starts.sort_values("starts", ascending=False).drop_duplicates(["season", "team"])
    att = reg.groupby(["season", "team", "player_id"]).attempts.sum()
    tot = reg.groupby(["season", "team"]).attempts.sum()
    names = reg.drop_duplicates("player_id").set_index("player_id").player
    tg = team_games.copy()
    tg = tg.merge(prim[["season", "team", "player_id"]].rename(columns={"player_id": "primary_id"}),
                  on=["season", "team"], how="left")
    tg["primary_qb"] = tg.primary_id.map(names)
    if "starter_id" not in tg.columns:
        post = p[p.season_type == "postseason"]
        played = post.groupby(["game_id", "team"]).player_id.apply(lambda x: set(x.astype(str)))
        top = starters(post).set_index(["game_id", "team"]).player_id
        sid = []
        for r in tg.itertuples():
            k = (r.game_id, r.team)
            if k not in played.index:
                sid.append(np.nan)
            elif pd.notna(r.primary_id) and str(r.primary_id) in played[k]:
                sid.append(r.primary_id)          # primary played: no change
            else:
                sid.append(top.get(k, np.nan))    # primary absent: who took over
        tg["starter_id"] = sid
    known = tg.starter_id.notna() & tg.primary_id.notna()
    tg["qb_change"] = np.where(known, (tg.starter_id.astype(str) != tg.primary_id.astype(str)).astype(float), np.nan)
    share = []
    for r in tg.itertuples():
        a = att.get((r.season, r.team, r.starter_id), 0) if pd.notna(r.starter_id) else np.nan
        t = tot.get((r.season, r.team), np.nan)
        share.append(a / t if t and pd.notna(a) else np.nan)
    tg["qb_starter_share"] = share
    return tg


def portal_features(team_games: pd.DataFrame, portal: pd.DataFrame | None = None) -> pd.DataFrame:
    """team_games needs season, team, t (kickoff, UTC)."""
    po = portal if portal is not None else pd.read_csv(RES / "cfbd_portal.csv")
    po = po.copy()
    po["pt"] = _t(po.transfer_date)
    q = (pd.to_numeric(po.stars, errors="coerce") >= 3) | (pd.to_numeric(po.rating, errors="coerce") >= 0.85)
    po["q"] = q.astype(int)
    po["qb"] = (po.position.astype(str).str.upper() == "QB").astype(int)
    have = set(po.season.unique())
    rows = []
    for r in team_games.itertuples():
        if r.season not in have:
            rows.append((np.nan, np.nan, np.nan))
            continue
        d = po[(po.season == r.season) & (po.origin == r.team) & (po.pt < r.t)]
        rows.append((len(d), int(d.q.sum()), int(d.qb.sum())))
    out = team_games.copy()
    out[["portal_out", "portal_out_q", "portal_qb_out"]] = rows
    return out


def coach_features(team_games: pd.DataFrame, coaches: pd.DataFrame | None = None) -> pd.DataFrame:
    c = coaches if coaches is not None else pd.read_csv(RES / "cfbd_coaches.csv")
    n = c.groupby(["season", "school"]).size().rename("n_coaches")
    interim = c[(c.games == 1)].groupby(["season", "school"]).size().rename("n_interim")
    out = team_games.merge(n, left_on=["season", "team"], right_index=True, how="left") \
                    .merge(interim, left_on=["season", "team"], right_index=True, how="left")
    out["coach_change"] = (out.n_coaches.fillna(1) > 1).astype(float)
    out["interim_coach"] = ((out.n_interim.fillna(0) > 0) & (out.n_coaches.fillna(1) > 1)).astype(float)
    return out.drop(columns=["n_coaches", "n_interim"])


def draft_features(team_games: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(RES / "cfbd_draft.csv")
    k = d[d["round"] <= 3].groupby(["season", "college_team"]).size().rename("draft_r1_3")
    out = team_games.merge(k, left_on=["season", "team"], right_index=True, how="left")
    have = set(d.season.unique())
    out["draft_r1_3"] = np.where(out.season.isin(have), out.draft_r1_3.fillna(0), np.nan)
    return out


def team_features(games: pd.DataFrame, passing=None, portal=None, coaches=None,
                  starters_override: pd.DataFrame | None = None,
                  research: bool = True) -> pd.DataFrame:
    """games: game_id, season, home_team, away_team, t. Returns team rows."""
    tg = pd.concat([games[["game_id", "season", "t"]].assign(team=games.home_team, side="home"),
                    games[["game_id", "season", "t"]].assign(team=games.away_team, side="away")],
                   ignore_index=True)
    if starters_override is not None:
        tg = tg.merge(starters_override[["game_id", "team", "starter_id"]], on=["game_id", "team"], how="left")
    tg = qb_features(tg, passing)
    tg = portal_features(tg, portal)
    tg = coach_features(tg, coaches)
    if research:
        tg = draft_features(tg)
    return tg


def game_features(games: pd.DataFrame, **kw) -> pd.DataFrame:
    tg = team_features(games, **kw)
    feats = [f for f in FEATS if f in tg.columns]
    out = games[["game_id"]].copy()
    for side in ("home", "away"):
        x = tg[tg.side == side][["game_id"] + feats].rename(columns={f: f"{side}_{f}" for f in feats})
        out = out.merge(x, on="game_id", how="left")
    for f in feats:
        out[f"{f}_diff"] = out[f"home_{f}"] - out[f"away_{f}"]
    return out
