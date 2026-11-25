"""Likely / confirmed starting QB per team, from free data.

Sources, strongest first:
  1. data/context/qb_overrides.csv  (hand-entered confirmed news:
     season,team,starter,status,note,updated_at; status = confirmed|likely|out)
  2. transfer portal (CFBD /player/portal): a primary QB who has entered the
     portal is treated as out for a bowl
  3. CFBD per-game passing: the QB who started the team's last game

CFBD has no injury or depth-chart feed and ESPN's college depth charts are
not published, so "confirmed" only comes from overrides. Everything else is
"likely" and is labeled that way in every output.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "data" / "context"
OVERRIDES = CTX / "qb_overrides.csv"
BASE = "https://api.collegefootballdata.com"
CALLS = {"n": 0}


def cfbd(endpoint: str, **params):
    key = os.environ.get("CFB_API_KEY", "")
    if not key:
        raise RuntimeError("CFB_API_KEY not set")
    for attempt in range(4):
        r = requests.get(f"{BASE}/{endpoint}", params=params,
                         headers={"Authorization": f"Bearer {key}"}, timeout=60)
        CALLS["n"] += 1
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _stat(types, name):
    for t in types:
        if t.get("name") == name:
            return {a.get("id"): (a.get("name"), a.get("stat")) for a in t.get("athletes", [])}
    return {}


def _passing_week(season: int, week: int, season_type: str = "regular") -> list[dict]:
    rows = []
    for g in cfbd("games/players", year=season, week=week, seasonType=season_type,
                  category="passing") or []:
        for t in g.get("teams", []):
            for cat in t.get("categories", []):
                if cat.get("name") != "passing":
                    continue
                types = cat.get("types", [])
                catt, yds = _stat(types, "C/ATT"), _stat(types, "YDS")
                for pid, (pname, ca) in catt.items():
                    try:
                        c, a = (int(x) for x in str(ca).split("/"))
                    except ValueError:
                        continue
                    rows.append({"game_id": g.get("id"), "season": season, "week": week,
                                 "season_type": season_type, "team": t.get("team"),
                                 "player_id": pid, "player": pname, "completions": c,
                                 "attempts": a,
                                 "yards": pd.to_numeric(yds.get(pid, (None, None))[1], errors="coerce")})
    return rows


def last_regular_week(season: int) -> int:
    """Latest regular-season week whose first game has kicked off (CFBD calendar)."""
    now = pd.Timestamp.now(tz="UTC")
    wk = 1
    for c in cfbd("calendar", year=season) or []:
        if c.get("seasonType") != "regular":
            continue
        start = pd.to_datetime(c.get("firstGameStart") or c.get("startDate"), utc=True, errors="coerce")
        if pd.notna(start) and start <= now:
            wk = max(wk, int(c["week"]))
    return wk


def refresh_passing(season: int, through_week: int) -> pd.DataFrame:
    """Season-to-date regular-season passing, cached in
    data/context/qb_passing_{season}.csv. Fetches only weeks not cached yet
    plus the latest cached week (late games / stat corrections)."""
    path = CTX / f"qb_passing_{season}.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(have["week"].unique()) if not have.empty else set()
    todo = [w for w in range(1, through_week + 1) if w not in done]
    if done:
        todo = sorted(set(todo) | {max(w for w in done if w <= through_week)} if any(
            w <= through_week for w in done) else set(todo))
    new = []
    for w in todo:
        try:
            new += _passing_week(season, w)
        except Exception as exc:  # keep what we have
            print(f"[qb] passing week {w} unavailable: {type(exc).__name__}")
    if new:
        nd = pd.DataFrame(new)
        have = pd.concat([have[~have["week"].isin(nd["week"].unique())] if not have.empty else have, nd],
                         ignore_index=True)
        CTX.mkdir(parents=True, exist_ok=True)
        have.sort_values(["week", "game_id", "team"]).to_csv(path, index=False)
    return have


def portal_entries(season: int) -> pd.DataFrame:
    """Portal entries for the cycle after `season` (CFBD portal year season+1)."""
    try:
        rows = cfbd("player/portal", year=season + 1) or []
    except Exception as exc:
        print(f"[qb] portal unavailable: {type(exc).__name__}")
        rows = []
    df = pd.DataFrame([{"season": season,
                        "player": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                        "position": p.get("position"), "origin": p.get("origin"),
                        "destination": p.get("destination"), "transfer_date": p.get("transferDate"),
                        "rating": p.get("rating"), "stars": p.get("stars")} for p in rows])
    return df


def load_overrides(season: int) -> pd.DataFrame:
    if not OVERRIDES.exists():
        return pd.DataFrame(columns=["season", "team", "starter", "status", "note", "updated_at"])
    o = pd.read_csv(OVERRIDES)
    return o[o["season"] == season]


def _norm(s) -> str:
    return " ".join(str(s).lower().replace(".", "").replace("'", "").split())


def team_status(passing: pd.DataFrame, teams: list[str], season: int,
                portal: pd.DataFrame | None = None, now=None) -> pd.DataFrame:
    """One row per team: primary QB, last starter, likely starter + source."""
    now = now or pd.Timestamp.now(tz="UTC")
    p = passing[passing["season_type"] == "regular"] if "season_type" in passing else passing
    p = p[p["player"].astype(str).str.strip().str.lower() != "team"]   # CFBD team-total rows
    st = p.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    ov = load_overrides(season)
    portal_names = set()
    if portal is not None and not portal.empty:
        t = pd.to_datetime(portal["transfer_date"], utc=True, errors="coerce", format="mixed")
        q = portal[(portal["position"].astype(str).str.upper() == "QB") & (t <= now)]
        portal_names = {(o, _norm(n)) for o, n in zip(q["origin"], q["player"])}
    names = p.drop_duplicates("player_id").set_index("player_id")["player"]
    out = []
    for team in teams:
        s = st[st["team"] == team].sort_values("week")
        row = {"season": season, "team": team, "primary_id": np.nan, "primary_qb": None,
               "primary_starts": 0, "last_starter_id": np.nan, "last_starter": None,
               "qb_out_last": np.nan, "primary_in_portal": False,
               "likely_starter_id": np.nan, "likely_starter": None, "source": "no data",
               "status": "unknown", "note": None}
        if not s.empty:
            vc = s["player_id"].value_counts()
            prim = vc.index[0]
            last = s.iloc[-1]
            played_last = set(p[(p["game_id"] == last["game_id"]) & (p["team"] == team)]["player_id"])
            row.update(primary_id=prim, primary_qb=names.get(prim), primary_starts=int(vc.iloc[0]),
                       last_starter_id=last["player_id"], last_starter=names.get(last["player_id"]),
                       qb_out_last=float(prim not in played_last))
            row["primary_in_portal"] = (team, _norm(row["primary_qb"])) in portal_names
            if row["primary_in_portal"]:
                # next QB by season attempts who is not in the portal
                att = p[p["team"] == team].groupby("player_id")["attempts"].sum().sort_values(ascending=False)
                nxt = [pid for pid in att.index if (team, _norm(names.get(pid))) not in portal_names]
                if nxt:
                    row.update(likely_starter_id=nxt[0], likely_starter=names.get(nxt[0]))
                row.update(source="primary QB in transfer portal", status="likely")
            else:
                row.update(likely_starter_id=last["player_id"], likely_starter=names.get(last["player_id"]),
                           source="started last game", status="likely")
        o = ov[ov["team"] == team]
        if not o.empty:
            o = o.sort_values("updated_at").iloc[-1]
            match = [pid for pid, n in names.items() if _norm(n) == _norm(o["starter"])]
            sid = match[0] if match else f"name:{o['starter']}"
            if str(o["status"]).lower() == "out":
                # named QB is out: if he is the likely starter, fall back to next QB
                if _norm(o["starter"]) == _norm(row["likely_starter"]):
                    att = p[p["team"] == team].groupby("player_id")["attempts"].sum().sort_values(ascending=False)
                    nxt = [pid for pid in att.index if _norm(names.get(pid)) != _norm(o["starter"])]
                    if nxt:
                        row.update(likely_starter_id=nxt[0], likely_starter=names.get(nxt[0]))
                row.update(source="override: QB out", status="likely", note=o.get("note"))
            else:
                row.update(likely_starter_id=sid, likely_starter=o["starter"],
                           source="override", status=str(o["status"]).lower(), note=o.get("note"))
        lid = row["likely_starter_id"]
        row["likely_starts"] = int((s["player_id"].astype(str) == str(lid)).sum()) if not s.empty else 0
        row["qb_change_likely"] = (float(str(row["likely_starter_id"]) != str(row["primary_id"]))
                                   if pd.notna(row["primary_id"]) and row["likely_starter"] else np.nan)
        out.append(row)
    return pd.DataFrame(out)
