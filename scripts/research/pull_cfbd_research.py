"""Pull research-only CFBD data (per-provider lines with openers; per-game
passing leaders) into data/research/. Run in CI (needs CFB_API_KEY).

  python scripts/research/pull_cfbd_research.py --seasons 2017-2026 [--min-remaining 300]

Calls: ~2 per season for lines, ~17 per season for passing. Aborts if the
key's remaining monthly calls (/info) are below --min-remaining.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
BASE = "https://api.collegefootballdata.com"
CALLS = 0


def get(endpoint, **params):
    global CALLS
    key = os.environ.get("CFB_API_KEY")
    if not key:
        sys.exit("CFB_API_KEY not set")
    for attempt in range(4):
        r = requests.get(f"{BASE}/{endpoint}", params=params,
                         headers={"Authorization": f"Bearer {key}"}, timeout=60)
        CALLS += 1
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def remaining():
    try:
        info = get("info")
        print("CFBD /info:", {k: info.get(k) for k in ("patronLevel", "remainingCalls")})
        return info.get("remainingCalls")
    except Exception as e:  # /info missing should not block
        print("CFBD /info unavailable:", e)
        return None


def pull_lines(season):
    rows = []
    for st in ("regular", "postseason"):
        for g in get("lines", year=season, seasonType=st) or []:
            for ln in g.get("lines", []) or []:
                rows.append({
                    "game_id": g.get("id"), "season": g.get("season"), "week": g.get("week"),
                    "season_type": g.get("seasonType"), "start_date": g.get("startDate"),
                    "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                    "home_score": g.get("homeScore"), "away_score": g.get("awayScore"),
                    "provider": ln.get("provider"), "spread": ln.get("spread"),
                    "spread_open": ln.get("spreadOpen"), "over_under": ln.get("overUnder"),
                    "over_under_open": ln.get("overUnderOpen"),
                    "home_moneyline": ln.get("homeMoneyline"),
                    "away_moneyline": ln.get("awayMoneyline"),
                })
    return rows


def _stat(types, name):
    for t in types:
        if t.get("name") == name:
            return {a.get("id"): (a.get("name"), a.get("stat")) for a in t.get("athletes", [])}
    return {}


def pull_passing(season, max_week):
    rows = []
    plan = [("regular", w) for w in range(1, max_week + 1)] + [("postseason", 1)]
    for st, week in plan:
        data = get("games/players", year=season, week=week, seasonType=st, category="passing") or []
        for g in data:
            for t in g.get("teams", []):
                for cat in t.get("categories", []):
                    if cat.get("name") != "passing":
                        continue
                    types = cat.get("types", [])
                    catt, yds, td, ints = (_stat(types, n) for n in ("C/ATT", "YDS", "TD", "INT"))
                    for pid, (pname, ca) in catt.items():
                        try:
                            c, a = (int(x) for x in str(ca).split("/"))
                        except ValueError:
                            continue
                        rows.append({
                            "game_id": g.get("id"), "season": season, "week": week,
                            "season_type": st, "team": t.get("team"),
                            "home_away": t.get("homeAway"), "player_id": pid,
                            "player": pname, "completions": c, "attempts": a,
                            "yards": pd.to_numeric(yds.get(pid, (None, None))[1], errors="coerce"),
                            "td": pd.to_numeric(td.get(pid, (None, None))[1], errors="coerce"),
                            "int": pd.to_numeric(ints.get(pid, (None, None))[1], errors="coerce"),
                        })
        print(f"  passing {season} {st} wk{week}: {len(data)} games")
    return rows


def pull_bowl_context(seasons):
    """Bowl research context (one call per season per endpoint):
    postseason games (bowl names in `notes`), head coaches by season,
    transfer portal entries (portal year = season + 1, with transferDate),
    and NFL draft picks (draft year = season + 1; a post-hoc opt-out proxy)."""
    games, coaches, portal, draft = [], [], [], []
    for s in seasons:
        for g in get("games", year=s, seasonType="postseason") or []:
            games.append({
                "game_id": g.get("id"), "season": g.get("season"), "week": g.get("week"),
                "start_date": g.get("startDate"), "neutral_site": g.get("neutralSite"),
                "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                "home_conference": g.get("homeConference"),
                "away_conference": g.get("awayConference"),
                "home_classification": g.get("homeClassification"),
                "away_classification": g.get("awayClassification"),
                "home_points": g.get("homePoints"), "away_points": g.get("awayPoints"),
                "notes": g.get("notes"), "venue": g.get("venue"),
            })
        for c in get("coaches", year=s) or []:
            for cs in c.get("seasons", []) or []:
                if cs.get("year") != s:
                    continue
                coaches.append({
                    "season": s, "school": cs.get("school"),
                    "coach": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
                    "hire_date": c.get("hireDate"), "games": cs.get("games"),
                    "wins": cs.get("wins"), "losses": cs.get("losses"),
                })
        for p in get("player/portal", year=s + 1) or []:
            portal.append({
                "season": s, "portal_year": s + 1,
                "player": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                "position": p.get("position"), "origin": p.get("origin"),
                "destination": p.get("destination"), "transfer_date": p.get("transferDate"),
                "rating": p.get("rating"), "stars": p.get("stars"),
                "eligibility": p.get("eligibility"),
            })
        if s + 1 <= 2026:
            for d in get("draft/picks", year=s + 1) or []:
                draft.append({
                    "season": s, "draft_year": s + 1, "college_team": d.get("collegeTeam"),
                    "name": d.get("name"), "position": d.get("position"),
                    "round": d.get("round"), "overall": d.get("overall"),
                    "pre_draft_ranking": d.get("preDraftRanking"),
                })
        print(f"  bowl context {s}: games={len(games)} coaches={len(coaches)} "
              f"portal={len(portal)} draft={len(draft)}")
    for name, rows in (("cfbd_postseason_games", games), ("cfbd_coaches", coaches),
                       ("cfbd_portal", portal), ("cfbd_draft", draft)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2017-2026")
    ap.add_argument("--what", default="lines,passing",
                    help="comma list of: lines, passing, bowl")
    ap.add_argument("--min-remaining", type=int, default=300)
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(a, b + 1))
    rem = remaining()
    if rem is not None and rem < args.min_remaining:
        sys.exit(f"Only {rem} CFBD calls left (< {args.min_remaining}); not pulling")
    OUT.mkdir(parents=True, exist_ok=True)
    what = set(args.what.split(","))
    if "lines" in what:
        rows = [r for s in seasons for r in pull_lines(s)]
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "cfbd_lines_by_provider.csv", index=False)
        print("lines rows", len(df), "with opener", int(df["spread_open"].notna().sum()))
        print(df.assign(o=df.spread_open.notna()).groupby(["season", "provider"]).o.sum().to_string())
    if "passing" in what:
        games = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False)
        rows = []
        for s in seasons:
            wk = games.loc[(games.season == s), "week"]
            max_week = int(wk.max()) if len(wk) else 15
            rows += pull_passing(s, min(max(max_week, 1), 16))
        pd.DataFrame(rows).to_csv(OUT / "cfbd_passing_by_game.csv", index=False)
        print("passing rows", len(rows))
    if "bowl" in what:
        pull_bowl_context(seasons)
    print("CFBD calls used:", CALLS)
    remaining()


if __name__ == "__main__":
    main()
