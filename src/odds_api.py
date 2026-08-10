"""
The Odds API (the-odds-api.com) — NCAAF live line fetcher.
=========================================================
Replaces the retired OddsBlaze feed. One request returns every US book for
every upcoming game (h2h/spreads/totals), so we collapse the books to a
single **consensus** line per game — median spread, median total, median
moneyline per side — giving a stable market reference that isn't tied to one
book's number. Spread/total are rounded to the nearest 0.5 to match the
0.5-increment lines the model was trained on.

Quota: a request costs 1 per market (3 markets = 3 credits). The free tier is
20k/month, so a weekly pull is negligible.

Shared by app.py and scripts/weekly_pipeline.py.
"""
from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher

import pandas as pd
import requests

ODDS_API_BASE = ("https://api.the-odds-api.com/v4/sports/"
                 "americanfootball_ncaaf/odds")


def _round_half(x):
    """Round to nearest 0.5 (book-line convention). None-safe."""
    return None if x is None else round(x * 2) / 2


def fetch_ncaaf_events(api_key: str, timeout: int = 25) -> list:
    """Raw event list from The Odds API (empty list on any non-list body)."""
    resp = requests.get(
        f"{ODDS_API_BASE}/",
        params={"apiKey": api_key, "regions": "us",
                "markets": "h2h,spreads,totals", "oddsFormat": "american"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def events_to_consensus(events: list) -> pd.DataFrame:
    """Collapse events to one consensus row per game.

    Columns: odds_home, odds_away, spread (home), over_under,
    home_moneyline, away_moneyline, provider, commence_time.
    """
    rows = []
    for e in events:
        home, away = e.get("home_team"), e.get("away_team")
        spreads, totals, home_mls, away_mls = [], [], [], []
        for bk in e.get("bookmakers", []):
            for m in bk.get("markets", []):
                k, outs = m.get("key"), m.get("outcomes", [])
                if k == "spreads":
                    for o in outs:
                        if o.get("name") == home and o.get("point") is not None:
                            spreads.append(float(o["point"]))
                elif k == "totals":
                    for o in outs:
                        if o.get("name") == "Over" and o.get("point") is not None:
                            totals.append(float(o["point"]))
                elif k == "h2h":
                    for o in outs:
                        if o.get("price") is None:
                            continue
                        if o.get("name") == home:
                            home_mls.append(float(o["price"]))
                        elif o.get("name") == away:
                            away_mls.append(float(o["price"]))
        if not (spreads or totals):
            continue
        rows.append({
            "odds_home": home, "odds_away": away,
            "spread": _round_half(statistics.median(spreads)) if spreads else None,
            "over_under": _round_half(statistics.median(totals)) if totals else None,
            "home_moneyline": round(statistics.median(home_mls)) if home_mls else None,
            "away_moneyline": round(statistics.median(away_mls)) if away_mls else None,
            "provider": "consensus", "commence_time": e.get("commence_time"),
        })
    return pd.DataFrame(rows)


def _tokens(s) -> list:
    return [t for t in re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split() if t]


def _team_match(cfbd_name: str, odds_name: str) -> float:
    """Match quality in [0, ~1.5]; 0 = no match.

    The Odds API appends a mascot to the school ("Ohio State Buckeyes"),
    while CFBD uses the school alone ("Ohio State"). So the CFBD name is
    normally a token *prefix* of the Odds name. A prefix hit scores 1.0 plus
    a specificity bonus (0.1 per matched token) so "Ohio State" outranks
    "Ohio" for "Ohio State Buckeyes". Spelling variants (St./State,
    Miss/Mississippi) fall back to a fuzzy ratio.
    """
    ct, ot = _tokens(cfbd_name), _tokens(odds_name)
    if not ct or not ot:
        return 0.0
    if len(ct) <= len(ot) and ot[:len(ct)] == ct:
        return 1.0 + 0.1 * len(ct)
    r = SequenceMatcher(None, " ".join(ct), " ".join(ot)).ratio()
    return r if r >= 0.72 else 0.0


def match_to_games(odds_df: pd.DataFrame, games_df: pd.DataFrame,
                   max_day_gap: int = 6) -> pd.DataFrame:
    """Match consensus rows to CFBD game_ids by team name.

    Requires BOTH teams to match (prefix or fuzzy) and, when commence_time
    and start_date are both present, the kickoff to fall within
    ``max_day_gap`` days — this prevents a future-week game for the same
    matchup from being matched to this week. Among candidates the most
    specific name match wins. Returns game_id, spread, over_under,
    home/away_moneyline, spread_open, provider — one row per matched game.
    """
    if odds_df.empty or games_df.empty:
        return pd.DataFrame()

    has_dates = "start_date" in games_df.columns
    cfbd = list(zip(games_df["game_id"], games_df["home_team"],
                    games_df["away_team"],
                    games_df["start_date"] if has_dates
                    else [None] * len(games_df)))

    matched = []
    for _, r in odds_df.iterrows():
        odds_dt = pd.to_datetime(r.get("commence_time"), utc=True, errors="coerce")
        best_id, best_score = None, 0.0
        for gid, ch, ca, sd in cfbd:
            if pd.notna(odds_dt) and sd is not None:
                game_dt = pd.to_datetime(sd, utc=True, errors="coerce")
                if pd.notna(game_dt) and abs((odds_dt - game_dt).days) > max_day_gap:
                    continue
            hs = _team_match(ch, r["odds_home"])
            as_ = _team_match(ca, r["odds_away"])
            if hs == 0.0 or as_ == 0.0:      # both teams must match
                continue
            score = hs + as_
            if score > best_score:
                best_score, best_id = score, gid
        if best_id is not None:
            matched.append({
                "game_id": best_id, "spread": r["spread"],
                "over_under": r["over_under"],
                "home_moneyline": r["home_moneyline"],
                "away_moneyline": r["away_moneyline"],
                "spread_open": None, "provider": r["provider"],
            })
    if not matched:
        return pd.DataFrame()
    # If two odds events map to the same game, keep the best-scoring one
    return pd.DataFrame(matched).drop_duplicates("game_id", keep="first")


def fetch_lines(api_key: str, games_df: pd.DataFrame,
                timeout: int = 25) -> pd.DataFrame:
    """One-call convenience: events → consensus → matched CFBD lines.
    Returns an empty DataFrame if the key is missing or nothing matches."""
    if not api_key:
        return pd.DataFrame()
    events = fetch_ncaaf_events(api_key, timeout=timeout)
    consensus = events_to_consensus(events)
    return match_to_games(consensus, games_df)


def merge_lines(primary: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    """Combine two line frames keyed on game_id: every ``primary`` row wins,
    ``fill`` contributes only game_ids not already present. Used to back the
    consensus lines with CFBD for any game The Odds API didn't cover."""
    if primary is None or primary.empty:
        return fill if fill is not None else pd.DataFrame()
    if fill is None or fill.empty:
        return primary
    extra = fill[~fill["game_id"].isin(set(primary["game_id"]))]
    return pd.concat([primary, extra], ignore_index=True)
