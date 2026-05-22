"""
CFB Weekly Predictor — Phase 7
================================
Pulls the upcoming week's schedule and current Vegas lines from the CFBD API,
builds feature vectors from the latest available team data, runs the saved
spread / totals / win-probability models, and prints ranked bet recommendations.

Usage:
    python3 src/predict.py                          # auto-detect next week
    python3 src/predict.py --season 2026 --week 1
    python3 src/predict.py --season 2026 --week 1 --show-all   # include no-bet games

How features are built for upcoming games
------------------------------------------
Pre-season (week 1 or before any 2026 games exist):
  • Ratings  (SP+, FPI, SRS, recruiting, HFA) come from the 2025 final season
    values — already shifted +1 year in features.py, so they slot in as 2026
    pre-season baseline automatically.
  • Elo      is recomputed through the end of 2025 using EloSystem.run().
  • EPA      falls back to each team's last-5-game rolling average from 2025.
    Once 2026 games are played, re-run data_collection.py + features.py and
    the in-season rolling EPA will replace these.
  • Rest     defaults to 14 days for week 1 (season opener).
  • Weather  is not available for future games — totals predictions will be
    slightly less calibrated until the game date approaches.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# ─── PATHS ───────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).parent.parent
DATA_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RAW_DIR   = ROOT_DIR / "data" / "raw"

CFB_API_KEY  = os.getenv("CFB_API_KEY",
    "uxvnvwwBh6dQBE/hxA+GK+srmnfZ1mkRSr8E7gOg/BuIL/TeNHw5aHbbZDbi4TMt")
CFB_BASE_URL = "https://api.collegefootballdata.com"

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "97fefeb9de733240ae640967ed5c1427")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NCAAF_SPORT   = "americanfootball_ncaaf"

# Book priority for line selection (best liquidity / sharpest lines first)
BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "caesars",
                 "pointsbetus", "betrivers", "bovada"]

# Team name differences between Odds API and CFBD
ODDS_TO_CFBD = {
    "Louisiana State":   "LSU",
    "Mississippi":       "Ole Miss",
    "Southern California": "USC",
    "Central Florida":   "UCF",
    "Southern Methodist": "SMU",
    "Texas Christian":   "TCU",
    "Brigham Young":     "BYU",
    "Nevada Las Vegas":  "UNLV",
    "Pittsburgh":        "Pittsburgh",
    "Massachusetts":     "UMass",
    "Florida International": "FIU",
    "Middle Tennessee State": "Middle Tennessee",
    "North Carolina State": "NC State",
    "Miami (OH)":        "Miami (OH)",
    "Miami":             "Miami",
}

# ─── BETTING THRESHOLDS (from backtesting) ───────────────────────────────────

SPREAD_EDGE_MIN = 2.0   # minimum model-vs-Vegas disagreement to flag
SPREAD_EDGE_MAX = 5.0   # above this, Vegas probably has info you don't
TOTALS_EDGE_MIN = 3.0
TOTALS_EDGE_MAX = 6.0
MONEYLINE_EV_MIN = 0.04  # minimum expected value per $1 bet (4%)
MONEYLINE_EV_MAX = 0.08  # above this, model likely overconfident vs. market


# ─── MONEYLINE MATH HELPERS ──────────────────────────────────────────────────

def american_to_implied_prob(odds: float) -> float:
    """Convert American moneyline to raw implied probability (includes vig)."""
    if pd.isna(odds):
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(home_prob: float, away_prob: float):
    """
    Divide out the bookmaker's juice so home + away sum to 1.0.
    Returns (fair_home_prob, fair_away_prob).
    """
    if pd.isna(home_prob) or pd.isna(away_prob):
        return np.nan, np.nan
    total = home_prob + away_prob
    if total <= 0:
        return np.nan, np.nan
    return home_prob / total, away_prob / total


def prob_to_american(prob: float) -> float:
    """Convert a win probability to American odds (no vig)."""
    if pd.isna(prob) or prob <= 0 or prob >= 1:
        return np.nan
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round(((1 - prob) / prob) * 100)


def moneyline_ev(model_prob: float, american_odds: float) -> float:
    """
    Expected value per $1 wagered at the given American odds.
    Positive EV means your model's probability implies you're being
    offered better-than-fair odds.

    EV = (model_prob × win_payout) − (1 − model_prob)
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
    if american_odds < 0:
        win_payout = 100 / abs(american_odds)
    else:
        win_payout = american_odds / 100
    return model_prob * win_payout - (1 - model_prob)


# ─── API HELPER ──────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    url     = f"{CFB_BASE_URL}/{endpoint}"
    resp    = requests.get(url, headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─── ODDS API HELPERS ────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Map Odds API team names to CFBD team names where they differ."""
    return ODDS_TO_CFBD.get(name, name)


def fetch_odds_api_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch current NCAAF spread + total odds from The Odds API and match
    them to the CFBD schedule by team name.

    Returns a DataFrame with columns:
      game_id, spread, over_under, home_moneyline, away_moneyline, book
    ready to drop into build_features() as the lines argument.
    """
    from difflib import SequenceMatcher

    url = f"{ODDS_API_BASE}/sports/{NCAAF_SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  Odds API error: {e}")
        return pd.DataFrame()

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"  Odds API: {remaining} requests remaining this month (used {used} total)")

    data = resp.json()
    if not data:
        return pd.DataFrame()

    # ── Parse each game from the API response ─────────────────────────────
    odds_records = []
    for game in data:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home     = _normalize_name(home_raw)
        away     = _normalize_name(away_raw)

        # Pick the sharpest available book
        bookmakers = {b["key"]: b for b in game.get("bookmakers", [])}
        book = next((bookmakers[k] for k in BOOK_PRIORITY if k in bookmakers),
                    next(iter(bookmakers.values()), None) if bookmakers else None)

        spread = over_under = home_ml = away_ml = None
        book_name = None

        if book:
            book_name = book.get("title", book.get("key"))
            for market in book.get("markets", []):
                if market["key"] == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:   # home team's point = spread
                            spread = o["point"]
                elif market["key"] == "totals":
                    if market["outcomes"]:
                        over_under = market["outcomes"][0]["point"]
                elif market["key"] == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:
                            home_ml = o["price"]
                        elif o["name"] == away_raw:
                            away_ml = o["price"]

        odds_records.append({
            "odds_home": home, "odds_away": away,
            "spread": spread, "over_under": over_under,
            "home_moneyline": home_ml, "away_moneyline": away_ml,
            "book": book_name,
            "spread_open": None,   # not available on free tier
        })

    if not odds_records:
        return pd.DataFrame()

    odds_df = pd.DataFrame(odds_records)

    # ── Match Odds API games to CFBD game_ids ─────────────────────────────
    def sim(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    matched = []
    cfbd_teams = list(zip(games_df["game_id"], games_df["home_team"], games_df["away_team"]))

    for _, odds_row in odds_df.iterrows():
        best_id, best_score = None, 0.0
        for gid, cfbd_home, cfbd_away in cfbd_teams:
            score = (sim(odds_row["odds_home"], cfbd_home) +
                     sim(odds_row["odds_away"], cfbd_away)) / 2
            if score > best_score:
                best_score, best_id = score, gid

        if best_score >= 0.70:   # require a reasonable match
            matched.append({
                "game_id":        best_id,
                "spread":         odds_row["spread"],
                "over_under":     odds_row["over_under"],
                "home_moneyline": odds_row["home_moneyline"],
                "away_moneyline": odds_row["away_moneyline"],
                "spread_open":    None,
                "provider":       odds_row["book"],
            })

    result = pd.DataFrame(matched)
    if not result.empty:
        result = result.drop_duplicates("game_id", keep="first")

    print(f"  Odds API matched {len(result)} of {len(games_df)} games")
    return result


# ─── 1. LOAD SAVED MODELS ────────────────────────────────────────────────────

def load_models():
    """Load the three saved models and their feature lists."""
    for f in ["spread_model.pkl", "totals_model.pkl",
              "win_prob_model.pkl", "feature_lists.json"]:
        if not (MODEL_DIR / f).exists():
            print(f"❌ Missing {f} — run python3 src/model.py first.")
            sys.exit(1)

    spread_model   = joblib.load(MODEL_DIR / "spread_model.pkl")
    totals_model   = joblib.load(MODEL_DIR / "totals_model.pkl")
    win_prob_model = joblib.load(MODEL_DIR / "win_prob_model.pkl")

    with open(MODEL_DIR / "feature_lists.json") as f:
        feature_lists = json.load(f)

    return spread_model, totals_model, win_prob_model, feature_lists


# ─── 2. LOAD TEAM DATA ───────────────────────────────────────────────────────

def load_team_ratings(pred_season: int) -> dict:
    """
    Load all per-team rating data for the prediction season.

    pred_season = the season we're predicting (e.g. 2026).
    Because features.py shifts SP+/FPI/SRS forward by +1, the 2025 raw file
    already represents 2026 pre-season ratings after the shift.
    We load with the shift already applied — so we query by pred_season directly.
    """
    ratings = {}

    # ── SP+ ──────────────────────────────────────────────────────────────────
    sp_path = DATA_DIR / "master_sp_ratings.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        import ast

        def safe_parse(val):
            if pd.isna(val): return {}
            if isinstance(val, dict): return val
            try: return ast.literal_eval(val)
            except: return {}

        sp["off_dict"] = sp["offense"].apply(safe_parse)
        sp["def_dict"] = sp["defense"].apply(safe_parse)
        sp["sp_offense"] = sp["off_dict"].apply(lambda d: d.get("rating"))
        sp["sp_defense"] = sp["def_dict"].apply(lambda d: d.get("rating"))

        year_col = "year" if "year" in sp.columns else "season"
        sp = sp.rename(columns={year_col: "season", "rating": "sp_rating"})
        sp["season"] = sp["season"] + 1   # apply same leakage shift as features.py

        sp_cur = sp[sp["season"] == pred_season][
            ["team", "sp_rating", "sp_offense", "sp_defense"]
        ].set_index("team")
        ratings["sp"] = sp_cur

    # ── FPI ───────────────────────────────────────────────────────────────────
    fpi_path = DATA_DIR / "master_fpi_ratings.csv"
    if fpi_path.exists():
        fpi = pd.read_csv(fpi_path)
        fpi.columns = [c.lower() for c in fpi.columns]
        if "school" in fpi.columns: fpi = fpi.rename(columns={"school": "team"})
        if "year"   in fpi.columns: fpi = fpi.rename(columns={"year": "season"})
        fpi["season"] = pd.to_numeric(fpi["season"], errors="coerce") + 1
        if "fpi" in fpi.columns:
            fpi_cur = fpi[fpi["season"] == pred_season][["team","fpi"]].set_index("team")
            ratings["fpi"] = fpi_cur

    # ── SRS ───────────────────────────────────────────────────────────────────
    srs_path = DATA_DIR / "master_srs_ratings.csv"
    if srs_path.exists():
        srs = pd.read_csv(srs_path)
        srs.columns = [c.lower() for c in srs.columns]
        if "school" in srs.columns: srs = srs.rename(columns={"school": "team"})
        if "year"   in srs.columns: srs = srs.rename(columns={"year": "season"})
        srs["season"] = pd.to_numeric(srs["season"], errors="coerce") + 1
        if "rating" in srs.columns: srs = srs.rename(columns={"rating": "srs"})
        if "srs" in srs.columns:
            srs_cur = srs[srs["season"] == pred_season][["team","srs"]].set_index("team")
            ratings["srs"] = srs_cur

    # ── Recruiting (4-year rolling) ───────────────────────────────────────────
    rec_path = DATA_DIR / "master_recruiting.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rec.columns = [c.lower() for c in rec.columns]
        if "points" not in rec.columns and "total" in rec.columns:
            rec = rec.rename(columns={"total": "points"})
        rec = rec.sort_values(["team", "year"])
        rec["recruiting_4yr"] = (
            rec.groupby("team")["points"]
               .transform(lambda x: x.rolling(4, min_periods=1).mean())
        )
        # Use most recent available year (pred_season - 1)
        rec_cur = rec[rec["year"] == pred_season - 1][
            ["team", "recruiting_4yr"]
        ].set_index("team")
        ratings["recruiting"] = rec_cur

    # ── Home Field Advantage (from feature matrix, last computed) ─────────────
    fm_path = DATA_DIR / "feature_matrix.csv"
    if fm_path.exists():
        fm = pd.read_csv(fm_path,
            usecols=lambda c: c in ["season","home_team","home_hfa"])
        fm = fm[fm["season"] == pred_season - 1]
        if not fm.empty:
            hfa = fm.groupby("home_team")["home_hfa"].last().reset_index()
            hfa.columns = ["team", "hfa_estimate"]
            ratings["hfa"] = hfa.set_index("team")

    return ratings


def load_current_elo(pred_season: int) -> pd.DataFrame:
    """
    Recompute Elo through the end of pred_season-1 and return current ratings.
    Returns DataFrame with index=team, column=elo.
    """
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from elo_ratings import EloSystem

    games_path = DATA_DIR / "master_games.csv"
    sp_path    = DATA_DIR / "master_sp_ratings.csv"

    if not games_path.exists():
        return pd.DataFrame(columns=["elo"])

    games = pd.read_csv(games_path)

    # Filter to FBS only
    if sp_path.exists():
        sp  = pd.read_csv(sp_path)
        fbs = set(sp["team"].unique())
        games = games[
            games["home_team"].isin(fbs) & games["away_team"].isin(fbs)
        ]

    # Only use completed games up to and including pred_season-1
    games = games[
        games["season"] <= (pred_season - 1)
    ].dropna(subset=["home_points", "away_points"])

    elo = EloSystem()
    elo.run(games)

    return elo.current_ratings_df().set_index("team")[["elo"]]


def load_recent_epa(pred_season: int) -> pd.DataFrame:
    """
    Get each team's rolling EPA from their last 5 games of pred_season-1.
    Used as a form proxy when no in-season pred_season data exists yet.
    Returns DataFrame indexed by team with epa columns.
    """
    ppa_path = DATA_DIR / "master_ppa_games.csv"
    if not ppa_path.exists():
        return pd.DataFrame()

    ppa = pd.read_csv(ppa_path)
    last_season = ppa[ppa["season"] == pred_season - 1].copy()

    if last_season.empty:
        return pd.DataFrame()

    last_season = last_season.sort_values(["team", "week"])

    # Last 5 games per team
    last5 = (
        last_season.groupby("team")
        .tail(5)
        .groupby("team")
        [["off_epa", "def_epa", "off_epa_pass", "off_epa_rush"]]
        .mean()
    )

    last5.columns = [
        "off_epa_roll5", "def_epa_roll5",
        "off_epa_pass_roll5", "off_epa_rush_roll5",
    ]

    # Last 3 games
    last3 = (
        last_season.groupby("team")
        .tail(3)
        .groupby("team")
        [["off_epa", "def_epa", "off_epa_pass", "off_epa_rush"]]
        .mean()
    )
    last3.columns = [
        "off_epa_roll3", "def_epa_roll3",
        "off_epa_pass_roll3", "off_epa_rush_roll3",
    ]

    return last3.join(last5, how="outer")


# ─── 3. FETCH SCHEDULE & LINES ───────────────────────────────────────────────

def fetch_schedule(season: int, week: int) -> pd.DataFrame:
    """Pull scheduled games for a given season and week from the CFBD API."""
    print(f"  Fetching schedule: {season} week {week}...")
    try:
        data = cfb_get("games", params={"year": season, "week": week,
                                         "seasonType": "regular"})
    except Exception as e:
        print(f"  ⚠️  Could not fetch schedule: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    records = []
    for g in data:
        records.append({
            "game_id":        g.get("id"),
            "season":         g.get("season"),
            "week":           g.get("week"),
            "home_team":      g.get("homeTeam"),
            "away_team":      g.get("awayTeam"),
            "home_conference": g.get("homeConference"),
            "away_conference": g.get("awayConference"),
            "neutral_site":   g.get("neutralSite", False),
            "start_date":     g.get("startDate"),
            "completed":      g.get("completed", False),
            # If the game is already played:
            "home_points":    g.get("homePoints"),
            "away_points":    g.get("awayPoints"),
            "home_pregame_elo": g.get("homePregameElo"),
            "away_pregame_elo": g.get("awayPregameElo"),
        })

    df = pd.DataFrame(records)
    df["neutral_site"]   = df["neutral_site"].fillna(False).astype(int)
    df["conference_game"] = (
        df["home_conference"].notna() &
        df["away_conference"].notna() &
        (df["home_conference"] == df["away_conference"])
    ).astype(int)

    return df


def fetch_lines(season: int, week: int) -> pd.DataFrame:
    """Pull betting lines for a given season and week."""
    print(f"  Fetching lines: {season} week {week}...")
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception as e:
        print(f"  ⚠️  Could not fetch lines: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet",
                "William Hill (New Jersey)", "FanDuel"]
    rank_map = {p: i for i, p in enumerate(priority)}

    records = []
    for game in data:
        game_id = game.get("id")
        for line in game.get("lines", []):
            records.append({
                "game_id":     game_id,
                "provider":    line.get("provider"),
                "spread":      line.get("spread"),
                "over_under":  line.get("overUnder"),
                "spread_open": line.get("spreadOpen"),
                "_rank": rank_map.get(line.get("provider", ""), len(priority)),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    best = (
        df.sort_values("_rank")
          .drop_duplicates("game_id", keep="first")
          .drop(columns=["_rank"])
          .reset_index(drop=True)
    )

    best["spread"]     = pd.to_numeric(best["spread"],     errors="coerce")
    best["over_under"] = pd.to_numeric(best["over_under"], errors="coerce")
    best["spread_open"]= pd.to_numeric(best["spread_open"],errors="coerce")

    return best[["game_id", "provider", "spread", "over_under", "spread_open"]]


# ─── 4. BUILD FEATURES FOR UPCOMING GAMES ────────────────────────────────────

def build_features(
    games:    pd.DataFrame,
    lines:    pd.DataFrame,
    ratings:  dict,
    epa:      pd.DataFrame,
    elo:      pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """
    Build a feature row for each game, matching the columns the saved models expect.
    Any feature that can't be computed is left as NaN (models handle via imputation).
    """
    # Merge lines onto games (including moneylines)
    if not lines.empty:
        line_cols = ["game_id", "spread", "over_under", "spread_open"]
        for ml_col in ["home_moneyline", "away_moneyline"]:
            if ml_col in lines.columns:
                line_cols.append(ml_col)
        df = games.merge(lines[line_cols], on="game_id", how="left")
    else:
        df = games.copy()
        df["spread"] = np.nan
        df["over_under"] = np.nan
        df["spread_open"] = np.nan

    if "home_moneyline" not in df.columns:
        df["home_moneyline"] = np.nan
    if "away_moneyline" not in df.columns:
        df["away_moneyline"] = np.nan

    # ── Add ratings for each team ──────────────────────────────────────────

    def get_rating(team, source_key, col, default=np.nan):
        src = ratings.get(source_key)
        if src is None or team not in src.index:
            return default
        if col not in src.columns:
            return default
        val = src.loc[team, col]
        # If duplicate index entries, take the first scalar value
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        try:
            return default if pd.isna(val) else val
        except Exception:
            return default

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_series = df[team_col]

        df[f"{side}_sp_rating"]  = team_series.map(lambda t: get_rating(t, "sp", "sp_rating"))
        df[f"{side}_sp_offense"] = team_series.map(lambda t: get_rating(t, "sp", "sp_offense"))
        df[f"{side}_sp_defense"] = team_series.map(lambda t: get_rating(t, "sp", "sp_defense"))
        df[f"{side}_fpi"]        = team_series.map(lambda t: get_rating(t, "fpi", "fpi"))
        df[f"{side}_srs"]        = team_series.map(lambda t: get_rating(t, "srs", "srs"))
        df[f"{side}_recruiting_4yr"] = team_series.map(
            lambda t: get_rating(t, "recruiting", "recruiting_4yr"))
        df[f"{side}_hfa"]        = team_series.map(
            lambda t: get_rating(t, "hfa", "hfa_estimate"))

        # Elo
        if not elo.empty:
            df[f"{side}_pregame_elo"] = team_series.map(
                lambda t: elo.loc[t, "elo"] if t in elo.index else np.nan)
        else:
            df[f"{side}_pregame_elo"] = np.nan

        # EPA from last season
        if not epa.empty:
            for col in epa.columns:
                df[f"{side}_{col}"] = team_series.map(
                    lambda t, c=col: epa.loc[t, c] if t in epa.index else np.nan)

    # ── Derived differentials ──────────────────────────────────────────────
    df["sp_diff"]     = df["home_sp_rating"]  - df["away_sp_rating"]
    df["sp_off_diff"] = df["home_sp_offense"] - df["away_sp_offense"]
    df["sp_def_diff"] = df["home_sp_defense"] - df["away_sp_defense"]
    df["elo_diff"]    = df["home_pregame_elo"]- df["away_pregame_elo"]
    df["fpi_diff"]    = df["home_fpi"]        - df["away_fpi"]
    df["srs_diff"]    = df["home_srs"]        - df["away_srs"]
    df["recruiting_diff"] = df["home_recruiting_4yr"] - df["away_recruiting_4yr"]
    df["hfa_diff"]    = df["home_hfa"].fillna(0) - df["away_hfa"].fillna(0)

    if "home_off_epa_roll3" in df.columns and "away_off_epa_roll3" in df.columns:
        df["epa_off_diff_roll3"] = df["home_off_epa_roll3"] - df["away_off_epa_roll3"]
        df["epa_def_diff_roll3"] = df["home_def_epa_roll3"] - df["away_def_epa_roll3"]

    # ── Rest days (default 14 for week 1, TBD for later weeks) ────────────
    df["home_rest_days"] = 14
    df["away_rest_days"] = 14
    df["rest_diff"]      = 0

    # ── Line movement ──────────────────────────────────────────────────────
    df["line_movement"]   = df["spread"] - df["spread_open"]
    df["line_moved_home"] = (df["line_movement"] < -1.0).astype(int)
    df["line_moved_away"] = (df["line_movement"] >  1.0).astype(int)

    # ── Vegas implied home margin ──────────────────────────────────────────
    df["vegas_home_margin"] = -df["spread"].fillna(0)

    # ── Only return the columns the models expect ──────────────────────────
    available = [f for f in feature_names if f in df.columns]
    feature_df = pd.DataFrame(index=df.index)
    for f in feature_names:
        feature_df[f] = df[f] if f in df.columns else np.nan

    return df, feature_df


# ─── 5. GENERATE PREDICTIONS ─────────────────────────────────────────────────

def generate_predictions(
    df:            pd.DataFrame,
    feature_df_sp: pd.DataFrame,
    feature_df_tot:pd.DataFrame,
    feature_df_win:pd.DataFrame,
    spread_model, totals_model, win_prob_model,
) -> pd.DataFrame:
    """Run all three models and attach predictions to the games DataFrame."""
    base_cols = ["game_id","season","week","home_team","away_team",
                 "neutral_site","conference_game",
                 "spread","over_under","spread_open",
                 "home_moneyline","away_moneyline"]
    out = df[[c for c in base_cols if c in df.columns]].copy()

    # Ensure moneyline columns exist
    if "home_moneyline" not in out.columns:
        out["home_moneyline"] = np.nan
    if "away_moneyline" not in out.columns:
        out["away_moneyline"] = np.nan

    out["pred_spread"] = spread_model.predict(feature_df_sp)
    out["pred_total"]  = totals_model.predict(feature_df_tot)
    out["pred_win_p"]  = win_prob_model.predict_proba(feature_df_win)[:, 1]
    out["pred_away_win_p"] = 1 - out["pred_win_p"]

    out["vegas_home_margin"] = -out["spread"]
    out["spread_edge"] = out["pred_spread"] - out["vegas_home_margin"]
    out["totals_edge"] = out["pred_total"]  - out["over_under"]

    # ── Moneyline Expected Value ───────────────────────────────────────────────
    # Step 1: implied probs from book (with vig baked in)
    out["implied_home_prob_raw"] = out["home_moneyline"].apply(american_to_implied_prob)
    out["implied_away_prob_raw"] = out["away_moneyline"].apply(american_to_implied_prob)

    # Step 2: remove vig → fair market probabilities
    out[["fair_home_prob", "fair_away_prob"]] = out.apply(
        lambda r: pd.Series(remove_vig(r["implied_home_prob_raw"], r["implied_away_prob_raw"])),
        axis=1,
    )

    # Step 3: model-implied American odds (no vig)
    out["model_home_ml"] = out["pred_win_p"].apply(prob_to_american)
    out["model_away_ml"] = out["pred_away_win_p"].apply(prob_to_american)

    # Step 4: EV for betting home vs. away
    out["home_ev"] = out.apply(
        lambda r: moneyline_ev(r["pred_win_p"], r["home_moneyline"]), axis=1)
    out["away_ev"] = out.apply(
        lambda r: moneyline_ev(r["pred_away_win_p"], r["away_moneyline"]), axis=1)

    # Step 5: best side and its EV
    def best_ml_bet(row):
        h, a = row["home_ev"], row["away_ev"]
        if pd.isna(h) and pd.isna(a):
            return pd.Series({"ml_bet_side": None, "ml_ev": np.nan, "ml_odds": np.nan})
        if pd.isna(a) or (not pd.isna(h) and h >= a):
            return pd.Series({"ml_bet_side": row["home_team"], "ml_ev": h, "ml_odds": row["home_moneyline"]})
        return pd.Series({"ml_bet_side": row["away_team"], "ml_ev": a, "ml_odds": row["away_moneyline"]})

    out[["ml_bet_side", "ml_ev", "ml_odds"]] = out.apply(best_ml_bet, axis=1)

    return out


# ─── 6. PRINT RECOMMENDATIONS ────────────────────────────────────────────────

def print_recommendations(preds: pd.DataFrame, show_all: bool = False):
    """Print a clean bet recommendation sheet."""
    if preds.empty:
        print("No games to display.")
        return

    has_lines = preds["spread"].notna().any()

    print("\n" + "═"*78)
    print(f"  CFB BET RECOMMENDATIONS — {int(preds['season'].iloc[0])} Season, "
          f"Week {int(preds['week'].iloc[0])}")
    print(f"  Model: spread (2–5pt edge window) | totals (3–6pt edge window)")
    print("═"*78)

    if not has_lines:
        print("\n  ⚠️  No Vegas lines found yet for this week.")
        print("  Lines typically appear 7–10 days before kickoff.")
        print("\n  Model projections (no edge calc without lines):")
        print(f"\n  {'Home Team':22s}  {'Away Team':22s}  {'Proj Spread':>12}  {'Proj Total':>10}  {'Home Win%':>10}")
        print("  " + "─"*80)
        for _, r in preds.sort_values("pred_spread", ascending=False).iterrows():
            sign = "+" if r["pred_spread"] > 0 else ""
            print(f"  {r['home_team']:22s}  {r['away_team']:22s}  "
                  f"{sign}{r['pred_spread']:>8.1f}      "
                  f"{r['pred_total']:>8.1f}      "
                  f"{r['pred_win_p']:>8.1%}")
        return

    # ── Moneyline bets ─────────────────────────────────────────────────────
    ml_bets = preds[
        preds["ml_ev"].notna() &
        (preds["ml_ev"] >= MONEYLINE_EV_MIN) &
        (preds["ml_ev"] <  MONEYLINE_EV_MAX)
    ].copy().sort_values("ml_ev", ascending=False)

    print(f"\n{'─'*78}")
    print(f"  MONEYLINES  ({len(ml_bets)} +EV bets | {MONEYLINE_EV_MIN:.0%}–{MONEYLINE_EV_MAX:.0%} EV window)")
    print(f"  Note: Underdogs dominate the edge (+52.7% ROI historical) — favorites rarely +EV")
    print(f"{'─'*78}")

    if ml_bets.empty:
        print("  No moneyline bets meet the EV threshold this week.")
        print("  (Favorites rarely offer +EV — look for underdog value)")
    else:
        print(f"  {'Bet on':22s}  {'Book ML':>8}  {'Model ML':>9}  {'EV%':>6}  {'Matchup'}")
        print("  " + "─"*70)
        for _, r in ml_bets.iterrows():
            model_ml_str = f"{int(r['model_home_ml']):>+d}" if r["ml_bet_side"] == r["home_team"] \
                           else f"{int(r['model_away_ml']):>+d}"
            book_ml_str  = f"{int(r['ml_odds']):>+d}"
            ev_str       = f"{r['ml_ev']:>+.1%}"
            flag         = " ★" if r["ml_ev"] >= 0.07 else ""
            print(f"  {r['ml_bet_side']:22s}  {book_ml_str:>8}  {model_ml_str:>9}  {ev_str}  "
                  f"{r['home_team']} vs {r['away_team']}{flag}")

    # ── Totals bets ────────────────────────────────────────────────────────
    tot_bets = preds[
        preds["totals_edge"].notna() &
        (preds["totals_edge"].abs() >= TOTALS_EDGE_MIN) &
        (preds["totals_edge"].abs() <= TOTALS_EDGE_MAX)
    ].copy()
    tot_bets["bet_side"] = tot_bets["totals_edge"].apply(
        lambda e: "OVER" if e > 0 else "UNDER")
    tot_bets = tot_bets.sort_values("totals_edge", key=abs, ascending=False)

    print(f"\n{'─'*78}")
    print(f"  TOTALS  ({len(tot_bets)} bets flagged | 3–6pt edge window)")
    print(f"  Note: model leans UNDER — unders win 59% historically")
    print(f"{'─'*78}")

    if tot_bets.empty:
        print("  No totals bets meet threshold this week.")
    else:
        print(f"  {'Side':6s}  {'Total':>6}  {'Model':>7}  {'Edge':>6}  "
              f"{'Matchup'}")
        print("  " + "─"*70)
        for _, r in tot_bets.iterrows():
            edge_str = f"{r['totals_edge']:>+5.1f}"
            flag = " ★" if abs(r["totals_edge"]) >= 5.0 else ""
            print(f"  {r['bet_side']:6s}  "
                  f"{r['over_under']:>6.1f}  "
                  f"{r['pred_total']:>7.1f}  "
                  f"{edge_str}  "
                  f"{r['home_team']} vs {r['away_team']}{flag}")

    # ── Spread bets ────────────────────────────────────────────────────────
    sp_bets = preds[
        preds["spread_edge"].notna() &
        (preds["spread_edge"].abs() >= SPREAD_EDGE_MIN) &
        (preds["spread_edge"].abs() <= SPREAD_EDGE_MAX)
    ].copy()
    sp_bets["bet_on"] = sp_bets.apply(
        lambda r: r["home_team"] if r["spread_edge"] > 0 else r["away_team"], axis=1)
    sp_bets["bet_line"] = sp_bets.apply(
        lambda r: f"{r['spread']:+.1f}" if not pd.isna(r["spread"]) else "N/A", axis=1)
    sp_bets = sp_bets.sort_values("spread_edge", key=abs, ascending=False)

    print(f"\n{'─'*78}")
    print(f"  SPREADS  ({len(sp_bets)} bets flagged | 2–5pt edge window | informational)")
    print(f"  Note: spread model near breakeven — use as secondary confirmation only")
    print(f"{'─'*78}")

    if sp_bets.empty:
        print("  No spread bets meet threshold this week.")
    else:
        print(f"  {'Bet on':22s}  {'Line':>6}  {'Model':>7}  {'Edge':>6}  "
              f"{'Matchup'}")
        print("  " + "─"*70)
        for _, r in sp_bets.iterrows():
            model_str = f"{r['pred_spread']:>+6.1f}"
            edge_str  = f"{r['spread_edge']:>+5.1f}"
            print(f"  {r['bet_on']:22s}  "
                  f"{r['bet_line']:>6}  "
                  f"{model_str}  "
                  f"{edge_str}  "
                  f"{r['home_team']} vs {r['away_team']}")

    # ── Show all games if requested ────────────────────────────────────────
    if show_all:
        print(f"\n{'─'*78}")
        print("  ALL GAMES THIS WEEK")
        print(f"{'─'*78}")
        print(f"  {'Home Team':22s}  {'Away Team':22s}  "
              f"{'Spread':>7}  {'Model':>7}  {'Edge':>6}  "
              f"{'Total':>6}  {'ProjTot':>7}  {'Win%':>6}")
        print("  " + "─"*78)
        for _, r in preds.sort_values("spread_edge", key=abs, ascending=False).iterrows():
            sp_str  = f"{r['spread']:>+6.1f}" if not pd.isna(r["spread"]) else "  N/A "
            mod_str = f"{r['pred_spread']:>+6.1f}"
            edg_str = f"{r['spread_edge']:>+5.1f}" if not pd.isna(r["spread_edge"]) else "  N/A"
            ou_str  = f"{r['over_under']:>5.1f}" if not pd.isna(r["over_under"]) else "  N/A"
            pt_str  = f"{r['pred_total']:>6.1f}"
            wp_str  = f"{r['pred_win_p']:.0%}"
            print(f"  {r['home_team']:22s}  {r['away_team']:22s}  "
                  f"{sp_str}  {mod_str}  {edg_str}  "
                  f"{ou_str}  {pt_str}  {wp_str}")

    print("\n" + "═"*78)
    print("  Moneyline ★ = EV ≥ 7%  |  Totals ★ = edge ≥ 5pts  |  UNDER bias: +59% historical")
    print("  EV >3% = worth considering  |  EV >7% = strong signal  |  EV <3% = skip")
    print("  Always cross-check: injuries, weather forecast, line movement direction")
    print("═"*78 + "\n")


# ─── 7. MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CFB bet recommendations for an upcoming week.")
    parser.add_argument("--season", type=int, default=2026,
        help="Season year (default: 2026)")
    parser.add_argument("--week", type=int, default=1,
        help="Week number (default: 1)")
    parser.add_argument("--show-all", action="store_true",
        help="Show all games, not just flagged bets")
    args = parser.parse_args()

    season = args.season
    week   = args.week

    print(f"\n{'='*55}")
    print(f"CFB Predictor — {season} Season, Week {week}")
    print(f"{'='*55}")

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading models...")
    spread_model, totals_model, win_prob_model, feature_lists = load_models()
    print("  ✓ Models loaded")

    # ── Load team data ────────────────────────────────────────────────────
    print("\nLoading team data...")
    ratings = load_team_ratings(season)
    print(f"  ✓ Ratings loaded "
          f"(SP+: {'✓' if 'sp' in ratings else '✗'}  "
          f"FPI: {'✓' if 'fpi' in ratings else '✗'}  "
          f"SRS: {'✓' if 'srs' in ratings else '✗'})")

    print("  Computing current Elo ratings...")
    elo = load_current_elo(season)
    print(f"  ✓ Elo ratings: {len(elo)} teams")

    print("  Loading recent EPA (end of prior season)...")
    epa = load_recent_epa(season)
    print(f"  ✓ EPA data: {len(epa)} teams")

    # ── Fetch schedule (CFBD) ────────────────────────────────────────────
    print(f"\nFetching {season} week {week} schedule from CFBD...")
    games = fetch_schedule(season, week)

    if games.empty:
        print(f"\n⚠️  No games found for {season} week {week}.")
        print("   The schedule may not be posted yet, or all games are listed")
        print("   under a different week number. Try --week 0 for week 0 games.")
        return

    print(f"  ✓ {len(games)} games on schedule")

    # ── Fetch lines: try Odds API first, fall back to CFBD ───────────────
    print("Fetching lines from The Odds API...")
    lines = fetch_odds_api_lines(games)

    if lines.empty:
        print("  Falling back to CFBD lines...")
        lines = fetch_lines(season, week)

    n_lines = lines["game_id"].nunique() if not lines.empty else 0
    source  = lines["provider"].iloc[0] if (not lines.empty and "provider" in lines.columns) else "unknown"
    print(f"  ✓ {n_lines} games with lines (source: {source})")

    # ── Build features ────────────────────────────────────────────────────
    print("\nBuilding feature vectors...")
    games_df, feat_sp  = build_features(
        games, lines, ratings, epa, elo, feature_lists["spread"])
    _,         feat_tot = build_features(
        games, lines, ratings, epa, elo, feature_lists["totals"])
    _,         feat_win = build_features(
        games, lines, ratings, epa, elo, feature_lists["win_prob"])

    # ── Run predictions ───────────────────────────────────────────────────
    print("Running models...")
    preds = generate_predictions(
        games_df, feat_sp, feat_tot, feat_win,
        spread_model, totals_model, win_prob_model)

    # ── Print recommendations ─────────────────────────────────────────────
    print_recommendations(preds, show_all=args.show_all)

    # ── Save to CSV ───────────────────────────────────────────────────────
    out_dir = ROOT_DIR / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"week_{season}_w{week:02d}_predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"Full predictions saved → {out_path}\n")


if __name__ == "__main__":
    main()
