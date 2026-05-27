"""
CFB Betting Model — Data Collection
====================================
Pulls game data, team stats, and betting lines from:
  - collegefootballdata.com (game results + advanced stats)
  - The Odds API (historical betting lines)

Setup:
  1. Get a free API key at https://collegefootballdata.com/key
  2. Get a free API key at https://the-odds-api.com
  3. Paste your keys below (or set them as environment variables)
  4. Run: python src/data_collection.py
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CFB_API_KEY  = os.getenv("CFB_API_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

CFB_BASE_URL = "https://api.collegefootballdata.com"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
LINES_DIR = DATA_DIR / "lines"

SEASONS = list(range(2015, 2026))  # 2015–2025

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    """Make a GET request to the CFB Data API."""
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    url = f"{CFB_BASE_URL}/{endpoint}"
    response = requests.get(url, headers=headers, params=params or {})
    response.raise_for_status()
    return response.json()


def save_json(data, filepath: Path):
    """Save data as a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} records → {filepath.name}")


def save_csv(df: pd.DataFrame, filepath: Path):
    """Save a DataFrame as CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} rows → {filepath.name}")


# ─── DATA COLLECTION FUNCTIONS ───────────────────────────────────────────────

def collect_games(season: int) -> pd.DataFrame:
    """
    Pull all regular season + bowl game results for a given season.
    Returns a DataFrame with: game_id, season, week, home_team, away_team,
    home_points, away_points, neutral_site, etc.
    """
    print(f"  Pulling games for {season}...")
    data = cfb_get("games", params={"year": season, "seasonType": "regular"})
    bowl_data = cfb_get("games", params={"year": season, "seasonType": "postseason"})
    all_games = data + bowl_data

    df = pd.DataFrame(all_games)

    # API returns camelCase — keep only completed games with scores
    df = df[df["completed"] == True].copy()

    # Rename camelCase → snake_case for consistency
    rename_map = {
        "id": "game_id",
        "seasonType": "season_type",
        "startDate": "start_date",
        "neutralSite": "neutral_site",
        "conferenceGame": "conference_game",
        "homeTeam": "home_team",
        "homeConference": "home_conference",
        "homePoints": "home_points",
        "awayTeam": "away_team",
        "awayConference": "away_conference",
        "awayPoints": "away_points",
        "homePregameElo": "home_pregame_elo",
        "awayPregameElo": "away_pregame_elo",
        "excitementIndex": "excitement_index",
    }
    # Only rename columns that exist in the response
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only the columns we care about
    keep = [
        "game_id", "season", "week", "season_type", "start_date",
        "neutral_site", "conference_game",
        "home_team", "home_conference", "home_points",
        "away_team", "away_conference", "away_points",
        "home_pregame_elo", "away_pregame_elo",
        "excitement_index",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df["point_diff"] = df["home_points"] - df["away_points"]
    df["total_points"] = df["home_points"] + df["away_points"]

    return df


def collect_team_stats(season: int) -> pd.DataFrame:
    """
    Pull season-level team stats (offense + defense).
    Returns per-team aggregated stats for the season.
    """
    print(f"  Pulling team stats for {season}...")
    data = cfb_get("stats/season", params={"year": season})
    df = pd.DataFrame(data)
    return df


def collect_advanced_stats(season: int) -> pd.DataFrame:
    """
    Pull EPA-based advanced stats from the PPA endpoint.
    These are the most predictive features for game modeling.
    Includes: offense EPA/play, defense EPA/play, success rate, explosiveness.
    """
    print(f"  Pulling advanced stats for {season}...")
    data = cfb_get("ppa/teams", params={"year": season})
    df = pd.DataFrame(data)
    return df


def collect_sp_ratings(season: int) -> pd.DataFrame:
    """
    Pull SP+ (Bill Connelly's efficiency ratings) — one of the best
    public team ratings. Great for cross-validating your own model.
    """
    print(f"  Pulling SP+ ratings for {season}...")
    data = cfb_get("ratings/sp", params={"year": season})
    df = pd.DataFrame(data)
    return df


def collect_fpi_ratings(season: int) -> pd.DataFrame:
    """
    Pull ESPN Football Power Index (FPI) ratings for all teams.
    FPI is ESPN's composite efficiency metric — it captures team quality
    independently of SP+, making it a useful second opinion.
    Available through the CFBD API (no extra key needed).
    """
    print(f"  Pulling FPI ratings for {season}...")
    data = cfb_get("ratings/fpi", params={"year": season})
    df = pd.DataFrame(data)
    if not df.empty and "year" not in df.columns:
        df["year"] = season
    return df


def collect_srs_ratings(season: int) -> pd.DataFrame:
    """
    Pull Simple Rating System (SRS) ratings.
    SRS = average point differential adjusted for strength of schedule.
    It's a clean, interpretable metric: +7.0 means the team is ~7 points
    better per game than an average opponent on a neutral field.
    """
    print(f"  Pulling SRS ratings for {season}...")
    data = cfb_get("ratings/srs", params={"year": season})
    df = pd.DataFrame(data)
    if not df.empty and "year" not in df.columns:
        df["year"] = season
    return df


def collect_recruiting(season: int) -> pd.DataFrame:
    """
    Pull team recruiting composite rankings.
    Used as a long-term talent proxy (typically averaged over 4 years).
    """
    print(f"  Pulling recruiting rankings for {season}...")
    data = cfb_get("recruiting/teams", params={"year": season})
    df = pd.DataFrame(data)
    return df


def collect_transfer_portal(season: int) -> pd.DataFrame:
    """
    Pull transfer portal entries for a given season.

    Each row = one player who entered the portal, with:
      - origin:      team they left
      - destination: team they joined (NaN if uncommitted)
      - position:    e.g. "QB", "WR", "OL"
      - stars:       2–5 star rating
      - rating:      composite recruiting rating (0.0–1.0 scale)

    Year convention: year=N returns players who transferred TO PLAY IN season N.
    A player entering the portal in Dec 2025 and committing in Jan 2026 appears
    under year=2026, and is used to predict 2026 games. No leakage shift needed.

    This is the single biggest unmodeled factor in CFB — teams that lost their
    starting QB via portal look identical to a team that returned everyone until
    actual games are played. Portal features fix that blind spot.
    """
    print(f"  Pulling transfer portal for {season}...")
    try:
        headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
        resp = requests.get(
            f"{CFB_BASE_URL}/player/portal",
            headers=headers,
            params={"year": season},
            timeout=15,
        )
        resp.raise_for_status()

        raw = resp.text.strip()
        if not raw or raw in ("null", "[]", ""):
            # API returned empty — free tier keys don't include portal data.
            # Upgrade to Patreon tier at https://www.patreon.com/collegefootballdata
            # or use the manual CSV approach described in the README.
            return pd.DataFrame()

        data = resp.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["season"] = season
        rename = {
            "firstName": "first_name", "lastName": "last_name",
            "transferDate": "transfer_date",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        print(f"    {len(df)} portal entries for {season}")
        return df

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status in (401, 403):
            print(f"    ⚠️  Portal data requires Patreon API access (status {status}).")
            print(f"         Upgrade at https://www.patreon.com/collegefootballdata")
        else:
            print(f"    Portal HTTP error for {season}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    Portal data unavailable for {season}: {e}")
        return pd.DataFrame()


def build_portal_team_features(portal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw portal entries into per-team per-season features.

    Returns a DataFrame keyed by (season, team) with:
      portal_talent_in    — sum of incoming player ratings (0–1 scale × 100 for readability)
      portal_talent_out   — sum of outgoing player ratings
      portal_net_rating   — incoming − outgoing talent (positive = net gain)
      portal_count_in     — # players gained
      portal_count_out    — # players lost
      portal_net_count    — count_in − count_out
      portal_stars_in_avg — avg star rating of incoming players
      portal_qb_in        — 1 if team gained a QB, else 0
      portal_qb_out       — 1 if team lost a QB, else 0
    """
    if portal_df.empty or "origin" not in portal_df.columns:
        return pd.DataFrame()

    df = portal_df.copy()
    # Ensure numeric
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0) * 100
    df["stars"]  = pd.to_numeric(df.get("stars",  0), errors="coerce").fillna(0)
    pos_col = "position" if "position" in df.columns else None

    # ── Incoming (destination = this team) ───────────────────────────────
    incoming = df[df["destination"].notna() & (df["destination"] != "")].copy()
    if incoming.empty:
        in_df = pd.DataFrame(columns=["season", "team", "portal_talent_in",
                                       "portal_count_in", "portal_stars_in_avg",
                                       "portal_qb_in"])
    else:
        def agg_in(g):
            return pd.Series({
                "portal_talent_in":    g["rating"].sum(),
                "portal_count_in":     len(g),
                "portal_stars_in_avg": g["stars"].mean(),
                "portal_qb_in":        int((g[pos_col].str.upper() == "QB").any()) if pos_col else 0,
            })
        in_df = (incoming.groupby(["season", "destination"])
                         .apply(agg_in).reset_index()
                         .rename(columns={"destination": "team"}))

    # ── Outgoing (origin = this team) ─────────────────────────────────────
    outgoing = df[df["origin"].notna() & (df["origin"] != "")].copy()
    if outgoing.empty:
        out_df = pd.DataFrame(columns=["season", "team", "portal_talent_out",
                                        "portal_count_out", "portal_qb_out"])
    else:
        def agg_out(g):
            return pd.Series({
                "portal_talent_out": g["rating"].sum(),
                "portal_count_out":  len(g),
                "portal_qb_out":     int((g[pos_col].str.upper() == "QB").any()) if pos_col else 0,
            })
        out_df = (outgoing.groupby(["season", "origin"])
                          .apply(agg_out).reset_index()
                          .rename(columns={"origin": "team"}))

    # ── Merge in/out ──────────────────────────────────────────────────────
    feat = in_df.merge(out_df, on=["season", "team"], how="outer")
    for col in ["portal_talent_in", "portal_count_in", "portal_stars_in_avg",
                "portal_qb_in", "portal_talent_out", "portal_count_out", "portal_qb_out"]:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = feat[col].fillna(0)

    feat["portal_net_rating"] = feat["portal_talent_in"] - feat["portal_talent_out"]
    feat["portal_net_count"]  = feat["portal_count_in"]  - feat["portal_count_out"]

    return feat[["season", "team", "portal_talent_in", "portal_talent_out",
                 "portal_net_rating", "portal_count_in", "portal_count_out",
                 "portal_net_count", "portal_stars_in_avg",
                 "portal_qb_in", "portal_qb_out"]]


def collect_wepa(season: int) -> pd.DataFrame:
    """
    Pull opponent-adjusted efficiency stats per team per season.

    The /wepa/team/season endpoint returns opponent-adjusted EPA (epa.total),
    EPA allowed (epaAllowed.total), success rates, and explosiveness.
    These are more predictive than raw EPA because they account for schedule quality.

    Field structure (confirmed from API):
      epa.total            → offensive EPA per play (opponent-adjusted)
      epaAllowed.total     → defensive EPA allowed per play (opponent-adjusted)
      successRate.total    → offensive success rate
      successRateAllowed.total → defensive success rate allowed
      explosiveness        → offensive play-by-play explosiveness
      explosivenessAllowed → defensive explosiveness allowed

    Available from 2014 onward.
    """
    print(f"  Pulling WEPA for {season}...")
    try:
        data = cfb_get("wepa/team/season", params={"year": season})
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        # Normalize team name and year → season
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = season

        # Extract from nested dicts — confirmed field names from API
        if "epa" in df.columns:
            df["wepa_offense"] = df["epa"].apply(
                lambda x: x.get("total") if isinstance(x, dict) else None)
        if "epaAllowed" in df.columns:
            df["wepa_defense"] = df["epaAllowed"].apply(
                lambda x: x.get("total") if isinstance(x, dict) else None)
        if "successRate" in df.columns:
            df["wepa_success_off"] = df["successRate"].apply(
                lambda x: x.get("total") if isinstance(x, dict) else None)
        if "successRateAllowed" in df.columns:
            df["wepa_success_def"] = df["successRateAllowed"].apply(
                lambda x: x.get("total") if isinstance(x, dict) else None)
        # Flat numeric fields
        df["wepa_explosiveness"]     = pd.to_numeric(df.get("explosiveness"),        errors="coerce")
        df["wepa_explosiveness_def"] = pd.to_numeric(df.get("explosivenessAllowed"), errors="coerce")

        keep = [c for c in ["season", "team",
                             "wepa_offense", "wepa_defense",
                             "wepa_success_off", "wepa_success_def",
                             "wepa_explosiveness", "wepa_explosiveness_def"]
                if c in df.columns]
        result = df[keep].copy()
        for col in keep:
            if col not in ("season", "team"):
                result[col] = pd.to_numeric(result[col], errors="coerce")

        print(f"    {len(result)} WEPA rows for {season}")
        return result

    except Exception as e:
        print(f"    WEPA unavailable for {season}: {e}")
        return pd.DataFrame()


def collect_talent(season: int) -> pd.DataFrame:
    """
    Pull team talent composite ratings.

    Based on 247Sports composite ratings for ALL scholarship players currently
    on the roster (not just one recruiting class). More predictive than a
    4-year recruiting average because it reflects actual current roster quality —
    captures both portal additions and attrition.

    Available from ~2015 onward. No year-shift needed in features.py:
    talent[year=N] reflects the roster heading into season N.
    """
    print(f"  Pulling talent composite for {season}...")
    try:
        data = cfb_get("talent", params={"year": season})
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        # Normalize column names
        rename = {"school": "team"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = season

        keep = [c for c in ["season", "team", "talent"] if c in df.columns]
        print(f"    {len(df)} talent rows for {season}")
        return df[keep].copy()

    except Exception as e:
        print(f"    Talent data unavailable for {season}: {e}")
        return pd.DataFrame()


def collect_havoc(season: int) -> pd.DataFrame:
    """
    Pull advanced team stats including havoc rate from the stats/season/advanced endpoint.

    Havoc rate = % of opponent plays resulting in a TFL, sack, forced fumble, or PBU.
    It's one of the most predictive defensive metrics — high havoc disrupts opposing
    offenses regardless of whether that shows up in EPA yet.

    Also captures rush/passing success rates and explosiveness.
    Garbage-time plays excluded (excludeGarbageTime=true) for cleaner signal.
    """
    print(f"  Pulling advanced stats (havoc) for {season}...")
    try:
        data = cfb_get("stats/season/advanced",
                       params={"year": season, "excludeGarbageTime": "true"})
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        # Flatten nested havoc sub-dict
        if "defense" in df.columns:
            df["havoc_total"]       = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("total")      if isinstance(x, dict) else None)
            df["havoc_front_seven"] = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("frontSeven") if isinstance(x, dict) else None)
            df["havoc_db"]          = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("db")         if isinstance(x, dict) else None)

        # Offensive success rate + explosiveness (from same response)
        if "offense" in df.columns:
            df["rush_success_rate"] = df["offense"].apply(
                lambda x: x.get("rushingPlays", {}).get("successRate") if isinstance(x, dict) else None)
            df["pass_success_rate"] = df["offense"].apply(
                lambda x: x.get("passingDowns", {}).get("successRate") if isinstance(x, dict) else None)
            # Explosiveness = average EPA on "big" plays (10+ yd pass, 5+ yd rush)
            # API returns explosiveness as either a nested dict {"total":x,"rushing":x,...}
            # or a plain float depending on season — handle both safely.
            def _exp(obj, subkey):
                if not isinstance(obj, dict):
                    return None
                val = obj.get("explosiveness")
                if isinstance(val, dict):
                    return val.get(subkey)
                if subkey == "total" and val is not None:
                    return val  # plain float IS the total
                return None

            df["explosiveness_off"]      = df["offense"].apply(lambda x: _exp(x, "total"))
            df["explosiveness_off_rush"] = df["offense"].apply(lambda x: _exp(x, "rushing"))
            df["explosiveness_off_pass"] = df["offense"].apply(lambda x: _exp(x, "passing"))

        # Defensive explosiveness allowed (lower = better D vs. big plays)
        if "defense" in df.columns:
            def _exp_def(obj):
                if not isinstance(obj, dict):
                    return None
                val = obj.get("explosiveness")
                if isinstance(val, dict):
                    return val.get("total")
                return val  # plain float
            df["explosiveness_def"] = df["defense"].apply(_exp_def)

        # Normalize column names
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = season

        keep = [c for c in ["season", "team",
                             "havoc_total", "havoc_front_seven", "havoc_db",
                             "rush_success_rate", "pass_success_rate",
                             "explosiveness_off", "explosiveness_off_rush",
                             "explosiveness_off_pass", "explosiveness_def"]
                if c in df.columns]
        result = df[keep].copy()
        # Convert to numeric
        for col in keep:
            if col not in ("season", "team"):
                result[col] = pd.to_numeric(result[col], errors="coerce")
        print(f"    {len(result)} havoc rows for {season}")
        return result

    except Exception as e:
        print(f"    Havoc data unavailable for {season}: {e}")
        return pd.DataFrame()


def collect_game_lines(season: int) -> pd.DataFrame:
    """
    Pull historical betting lines (spread + total) for each game.
    Source: CFB Data API's lines endpoint (aggregates multiple books).
    """
    print(f"  Pulling betting lines for {season}...")
    data = cfb_get("lines", params={"year": season})

    records = []
    for game in data:
        game_id = game.get("id")
        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")
        season_val = game.get("season")
        week = game.get("week")

        for line in game.get("lines", []):
            records.append({
                "game_id": game_id,
                "season": season_val,
                "week": week,
                "home_team": home_team,
                "away_team": away_team,
                "provider": line.get("provider"),
                "spread": line.get("spread"),
                "formatted_spread": line.get("formattedSpread"),
                "spread_open": line.get("spreadOpen"),
                "over_under": line.get("overUnder"),
                "over_under_open": line.get("overUnderOpen"),
                "home_moneyline": line.get("homeMoneyline"),
                "away_moneyline": line.get("awayMoneyline"),
            })

    df = pd.DataFrame(records)
    return df


def collect_ppa_games(season: int) -> pd.DataFrame:
    """
    Pull per-game EPA (PPA) data for every team in a season.
    This gives us game-by-game efficiency numbers we can use to build
    rolling averages (last 3 games, season-to-date, etc.).

    Returns one row per team per game with offensive and defensive EPA/play.
    """
    print(f"  Pulling per-game EPA for {season}...")
    data = cfb_get("ppa/games", params={"year": season, "seasonType": "regular"})
    bowl = cfb_get("ppa/games", params={"year": season, "seasonType": "postseason"})
    all_data = data + bowl

    records = []
    for row in all_data:
        records.append({
            "game_id":        row.get("gameId"),
            "season":         row.get("season"),
            "week":           row.get("week"),
            "team":           row.get("team"),
            "conference":     row.get("conference"),
            "opponent":       row.get("opponent"),
            "off_epa":        row.get("offense", {}).get("overall"),
            "off_epa_pass":   row.get("offense", {}).get("passing"),
            "off_epa_rush":   row.get("offense", {}).get("rushing"),
            "def_epa":        row.get("defense", {}).get("overall"),
            "def_epa_pass":   row.get("defense", {}).get("passing"),
            "def_epa_rush":   row.get("defense", {}).get("rushing"),
        })

    df = pd.DataFrame(records)
    return df


# ─── MASTER COLLECTION RUNNER ─────────────────────────────────────────────────

def collect_all_seasons():
    """
    Pull all data for every season in SEASONS list.
    Saves raw files to data/raw/ and data/lines/.
    Then merges everything into a master dataset.
    """
    all_games = []
    all_advanced = []
    all_ratings = []
    all_recruiting = []
    all_lines = []
    all_ppa_games = []
    all_fpi = []
    all_srs = []
    all_portal = []
    all_wepa = []
    all_talent = []
    all_havoc = []

    for season in SEASONS:
        print(f"\n{'='*50}")
        print(f"Season: {season}")
        print(f"{'='*50}")

        try:
            # Games
            games_df = collect_games(season)
            save_csv(games_df, RAW_DIR / f"games_{season}.csv")
            all_games.append(games_df)
            time.sleep(0.5)  # Be polite to the API

            # Advanced stats (EPA season-level)
            adv_df = collect_advanced_stats(season)
            save_csv(adv_df, RAW_DIR / f"advanced_stats_{season}.csv")
            all_advanced.append(adv_df)
            time.sleep(0.5)

            # Per-game EPA (for rolling averages)
            ppa_df = collect_ppa_games(season)
            save_csv(ppa_df, RAW_DIR / f"ppa_games_{season}.csv")
            all_ppa_games.append(ppa_df)
            time.sleep(0.5)

            # SP+ ratings
            ratings_df = collect_sp_ratings(season)
            save_csv(ratings_df, RAW_DIR / f"sp_ratings_{season}.csv")
            all_ratings.append(ratings_df)
            time.sleep(0.5)

            # Recruiting
            recruiting_df = collect_recruiting(season)
            save_csv(recruiting_df, RAW_DIR / f"recruiting_{season}.csv")
            all_recruiting.append(recruiting_df)
            time.sleep(0.5)

            # Betting lines
            lines_df = collect_game_lines(season)
            save_csv(lines_df, LINES_DIR / f"lines_{season}.csv")
            all_lines.append(lines_df)
            time.sleep(0.5)

            # ESPN FPI ratings
            fpi_df = collect_fpi_ratings(season)
            save_csv(fpi_df, RAW_DIR / f"fpi_ratings_{season}.csv")
            all_fpi.append(fpi_df)
            time.sleep(0.5)

            # SRS ratings
            srs_df = collect_srs_ratings(season)
            save_csv(srs_df, RAW_DIR / f"srs_ratings_{season}.csv")
            all_srs.append(srs_df)
            time.sleep(0.5)

            # Transfer portal (available from 2018 onward)
            if season >= 2018:
                portal_df = collect_transfer_portal(season)
                if not portal_df.empty:
                    save_csv(portal_df, RAW_DIR / f"portal_{season}.csv")
                    all_portal.append(portal_df)
                time.sleep(0.5)

            # WEPA (opponent-adjusted EPA) — available 2014+
            wepa_df = collect_wepa(season)
            if not wepa_df.empty:
                save_csv(wepa_df, RAW_DIR / f"wepa_{season}.csv")
                all_wepa.append(wepa_df)
            time.sleep(0.5)

            # Talent composite (247Sports roster ratings) — available ~2015+
            talent_df = collect_talent(season)
            if not talent_df.empty:
                save_csv(talent_df, RAW_DIR / f"talent_{season}.csv")
                all_talent.append(talent_df)
            time.sleep(0.5)

            # Advanced stats — havoc rate, success rates — available ~2014+
            havoc_df = collect_havoc(season)
            if not havoc_df.empty:
                save_csv(havoc_df, RAW_DIR / f"havoc_{season}.csv")
                all_havoc.append(havoc_df)
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR on {season}: {e}")
            continue

    # ─── Merge into master datasets ─────────────────────────────────────────

    print("\n\nMerging all seasons into master files...")

    if all_games:
        master_games = pd.concat(all_games, ignore_index=True)
        save_csv(master_games, DATA_DIR / "processed" / "master_games.csv")

    if all_advanced:
        master_advanced = pd.concat(all_advanced, ignore_index=True)
        save_csv(master_advanced, DATA_DIR / "processed" / "master_advanced_stats.csv")

    if all_ratings:
        master_ratings = pd.concat(all_ratings, ignore_index=True)
        save_csv(master_ratings, DATA_DIR / "processed" / "master_sp_ratings.csv")

    if all_recruiting:
        master_recruiting = pd.concat(all_recruiting, ignore_index=True)
        save_csv(master_recruiting, DATA_DIR / "processed" / "master_recruiting.csv")

    if all_lines:
        master_lines = pd.concat(all_lines, ignore_index=True)
        save_csv(master_lines, DATA_DIR / "processed" / "master_lines.csv")

    if all_ppa_games:
        master_ppa = pd.concat(all_ppa_games, ignore_index=True)
        save_csv(master_ppa, DATA_DIR / "processed" / "master_ppa_games.csv")

    if all_fpi:
        master_fpi = pd.concat(all_fpi, ignore_index=True)
        save_csv(master_fpi, DATA_DIR / "processed" / "master_fpi_ratings.csv")

    if all_srs:
        master_srs = pd.concat(all_srs, ignore_index=True)
        save_csv(master_srs, DATA_DIR / "processed" / "master_srs_ratings.csv")

    if all_portal:
        master_portal_raw = pd.concat(all_portal, ignore_index=True)
        save_csv(master_portal_raw, DATA_DIR / "processed" / "master_portal.csv")
        # Also pre-compute team-level features and save
        portal_features = build_portal_team_features(master_portal_raw)
        if not portal_features.empty:
            save_csv(portal_features, DATA_DIR / "processed" / "master_portal_features.csv")
            print(f"   Portal feature rows: {len(portal_features)}")

    if all_wepa:
        master_wepa = pd.concat(all_wepa, ignore_index=True)
        save_csv(master_wepa, DATA_DIR / "processed" / "master_wepa.csv")

    if all_talent:
        master_talent = pd.concat(all_talent, ignore_index=True)
        save_csv(master_talent, DATA_DIR / "processed" / "master_talent.csv")

    if all_havoc:
        master_havoc = pd.concat(all_havoc, ignore_index=True)
        save_csv(master_havoc, DATA_DIR / "processed" / "master_havoc.csv")

    print("\n✅ Data collection complete!")
    print(f"   Games collected:    {len(master_games) if all_games else 0}")
    print(f"   Lines collected:    {len(master_lines) if all_lines else 0}")
    print(f"   Per-game EPA rows:  {len(master_ppa) if all_ppa_games else 0}")
    print(f"\nNext step: run python3 src/features.py to build the feature matrix.")


def collect_single_season_quick(season: int = 2024):
    """
    Quick test — pull just one season to verify your API key works.
    Run this first before doing the full multi-season pull.
    """
    print(f"Quick test: pulling {season} games and lines...")
    games_df = collect_games(season)
    print(f"\nSample of {season} games:")
    print(games_df[["home_team", "away_team", "home_points", "away_points"]].head(10))

    lines_df = collect_game_lines(season)
    print(f"\nSample of {season} betting lines:")
    print(lines_df[["home_team", "away_team", "spread", "over_under"]].head(10))

    return games_df, lines_df


def refresh_portal_only(seasons: list = None):
    """
    Pull/refresh transfer portal data without re-running the full pipeline.
    Useful each offseason to capture the latest portal activity.

    Usage:
        python3 src/data_collection.py  # (uncomment refresh_portal_only() in __main__)
    """
    if seasons is None:
        seasons = list(range(2018, 2027))
    all_portal = []
    for season in seasons:
        df = collect_transfer_portal(season)
        if not df.empty:
            save_csv(df, RAW_DIR / f"portal_{season}.csv")
            all_portal.append(df)
        time.sleep(0.5)

    if not all_portal:
        print("No portal data collected.")
        return

    master_raw = pd.concat(all_portal, ignore_index=True)
    save_csv(master_raw, DATA_DIR / "processed" / "master_portal.csv")

    features = build_portal_team_features(master_raw)
    if not features.empty:
        save_csv(features, DATA_DIR / "processed" / "master_portal_features.csv")
        print(f"\n✅ Portal features saved — {len(features)} team-season rows")
        seasons_covered = sorted(features["season"].unique())
        print(f"   Seasons: {seasons_covered[0]}–{seasons_covered[-1]}")
    return features


def refresh_advanced_stats(seasons: list = None):
    """
    Pull/refresh WEPA, talent composite, and havoc rate without re-running
    the full data pipeline.

    Run this each offseason (or once to bootstrap) before re-running features.py.
    Typically takes ~2 minutes for all seasons.

    Usage:
        python3 -c "from src.data_collection import refresh_advanced_stats; refresh_advanced_stats()"
    """
    if seasons is None:
        seasons = list(range(2015, 2027))

    all_wepa, all_talent, all_havoc = [], [], []

    for season in seasons:
        print(f"\nSeason {season}:")

        wepa_df = collect_wepa(season)
        if not wepa_df.empty:
            save_csv(wepa_df, RAW_DIR / f"wepa_{season}.csv")
            all_wepa.append(wepa_df)
        time.sleep(0.3)

        talent_df = collect_talent(season)
        if not talent_df.empty:
            save_csv(talent_df, RAW_DIR / f"talent_{season}.csv")
            all_talent.append(talent_df)
        time.sleep(0.3)

        havoc_df = collect_havoc(season)
        if not havoc_df.empty:
            save_csv(havoc_df, RAW_DIR / f"havoc_{season}.csv")
            all_havoc.append(havoc_df)
        time.sleep(0.3)

    # Save master files
    results = {}
    if all_wepa:
        master = pd.concat(all_wepa, ignore_index=True)
        save_csv(master, DATA_DIR / "processed" / "master_wepa.csv")
        results["wepa"] = len(master)
    if all_talent:
        master = pd.concat(all_talent, ignore_index=True)
        save_csv(master, DATA_DIR / "processed" / "master_talent.csv")
        results["talent"] = len(master)
    if all_havoc:
        master = pd.concat(all_havoc, ignore_index=True)
        save_csv(master, DATA_DIR / "processed" / "master_havoc.csv")
        results["havoc"] = len(master)

    print("\n✅ Advanced stats refresh complete!")
    for k, v in results.items():
        print(f"   {k}: {v} rows saved")
    print("\nNext step: python3 src/features.py  (rebuilds feature matrix with new data)")
    return results


def backfill_sp_ratings(extra_seasons: list = [2018]):
    """
    Pull SP+ ratings for seasons before our main data window.
    We need 2018 SP+ so that 2019 games get a valid prior-year rating
    after the one-year shift applied in features.py.
    Appends to master_sp_ratings.csv without re-pulling everything else.
    """
    print(f"Backfilling SP+ ratings for seasons: {extra_seasons}")
    new_rows = []
    for season in extra_seasons:
        print(f"  Pulling SP+ for {season}...")
        df = collect_sp_ratings(season)
        df["year"] = df.get("year", season) if "year" in df.columns else season
        new_rows.append(df)
        time.sleep(0.5)

    if not new_rows:
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    master_path = DATA_DIR / "processed" / "master_sp_ratings.csv"

    # Append to existing master, drop duplicates
    existing = pd.read_csv(master_path)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicate on year + team
    year_col = "year" if "year" in combined.columns else "season"
    combined = combined.drop_duplicates(subset=[year_col, "team"], keep="first")
    combined = combined.sort_values([year_col, "team"]).reset_index(drop=True)

    save_csv(combined, master_path)
    print(f"✅ master_sp_ratings.csv now has {len(combined)} rows "
          f"(seasons {combined[year_col].min()}–{combined[year_col].max()})")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Uncomment one of these:

    # Option A: Quick test with just 2024 data
    # collect_single_season_quick(2024)

    # Option B: Full historical pull (takes ~5 minutes)
    # collect_all_seasons()

    # Full pull including 2025 season
    collect_all_seasons()
