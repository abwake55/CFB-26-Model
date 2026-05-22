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

CFB_API_KEY = os.getenv("CFB_API_KEY", "uxvnvwwBh6dQBE/hxA+GK+srmnfZ1mkRSr8E7gOg/BuIL/TeNHw5aHbbZDbi4TMt")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY_HERE")

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
