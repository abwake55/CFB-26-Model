#!/usr/bin/env python3
"""
CFB Betting Model — 2025 Season Data Refresh
=============================================
Pulls the FULL 2025 season (regular + postseason / bowls / CFP)
and updates the master CSV files in data/processed/.

Run once before retraining for the 2026 season:
    cd /Users/alex/Desktop/CFB-Betting-Model
    /usr/local/bin/python3 scripts/refresh_2025.py

Only touches 2025 rows — all other seasons are left intact.
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR  = ROOT_DIR / "data" / "raw"

sys.path.insert(0, str(ROOT_DIR / "src"))

# ─── Secrets ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv("CFB_API_KEY", "")

CFB_API_KEY  = load_api_key()
CFB_BASE_URL = "https://api.collegefootballdata.com"
SEASON       = 2025

# ─── HTTP helper ─────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers=headers, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def update_master(new_df: pd.DataFrame, master_path: Path,
                  season_col: str = "season") -> pd.DataFrame:
    """
    Replace all rows where season_col == SEASON in the master file with new_df.
    If the master file doesn't exist, create it from new_df.
    Returns the updated DataFrame.
    """
    if master_path.exists() and not new_df.empty:
        master = pd.read_csv(master_path)
        # Drop old 2025 rows and append fresh ones
        master = master[master[season_col] != SEASON].copy()
        updated = pd.concat([master, new_df], ignore_index=True)
    else:
        updated = new_df.copy()
    updated.to_csv(master_path, index=False)
    print(f"  ✅ {master_path.name}: {len(new_df)} new 2025 rows  ({len(updated)} total)")
    return updated

# ─── 1. Games (regular + postseason) ─────────────────────────────────────────

def refresh_games() -> pd.DataFrame:
    print("\n📅 Games (regular + postseason)...")
    regular    = cfb_get("games", params={"year": SEASON, "seasonType": "regular"})
    postseason = cfb_get("games", params={"year": SEASON, "seasonType": "postseason"})
    all_games  = regular + postseason

    df = pd.DataFrame(all_games)
    df = df[df["completed"] == True].copy()

    rename = {
        "id": "game_id", "seasonType": "season_type", "startDate": "start_date",
        "neutralSite": "neutral_site", "conferenceGame": "conference_game",
        "homeTeam": "home_team", "homeConference": "home_conference",
        "homePoints": "home_points", "awayTeam": "away_team",
        "awayConference": "away_conference", "awayPoints": "away_points",
        "homePregameElo": "home_pregame_elo", "awayPregameElo": "away_pregame_elo",
        "excitementIndex": "excitement_index",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    keep = ["game_id", "season", "week", "season_type", "start_date",
            "neutral_site", "conference_game",
            "home_team", "home_conference", "home_points",
            "away_team", "away_conference", "away_points",
            "home_pregame_elo", "away_pregame_elo", "excitement_index"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["point_diff"]   = df["home_points"] - df["away_points"]
    df["total_points"] = df["home_points"] + df["away_points"]

    print(f"   Regular: {len(regular)} games, Postseason: {len(postseason)} games")
    print(f"   Completed: {len(df)}  |  Max week: {df['week'].max()}")
    update_master(df, PROC_DIR / "master_games.csv")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"games_{SEASON}.csv", index=False)
    return df

# ─── 2. Betting lines ─────────────────────────────────────────────────────────

def refresh_lines():
    print("\n💰 Betting lines...")
    data = cfb_get("lines", params={"year": SEASON})
    records = []
    for game in data:
        for line in game.get("lines", []):
            records.append({
                "game_id":        game.get("id"),
                "season":         game.get("season"),
                "week":           game.get("week"),
                "home_team":      game.get("homeTeam"),
                "away_team":      game.get("awayTeam"),
                "provider":       line.get("provider"),
                "spread":         line.get("spread"),
                "formatted_spread": line.get("formattedSpread"),
                "spread_open":    line.get("spreadOpen"),
                "over_under":     line.get("overUnder"),
                "over_under_open": line.get("overUnderOpen"),
                "home_moneyline": line.get("homeMoneyline"),
                "away_moneyline": line.get("awayMoneyline"),
            })
    df = pd.DataFrame(records)
    update_master(df, PROC_DIR / "master_lines.csv")
    df.to_csv(RAW_DIR / f"lines_{SEASON}.csv", index=False)

# ─── 3. Per-game EPA (for rolling averages) ──────────────────────────────────

def refresh_ppa_games():
    print("\n📈 Per-game EPA (PPA)...")
    regular    = cfb_get("ppa/games", params={"year": SEASON, "seasonType": "regular"})
    postseason = cfb_get("ppa/games", params={"year": SEASON, "seasonType": "postseason"})
    all_data   = regular + postseason

    records = []
    for row in all_data:
        offense = row.get("offense") or {}
        defense = row.get("defense") or {}
        records.append({
            "game_id":        row.get("gameId"),
            "season":         row.get("season"),
            "week":           row.get("week"),
            "team":           row.get("team"),
            "opponent":       row.get("opponent"),
            "off_epa":        offense.get("overall"),
            "off_epa_pass":   offense.get("passing"),
            "off_epa_rush":   offense.get("rushing"),
            "def_epa":        defense.get("overall"),
            "def_epa_pass":   defense.get("passing"),
            "def_epa_rush":   defense.get("rushing"),
        })
    df = pd.DataFrame(records)
    # Use game_id + team as unique key for deduplication
    update_master(df, PROC_DIR / "master_ppa_games.csv", season_col="season")
    df.to_csv(RAW_DIR / f"ppa_games_{SEASON}.csv", index=False)

# ─── 4. Advanced stats (season-level EPA) ────────────────────────────────────

def refresh_advanced_stats():
    print("\n🔬 Advanced stats (season EPA)...")
    data = cfb_get("ppa/teams", params={"year": SEASON})
    df   = pd.DataFrame(data)
    if not df.empty and "season" not in df.columns:
        df["season"] = SEASON
    update_master(df, PROC_DIR / "master_advanced_stats.csv")
    df.to_csv(RAW_DIR / f"advanced_stats_{SEASON}.csv", index=False)

# ─── 5. SP+ ratings ──────────────────────────────────────────────────────────

def refresh_sp_ratings():
    print("\n📊 SP+ ratings...")
    data = cfb_get("ratings/sp", params={"year": SEASON})
    df   = pd.DataFrame(data)
    if not df.empty and "season" not in df.columns:
        df["season"] = SEASON
    update_master(df, PROC_DIR / "master_sp_ratings.csv")
    df.to_csv(RAW_DIR / f"sp_ratings_{SEASON}.csv", index=False)

# ─── 6. FPI ratings ──────────────────────────────────────────────────────────

def refresh_fpi_ratings():
    print("\n🏈 FPI ratings...")
    data = cfb_get("ratings/fpi", params={"year": SEASON})
    df   = pd.DataFrame(data)
    if not df.empty:
        if "year" not in df.columns:
            df["year"] = SEASON
        if "season" not in df.columns:
            df["season"] = SEASON
    update_master(df, PROC_DIR / "master_fpi_ratings.csv")
    df.to_csv(RAW_DIR / f"fpi_ratings_{SEASON}.csv", index=False)

# ─── 7. SRS ratings ──────────────────────────────────────────────────────────

def refresh_srs_ratings():
    print("\n📉 SRS ratings...")
    data = cfb_get("ratings/srs", params={"year": SEASON})
    df   = pd.DataFrame(data)
    if not df.empty and "season" not in df.columns:
        df["season"] = SEASON
    update_master(df, PROC_DIR / "master_srs_ratings.csv")
    df.to_csv(RAW_DIR / f"srs_ratings_{SEASON}.csv", index=False)

# ─── 8. Recruiting ────────────────────────────────────────────────────────────

def refresh_recruiting():
    print("\n🎓 Recruiting...")
    data = cfb_get("recruiting/teams", params={"year": SEASON})
    df   = pd.DataFrame(data)
    if not df.empty and "season" not in df.columns:
        df["season"] = SEASON
    update_master(df, PROC_DIR / "master_recruiting.csv")
    df.to_csv(RAW_DIR / f"recruiting_{SEASON}.csv", index=False)

# ─── 9. WEPA ─────────────────────────────────────────────────────────────────

def refresh_wepa():
    print("\n⚡ WEPA (opponent-adjusted EPA)...")
    try:
        data = cfb_get("wepa/team/season", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            print("   No WEPA data returned")
            return
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = SEASON

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
        df["wepa_explosiveness"]     = pd.to_numeric(df.get("explosiveness"),        errors="coerce")
        df["wepa_explosiveness_def"] = pd.to_numeric(df.get("explosivenessAllowed"), errors="coerce")

        keep = [c for c in ["season", "team", "wepa_offense", "wepa_defense",
                             "wepa_success_off", "wepa_success_def",
                             "wepa_explosiveness", "wepa_explosiveness_def"]
                if c in df.columns]
        df = df[keep].copy()
        update_master(df, PROC_DIR / "master_wepa.csv")
        df.to_csv(RAW_DIR / f"wepa_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   WEPA unavailable: {e}")

# ─── 10. Talent composite ────────────────────────────────────────────────────

def refresh_talent():
    print("\n⭐ Talent composite...")
    try:
        data = cfb_get("talent", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            return
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = SEASON
        keep = [c for c in ["season", "team", "talent"] if c in df.columns]
        df = df[keep].copy()
        update_master(df, PROC_DIR / "master_talent.csv")
        df.to_csv(RAW_DIR / f"talent_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   Talent unavailable: {e}")

# ─── 11. Havoc ───────────────────────────────────────────────────────────────

def refresh_havoc():
    print("\n💥 Havoc rates...")
    try:
        data = cfb_get("stats/season/advanced",
                       params={"year": SEASON, "excludeGarbageTime": "true"})
        df = pd.DataFrame(data)
        if df.empty:
            return
        if "defense" in df.columns:
            df["havoc_total"]       = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("total")      if isinstance(x, dict) else None)
            df["havoc_front_seven"] = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("frontSeven") if isinstance(x, dict) else None)
            df["havoc_db"]          = df["defense"].apply(
                lambda x: x.get("havoc", {}).get("db")         if isinstance(x, dict) else None)
        if "offense" in df.columns:
            df["rush_success_rate"] = df["offense"].apply(
                lambda x: x.get("rushingPlays", {}).get("successRate") if isinstance(x, dict) else None)
            df["pass_success_rate"] = df["offense"].apply(
                lambda x: x.get("passingDowns", {}).get("successRate") if isinstance(x, dict) else None)
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = SEASON
        keep = [c for c in ["season", "team", "havoc_total", "havoc_front_seven",
                             "havoc_db", "rush_success_rate", "pass_success_rate"]
                if c in df.columns]
        df = df[keep].copy()
        for col in keep:
            if col not in ("season", "team"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        update_master(df, PROC_DIR / "master_havoc.csv")
        df.to_csv(RAW_DIR / f"havoc_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   Havoc unavailable: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not CFB_API_KEY:
        print("❌ CFB_API_KEY not found. Check .streamlit/secrets.toml")
        sys.exit(1)

    print(f"🏈 Refreshing {SEASON} data from College Football Data API")
    print(f"   Key: {CFB_API_KEY[:8]}...")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("Games",            refresh_games),
        ("Betting lines",    refresh_lines),
        ("Per-game EPA",     refresh_ppa_games),
        ("Advanced stats",   refresh_advanced_stats),
        ("SP+ ratings",      refresh_sp_ratings),
        ("FPI ratings",      refresh_fpi_ratings),
        ("SRS ratings",      refresh_srs_ratings),
        ("Recruiting",       refresh_recruiting),
        ("WEPA",             refresh_wepa),
        ("Talent",           refresh_talent),
        ("Havoc",            refresh_havoc),
    ]

    errors = []
    for name, fn in steps:
        try:
            fn()
            time.sleep(0.5)   # polite API rate limiting
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            errors.append((name, str(e)))

    print(f"\n{'='*55}")
    print(f"✅ 2025 data refresh complete")
    if errors:
        print(f"⚠️  {len(errors)} step(s) had errors:")
        for name, err in errors:
            print(f"   - {name}: {err}")
    print(f"\nNext step: run /usr/local/bin/python3 src/features.py")
    print(f"           then /usr/local/bin/python3 src/model.py")

if __name__ == "__main__":
    main()
