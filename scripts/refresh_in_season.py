#!/usr/bin/env python3
"""
CFB Model — In-Season Data Refresh (2026)
==========================================
Pulls weekly game results, betting lines, and per-game EPA for the 2026 season.
Called automatically by the GitHub Actions weekly workflow during Sep–Jan.

Also works for any season: set SEASON env var to override.
    SEASON=2025 python scripts/refresh_in_season.py

Run manually after results are in (typically Tuesday morning):
    /opt/homebrew/bin/python3 scripts/refresh_in_season.py
"""

import os, sys, time
import requests
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR  = ROOT_DIR / "data" / "raw"
sys.path.insert(0, str(ROOT_DIR / "src"))

# Allow override via environment variable (e.g. SEASON=2025 python ...)
SEASON = int(os.getenv("SEASON", "2026"))

# ─── API key ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    # Prefer environment variable (GitHub Actions injects this)
    key = os.getenv("CFB_API_KEY", "")
    if key:
        return key
    # Fall back to secrets.toml for local runs
    path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

CFB_API_KEY  = load_api_key()
CFB_BASE_URL = "https://api.collegefootballdata.com"

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers=headers, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def update_master(new_df: pd.DataFrame, master_path: Path,
                  season_col: str = "season") -> None:
    if new_df.empty:
        print(f"   (no data — skipping {master_path.name})")
        return
    if master_path.exists():
        master = pd.read_csv(master_path)
        master = master[master[season_col] != SEASON].copy()
        updated = pd.concat([master, new_df], ignore_index=True)
    else:
        updated = new_df.copy()
    updated.to_csv(master_path, index=False)
    print(f"   ✅ {master_path.name}: {len(new_df)} rows for {SEASON} ({len(updated)} total)")

# ─── Refresh functions ────────────────────────────────────────────────────────

def refresh_games():
    print(f"\n📅 Games (regular + postseason)...")
    regular    = cfb_get("games", params={"year": SEASON, "seasonType": "regular"})
    postseason = cfb_get("games", params={"year": SEASON, "seasonType": "postseason"})
    all_games  = regular + postseason
    df = pd.DataFrame(all_games)
    if df.empty:
        print("   No games yet")
        return
    df = df[df.get("completed", pd.Series([False]*len(df))) == True].copy()

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
            "neutral_site", "conference_game", "home_team", "home_conference",
            "home_points", "away_team", "away_conference", "away_points",
            "home_pregame_elo", "away_pregame_elo", "excitement_index"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["point_diff"]   = df["home_points"] - df["away_points"]
    df["total_points"] = df["home_points"] + df["away_points"]
    print(f"   Completed games: {len(df)}  |  Max week: {df['week'].max() if not df.empty else 0}")
    update_master(df, PROC_DIR / "master_games.csv")
    df.to_csv(RAW_DIR / f"games_{SEASON}.csv", index=False)

def refresh_lines():
    print(f"\n💰 Betting lines...")
    data = cfb_get("lines", params={"year": SEASON})
    records = []
    for game in data:
        for line in game.get("lines", []):
            records.append({
                "game_id": game.get("id"), "season": game.get("season"),
                "week": game.get("week"),
                "home_team": game.get("homeTeam"), "away_team": game.get("awayTeam"),
                "provider": line.get("provider"),
                "spread": line.get("spread"), "formatted_spread": line.get("formattedSpread"),
                "spread_open": line.get("spreadOpen"),
                "over_under": line.get("overUnder"), "over_under_open": line.get("overUnderOpen"),
                "home_moneyline": line.get("homeMoneyline"), "away_moneyline": line.get("awayMoneyline"),
            })
    df = pd.DataFrame(records)
    update_master(df, PROC_DIR / "master_lines.csv")
    df.to_csv(RAW_DIR / f"lines_{SEASON}.csv", index=False)

def refresh_ppa_games():
    print(f"\n📈 Per-game EPA...")
    regular    = cfb_get("ppa/games", params={"year": SEASON, "seasonType": "regular"})
    postseason = cfb_get("ppa/games", params={"year": SEASON, "seasonType": "postseason"})
    records = []
    for row in regular + postseason:
        off = row.get("offense") or {}
        def_ = row.get("defense") or {}
        records.append({
            "game_id": row.get("gameId"), "season": row.get("season"),
            "week": row.get("week"), "team": row.get("team"),
            "opponent": row.get("opponent"),
            "off_epa": off.get("overall"), "off_epa_pass": off.get("passing"),
            "off_epa_rush": off.get("rushing"),
            "def_epa": def_.get("overall"), "def_epa_pass": def_.get("passing"),
            "def_epa_rush": def_.get("rushing"),
        })
    df = pd.DataFrame(records)
    update_master(df, PROC_DIR / "master_ppa_games.csv")
    df.to_csv(RAW_DIR / f"ppa_games_{SEASON}.csv", index=False)

def refresh_sp_ratings():
    print(f"\n📊 SP+ ratings...")
    try:
        data = cfb_get("ratings/sp", params={"year": SEASON})
        df = pd.DataFrame(data)
        if df.empty: return
        if "season" not in df.columns: df["season"] = SEASON
        update_master(df, PROC_DIR / "master_sp_ratings.csv")
        df.to_csv(RAW_DIR / f"sp_ratings_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   ⚠️  SP+ unavailable: {e}")

def refresh_advanced_stats():
    print(f"\n🔬 Season-level EPA...")
    try:
        data = cfb_get("ppa/teams", params={"year": SEASON})
        df = pd.DataFrame(data)
        if df.empty: return
        if "season" not in df.columns: df["season"] = SEASON
        update_master(df, PROC_DIR / "master_advanced_stats.csv")
        df.to_csv(RAW_DIR / f"advanced_stats_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   ⚠️  Advanced stats unavailable: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not CFB_API_KEY:
        print("❌ CFB_API_KEY not set. Add it as a GitHub secret or in .streamlit/secrets.toml")
        sys.exit(1)

    print(f"🏈 In-season refresh — {SEASON}")
    print(f"   Key: {CFB_API_KEY[:8]}...")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("Games",          refresh_games),
        ("Lines",          refresh_lines),
        ("Per-game EPA",   refresh_ppa_games),
        ("Season EPA",     refresh_advanced_stats),
        ("SP+ ratings",    refresh_sp_ratings),
    ]

    errors = []
    for name, fn in steps:
        try:
            fn()
            time.sleep(0.4)
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            errors.append((name, str(e)))

    print(f"\n{'='*50}")
    print(f"✅ In-season refresh complete for {SEASON}")
    if errors:
        print(f"⚠️  {len(errors)} error(s): {[n for n, _ in errors]}")
    print("Next: python src/features.py  →  python src/model.py")

if __name__ == "__main__":
    main()
