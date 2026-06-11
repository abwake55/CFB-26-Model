#!/usr/bin/env python3
"""
Refresh returning-production data from CFBD.

GET /player/returning?year=YYYY → one row per team with the share of last
season's production (PPA) that returns on this season's roster. This is the
core ingredient of SP+'s preseason projections and directly targets
early-season error, when rolling-form features are still prior-season noise.

NO year shift needed: returning[season=N] describes the roster entering
season N — it's known before any games are played.

Writes: data/processed/master_returning.csv
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
CFB_BASE_URL = "https://api.collegefootballdata.com"

FIRST_SEASON = 2015
LAST_SEASON  = int(os.getenv("SEASON", "2026"))


def load_api_key() -> str:
    key = os.getenv("CFB_API_KEY", "")
    if key:
        return key
    secrets = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main():
    api_key = load_api_key()
    if not api_key:
        print("❌ CFB_API_KEY not found")
        sys.exit(1)
    headers = {"Authorization": f"Bearer {api_key}"}

    rows = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            resp = requests.get(f"{CFB_BASE_URL}/player/returning",
                                headers=headers, params={"year": year}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  ⚠️  {year}: {exc}")
            continue
        for r in data:
            rows.append({
                "season":              r.get("season", year),
                "team":                r.get("team"),
                "conference":          r.get("conference"),
                "ret_total_ppa":       r.get("totalPPA"),
                "ret_pass_ppa":        r.get("totalPassingPPA"),
                "ret_rush_ppa":        r.get("totalRushingPPA"),
                "ret_recv_ppa":        r.get("totalReceivingPPA"),
                "ret_ppa_pct":         r.get("percentPPA"),
                "ret_pass_ppa_pct":    r.get("percentPassingPPA"),
                "ret_usage":           r.get("usage"),
                "ret_pass_usage":      r.get("passingUsage"),
            })
        print(f"  {year}: {len(data)} teams")
        time.sleep(0.4)   # be polite to the API

    if not rows:
        print("❌ No returning-production data fetched")
        sys.exit(1)

    df = pd.DataFrame(rows).dropna(subset=["team"])
    out = PROC_DIR / "master_returning.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Saved {len(df):,} team-season rows → {out.name} "
          f"({df['season'].min()}–{df['season'].max()})")
    print(df[["ret_ppa_pct", "ret_pass_ppa_pct"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
