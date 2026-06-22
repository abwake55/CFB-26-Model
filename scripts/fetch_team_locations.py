#!/usr/bin/env python3
"""
Fetch FBS team home stadium coordinates from CFBD API and cache to JSON.

This expands travel feature coverage from ~29% (hardcoded ~130 teams)
to ~100% (all FBS programs CFBD knows about).

Run once before training (or anytime new programs join FBS):
    python3 scripts/fetch_team_locations.py

Output:
    data/processed/team_locations_cfbd.json
    { "Alabama": [33.2082, -87.5503], "Ohio State": [40.0014, -83.0196], ... }
"""

import json
import os
import sys
import requests
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
OUT_FILE = PROC_DIR / "team_locations_cfbd.json"

CFB_BASE = "https://api.collegefootballdata.com"

# ── try to get key from secrets / env ────────────────────────────────────────
def _get_key() -> str:
    # 1. streamlit secrets file
    secrets_file = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            if line.startswith("CFB_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # 2. environment variable
    return os.getenv("CFB_API_KEY", "")


def fetch_fbs_teams(key: str, year: int = 2024) -> list[dict]:
    """Fetch all FBS teams with location data from CFBD."""
    resp = requests.get(
        f"{CFB_BASE}/teams/fbs",
        headers={"Authorization": f"Bearer {key}"},
        params={"year": year},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def build_coords_map(teams: list[dict]) -> dict[str, list[float]]:
    """
    Extract team name → [lat, lon] from CFBD team records.
    Skips teams with no location or coordinates.
    """
    coords: dict[str, list[float]] = {}
    for t in teams:
        school = t.get("school", "")
        loc = t.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if school and lat is not None and lon is not None:
            try:
                coords[school] = [round(float(lat), 4), round(float(lon), 4)]
            except (ValueError, TypeError):
                pass
    return coords


def main():
    key = _get_key()
    if not key:
        print("ERROR: CFB_API_KEY not found. Set it in .streamlit/secrets.toml or as an env var.")
        sys.exit(1)

    print("Fetching FBS team locations from CFBD API...")
    try:
        teams = fetch_fbs_teams(key, year=2024)
    except Exception as e:
        print(f"ERROR fetching teams: {e}")
        sys.exit(1)

    coords = build_coords_map(teams)
    print(f"  Got coordinates for {len(coords)} teams")

    # Merge with any existing JSON (keeps entries CFBD missed)
    existing: dict = {}
    if OUT_FILE.exists():
        try:
            existing = json.loads(OUT_FILE.read_text())
        except Exception:
            pass

    # CFBD data takes priority over existing cache
    merged = {**existing, **coords}
    print(f"  Total teams in cache after merge: {len(merged)}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"  Saved → {OUT_FILE}")

    # Quick coverage estimate against a training feature matrix if it exists
    fm_path = PROC_DIR / "feature_matrix.csv"
    if fm_path.exists():
        import pandas as pd
        try:
            fm = pd.read_csv(fm_path, usecols=["home_team", "away_team"])
            all_teams = set(fm["home_team"].dropna()) | set(fm["away_team"].dropna())
            covered = sum(1 for t in all_teams if t in merged)
            print(f"  Travel coverage on feature matrix: {covered}/{len(all_teams)} "
                  f"teams ({covered/len(all_teams):.0%})")
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
