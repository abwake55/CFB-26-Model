#!/usr/bin/env python3
"""
CFB Betting Model — 2026 Preseason Data Refresh
================================================
Pulls preseason data available before the 2026 season kicks off:

  ✅ Recruiting class      (2026 signing class — complete by February)
  ✅ Transfer portal       (winter/spring portal period — complete by May)
  ✅ Talent composite      (247Sports roster ratings — updated for 2026 rosters)
  ✅ SP+ preseason ratings (if published by Bill Connelly, usually July-Aug)
  ✅ FPI preseason ratings (ESPN, if available in CFBD)
  ✅ 2025 final stats      (SP+, WEPA, havoc — used via +1 shift for 2026 predictions)

Does NOT pull: games, betting lines, per-game EPA, WEPA, SRS, havoc
(all require completed season data).

Run each summer before the season:
    /opt/homebrew/bin/python3 scripts/refresh_2026.py

Then rebuild the feature matrix:
    /opt/homebrew/bin/python3 src/features.py

The live model (trained on 2017-2025) will automatically use the fresh
2026 team-level data when generating This Week's Picks.
"""

import os, sys, time
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR  = ROOT_DIR / "data" / "raw"
sys.path.insert(0, str(ROOT_DIR / "src"))

SEASON = 2026

# ─── API key ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv("CFB_API_KEY", "")

CFB_API_KEY  = load_api_key()
CFB_BASE_URL = "https://api.collegefootballdata.com"

import requests

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers=headers, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ─── Upsert helper ─────────────────────────────────────────────────────────────

def update_master(new_df: pd.DataFrame, master_path: Path,
                  season_col: str = "season") -> None:
    """Replace rows for SEASON in master CSV; create file if missing."""
    if new_df.empty:
        print(f"   (no data returned — skipping {master_path.name})")
        return
    if master_path.exists():
        master = pd.read_csv(master_path)
        master = master[master[season_col] != SEASON].copy()
        updated = pd.concat([master, new_df], ignore_index=True)
    else:
        updated = new_df.copy()
    updated.to_csv(master_path, index=False)
    print(f"   ✅ {master_path.name}: {len(new_df)} new {SEASON} rows ({len(updated)} total)")

# ─── 1. SP+ preseason ratings ─────────────────────────────────────────────────

def refresh_sp_ratings():
    print(f"\n📊 SP+ ratings ({SEASON})...")
    try:
        data = cfb_get("ratings/sp", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            print("   ⚠️  No SP+ data yet — will be available closer to season start")
            return
        if "season" not in df.columns:
            df["season"] = SEASON
        update_master(df, PROC_DIR / "master_sp_ratings.csv")
        df.to_csv(RAW_DIR / f"sp_ratings_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   ⚠️  SP+ unavailable: {e}")

# ─── 2. FPI preseason ratings ─────────────────────────────────────────────────

def refresh_fpi_ratings():
    print(f"\n🏈 FPI ratings ({SEASON})...")
    try:
        data = cfb_get("ratings/fpi", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            print("   ⚠️  No FPI data yet")
            return
        df.columns = [c.lower() for c in df.columns]
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "year" in df.columns and "season" not in df.columns:
            df = df.rename(columns={"year": "season"})
        if "season" not in df.columns:
            df["season"] = SEASON
        update_master(df, PROC_DIR / "master_fpi_ratings.csv")
        df.to_csv(RAW_DIR / f"fpi_ratings_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   ⚠️  FPI unavailable: {e}")

# ─── 3. Recruiting class ──────────────────────────────────────────────────────

def refresh_recruiting():
    print(f"\n🎓 Recruiting class ({SEASON})...")
    try:
        data = cfb_get("recruiting/teams", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            print("   ⚠️  No recruiting data yet")
            return
        if "season" not in df.columns:
            df["season"] = SEASON
        update_master(df, PROC_DIR / "master_recruiting.csv")
        df.to_csv(RAW_DIR / f"recruiting_{SEASON}.csv", index=False)
    except Exception as e:
        print(f"   ⚠️  Recruiting unavailable: {e}")

# ─── 4. Transfer portal ───────────────────────────────────────────────────────

def refresh_portal():
    print(f"\n🔄 Transfer portal ({SEASON})...")
    try:
        resp = requests.get(
            f"{CFB_BASE_URL}/player/portal",
            headers={"Authorization": f"Bearer {CFB_API_KEY}"},
            params={"year": SEASON}, timeout=20)
        resp.raise_for_status()
        raw = resp.text.strip()
        if not raw or raw in ("null", "[]", ""):
            print("   ⚠️  No portal data (may require Patreon API tier)")
            return
        data = resp.json()
        if not data:
            return
        df = pd.DataFrame(data)
        df["season"] = SEASON
        rename = {"firstName": "first_name", "lastName": "last_name",
                  "transferDate": "transfer_date"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        print(f"   {len(df)} portal entries")
        df.to_csv(RAW_DIR / f"portal_{SEASON}.csv", index=False)

        # Update raw master
        raw_master = PROC_DIR / "master_portal.csv"
        if raw_master.exists():
            existing = pd.read_csv(raw_master)
            existing = existing[existing["season"] != SEASON]
            updated = pd.concat([existing, df], ignore_index=True)
        else:
            updated = df.copy()
        updated.to_csv(raw_master, index=False)

        # Rebuild portal features
        from data_collection import build_portal_team_features
        feats = build_portal_team_features(updated)
        if not feats.empty:
            feats.to_csv(PROC_DIR / "master_portal_features.csv", index=False)
            n_2026 = feats[feats["season"] == SEASON]
            print(f"   ✅ Portal features: {len(n_2026)} teams for {SEASON}")

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status in (401, 403):
            print(f"   ⚠️  Portal requires Patreon API access (HTTP {status})")
            print(f"        Upgrade at https://www.patreon.com/collegefootballdata")
        else:
            print(f"   ⚠️  Portal error: {e}")
    except Exception as e:
        print(f"   ⚠️  Portal unavailable: {e}")

# ─── 5. Talent composite ──────────────────────────────────────────────────────

def refresh_talent():
    print(f"\n⭐ Talent composite ({SEASON})...")
    try:
        data = cfb_get("talent", params={"year": SEASON})
        df   = pd.DataFrame(data)
        if df.empty:
            print("   ⚠️  No talent data yet")
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
        print(f"   ⚠️  Talent unavailable: {e}")

# ─── 6. Confirm 2025 final stats are current ──────────────────────────────────

def check_2025_coverage():
    """
    The most important preseason data for 2026 predictions is actually the
    2025 final stats (SP+, havoc, WEPA) which feed into 2026 via +1yr shift.
    Just verify they're present — refresh_2025.py handles updating them.
    """
    print(f"\n🔍 Checking 2025 final stats coverage...")
    checks = [
        ("master_sp_ratings.csv",   "SP+"),
        ("master_havoc.csv",        "Havoc/Tempo/Turnovers"),
        ("master_wepa.csv",         "WEPA"),
        ("master_fpi_ratings.csv",  "FPI"),
        ("master_talent.csv",       "Talent"),
    ]
    for fname, label in checks:
        path = PROC_DIR / fname
        if not path.exists():
            print(f"   ❌ {label}: {fname} missing — run refresh_2025.py")
            continue
        df = pd.read_csv(path)
        season_col = "year" if "year" in df.columns and "season" not in df.columns else "season"
        has_2025 = 2025 in df[season_col].values if season_col in df.columns else False
        count = len(df[df[season_col] == 2025]) if has_2025 else 0
        mark = "✅" if has_2025 else "❌"
        print(f"   {mark} {label}: {count} teams for 2025 "
              f"{'(will be used for 2026 game predictions)' if has_2025 else '— run refresh_2025.py!'}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not CFB_API_KEY:
        print("❌ CFB_API_KEY not found. Check .streamlit/secrets.toml")
        sys.exit(1)

    print("=" * 60)
    print(f"🏈 CFB Model — {SEASON} Preseason Data Refresh")
    print("=" * 60)
    print(f"   Key: {CFB_API_KEY[:8]}...")
    print(f"   Note: SP+/FPI may not be published until July-August")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("SP+ ratings",      refresh_sp_ratings),
        ("FPI ratings",      refresh_fpi_ratings),
        ("Recruiting",       refresh_recruiting),
        ("Transfer portal",  refresh_portal),
        ("Talent",           refresh_talent),
    ]

    errors = []
    for name, fn in steps:
        try:
            fn()
            time.sleep(0.5)
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            errors.append((name, str(e)))

    check_2025_coverage()

    print(f"\n{'=' * 60}")
    print(f"✅ {SEASON} preseason refresh complete")
    if errors:
        print(f"⚠️  {len(errors)} step(s) had errors:")
        for name, err in errors:
            print(f"   - {name}: {err}")

    print(f"""
Next steps:
  1. Rebuild feature matrix:
     /opt/homebrew/bin/python3 src/features.py

  2. Re-run walk-forward (optional, adds 2026 context):
     /opt/homebrew/bin/python3 scripts/walk_forward.py

  3. Each week during the season, update the model training cutoff
     in src/model.py (TRAIN_SEASONS, TEST_SEASONS) and retrain.

  4. When the season starts (~August), run refresh_2025.py with
     SEASON=2026 to pull week-by-week results as they come in.
""")

if __name__ == "__main__":
    main()
