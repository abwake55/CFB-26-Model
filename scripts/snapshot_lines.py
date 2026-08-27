#!/usr/bin/env python3
"""
Line movement snapshots
=======================
Captures the current consensus lines for the upcoming week's games and keeps a
per-week JSON under data/lines_snapshots/ with:

  first_seen / first  — the lines as they were when this snapshotter first saw
                        the game (≈ when the model first flagged its picks)
  current / updated   — the latest lines

The app diffs `first` vs the live line to warn when a flagged pick's edge has
been bet away ("flagged UNDER 59 on Tue, now 56 — edge gone").

Run by .github/workflows/line_snapshots.yml (daily). No-ops quietly in the
offseason or when no lines are posted yet, so the cron can stay on year-round.

Usage:
    python3 scripts/snapshot_lines.py            # auto-detect current week
    python3 scripts/snapshot_lines.py --season 2026 --week 1
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR  = Path(__file__).parent.parent
SRC_DIR   = ROOT_DIR / "src"
SNAP_DIR  = ROOT_DIR / "data" / "lines_snapshots"
CFB_BASE_URL = "https://api.collegefootballdata.com"

sys.path.insert(0, str(SRC_DIR))
import odds_api  # noqa: E402


def _secret(key: str) -> str:
    if os.getenv(key):
        return os.getenv(key)
    path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith(key) and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _cfb_get(endpoint: str, params: dict, api_key: str) -> list:
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def current_week(cfb_key: str, season: int) -> int | None:
    """Earliest regular-season week with any uncompleted game."""
    try:
        games = _cfb_get("games", {"year": season, "seasonType": "regular"}, cfb_key)
    except Exception as exc:
        print(f"⚠️  games fetch failed: {exc}")
        return None
    weeks = sorted({g["week"] for g in games if not g.get("completed")})
    return weeks[0] if weeks else None


def fetch_week_games(cfb_key: str, season: int, week: int) -> pd.DataFrame:
    data = _cfb_get("games", {"year": season, "week": week,
                              "seasonType": "regular"}, cfb_key)
    rows = [{"game_id": g.get("id"), "season": g.get("season"),
             "week": g.get("week"), "home_team": g.get("homeTeam"),
             "away_team": g.get("awayTeam"), "start_date": g.get("startDate")}
            for g in data]
    return pd.DataFrame(rows)


def cfbd_lines(cfb_key: str, season: int, week: int) -> pd.DataFrame:
    """Fallback consensus-ish lines from CFBD (first preferred provider)."""
    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
    rank = {p: i for i, p in enumerate(priority)}
    try:
        data = _cfb_get("lines", {"year": season, "week": week,
                                  "seasonType": "regular"}, cfb_key)
    except Exception:
        return pd.DataFrame()
    rows = []
    for game in data:
        for line in game.get("lines", []):
            rows.append({"game_id": game.get("id"),
                         "spread": line.get("spread"),
                         "over_under": line.get("overUnder"),
                         "home_moneyline": line.get("homeMoneyline"),
                         "away_moneyline": line.get("awayMoneyline"),
                         "_rank": rank.get(line.get("provider", ""), 99)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("_rank")
            .drop_duplicates("game_id", keep="first").drop(columns=["_rank"]))


def snap_path(season: int, week: int) -> Path:
    return SNAP_DIR / f"lines_{season}_W{week:02d}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    cfb_key = _secret("CFB_API_KEY")
    if not cfb_key:
        print("❌ CFB_API_KEY not found (env or .streamlit/secrets.toml)")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    season = args.season or now.year
    week = args.week or current_week(cfb_key, season)
    if week is None:
        print(f"No upcoming games for {season} (offseason or season over) — nothing to snapshot.")
        return

    games = fetch_week_games(cfb_key, season, week)
    if games.empty:
        print(f"No games found for {season} W{week} — nothing to snapshot.")
        return

    # Prefer The Odds API consensus (median across books); CFBD as fallback.
    lines = pd.DataFrame()
    odds_key = _secret("ODDS_API_KEY")
    if odds_key:
        try:
            lines = odds_api.fetch_lines(odds_key, games)
        except Exception as exc:
            print(f"⚠️  The Odds API failed ({exc}) — falling back to CFBD lines")
    if lines.empty or lines["spread"].isna().all():
        lines = cfbd_lines(cfb_key, season, week)
    if lines.empty:
        print(f"No lines posted yet for {season} W{week} — nothing to snapshot.")
        return

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    path = snap_path(season, week)
    snap = {"season": season, "week": week, "games": {}}
    if path.exists():
        try:
            snap = json.loads(path.read_text())
        except Exception:
            pass

    ts = now.isoformat(timespec="seconds")
    n_new, n_moved = 0, 0
    by_id = lines.set_index("game_id")
    for _, g in games.iterrows():
        gid = g["game_id"]
        if gid not in by_id.index:
            continue
        ln = by_id.loc[gid]
        cur = {"spread":       ln.get("spread") if pd.notna(ln.get("spread")) else None,
               "over_under":   ln.get("over_under") if pd.notna(ln.get("over_under")) else None,
               "home_ml":      ln.get("home_moneyline") if pd.notna(ln.get("home_moneyline")) else None,
               "away_ml":      ln.get("away_moneyline") if pd.notna(ln.get("away_moneyline")) else None}
        if cur["spread"] is None and cur["over_under"] is None:
            continue
        rec = snap["games"].get(str(gid))
        if rec is None:
            snap["games"][str(gid)] = {
                "home": g["home_team"], "away": g["away_team"],
                "first_seen": ts, "first": cur, "current": cur, "updated": ts,
            }
            n_new += 1
        else:
            prev = rec.get("current") or {}
            if prev != cur:
                n_moved += 1
            rec["current"] = cur
            rec["updated"] = ts

    path.write_text(json.dumps(snap, indent=2))
    print(f"✅ {season} W{week}: {len(snap['games'])} games snapshotted "
          f"({n_new} new, {n_moved} moved) → {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
