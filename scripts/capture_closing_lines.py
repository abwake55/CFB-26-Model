"""
Automated Closing Line Value (CLV) capture
============================================
Fills the `closing_line` field on tracked bets using CFBD's lines API.
CFBD lines stop updating at kickoff, so post-kickoff they ARE the closing
lines — fetching them after games complete gives every bet its CLV for free.

Used two ways:
  • Streamlit app — "Auto-fill closing lines" button in My Bets
  • CLI          — python3 scripts/capture_closing_lines.py

The closing_line string formats match what app.compute_clv() parses:
  Spread    → signed points from the bet team's perspective, e.g. "-4.5"
  Total     → the closing over/under, e.g. "55.5"
  Moneyline → American odds for the bet team, e.g. "+145" / "-130"
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR  = Path(__file__).parent.parent
BETS_FILE = ROOT_DIR / "tracked_bets.json"
CFB_BASE_URL = "https://api.collegefootballdata.com"

PROVIDER_PRIORITY = ["consensus", "Bovada", "DraftKings", "ESPN Bet",
                     "William Hill (New Jersey)", "FanDuel"]


def _load_api_key() -> str:
    key = os.getenv("CFB_API_KEY", "")
    if key:
        return key
    secrets = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _cfb_get(endpoint: str, params: dict, api_key: str) -> list:
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _parse_game_teams(game_label: str) -> tuple[str, str] | None:
    """Parse a bet's game label into (home, away). Handles both formats
    the app has used: 'Away @ Home' and 'Home vs Away'."""
    if " @ " in game_label:
        away, home = game_label.split(" @ ", 1)
        return home.strip(), away.strip()
    if " vs " in game_label:
        home, away = game_label.split(" vs ", 1)
        return home.strip(), away.strip()
    return None


def _fetch_week_closing_lines(season: int, week: int, api_key: str) -> dict:
    """Return {(home, away): {spread, over_under, home_ml, away_ml, completed}}
    for one week, using the best available provider per game."""
    out: dict = {}

    # Completion status from /games (don't fill CLV for games not yet played)
    completed_pairs = set()
    for season_type in ("regular", "postseason"):
        try:
            games = _cfb_get("games", {"year": season, "week": week,
                                       "seasonType": season_type}, api_key)
        except Exception:
            games = []
        for g in games:
            if g.get("completed"):
                completed_pairs.add((g.get("homeTeam"), g.get("awayTeam")))

    rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
    for season_type in ("regular", "postseason"):
        try:
            data = _cfb_get("lines", {"year": season, "week": week,
                                      "seasonType": season_type}, api_key)
        except Exception:
            data = []
        for game in data:
            home, away = game.get("homeTeam"), game.get("awayTeam")
            lines = game.get("lines") or []
            if not lines:
                continue
            best = min(lines, key=lambda l: rank.get(l.get("provider", ""), 99))
            key = (home, away)
            if key not in out:   # regular takes precedence over postseason dupes
                out[key] = {
                    "spread":     pd.to_numeric(best.get("spread"),        errors="coerce"),
                    "over_under": pd.to_numeric(best.get("overUnder"),     errors="coerce"),
                    "home_ml":    pd.to_numeric(best.get("homeMoneyline"), errors="coerce"),
                    "away_ml":    pd.to_numeric(best.get("awayMoneyline"), errors="coerce"),
                    "completed":  key in completed_pairs,
                }
    return out


def _closing_value_for_bet(bet: dict, game_lines: dict) -> str | None:
    """Format the closing line for one bet, matching compute_clv()'s parsing."""
    teams = _parse_game_teams(str(bet.get("game", "")))
    if teams is None:
        return None
    home, away = teams

    # Find the fetched game — exact match either orientation
    rec = game_lines.get((home, away)) or game_lines.get((away, home))
    if rec is None or not rec["completed"]:
        return None

    btype = bet.get("bet_type", "")
    pick  = str(bet.get("pick", ""))

    if btype == "Total":
        ou = rec["over_under"]
        return f"{ou:.1f}" if pd.notna(ou) else None

    # Spread / Moneyline need to know which team the bet is on
    bet_on_home = pick.startswith(home)
    bet_on_away = pick.startswith(away)
    if not bet_on_home and not bet_on_away:
        return None

    if btype == "Spread":
        sp = rec["spread"]   # home team's perspective (negative = home favored)
        if pd.isna(sp):
            return None
        val = sp if bet_on_home else -sp
        return f"{val:+.1f}"

    if btype == "Moneyline":
        ml = rec["home_ml"] if bet_on_home else rec["away_ml"]
        return f"{int(ml):+d}" if pd.notna(ml) else None

    return None


def fill_closing_lines(bets: list, api_key: str) -> tuple[list, int, list]:
    """
    Fill missing closing_line on all bets whose games have completed.
    Returns (bets, n_filled, notes). Mutates and returns the same list.
    """
    notes: list[str] = []
    todo = [b for b in bets if not str(b.get("closing_line", "")).strip()]
    if not todo:
        return bets, 0, ["All bets already have closing lines."]

    # Fetch each (season, week) only once
    weeks_needed = sorted({(int(b["season"]), int(b["week"]))
                           for b in todo
                           if str(b.get("season", "")).strip() and str(b.get("week", "")).strip()})
    week_lines: dict = {}
    for season, week in weeks_needed:
        try:
            week_lines[(season, week)] = _fetch_week_closing_lines(season, week, api_key)
        except Exception as exc:
            notes.append(f"⚠️ {season} W{week}: lines fetch failed ({exc})")

    n_filled = 0
    for bet in todo:
        try:
            key = (int(bet["season"]), int(bet["week"]))
        except (KeyError, ValueError, TypeError):
            continue
        lines = week_lines.get(key)
        if not lines:
            continue
        val = _closing_value_for_bet(bet, lines)
        if val is not None:
            bet["closing_line"] = val
            n_filled += 1
        else:
            notes.append(f"· No closing line matched: {bet.get('pick')} ({bet.get('game')})")

    return bets, n_filled, notes


def main():
    api_key = _load_api_key()
    if not api_key:
        print("❌ CFB_API_KEY not found (env or .streamlit/secrets.toml)")
        sys.exit(1)
    if not BETS_FILE.exists():
        print("No tracked_bets.json — nothing to fill.")
        return

    bets = json.loads(BETS_FILE.read_text())
    bets, n, notes = fill_closing_lines(bets, api_key)
    if n:
        BETS_FILE.write_text(json.dumps(bets, indent=2))
    print(f"✅ Filled closing lines on {n} bet(s).")
    for note in notes:
        print(f"  {note}")


if __name__ == "__main__":
    main()
