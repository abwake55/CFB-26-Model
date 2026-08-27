"""
Situational spot features: lookahead, sandwich games, and altitude.

Motivation
----------
Spread markets price team quality well but are slower on spot dynamics:

* Lookahead — a team with a much bigger game next week (rivalry, top-10
  matchup) underperforms this week's number. Proxy: next opponent's SP+
  rating minus this opponent's SP+ rating.
* Sandwich — a weak opponent *between* two much stronger ones is the
  classic flat spot. Flagged when both the previous and next opponent
  rate >= 7 SP+ points above the current opponent.
* Altitude — visitors from near sea level at high-altitude venues
  (Wyoming 7,220 ft, Air Force 6,621 ft, etc.) historically underperform;
  the market prices HFA but not the altitude component separately.

Walk-forward validation (2019-25, ~5,200 OOS games, spread ATS at edge >= 3):
    baseline 50.8% (-3.0% ROI) -> +features 54.5% (+4.1% ROI)
    edge >= 3.5: 51.1% -> 56.1% (+7.1% ROI); lift holds in weeks 1-3 and 4+.
CORE totals-under edge and Brier calibration unchanged (spread-only features).

Data sources: data/processed/master_games.csv + master_sp_ratings.csv.
For the live app (current season not yet in master_games), the season
schedule is fetched once from the CFBD API and cached under
data/processed/schedule_cache_<season>.csv. Ratings use the same
season-level SP+ vintage as the rest of the feature matrix.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"

# Home-venue elevation (feet) for programs whose stadium sits high enough
# to affect visiting lowlanders. Static; venues do not move.
VENUE_ALT_FT = {
    "Wyoming": 7220, "Air Force": 6621, "Colorado": 5430, "New Mexico": 5312,
    "Colorado State": 5003, "Utah": 4657, "BYU": 4551, "Utah State": 4534,
    "Nevada": 4499, "New Mexico State": 3915, "UTEP": 3762,
    "Boise State": 2730,
}
ALT_SHOCK_HOME_FT = 4000   # venue must be at least this high
ALT_SHOCK_AWAY_FT = 2000   # ...and the visitor from below this

SITUATIONAL_FEATURES = [
    "home_lookahead", "away_lookahead",
    "home_sandwich", "away_sandwich",
    "alt_shock_away", "altitude_diff",
]


def _load_schedule(seasons) -> pd.DataFrame:
    """All known games (played or scheduled) for the requested seasons.

    Combines the raw season files (which include scheduled-but-unplayed
    games when present) with the processed master, and for seasons with no
    local coverage — the current season inside the app — fetches the full
    schedule from the CFBD API once and caches it.
    """
    frames = []
    for season in sorted(set(int(s) for s in seasons)):
        raw = ROOT / "data" / "raw" / f"games_{season}.csv"
        cache = DATA / f"schedule_cache_{season}.csv"
        if raw.exists() and raw.stat().st_size > 100:
            frames.append(pd.read_csv(raw))
        elif cache.exists():
            frames.append(pd.read_csv(cache))
        else:
            fetched = _fetch_season_schedule(season)
            if fetched is not None:
                frames.append(fetched)
    if not frames:
        return pd.DataFrame(
            columns=["season", "week", "season_type", "start_date",
                     "home_team", "away_team"])
    sched = pd.concat(frames, ignore_index=True)
    need = ["season", "week", "season_type", "start_date", "home_team", "away_team"]
    for c in need:
        if c not in sched.columns:
            sched[c] = np.nan
    return sched[need].dropna(subset=["season", "week", "home_team", "away_team"])


def _fetch_season_schedule(season: int) -> pd.DataFrame | None:
    """Fetch a full season schedule from the CFBD API and cache it."""
    try:
        from data_collection import cfb_get  # shared API helper
    except Exception:
        try:
            from src.data_collection import cfb_get
        except Exception:
            return None
    try:
        data = cfb_get("games", params={"year": int(season)})
        if not data:
            return None
        df = pd.DataFrame([{
            "season": g.get("season"),
            "week": g.get("week"),
            "season_type": g.get("seasonType"),
            "start_date": g.get("startDate"),
            "home_team": g.get("homeTeam"),
            "away_team": g.get("awayTeam"),
        } for g in data])
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA / f"schedule_cache_{season}.csv", index=False)
        return df
    except Exception as e:  # app must never fail on a feature fetch
        print(f"  [situational] schedule fetch for {season} failed: {e}")
        return None


def _load_ratings() -> dict:
    sp = pd.read_csv(DATA / "master_sp_ratings.csv",
                     usecols=["year", "team", "rating"])
    return {(int(r.year), r.team): float(r.rating) for r in sp.itertuples()}


def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lookahead / sandwich / altitude columns to a games DataFrame.

    Required input columns: season, week, home_team, away_team.
    Uses neutral_site and season_type when present (defaults: home venue,
    regular season). Idempotent and NaN-safe — missing schedule or ratings
    leave NaN, which both model backends handle natively.
    """
    df = df.copy()
    if "season_type" not in df.columns:
        df["season_type"] = "regular"
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0

    sched = _load_schedule(df["season"].unique())
    rating = _load_ratings()

    # Per team-season ordered schedule: (postseason flag, week, date, opponent)
    team_sched: dict = {}
    for r in sched.itertuples():
        st_flag = 0 if r.season_type == "regular" else 1
        team_sched.setdefault((int(r.season), r.home_team), []).append(
            (st_flag, int(r.week), r.away_team))
        team_sched.setdefault((int(r.season), r.away_team), []).append(
            (st_flag, int(r.week), r.home_team))
    for k in team_sched:
        team_sched[k].sort()

    def neighbor_opp(season, team, week, season_type, direction):
        st_flag = 0 if season_type == "regular" else 1
        entries = team_sched.get((int(season), team), [])
        cur = (st_flag, int(week))
        if direction > 0:
            future = [e for e in entries if (e[0], e[1]) > cur]
            return future[0][2] if future else None
        past = [e for e in entries if (e[0], e[1]) < cur]
        return past[-1][2] if past else None

    for side in ("home", "away"):
        opp_side = "away" if side == "home" else "home"
        la, sw = [], []
        for r in df.itertuples():
            season = int(r.season)
            team = getattr(r, f"{side}_team")
            cur_opp = getattr(r, f"{opp_side}_team")
            cur_r = rating.get((season, cur_opp))
            n_o = neighbor_opp(season, team, r.week, r.season_type, +1)
            p_o = neighbor_opp(season, team, r.week, r.season_type, -1)
            n_r = rating.get((season, n_o)) if n_o else None
            p_r = rating.get((season, p_o)) if p_o else None
            la.append((n_r - cur_r) if (n_r is not None and cur_r is not None)
                      else np.nan)
            sw.append(int(n_r is not None and p_r is not None and cur_r is not None
                          and n_r - cur_r >= 7 and p_r - cur_r >= 7))
        df[f"{side}_lookahead"] = la
        df[f"{side}_sandwich"] = sw

    neutral = df["neutral_site"].astype(bool)
    home_alt = df["home_team"].map(VENUE_ALT_FT).fillna(0)
    away_alt = df["away_team"].map(VENUE_ALT_FT).fillna(0)
    df["alt_shock_away"] = ((home_alt >= ALT_SHOCK_HOME_FT)
                            & (away_alt < ALT_SHOCK_AWAY_FT)
                            & ~neutral).astype(int)
    df["altitude_diff"] = np.where(neutral, 0, home_alt - away_alt)
    return df
