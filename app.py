"""
CFB Betting Model — Streamlit Web App
======================================
Dad-friendly interface for weekly bet recommendations.
Hosted on Streamlit Community Cloud — no Python knowledge required.

Deploy:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io → New app → select this repo → app.py
  3. Add API keys in Settings → Secrets (see .streamlit/secrets_template.toml)
"""

import sys
import os
import json
import uuid
import warnings
warnings.filterwarnings("ignore")

# ── Shared feature builder (single source of truth for feature construction) ──
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))
from model import EnsembleRegressor, EnsembleClassifier, MarketAnchoredEnsemble  # required so joblib can unpickle saved models
import __main__
__main__.EnsembleRegressor       = EnsembleRegressor   # joblib looks in __main__ when model was trained via python3 src/model.py
__main__.EnsembleClassifier      = EnsembleClassifier
__main__.MarketAnchoredEnsemble  = MarketAnchoredEnsemble
from feature_builder import (
    load_rating_sources,
    load_recent_epa    as _fb_load_recent_epa,
    load_current_elo   as _fb_load_current_elo,
    attach_team_features,
    feature_coverage_report,
)
import odds_api   # The Odds API line fetcher (the-odds-api.com)

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from datetime import date

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CFB Picks",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PATHS ────────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).parent
DATA_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
BETS_FILE = ROOT_DIR / "tracked_bets.json"

# ─── API KEYS ─────────────────────────────────────────────────────────────────

def get_secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, fallback)

# Keys are read at call-time (not module load) so Streamlit secrets are always initialised first
def _cfb_api_key()      -> str: return get_secret("CFB_API_KEY",   "")
def _theodds_api_key()  -> str: return get_secret("ODDS_API_KEY",  "")   # the-odds-api.com

CFB_BASE_URL  = "https://api.collegefootballdata.com"

SPREAD_EDGE_MIN, SPREAD_EDGE_MAX = 4.0, 7.0
# Totals flag range must span the validated CORE gate (under, edge 2-7): a
# 3.0 floor silently dropped CORE unders at edge 2-3 before they could render.
TOTALS_EDGE_MIN, TOTALS_EDGE_MAX = 2.0, 7.0
MONEYLINE_EV_MIN = 0.04
MONEYLINE_EV_MAX = 0.08

BETTORS = ["Alex", "Joe", "Zou", "Pat"]


# ─── KELLY CRITERION SIZING ───────────────────────────────────────────────────

def unit_dollar_label(units: int) -> str:
    """Return '2u ≈ $200' string using the bankroll from session state."""
    bankroll = st.session_state.get("bankroll", 1000)
    unit_val = bankroll / 100
    dollars  = units * unit_val
    if dollars >= 1000:
        return f"{units}u ≈ ${dollars:,.0f}"
    return f"{units}u ≈ ${dollars:.0f}"


def kelly_units_spread(edge_abs: float, fraction: float = 0.25) -> int:
    """
    Quarter-Kelly bet sizing for ATS bets at standard -110 juice.

    Realistic calibration: each point of spread edge ≈ 0.5% improvement
    in ATS cover probability beyond the 52.38% breakeven baseline.
    (Model spread direction accuracy is ~51–53%, not 58%.)

    Full Kelly formula at -110:
        b = 100/110 ≈ 0.909 (net payout per unit)
        f = (p·b − q) / b  where q = 1 − p

    Uses quarter-Kelly (25%) as a conservative default. Capped at 3 units.
    Tiered output:
      4–5.9 pt edge → 1u
      6–7.9 pt edge → 2u
      8+ pt edge    → 3u
    """
    win_prob = min(0.5238 + edge_abs * 0.005, 0.60)
    b = 100 / 110  # -110 payout
    kelly_f = max((win_prob * b - (1 - win_prob)) / b, 0.0)
    units = kelly_f * fraction * 100  # bankroll assumed = 100 units
    return max(1, min(3, round(units)))


def kelly_units_ml(ev: float, fraction: float = 0.25) -> int:
    """
    Quarter-Kelly bet sizing for moneyline bets given expected value.

    Tiered to be conservative at the margin (4% EV is the minimum threshold):
      4–5.9% EV → 1u  (borderline, keep small)
      6–7.9% EV → 2u  (solid edge)
      8%+ EV    → 3u  (strong edge, capped at 3 due to ML variance)
    """
    if ev >= 0.08: return 3
    if ev >= 0.06: return 2
    return 1


# ─── BET TRACKER ─────────────────────────────────────────────────────────────

def load_bets() -> list:
    if BETS_FILE.exists():
        try:
            return json.loads(BETS_FILE.read_text())
        except Exception:
            return []
    return []

def save_bets(bets: list):
    BETS_FILE.write_text(json.dumps(bets, indent=2))
    sync_bets_to_github()


# ─── GITHUB PERSISTENCE ─────────────────────────────────────────────────────
# Streamlit Cloud's filesystem is ephemeral: tracked_bets.json written at
# runtime is wiped on every redeploy, and the weekly refresh workflow pushes
# a commit (triggering a redeploy) every Tuesday. To keep bet + CLV history
# durable, every save also commits tracked_bets.json back to the repo when a
# GITHUB_TOKEN secret is configured (Streamlit Cloud -> Settings -> Secrets;
# use a fine-grained PAT with Contents: read/write on this repo only).
# Without the token everything still works locally; history just is not
# backed up between deploys, and My Bets shows a reminder.

GITHUB_REPO      = "abwake55/CFB-26-Model"
GITHUB_BETS_PATH = "tracked_bets.json"


def _github_token() -> str:
    return get_secret("GITHUB_TOKEN", "")


def github_backup_configured() -> bool:
    return bool(_github_token())


def sync_bets_to_github() -> tuple[bool, str]:
    """Commit the current tracked_bets.json to the repo. Best-effort: any
    failure returns (False, reason) and never breaks a local save."""
    token = _github_token()
    if not token:
        return False, "no GITHUB_TOKEN secret"
    import base64
    api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_BETS_PATH}"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    try:
        content = BETS_FILE.read_text()
        cur = requests.get(api, headers=headers, timeout=15)
        sha = None
        if cur.status_code == 200:
            sha = cur.json().get("sha")
            existing = base64.b64decode(cur.json().get("content", "")).decode()
            if existing.strip() == content.strip():
                return True, "already in sync"
        payload = {
            "message": "Update tracked bets from app",
            "content": base64.b64encode(content.encode()).decode(),
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(api, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            return True, "synced"
        return False, f"github sync failed: HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"github sync failed: {exc}"

def add_bet(game: str, bet_type: str, pick: str, line: str,
            units: int, season: int, week: int, edge: str = "", bettor: str = ""):
    bets = load_bets()
    bets.append({
        "id":       str(uuid.uuid4())[:8],
        "date":     str(date.today()),
        "season":   season,
        "week":     week,
        "game":     game,
        "bet_type": bet_type,
        "pick":     pick,
        "line":     line,
        "edge":     edge,
        "units":    units,
        "status":   "Pending",
        "bettor":   bettor,
    })
    save_bets(bets)

def update_bet_status(bet_id: str, status: str):
    bets = load_bets()
    for b in bets:
        if b["id"] == bet_id:
            b["status"] = status
            break
    save_bets(bets)

def update_bet_bettor(bet_id: str, bettor: str):
    bets = load_bets()
    for b in bets:
        if b["id"] == bet_id:
            b["bettor"] = bettor
            break
    save_bets(bets)

def delete_bet(bet_id: str):
    bets = load_bets()
    save_bets([b for b in bets if b["id"] != bet_id])

def update_bet_closing_line(bet_id: str, closing_line: str):
    bets = load_bets()
    for b in bets:
        if b["id"] == bet_id:
            b["closing_line"] = closing_line.strip()
            break
    save_bets(bets)

def compute_clv(bet: dict) -> float | None:
    """
    Closing Line Value: how much better (or worse) was your line vs. the closing line.
    Positive = you beat the close (good). Negative = line moved against you.

    Spreads / Totals → returned in points.
    Moneylines       → returned in implied-probability percentage points.

    Sign convention
    ───────────────
    Spread:  CLV = bet_line − closing_line
             e.g. bet −7, closes −9  → CLV = +2.0 (you got the better number)
             e.g. bet +7, closes +5  → CLV = +2.0 (same logic, dog side)
    Total:   CLV = (closing − bet) for OVER, (bet − closing) for UNDER
             e.g. OVER 45, closes 47 → CLV = +2.0
             e.g. UNDER 45, closes 43 → CLV = +2.0
    ML:      CLV = (closing implied prob − bet implied prob) × 100
             e.g. bet +150 (40%), closes +120 (45.5%) → CLV = +5.5 ppts
    """
    closing_str = bet.get("closing_line", "").strip()
    if not closing_str:
        return None
    try:
        close    = float(closing_str.replace("+", ""))
        line_str = str(bet.get("line", "")).replace("+", "")
        bet_line = float(line_str)
        btype    = bet.get("bet_type", "")

        if btype == "Spread":
            return bet_line - close
        elif btype == "Total":
            is_over = "OVER" in str(bet.get("pick", "")).upper()
            return (close - bet_line) if is_over else (bet_line - close)
        elif btype == "Moneyline":
            def impl(o: float) -> float:
                return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)
            return (impl(close) - impl(bet_line)) * 100
    except (ValueError, TypeError):
        return None

def bet_pnl(bet: dict) -> float:
    u = bet.get("units", 1)
    if bet["status"] == "Won":   return u * 0.91  # standard -110 juice
    if bet["status"] == "Lost":  return -u
    if bet["status"] == "Push":  return 0.0
    return 0.0  # Pending


# ─── MONEYLINE MATH ───────────────────────────────────────────────────────────

def american_to_implied_prob(odds):
    if pd.isna(odds): return np.nan
    if odds < 0: return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def remove_vig(hp, ap):
    if pd.isna(hp) or pd.isna(ap): return np.nan, np.nan
    t = hp + ap
    return (hp / t, ap / t) if t > 0 else (np.nan, np.nan)

def prob_to_american(p):
    if pd.isna(p) or p <= 0 or p >= 1: return np.nan
    return round(-(p / (1-p)) * 100) if p >= 0.5 else round(((1-p) / p) * 100)

def ml_ev(model_prob, american_odds):
    if pd.isna(model_prob) or pd.isna(american_odds): return np.nan
    payout = 100 / abs(american_odds) if american_odds < 0 else american_odds / 100
    return model_prob * payout - (1 - model_prob)


# ─── API HELPERS ──────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {_cfb_api_key()}"}
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─── CACHED LOADERS ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # Ensure all ensemble classes are findable in __main__ before joblib unpickling
    import sys
    sys.modules["__main__"].__dict__.setdefault("EnsembleRegressor",       EnsembleRegressor)
    sys.modules["__main__"].__dict__.setdefault("EnsembleClassifier",      EnsembleClassifier)
    sys.modules["__main__"].__dict__.setdefault("MarketAnchoredEnsemble",  MarketAnchoredEnsemble)
    missing = [f for f in ["spread_model.pkl", "totals_model.pkl",
                            "win_prob_model.pkl", "feature_lists.json"]
               if not (MODEL_DIR / f).exists()]
    if missing:
        return None, None, None, None
    spread   = joblib.load(MODEL_DIR / "spread_model.pkl")
    totals   = joblib.load(MODEL_DIR / "totals_model.pkl")
    win_prob = joblib.load(MODEL_DIR / "win_prob_model.pkl")
    with open(MODEL_DIR / "feature_lists.json") as f:
        feat_lists = json.load(f)
    return spread, totals, win_prob, feat_lists


@st.cache_data(show_spinner="Loading team ratings...", ttl=86400)
def load_team_ratings(pred_season: int) -> dict:
    """Thin wrapper — delegates entirely to feature_builder.load_rating_sources."""
    return load_rating_sources(pred_season, DATA_DIR)


@st.cache_data(show_spinner="Computing Elo ratings...", ttl=86400)
def load_current_elo(pred_season: int) -> pd.DataFrame:
    """Thin wrapper — delegates entirely to feature_builder.load_current_elo."""
    return _fb_load_current_elo(pred_season, DATA_DIR)


@st.cache_data(show_spinner="Loading recent form...", ttl=86400)
def load_recent_epa(pred_season: int) -> pd.DataFrame:
    """Thin wrapper — delegates entirely to feature_builder.load_recent_epa."""
    return _fb_load_recent_epa(pred_season, DATA_DIR)


# ─── SCHEDULE & LINES ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching schedule...", ttl=3600)
def fetch_schedule(season: int, week: int) -> pd.DataFrame:
    try:
        data = cfb_get("games", params={"year": season, "week": week,
                                         "seasonType": "regular"})
    except Exception as e:
        st.warning(f"Could not fetch schedule: {e}")
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    records = []
    for g in data:
        records.append({
            "game_id": g.get("id"), "season": g.get("season"),
            "week": g.get("week"),
            "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
            "home_conference": g.get("homeConference"),
            "away_conference": g.get("awayConference"),
            "neutral_site": int(g.get("neutralSite") or False),
            "start_date": g.get("startDate"),
            "home_pregame_elo": g.get("homePregameElo"),
            "away_pregame_elo": g.get("awayPregameElo"),
        })
    df = pd.DataFrame(records)
    df["conference_game"] = (
        df["home_conference"].notna() & df["away_conference"].notna() &
        (df["home_conference"] == df["away_conference"])
    ).astype(int)

    # Deduplicate — a team can only play one game per week.
    # CFBD occasionally returns duplicate entries for Week 1 / neutral-site games.
    df = df.drop_duplicates(subset=["home_team", "away_team"])
    seen_teams: set = set()
    clean: list = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in seen_teams and a not in seen_teams:
            clean.append(row)
            seen_teams.update([h, a])
    df = pd.DataFrame(clean).reset_index(drop=True)

    # Capture venue_id for weather lookup (CFBD returns venueId on each game)
    venue_ids = {g.get("id"): g.get("venueId") for g in data}
    df["venue_id"] = df["game_id"].map(venue_ids)

    return df


# ─── VENUE & WEATHER ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)  # cache venues for 24 h
def fetch_venues() -> pd.DataFrame:
    """
    Pull all CFB venues from CFBD. Returns DataFrame with:
      venue_id (int), name, lat, lon, is_dome (bool)
    Cached for 24 hours since venues rarely change.
    """
    try:
        data = cfb_get("venues")
    except Exception:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    rows = []
    for v in data:
        loc = v.get("location") or {}
        # CFBD venue roof types that count as enclosed/domed
        dome_roofs = {"dome", "retractable dome", "closed"}
        roof = (v.get("roofType") or "").lower()
        rows.append({
            "venue_id": v.get("id"),
            "venue_name": v.get("name"),
            "lat": loc.get("lat") or loc.get("x"),
            "lon": loc.get("lon") or loc.get("y"),
            "is_dome": int(roof in dome_roofs),
        })
    df = pd.DataFrame(rows)
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce")
    df["lat"]      = pd.to_numeric(df["lat"],      errors="coerce")
    df["lon"]      = pd.to_numeric(df["lon"],      errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=3600)  # re-fetch weather every hour
def fetch_game_weather(game_id: int, lat: float, lon: float,
                       game_date: str, is_dome: int) -> dict:
    """
    Fetch wind speed for a single game via Open-Meteo (free, no API key).
    game_date: ISO date string 'YYYY-MM-DD' or full ISO timestamp.
    Returns dict with wind_speed (mph) and is_dome.
    """
    if is_dome:
        return {"wind_speed": 0.0, "is_dome": 1}
    if not lat or not lon or pd.isna(lat) or pd.isna(lon):
        return {"wind_speed": None, "is_dome": 0}

    try:
        date_str = str(game_date)[:10]  # 'YYYY-MM-DD'
        today    = date.today().isoformat()
        if date_str <= today:
            # Historical — use archive endpoint
            url = (f"https://archive-api.open-meteo.com/v1/archive"
                   f"?latitude={lat}&longitude={lon}"
                   f"&start_date={date_str}&end_date={date_str}"
                   f"&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=auto")
        else:
            # Future — use forecast endpoint
            url = (f"https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}"
                   f"&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=auto"
                   f"&forecast_days=16")
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return {"wind_speed": None, "is_dome": 0}
        j = resp.json()
        speeds = j.get("hourly", {}).get("wind_speed_10m", [])
        times  = j.get("hourly", {}).get("time", [])
        if not speeds:
            return {"wind_speed": None, "is_dome": 0}
        # Pick the hour closest to 3 PM local (typical CFB kickoff window)
        target_hour = f"{date_str}T15:00"
        if target_hour in times:
            idx = times.index(target_hour)
        else:
            # Fall back to afternoon average (hours 12-20)
            afternoon = [s for t, s in zip(times, speeds)
                         if t.startswith(date_str) and "T12" <= t <= "T20"]
            if afternoon:
                return {"wind_speed": round(sum(afternoon) / len(afternoon), 1), "is_dome": 0}
            idx = len(speeds) // 2  # midday fallback
        return {"wind_speed": round(float(speeds[idx]), 1), "is_dome": 0}
    except Exception:
        return {"wind_speed": None, "is_dome": 0}


def attach_weather_to_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Given the schedule DataFrame, fetch venue lat/lon and wind speed for
    each game. Adds 'wind_speed' and 'is_dome' columns in-place.
    Returns a copy with weather columns attached.
    """
    df = games.copy()
    venues = fetch_venues()

    if not venues.empty and "venue_id" in df.columns:
        df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce")
        df = df.merge(venues[["venue_id", "lat", "lon", "is_dome"]],
                      on="venue_id", how="left")
    else:
        df["lat"] = df["lon"] = df["is_dome"] = np.nan

    df["is_dome"]    = pd.to_numeric(df.get("is_dome", 0), errors="coerce").fillna(0).astype(int)
    df["wind_speed"] = np.nan

    for idx, row in df.iterrows():
        w = fetch_game_weather(
            game_id  = row.get("game_id", 0),
            lat      = row.get("lat"),
            lon      = row.get("lon"),
            game_date= str(row.get("start_date", ""))[:10],
            is_dome  = int(row.get("is_dome", 0)),
        )
        df.at[idx, "wind_speed"] = w.get("wind_speed")
        df.at[idx, "is_dome"]    = w.get("is_dome", 0)

    return df


@st.cache_data(show_spinner="Fetching odds...", ttl=1800)
def fetch_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consensus lines from The Odds API (median across US books), backed by
    CFBD for any game it doesn't cover. CFBD alone if the key is missing/down.
    """
    odds_df = pd.DataFrame()
    key = _theodds_api_key()
    if key:
        try:
            odds_df = odds_api.fetch_lines(key, games_df)
        except Exception as exc:
            print(f"The Odds API unavailable — using CFBD lines: {exc}")

    # ── CFBD lines (fill gaps / full fallback) ────────────────────────────────
    season = int(games_df["season"].iloc[0])
    week   = int(games_df["week"].iloc[0])
    cfbd_df = pd.DataFrame()
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
        priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
        rank_map = {p: i for i, p in enumerate(priority)}
        rows = []
        for game in data:
            for line in game.get("lines", []):
                rows.append({"game_id": game.get("id"),
                             "spread": line.get("spread"),
                             "over_under": line.get("overUnder"),
                             "spread_open": line.get("spreadOpen"),
                             "home_moneyline": line.get("homeMoneyline"),
                             "away_moneyline": line.get("awayMoneyline"),
                             "provider": line.get("provider"),
                             "_rank": rank_map.get(line.get("provider", ""), 99)})
        if rows:
            cfbd_df = (pd.DataFrame(rows).sort_values("_rank")
                       .drop_duplicates("game_id", keep="first")
                       .drop(columns=["_rank"]))
    except Exception:
        cfbd_df = pd.DataFrame()

    return odds_api.merge_lines(odds_df, cfbd_df)


# ─── FEATURE BUILDING & PREDICTION ───────────────────────────────────────────

def build_and_predict(games, lines, ratings, epa, elo,
                      spread_model, totals_model, win_prob_model, feature_lists,
                      weather: pd.DataFrame | None = None):
    """
    Merge lines onto games, build feature vectors via feature_builder, run
    the three models, and return a predictions DataFrame.
    weather: optional DataFrame with game_id, wind_speed, is_dome columns.
    """
    # ── Merge lines ───────────────────────────────────────────────────────
    if not lines.empty:
        ml_avail = [c for c in ["home_moneyline", "away_moneyline"] if c in lines.columns]
        # Best-available numbers for line shopping (present when lines came
        # from The Odds API; absent for CFBD-fill games).
        best_cols = [c for c in ["best_under_total", "best_under_book",
                                 "best_over_total", "best_over_book",
                                 "best_home_ml", "best_home_ml_book",
                                 "best_away_ml", "best_away_ml_book", "n_books"]
                     if c in lines.columns]
        line_cols = ["game_id", "spread", "over_under", "spread_open"] + ml_avail + best_cols
        if "provider" in lines.columns:
            line_cols.append("provider")
        df = games.merge(
            lines[[c for c in line_cols if c in lines.columns]],
            on="game_id", how="left"
        )
    else:
        df = games.copy()
        df["spread"] = df["over_under"] = df["spread_open"] = np.nan

    if "home_moneyline" not in df.columns:
        df["home_moneyline"] = np.nan
    if "away_moneyline" not in df.columns:
        df["away_moneyline"] = np.nan

    # ── Build all team features via shared feature_builder ────────────────
    df = attach_team_features(df, ratings, epa, elo if not elo.empty else None)

    # ── Merge weather (wind_speed, is_dome) ──────────────────────────────
    if weather is not None and not weather.empty:
        wcols = [c for c in ["game_id", "wind_speed", "is_dome"] if c in weather.columns]
        df = df.merge(weather[wcols], on="game_id", how="left")
        # Dome games: hard-zero wind so model gets the same signal as training
        if "is_dome" in df.columns and "wind_speed" in df.columns:
            df["is_dome"] = df["is_dome"].fillna(0).astype(int)
            df.loc[df["is_dome"] == 1, "wind_speed"] = 0.0
    else:
        if "wind_speed" not in df.columns:
            df["wind_speed"] = np.nan
        if "is_dome" not in df.columns:
            df["is_dome"] = 0

    # ── Assemble feature matrices for each model ──────────────────────────
    def make_feat(feat_names):
        out = pd.DataFrame(index=df.index)
        for f in feat_names:
            out[f] = df[f] if f in df.columns else np.nan
        return out

    feat_sp  = make_feat(feature_lists["spread"])
    feat_tot = make_feat(feature_lists["totals"])
    feat_win = make_feat(feature_lists["win_prob"])

    # ── Build output frame ────────────────────────────────────────────────
    out_cols = ["game_id", "season", "week", "start_date",
                "home_team", "away_team", "home_conference", "away_conference",
                "neutral_site", "conference_game", "spread", "over_under",
                "spread_open", "home_moneyline", "away_moneyline",
                "best_under_total", "best_under_book", "best_over_total",
                "best_over_book", "best_home_ml", "best_home_ml_book",
                "best_away_ml", "best_away_ml_book", "n_books",
                "home_unrated", "away_unrated", "has_unrated_opponent",
                "wind_speed", "is_dome",
                # Driver columns — power the "why this pick" chips on cards
                "sp_diff", "elo_diff", "fpi_diff", "srs_diff",
                "wepa_off_diff", "wepa_def_diff", "rest_diff", "hfa_diff",
                "talent_diff", "portal_net_rating_diff", "line_movement",
                "sharp_move_home", "sharp_move_away",
                "sharp_total_under", "sharp_total_over",
                "spread_open_val", "total_movement",
                "tempo_combined", "rush_rate_combined"]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    if "provider" in df.columns:
        out["provider"] = df["provider"]

    # Spread model: legacy models predict the raw home margin; current models
    # (feature_lists.json: spread_target=margin_residual) predict the residual
    # vs the Vegas line, so the line is added back here. NaN when no line yet.
    _sp_raw = spread_model.predict(feat_sp)
    if feature_lists.get("spread_target") == "margin_residual":
        _vm = -pd.to_numeric(out["spread"], errors="coerce")
        out["pred_spread"] = _vm + _sp_raw
    else:
        out["pred_spread"] = _sp_raw
    # Totals model predicts deviation from the O/U line, not the raw total.
    # Add the line back so pred_total is the expected combined score
    # (same handling as weekly_pipeline.py; NaN when no line is posted yet).
    _ou_vals = pd.to_numeric(out["over_under"], errors="coerce")
    out["pred_total"]      = _ou_vals + totals_model.predict(feat_tot)
    out["pred_win_p"]      = win_prob_model.predict_proba(feat_win)[:, 1]
    out["pred_away_win_p"] = 1 - out["pred_win_p"]

    # ── Cross-calibration: blend spread-implied win prob with classifier ──────
    # Ensures spread prediction and win probability are internally consistent.
    # Parameters (sigma, alpha) are tuned on the validation season in model.py.
    _calib_path = MODEL_DIR / "win_prob_calibration.json"
    if _calib_path.exists():
        import json as _json
        from math import erf as _erf, sqrt as _msqrt
        def _norm_cdf(x): return 0.5 * (1 + _erf(float(x) / _msqrt(2)))
        _calib  = _json.load(open(_calib_path))
        _sigma  = _calib["spread_sigma"]
        _alpha  = _calib["blend_alpha"]
        _s_impl = out["pred_spread"].apply(
            lambda s: _norm_cdf(s / _sigma) if pd.notna(s) else np.nan)
        _blend  = _alpha * _s_impl + (1 - _alpha) * out["pred_win_p"]
        # Games without a line have no spread-implied prob — keep classifier-only
        out["pred_win_p"]      = _blend.fillna(out["pred_win_p"]).clip(0.01, 0.99)
        out["pred_away_win_p"] = 1 - out["pred_win_p"]

    out["spread_edge"]     = out["pred_spread"] - (-out["spread"])
    out["totals_edge"]     = out["pred_total"]  - out["over_under"]

    out["home_ml_ev"] = out.apply(
        lambda r: ml_ev(r["pred_win_p"], r["home_moneyline"]), axis=1)
    out["away_ml_ev"] = out.apply(
        lambda r: ml_ev(r["pred_away_win_p"], r["away_moneyline"]), axis=1)
    out["model_home_ml"] = out["pred_win_p"].apply(prob_to_american)
    out["model_away_ml"] = out["pred_away_win_p"].apply(prob_to_american)

    def best_ml(r):
        h, a = r["home_ml_ev"], r["away_ml_ev"]
        if pd.isna(h) and pd.isna(a):
            return pd.Series({"ml_team": None, "ml_ev": np.nan,
                               "ml_book_odds": np.nan, "ml_model_odds": np.nan})
        if pd.isna(a) or (not pd.isna(h) and h >= a):
            return pd.Series({"ml_team": r["home_team"], "ml_ev": h,
                               "ml_book_odds": r["home_moneyline"],
                               "ml_model_odds": r["model_home_ml"]})
        return pd.Series({"ml_team": r["away_team"], "ml_ev": a,
                           "ml_book_odds": r["away_moneyline"],
                           "ml_model_odds": r["model_away_ml"]})

    out[["ml_team", "ml_ev", "ml_book_odds", "ml_model_odds"]] = out.apply(best_ml, axis=1)

    # Coverage must be measured on the FULL feature frame (df) — `out` keeps
    # only prediction columns, so checking it would misreport every rating
    # source as 0%. Stash the report in attrs for the picks tab to read.
    out.attrs["coverage"] = feature_coverage_report(df)
    return out


def apply_qb_adjustments(preds: pd.DataFrame, qb_out_teams: list,
                         pts_per_team: float) -> pd.DataFrame:
    """
    Shift predictions for teams whose starting QB is out/doubtful.

    The model cannot see injury news — the market prices a backup QB at
    roughly 4–7 points, so an unadjusted model produces false 'edges' on
    exactly the games where it knows the least. This overlay:
      • shifts pred_spread by ±pts (home margin down if home QB out, up if away)
      • shifts win probability by the EXACT spread-implied blend amount:
            Δwp = α · [Φ((s+δ)/σ) − Φ(s/σ)]
        using the σ/α saved by model.py's cross-calibration
      • recomputes spread_edge and all moneyline EV columns
      • tags rows with qb_adjustment so cards can show why
    """
    preds = preds.copy()
    delta = pd.Series(0.0, index=preds.index)
    for team in qb_out_teams:
        delta = delta - pts_per_team * (preds["home_team"] == team).astype(float)
        delta = delta + pts_per_team * (preds["away_team"] == team).astype(float)

    affected = delta != 0
    if not affected.any():
        return preds

    # Win probability: exact blend shift (needs original pred_spread + σ/α)
    calib_path = MODEL_DIR / "win_prob_calibration.json"
    if calib_path.exists() and preds["pred_spread"].notna().any():
        from math import erf as _erf, sqrt as _msqrt
        def _ncdf(x): return 0.5 * (1 + _erf(float(x) / _msqrt(2)))
        calib = json.loads(calib_path.read_text())
        sigma, alpha = calib["spread_sigma"], calib["blend_alpha"]
        s = preds["pred_spread"]
        shift = pd.Series([
            alpha * (_ncdf((sv + dv) / sigma) - _ncdf(sv / sigma))
            if pd.notna(sv) and dv != 0 else 0.0
            for sv, dv in zip(s, delta)
        ], index=preds.index)
        preds["pred_win_p"]      = (preds["pred_win_p"] + shift).clip(0.01, 0.99)
        preds["pred_away_win_p"] = 1 - preds["pred_win_p"]

    preds["pred_spread"] = preds["pred_spread"] + delta.where(preds["pred_spread"].notna(), 0)
    preds["spread_edge"] = preds["pred_spread"] - (-preds["spread"])
    preds["qb_adjustment"] = delta

    # Recompute moneyline EV / model odds with the shifted win prob
    preds["home_ml_ev"] = preds.apply(
        lambda r: ml_ev(r["pred_win_p"], r["home_moneyline"]), axis=1)
    preds["away_ml_ev"] = preds.apply(
        lambda r: ml_ev(r["pred_away_win_p"], r["away_moneyline"]), axis=1)
    preds["model_home_ml"] = preds["pred_win_p"].apply(prob_to_american)
    preds["model_away_ml"] = preds["pred_away_win_p"].apply(prob_to_american)

    def _best_ml(r):
        h, a = r["home_ml_ev"], r["away_ml_ev"]
        if pd.isna(h) and pd.isna(a):
            return pd.Series({"ml_team": None, "ml_ev": np.nan,
                               "ml_book_odds": np.nan, "ml_model_odds": np.nan})
        if pd.isna(a) or (not pd.isna(h) and h >= a):
            return pd.Series({"ml_team": r["home_team"], "ml_ev": h,
                               "ml_book_odds": r["home_moneyline"],
                               "ml_model_odds": r["model_home_ml"]})
        return pd.Series({"ml_team": r["away_team"], "ml_ev": a,
                           "ml_book_odds": r["away_moneyline"],
                           "ml_model_odds": r["model_away_ml"]})
    preds[["ml_team", "ml_ev", "ml_book_odds", "ml_model_odds"]] = preds.apply(_best_ml, axis=1)

    return preds


# ─── UI HELPERS ───────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap');

    :root {
        --bg:        #0b0e14;
        --bg-raised: #11151f;
        --card:      #161b28;
        --card-hi:   #1c2333;
        --line:      #232b3d;
        --line-hi:   #2e3950;
        --ink:       #f3f5f9;
        --ink-2:     #aab4c5;
        --ink-3:     #67738a;
        --ink-4:     #454f63;
        --gold:      #f5c518;
        --gold-deep: #b8930a;
        --green:     #34d399;
        --red:       #f87171;
        --cyan:      #22d3ee;
        --orange:    #fb923c;
        --blue:      #60a5fa;
        --violet:    #a78bfa;
        --mono:      'JetBrains Mono', ui-monospace, monospace;
    }

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background-color: var(--bg);
        color: var(--ink);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 500px at 75% -10%, rgba(245,197,24,0.05), transparent 60%),
            radial-gradient(900px 400px at 10% -5%, rgba(96,165,250,0.04), transparent 55%),
            var(--bg);
    }
    [data-testid="stHeader"] {
        background-color: transparent;
        border-bottom: none;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-raised);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stSelectbox label {
        color: var(--ink-3) !important;
        font-size: 0.78em !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700 !important;
    }

    /* ── Tabs: underline style ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-radius: 0;
        padding: 0;
        gap: 26px;
        border: none;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--ink-3);
        background: transparent;
        border-radius: 0;
        font-weight: 600;
        font-size: 0.9em;
        padding: 10px 2px;
        border-bottom: 2px solid transparent;
        transition: color 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--ink-2); }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: transparent;
        color: var(--gold);
        border-bottom: 2px solid var(--gold);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* ── Sub-nav radio as segmented pills ── */
    .stRadio > label { display: none !important; }
    .stRadio > div {
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        padding: 8px 0 12px 0 !important;
    }
    .stRadio > div > label {
        background: var(--card) !important;
        border: 1px solid var(--line) !important;
        border-radius: 999px !important;
        padding: 6px 18px !important;
        cursor: pointer !important;
        font-size: 0.82em !important;
        font-weight: 600 !important;
        color: var(--ink-3) !important;
        margin: 0 !important;
        transition: all 0.15s !important;
    }
    .stRadio > div > label:hover {
        border-color: var(--line-hi) !important;
        color: var(--ink-2) !important;
        transform: translateY(-1px);
    }
    .stRadio > div > label:has(input:checked) {
        background: linear-gradient(135deg, var(--gold), #e3ae09) !important;
        border-color: var(--gold) !important;
        color: #0b0e14 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 14px rgba(245,197,24,0.25);
    }
    .stRadio > div > label > div:first-child { display: none !important; }

    /* ── Metric tiles ── */
    [data-testid="metric-container"] {
        background: linear-gradient(180deg, var(--card-hi), var(--card));
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid var(--line);
        box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.25);
    }
    [data-testid="metric-container"] label {
        color: var(--ink-3) !important;
        font-size: 0.7em !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-size: 1.6em !important;
        font-weight: 800 !important;
        font-family: var(--mono);
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        margin-bottom: 8px;
        transition: border-color 0.15s;
    }
    [data-testid="stExpander"]:hover { border-color: var(--line-hi) !important; }
    [data-testid="stExpander"] summary {
        color: var(--ink-2);
        font-weight: 600;
        font-size: 0.92em;
    }
    [data-testid="stExpander"] summary:hover { color: var(--gold); }

    /* ── Buttons ── */
    [data-testid="stButton"] > button {
        background: var(--card);
        color: var(--ink-3);
        border: 1px solid var(--line);
        border-radius: 8px;
        font-size: 0.8em;
        font-weight: 600;
        padding: 4px 14px;
        transition: all 0.15s;
    }
    [data-testid="stButton"] > button:hover {
        background: var(--card-hi);
        color: var(--ink);
        border-color: var(--line-hi);
        transform: translateY(-1px);
    }
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, var(--gold), #e3ae09);
        color: #0b0e14;
        border: none;
        font-weight: 800;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 14px rgba(245,197,24,0.25);
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #ffd83d, var(--gold));
        box-shadow: 0 4px 20px rgba(245,197,24,0.35);
    }

    /* ── Selects / Slider / Inputs ── */
    [data-baseweb="select"] > div {
        background: var(--card) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSlider"] > div > div > div { background: var(--gold) !important; }
    [data-testid="stTextInput"] input {
        background: var(--card) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        border-radius: 8px !important;
    }

    /* ── Alerts / dividers / typography ── */
    [data-testid="stAlert"] { background: var(--card); border-radius: 12px; }
    hr { border-color: var(--line) !important; margin: 16px 0 !important; }
    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.01em; }
    p, li, .stMarkdown { color: var(--ink-2); }
    [data-testid="stCaptionContainer"] { color: var(--ink-4) !important; }
    [data-testid="stDownloadButton"] > button {
        background: var(--card); color: var(--ink-3);
        border: 1px solid var(--line); border-radius: 8px;
    }
    code { background: var(--card); color: var(--gold); border-radius: 4px; padding: 1px 5px; }

    /* ── Pick cards ── */
    .pick-card {
        background: linear-gradient(180deg, var(--card-hi), var(--card));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px 20px 14px 20px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    }
    .pick-card:hover {
        transform: translateY(-2px);
        border-color: var(--line-hi);
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
    }
    .pick-card .accent {
        position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    }
    .chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--line);
        color: var(--ink-2);
        font-size: 0.7em; font-weight: 600;
        padding: 3px 9px; border-radius: 999px;
        margin: 2px 4px 2px 0; white-space: nowrap;
    }
    .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

    /* ── Panel cards (right column) ── */
    .panel-card {
        background: linear-gradient(180deg, var(--card-hi), var(--card));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 16px 12px 16px;
        margin-bottom: 10px;
    }
    .panel-card .panel-label {
        color: var(--ink-4);
        font-size: 0.62em;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .panel-bet-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 5px 0;
        border-bottom: 1px solid var(--line);
    }
    .panel-bet-row:last-child { border-bottom: none; }

    /* ── Edge strength bar on pick cards ── */
    .edge-bar-track {
        height: 3px;
        background: var(--line);
        border-radius: 999px;
        margin: 6px 0 10px 0;
        overflow: hidden;
    }
    .edge-bar-fill {
        height: 100%;
        border-radius: 999px;
    }
    </style>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    sub = (f'<span style="color:#4b5563;font-size:0.8em;margin-left:10px">{subtitle}</span>'
           if subtitle else "")
    st.markdown(f"""
    <div style="margin:28px 0 14px 0;display:flex;align-items:center;gap:12px">
        <span style="background:#1a1f2e;color:#9ca3af;font-size:0.7em;font-weight:700;
                     letter-spacing:0.12em;text-transform:uppercase;white-space:nowrap;
                     padding:4px 12px;border-radius:20px;border:1px solid #252d3d">{title}</span>
        <div style="flex:1;height:1px;background:#1e2537"></div>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def confidence_stars(edge_abs: float) -> str:
    if edge_abs >= 5.5: return "★★★"
    if edge_abs >= 4.5: return "★★"
    return "★"

def format_kickoff(start_date) -> str:
    """Format ISO start_date → 'Sat 8/30 · 3:30 PM ET' for card display."""
    if not start_date or pd.isna(start_date):
        return ""
    try:
        import datetime as _dt
        # API returns ISO string like "2026-08-30T19:30:00.000Z" (UTC)
        ts = pd.to_datetime(start_date, utc=True)
        # Convert UTC → Eastern Time (ET = UTC-4 in summer, UTC-5 in winter)
        # CFB season runs Aug-Jan; most games are EDT (UTC-4)
        et = ts - _dt.timedelta(hours=4)
        day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][et.weekday()]
        hour, minute = et.hour, et.minute
        am_pm = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        time_str = f"{hour12}:{minute:02d} {am_pm} ET"
        return f"{day_name} {et.month}/{et.day} · {time_str}"
    except Exception:
        return ""

def ev_stars(ev: float) -> str:
    if ev >= 0.07: return "★★★"
    if ev >= 0.05: return "★★"
    return "★"

def track_button(label: str, game: str, bet_type: str, pick: str,
                 line: str, units: int, season: int, week: int, edge: str = "", key_prefix: str = ""):
    """Render a small Track button. Returns True if clicked."""
    key = f"{key_prefix}track_{game}_{bet_type}_{pick}".replace(" ", "_")
    bettor = st.session_state.get("bettor", BETTORS[0])
    if st.button(f"+ Track  {label}", key=key, width='content'):
        add_bet(game, bet_type, pick, line, units, season, week, edge, bettor)
        st.toast(f"Added: {pick} — {bettor}", icon="✅")
        return True
    return False

# ─── UNIFIED PICK CARD ────────────────────────────────────────────────────────
# One renderer for all bet types. Design goals:
#   1. Show WHY the pick exists (driver chips computed from model features)
#   2. Show model-vs-market visually (edge meter) instead of just a number
#   3. Confidence tiers grounded in the actual 2025 holdout backtest —
#      the UI itself encodes which bet types have historically worked.

# Power-4 (and legacy power) conferences for segment gating. Walk-forward
# 2019-25: totals edge only exists when a power team is involved — G5vG5
# totals hit 48.2% and are excluded from CORE.
_POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
                "Big East", "FBS Independents"}

def _power_involved(row) -> bool:
    return (row.get("home_conference") in _POWER_CONFS
            or row.get("away_conference") in _POWER_CONFS)


def _wind15(row) -> bool:
    """Forecast wind >= 15 mph outdoors. CORE unders hit only 50.0% in
    high wind (n=88) — the market prices obvious weather itself."""
    ws = row.get("wind_speed")
    if pd.isna(ws) or bool(row.get("is_dome", 0)):
        return False
    return float(ws) >= 15


def _low_total(row) -> bool:
    """Market total < 48. Low-total games have no over-bias to fade — CORE
    unders there hit ~49%. The inflation the edge exploits lives in higher
    totals (public backs overs in expected shootouts)."""
    ou = row.get("over_under")
    return pd.notna(ou) and float(ou) < 48


def _core_total(row) -> bool:
    """Refined CORE gate, walk-forward 2019-25: under, edge 2-7 pts,
    power-conf involved, wind < 15, market total >= 48 → 54.9% (n=559,
    +4.8% ROI, z=2.33), profitable 6 of 7 seasons. Excluded because they add no
    edge: edges >7 pts (50.7%, winner's curse), wind>=15 (50.0%, priced
    in), totals <48 (49%, no over-bias to fade)."""
    edge = row["totals_edge"]
    return bool(edge <= -2 and edge >= -7 and _power_involved(row)
                and not _wind15(row) and not _low_total(row))


def _both_power(row) -> bool:
    return (row.get("home_conference") in _POWER_CONFS
            and row.get("away_conference") in _POWER_CONFS)


def _core_premium(row) -> bool:
    """RETIRED 2026-08-24. A both-power CORE tier looked strong on the Jul-01
    feature matrix (59.8% vs 55.0%), but after CFBD revised returning-production
    for 1,227 historical rows the split collapsed to 55.1% vs 54.5% — i.e. it
    was an artifact of stale inputs, not a real effect. Kept as a stub returning
    False so CORE sizes uniformly; do not re-add without fresh validation."""
    return False


def _high_total_paper(row) -> bool:
    """Pure market-bias play, PAPER only: bet UNDER whenever the market total
    is >= 60, independent of the model. Walk-forward 2019-25 (excluding games
    already in CORE): 53.8% over 974 bets, +2.7% ROI, p=0.019, profitable 5/7
    seasons. The edge is uniform whether the model leans under (53.5%) or over
    (54.0%), confirming it is book/public over-inflation on shootout-hyped
    games, not model skill. Thin (~1.4 pts above breakeven) and it failed two
    seasons, so it is tracked at 0u until a live season corroborates it."""
    ou = row.get("over_under")
    return bool(pd.notna(ou) and float(ou) >= 60 and not _core_total(row))


def _tier_badge(kind: str, row) -> tuple[str, str]:
    """Return (label, color) tier gated by walk-forward results 2019-25
    (5,100 games). Segments below are the only ones that cleared breakeven:
      - Unders, edge>=2, power-conf involved: 55.8% (n=868, +6.4% ROI)
      - Everything else totals: no edge (overs 49.4%, G5vG5 48.2%)
      - Spreads wk10+: 44.7% at edge>=3 — actively bad, flagged PASS
      - Spreads wk1-3 edge>=3: 53.5% (n=484) — marginal, watch-list only
    """
    if kind == "total":
        edge = row["totals_edge"]
        if row.get("_force_under") and not _core_total(row):
            return "PAPER · HIGH-TOTAL 54.3%", "var(--orange)"
        if edge < 0:  # under pick
            if _core_total(row):
                return "CORE PLAY · 54.9% '19-'25", "var(--green)"
            # Explain the specific disqualifier
            if abs(edge) > 7:
                return "CAUTION · 7+PT EDGES 51%", "var(--orange)"
            if _wind15(row):
                return "CAUTION · WIND PRICED IN", "var(--orange)"
            if _low_total(row):
                return "CAUTION · LOW TOTAL 49%", "var(--orange)"
            if not _power_involved(row):
                return "MARGINAL · G5 NO EDGE", "var(--ink-3)"
            if abs(edge) >= 2:
                return "MARGINAL · BELOW GATE", "var(--ink-3)"
            return "LEAN · BELOW EDGE MIN", "var(--ink-3)"
        return "CAUTION · OVERS 49% '19-'25", "var(--orange)"
    if kind == "spread":
        wk = int(row.get("week", 0) or 0)
        if wk >= 10:
            return "PASS · 44.7% ATS WK10+", "var(--red)"
        if wk <= 3 and abs(row.get("spread_edge", 0) or 0) >= 3:
            return "WATCH · 53.5% EARLY SZN", "var(--blue)"
        return "INFO ONLY · ~50% ATS", "var(--ink-3)"
    # moneyline — EV>=4% strategy backtested on walk-forward 2023-25 (n=868):
    # +13.5% ROI in 2023, +0.6% in 2024, −3.7% in 2025, z=0.77. Not validated;
    # tracked as a paper record only, never sized.
    return "PAPER · '25 −4% ROI", "var(--orange)"


def _is_play(kind: str, row) -> bool:
    """True when the pick's segment has a validated walk-forward edge.
    Non-plays render with 0u — shown for research, not sized."""
    if kind == "total":
        return _core_total(row)
    if kind == "spread":
        return int(row.get("week", 0) or 0) <= 9
    return False  # moneyline: paper record only


def _winprob_bar_html(row) -> str:
    """Horizontal home/away win probability bar."""
    p = row.get("pred_win_p")
    if pd.isna(p):
        return ""
    p = float(p)
    home, away = row["home_team"], row["away_team"]
    home_pct, away_pct = p * 100, (1 - p) * 100
    hc, ac = ("var(--gold)", "var(--line-hi)") if p >= 0.5 else ("var(--line-hi)", "var(--gold)")
    return f"""
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;font-size:0.72em;margin-bottom:4px">
        <span style="color:var(--ink-2);font-weight:600">{home} <span class="num" style="color:var(--ink)">{home_pct:.0f}%</span></span>
        <span style="color:var(--ink-2);font-weight:600"><span class="num" style="color:var(--ink)">{away_pct:.0f}%</span> {away}</span>
      </div>
      <div style="display:flex;height:6px;border-radius:999px;overflow:hidden;background:var(--line)">
        <div style="width:{home_pct:.1f}%;background:{hc}"></div>
        <div style="width:{away_pct:.1f}%;background:{ac}"></div>
      </div>
    </div>"""


def _edge_meter_html(market_val: float, model_val: float, color: str,
                     market_label: str = "Vegas", model_label: str = "Model",
                     span: float = 8.0, unit: str = "") -> str:
    """
    Number-line showing where the model's number sits relative to the market.
    Market anchors the center; the model dot is offset by the edge (clamped).
    """
    if pd.isna(market_val) or pd.isna(model_val):
        return ""
    diff = float(model_val) - float(market_val)
    off  = max(-span, min(span, diff)) / span * 44   # ±44% from center
    mpct = 50 + off
    lo, hi = min(50.0, mpct), max(50.0, mpct)
    return f"""
    <div style="margin-top:12px">
      <div style="position:relative;height:6px;border-radius:999px;background:var(--line)">
        <div style="position:absolute;left:{lo:.1f}%;width:{hi - lo:.1f}%;top:0;bottom:0;
                    background:{color};opacity:0.35;border-radius:999px"></div>
        <div style="position:absolute;left:50%;top:50%;width:2px;height:14px;
                    transform:translate(-50%,-50%);background:var(--ink-3);border-radius:2px"></div>
        <div style="position:absolute;left:{mpct:.1f}%;top:50%;width:12px;height:12px;
                    transform:translate(-50%,-50%);background:{color};border-radius:50%;
                    box-shadow:0 0 8px {color}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.68em;margin-top:5px;color:var(--ink-4)">
        <span>{market_label} <span class="num" style="color:var(--ink-2)">{market_val:.1f}{unit}</span></span>
        <span style="color:{color};font-weight:700">{model_label} <span class="num">{model_val:.1f}{unit}</span> ({diff:+.1f})</span>
      </div>
    </div>"""


def _driver_chips_html(row, kind: str) -> str:
    """Plain-English 'why this pick' chips computed from the model's inputs."""
    chips: list[str] = []

    def _get(col):
        v = row.get(col)
        return float(v) if v is not None and pd.notna(v) else None

    def add(icon, text, color="var(--ink-2)"):
        chips.append(f'<span class="chip" style="color:{color}">{icon} {text}</span>')

    def team_for(diff):  # positive diff favors home
        return row["home_team"] if diff > 0 else row["away_team"]

    if kind == "total":
        wind = _get("wind_speed")
        if int(row.get("is_dome", 0) or 0):
            add("🏟", "Dome — weather neutral")
        elif wind is not None and wind >= 12:
            add("💨", f"{wind:.0f} mph wind → under lean",
                "var(--red)" if wind >= 20 else "var(--orange)")
        rr = _get("rush_rate_combined")
        if rr is not None and rr > 0:
            if rr >= 1.12: add("🏃", "Run-heavy matchup → shorter game")
            elif rr <= 0.88: add("🎯", "Pass-heavy matchup → longer game")
        tempo = _get("tempo_combined")
        if tempo is not None and tempo >= 13.5:
            add("⚡", "Both offenses sustain drives")
    else:
        sp  = _get("sp_diff");  elo = _get("elo_diff");  fpi = _get("fpi_diff")
        if sp is not None and abs(sp) >= 4:
            add("📊", f"SP+ favors {team_for(sp)} by {abs(sp):.1f}")
        if elo is not None and abs(elo) >= 80:
            add("♟", f"Elo gap {abs(elo):.0f} → {team_for(elo)}")
        if fpi is not None and abs(fpi) >= 4 and (sp is None or (fpi > 0) != (sp > 0)):
            add("⚠️", f"FPI disagrees — favors {team_for(fpi)}", "var(--orange)")
        rest = _get("rest_diff")
        if rest is not None and abs(rest) >= 3:
            add("🛌", f"{team_for(rest)} +{abs(rest):.0f} days rest")
        hfa = _get("hfa_diff")
        if hfa is not None and abs(hfa) >= 2.5:
            add("🏟", f"Venue edge {team_for(hfa)} ({abs(hfa):.1f} pts)")
        portal = _get("portal_net_rating_diff")
        if portal is not None and abs(portal) >= 8:
            add("🔄", f"Portal winner: {team_for(portal)}")

    # Line movement shown as neutral fact only. Walk-forward validation
    # (2023-25 openers, n=2353) found movement-agrees-with-pick adds no edge
    # on spreads and is *inverted* on totals — never present it as confirmation.
    lm = _get("line_movement")
    if lm is not None and abs(lm) >= 1.5:
        toward = row["home_team"] if lm < 0 else row["away_team"]
        add("📈", f"Line moved {abs(lm):.1f} toward {toward}", "var(--blue)")
    if kind == "total":
        tm = _get("total_movement")
        if tm is not None and abs(tm) >= 2.0:
            direction = "Over" if tm > 0 else "Under"
            add("📈", f"Total moved {abs(tm):.1f} toward {direction}", "var(--blue)")
    if bool(row.get("has_unrated_opponent", False)):
        add("❓", "Unrated opponent — low data confidence", "var(--red)")
    qb_adj = _get("qb_adjustment")
    if qb_adj is not None and qb_adj != 0:
        add("🏥", f"QB-out adjustment {qb_adj:+.1f} pts applied", "var(--orange)")

    return ("<div style='margin-top:12px'>" + "".join(chips[:5]) + "</div>") if chips else ""


def _stat_footer_html(cells: list[tuple[str, str, str]]) -> str:
    """cells = [(label, value, color)]"""
    cols = []
    for i, (label, value, color) in enumerate(cells):
        border = "border-left:1px solid var(--line);" if i else ""
        cols.append(f"""
        <div style="flex:1;text-align:center;{border}">
            <div style="color:var(--ink-4);font-size:0.62em;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px">{label}</div>
            <div class="num" style="color:{color};font-size:0.9em;font-weight:700">{value}</div>
        </div>""")
    return ('<div style="display:flex;margin-top:12px;border-top:1px solid var(--line);'
            'padding-top:10px">' + "".join(cols) + "</div>")


def _pick_card_html(*, accent: str, kind_badge: str, kind_color: str,
                    tier: tuple[str, str], headline: str, big_num: str,
                    sub: str, meters: str, chips: str, footer: str,
                    metric_html: str, edge_pct: float = 0.0) -> str:
    """edge_pct 0-100: drives the thin strength bar under the badge row."""
    tier_label, tier_color = tier
    bar_color = ("var(--green)" if edge_pct >= 60 else
                 "var(--gold)"  if edge_pct >= 35 else "var(--ink-3)")
    return f"""
    <div class="pick-card">
        <div class="accent" style="background:{accent}"></div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span style="background:{kind_color};color:#0b0e14;font-size:0.62em;
                             font-weight:800;letter-spacing:0.1em;padding:3px 9px;
                             border-radius:5px">{kind_badge}</span>
                <span style="border:1px solid {tier_color};color:{tier_color};font-size:0.6em;
                             font-weight:800;letter-spacing:0.08em;padding:2px 8px;
                             border-radius:5px">{tier_label}</span>
            </div>
            {metric_html}
        </div>
        <div class="edge-bar-track">
            <div class="edge-bar-fill" style="width:{min(edge_pct,100):.0f}%;background:{bar_color}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:var(--ink);font-size:1.15em;font-weight:800;letter-spacing:-0.01em">{headline}</span>
            <span class="num" style="color:{accent};font-size:1.35em;font-weight:800">{big_num}</span>
        </div>
        <div style="color:var(--ink-4);font-size:0.78em;margin-top:3px">{sub}</div>
        {meters}{chips}{footer}
    </div>"""


# ─── LINE SHOPPING ────────────────────────────────────────────────────────────
# Model edges are computed vs the consensus (median) line — the validated
# reference. These helpers surface the single best number available across the
# 11 US books so the user bets the best price, not the average. Betting the
# best total (median book-to-book gap is ~1 pt) is free ROI on a validated pick.

_BOOK_NAMES = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
    "caesars": "Caesars", "williamhill_us": "Caesars", "pointsbetus": "PointsBet",
    "betrivers": "BetRivers", "bovada": "Bovada", "betonlineag": "BetOnline",
    "mybookieag": "MyBookie", "lowvig": "LowVig", "espnbet": "ESPN BET",
    "hardrockbet": "Hard Rock", "fanatics": "Fanatics", "unibet_us": "Unibet",
    "superbook": "SuperBook", "wynnbet": "WynnBET", "ballybet": "Bally Bet",
}


def _book_name(k) -> str:
    if k is None or k == "" or (isinstance(k, float) and pd.isna(k)):
        return ""
    return _BOOK_NAMES.get(str(k), str(k).replace("_", " ").title())


def _shop_chip(text: str, color: str) -> str:
    return (f"<div style='margin-top:12px'><span class=\"chip\" "
            f"style=\"color:{color}\">{text}</span></div>")


def _line_shop_total_html(row, is_under: bool) -> str:
    """Chip: best available total + book for the picked side, vs consensus."""
    cons = row.get("over_under")
    best = row.get("best_under_total" if is_under else "best_over_total")
    book = _book_name(row.get("best_under_book" if is_under else "best_over_book"))
    if pd.isna(cons) or best is None or pd.isna(best) or not book:
        return ""
    side = "U" if is_under else "O"
    better = (float(best) > float(cons)) if is_under else (float(best) < float(cons))
    if better:
        gain = abs(float(best) - float(cons))
        return _shop_chip(
            f"🛒 Best number: {side}{float(best):.1f} at {book} "
            f"<b style='color:var(--green)'>+{gain:.1f} pts</b> vs {float(cons):.1f} consensus",
            "var(--ink-2)")
    return _shop_chip(f"🛒 Best number: {side}{float(best):.1f} at {book} "
                      f"(matches consensus)", "var(--ink-3)")


def _line_shop_ml_html(row, is_home: bool) -> str:
    """Chip: best available moneyline price + book for the picked side."""
    best = row.get("best_home_ml" if is_home else "best_away_ml")
    book = _book_name(row.get("best_home_ml_book" if is_home else "best_away_ml_book"))
    cons = row.get("home_moneyline" if is_home else "away_moneyline")
    if best is None or pd.isna(best) or not book:
        return ""
    best_s = f"+{int(best)}" if best > 0 else str(int(best))
    better = pd.notna(cons) and float(best) > float(cons)
    tag = (f" <b style='color:var(--green)'>best price</b>" if better
           else " (matches consensus)")
    return _shop_chip(f"🛒 Best price: {best_s} at {book}{tag}", "var(--ink-2)")


def render_moneyline_card(row, season, week):
    ev      = row["ml_ev"]
    team    = row["ml_team"]
    book_ml = row["ml_book_odds"]
    mdl_ml  = row["ml_model_odds"]
    is_dog  = book_ml > 0
    label   = f"+{int(book_ml)}" if is_dog else str(int(book_ml))
    model_label = (f"+{int(mdl_ml)}" if (not pd.isna(mdl_ml) and mdl_ml > 0)
                   else str(int(mdl_ml)) if not pd.isna(mdl_ml) else "—")
    matchup = f"{row['away_team']} @ {row['home_team']}"
    kickoff = format_kickoff(row.get("start_date"))
    sub     = matchup + (f" &nbsp;·&nbsp; {kickoff}" if kickoff else "")
    units   = 0   # paper record only — ML EV strategy unvalidated ('25 −4%)
    ev_str  = f"{ev:+.1%}"
    ev_color = "var(--green)" if ev >= 0.05 else "var(--ink-2)"
    accent   = "var(--gold)" if ev >= 0.07 else "var(--blue)"

    metric = (f'<div style="display:flex;align-items:center;gap:8px">'
              f'<span class="num" style="color:{ev_color};font-size:0.85em;font-weight:800">EV {ev_str}</span>'
              f'<span style="color:var(--gold);font-size:0.88em">{ev_stars(ev)}</span></div>')

    _ev_pct = min(float(ev) / 0.07 * 100, 100) if ev else 0
    st.html(_pick_card_html(
        accent=accent, kind_badge="MONEYLINE", kind_color="var(--blue)",
        tier=_tier_badge("moneyline", row),
        headline=f"{team} ML", big_num=label, sub=sub,
        meters=_winprob_bar_html(row),
        chips=_line_shop_ml_html(row, team == row["home_team"])
              + _driver_chips_html(row, "moneyline"),
        footer=_stat_footer_html([
            ("Book", label, "var(--ink)"),
            ("Model fair", model_label, "var(--ink)"),
            ("EV", ev_str, ev_color),
            ("Kelly", "0u · paper", "var(--ink-3)"),
        ]),
        metric_html=metric,
        edge_pct=_ev_pct,
    ))
    track_button(f"{team} ML {label}", matchup, "Moneyline", f"{team} ML {label}",
                 label, units, season, week, f"EV {ev:+.1%}")

def render_totals_card(row, season, week):
    # High-total paper picks are always UNDER regardless of model direction —
    # the edge is market over-inflation, not a model read.
    force_under = bool(row.get("_force_under"))
    is_under = force_under or row["totals_edge"] < 0
    side_str = "UNDER" if is_under else "OVER"
    edge_abs = abs(row["totals_edge"])
    _play    = _is_play("total", row)
    # Flat 1u on CORE: hit rate does NOT rise with edge size, so edge-scaled
    # Kelly is backwards. At the measured 54.9% quarter-Kelly is ~1.3% of
    # bankroll, so 1u (=1%) is the conservative size. Non-CORE = research only.
    units    = 1 if _play else 0
    matchup  = f"{row['away_team']} @ {row['home_team']}"
    ou_str   = f"{row['over_under']:.1f}" if pd.notna(row["over_under"]) else "TBD"
    edge_str = f"{row['totals_edge']:+.1f}"
    kickoff  = format_kickoff(row.get("start_date"))
    sub = matchup
    if row.get("neutral_site"):
        sub += " &nbsp;·&nbsp; Neutral site"
    if kickoff:
        sub += f" &nbsp;·&nbsp; {kickoff}"

    accent     = "var(--cyan)" if is_under else "var(--orange)"
    edge_color = "var(--green)" if edge_abs >= 4.5 else "var(--ink-2)"

    if force_under and not _play:
        # Paper high-total fade: the model's edge is irrelevant to the play,
        # so don't imply model support with an edge number or stars.
        metric = ('<div style="color:var(--ink-3);font-size:0.78em;'
                  'font-weight:700;letter-spacing:0.06em">MARKET FADE</div>')
        _tot_pct = 0.0
    else:
        metric = (f'<div style="display:flex;align-items:center;gap:8px">'
                  f'<span class="num" style="color:{edge_color};font-size:0.85em;font-weight:800">'
                  f'Edge {edge_str}</span>'
                  f'<span style="color:var(--gold);font-size:0.88em">{confidence_stars(edge_abs)}</span></div>')
        _tot_pct = min(edge_abs / 8.0 * 100, 100)
    st.html(_pick_card_html(
        accent=accent, kind_badge=f"TOTAL · {side_str}", kind_color=accent,
        edge_pct=_tot_pct,
        tier=_tier_badge("total", row),
        headline=f"{side_str} {ou_str}", big_num=ou_str, sub=sub,
        meters=_edge_meter_html(
            row["over_under"], row["pred_total"], accent,
            market_label="Market total", model_label="Model total", span=8.0),
        chips=_line_shop_total_html(row, is_under) + _driver_chips_html(row, "total"),
        footer=_stat_footer_html([
            ("Line", ou_str, "var(--ink)"),
            ("Model", f"{row['pred_total']:.1f}" if pd.notna(row["pred_total"]) else "—", "var(--ink)"),
            ("Edge", "n/a" if (force_under and not _play) else f"{edge_str} pts", edge_color),
            ("Kelly", unit_dollar_label(units) if _play
             else ("0u · paper" if force_under else "0u · pass"),
             "var(--ink)" if _play else "var(--ink-3)"),
        ]),
        metric_html=metric,
    ))
    track_button(f"{side_str} {ou_str}", matchup, "Total",
                 f"{side_str} {ou_str}", ou_str, units, season, week,
                 f"{row['totals_edge']:+.1f} pts")


def render_spread_card(row, season, week):
    """Spread card (reference only — near breakeven historically)."""
    is_home  = row["spread_edge"] > 0
    bet_on   = row["home_team"] if is_home else row["away_team"]
    edge     = row["spread_edge"]
    spread   = row["spread"]
    pred_sp  = row["pred_spread"]
    matchup  = f"{row['away_team']} @ {row['home_team']}"
    edge_str = f"{edge:+.1f}"
    kickoff  = format_kickoff(row.get("start_date"))
    sub      = matchup + (f" &nbsp;·&nbsp; {kickoff}" if kickoff else "")

    # Vegas / model lines from bet_on's perspective
    vl_bet  = (f"{spread:+.1f}" if is_home else f"{-spread:+.1f}") if pd.notna(spread) else "N/A"
    mdl_str = (f"{-pred_sp:+.1f}" if is_home else f"{pred_sp:+.1f}") if pd.notna(pred_sp) else "—"

    edge_color = "var(--green)" if abs(edge) >= 5.5 else "var(--ink-2)"
    accent     = "var(--violet)"
    _play      = _is_play("spread", row)
    _units     = 1 if _play else 0

    metric = (f'<div style="display:flex;align-items:center;gap:8px">'
              f'<span class="num" style="color:{edge_color};font-size:0.85em;font-weight:800">'
              f'Edge {edge_str}</span>'
              f'<span style="color:var(--gold);font-size:0.88em">{confidence_stars(abs(edge))}</span></div>')

    # Edge meter on the home-margin scale (positive = home better)
    vegas_margin = -float(spread) if pd.notna(spread) else np.nan
    meters = _edge_meter_html(vegas_margin, pred_sp, accent,
                              market_label="Vegas margin", model_label="Model margin",
                              span=7.0) + _winprob_bar_html(row)

    _sp_pct = min(abs(edge) / 10.0 * 100, 100)
    st.html(_pick_card_html(
        accent=accent, kind_badge="SPREAD", kind_color=accent,
        edge_pct=_sp_pct,
        tier=_tier_badge("spread", row),
        headline=f"{bet_on} {vl_bet}", big_num=vl_bet, sub=sub,
        meters=meters,
        chips=_driver_chips_html(row, "spread"),
        footer=_stat_footer_html([
            ("Vegas", vl_bet, "var(--ink)"),
            ("Model", mdl_str, "var(--ink)"),
            ("Edge", f"{edge_str} pts", edge_color),
            ("Kelly", unit_dollar_label(_units) if _play else "0u · pass", "var(--ink)" if _play else "var(--ink-3)"),
        ]),
        metric_html=metric,
    ))
    track_button(f"{bet_on} {vl_bet}", matchup, "Spread",
                 f"{bet_on} {vl_bet}", vl_bet, _units, season, week, edge_str)


# ─── MY BETS TAB ──────────────────────────────────────────────────────────────

def render_bets_tab():
    bets = load_bets()

    if not github_backup_configured():
        st.caption("⚠️ Bet history is only stored on this app's temporary filesystem and is "
                   "wiped on every weekly redeploy. Add a `GITHUB_TOKEN` secret (fine-grained "
                   "token, Contents read/write on this repo) in Streamlit Cloud → Settings → "
                   "Secrets to back it up to GitHub on every change.")

    if not bets:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.info("No bets tracked yet. Load a week on the Picks tab and hit **+ Track** on any pick.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    settled   = [b for b in bets if b["status"] != "Pending"]
    pending   = [b for b in bets if b["status"] == "Pending"]
    wins      = [b for b in settled if b["status"] == "Won"]
    losses    = [b for b in settled if b["status"] == "Lost"]
    total_pnl = sum(bet_pnl(b) for b in settled)
    win_rate  = len(wins) / len(settled) if settled else 0

    clv_vals   = [v for b in bets if (v := compute_clv(b)) is not None]
    avg_clv    = sum(clv_vals) / len(clv_vals) if clv_vals else None
    clv_beat   = sum(1 for v in clv_vals if v > 0)
    clv_label  = (f"{avg_clv:+.1f}" if avg_clv is not None else "—")
    clv_delta  = (f"{clv_beat}/{len(clv_vals)} beat close" if clv_vals else None)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Bets",  len(bets))
    c2.metric("Pending",     len(pending))
    c3.metric("Record",      f"{len(wins)}-{len(losses)}" if settled else "—")
    c4.metric("Win Rate",    f"{win_rate:.0%}" if settled else "—")
    c5.metric("Units P&L",   f"{total_pnl:+.2f}u",
              delta_color="normal" if total_pnl >= 0 else "inverse")
    c6.metric("Avg CLV",     clv_label, delta=clv_delta,
              delta_color="normal" if (avg_clv or 0) >= 0 else "inverse",
              help="Closing Line Value — how much better your line was vs. the closing line. "
                   "Positive = beat the close. Enter closing lines on each bet below.")

    # ── Filters + CLV auto-fill ──────────────────────────────────────────
    section_header("Bet History")
    col_f1, col_f2, col_clv, _ = st.columns([1, 1, 1.2, 0.8])
    status_filter = col_f1.selectbox("Status", ["All", "Pending", "Won", "Lost", "Push"])
    bettor_filter = col_f2.selectbox("Bettor",  ["All"] + BETTORS)
    with col_clv:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("⚡ Auto-fill closing lines",
                     help="Fetch closing lines from CFBD for completed games and "
                          "fill CLV on every bet missing one"):
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "capture_closing_lines", ROOT_DIR / "scripts" / "capture_closing_lines.py")
            _ccl = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_ccl)
            with st.spinner("Fetching closing lines from CFBD..."):
                updated, n_filled, notes = _ccl.fill_closing_lines(bets, _cfb_api_key())
            if n_filled:
                save_bets(updated)
                st.toast(f"Filled closing lines on {n_filled} bet(s)", icon="⚡")
                st.rerun()
            else:
                st.toast("No closing lines to fill yet (games not completed, "
                         "or all bets already filled)", icon="ℹ️")

    filtered = bets
    if status_filter != "All":
        filtered = [b for b in filtered if b["status"] == status_filter]
    if bettor_filter != "All":
        filtered = [b for b in filtered if b.get("bettor", "") == bettor_filter]

    if not filtered:
        st.info("No bets match the selected filters.")
        return

    # ── Bet rows ─────────────────────────────────────────────────────────
    status_accent = {"Pending": "var(--ink-4)", "Won": "var(--green)",
                     "Lost": "var(--red)", "Push": "var(--orange)"}
    status_label  = {"Pending": "PENDING", "Won": "WON", "Lost": "LOST", "Push": "PUSH"}

    for bet in reversed(filtered):
        accent  = status_accent.get(bet["status"], "var(--ink-4)")
        slabel  = status_label.get(bet["status"], bet["status"])
        pending = bet["status"] == "Pending"
        pnl     = bet_pnl(bet)
        pnl_str = f"{pnl:+.2f}u" if not pending else "—"
        pnl_col = ("var(--green)" if pnl > 0 else "var(--red)" if pnl < 0
                   else "var(--ink-3)")
        bettor  = bet.get("bettor", "—")
        edge_tag = f" · {bet['edge']}" if bet.get("edge") else ""

        clv      = compute_clv(bet)
        clv_col  = ("var(--green)" if (clv or 0) > 0 else "var(--red)" if (clv or 0) < 0
                    else "var(--ink-3)")
        clv_unit = "ppts" if bet.get("bet_type") == "Moneyline" else "pts"
        clv_html = (
            f'<span class="chip num" style="color:{clv_col};border-color:{clv_col}">'
            f'CLV {clv:+.1f}{clv_unit}</span>'
            if clv is not None else
            ('<span class="chip" style="color:var(--ink-4)">no closing line yet</span>'
             if not pending else "")
        )

        st.html(f"""
        <div class="pick-card" style="padding:13px 18px 12px 18px">
            <div class="accent" style="background:{accent}"></div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap">
                    <span style="border:1px solid {accent};color:{accent};font-size:0.6em;
                                 font-weight:800;letter-spacing:0.1em;padding:2px 8px;
                                 border-radius:5px">{slabel}</span>
                    <span style="color:var(--ink);font-weight:800;font-size:1em;
                                 letter-spacing:-0.01em">{bet['pick']}</span>
                    <span style="background:rgba(255,255,255,0.05);color:var(--ink-3);
                                 font-size:0.64em;font-weight:700;letter-spacing:0.08em;
                                 padding:2px 7px;border-radius:4px">{bet['bet_type'].upper()}</span>
                    <span class="num" style="color:var(--ink-3);font-size:0.8em">{bet['units']}u</span>
                    {clv_html}
                </div>
                <span class="num" style="color:{pnl_col};font-weight:800;font-size:1em">{pnl_str}</span>
            </div>
            <div style="color:var(--ink-3);font-size:0.8em;margin-top:6px">{bet['game']}</div>
            <div style="color:var(--ink-4);font-size:0.72em;margin-top:2px">
                {bettor} · Wk {bet['week']} · {bet['date']} ·
                Line <span class="num">{bet['line']}</span>{edge_tag}
            </div>
        </div>
        """)

        b_cols = st.columns([1, 1, 1, 1, 2, 2])
        if bet["status"] != "Won":
            if b_cols[0].button("Won",  key=f"won_{bet['id']}"):
                update_bet_status(bet["id"], "Won");  st.rerun()
        if bet["status"] != "Lost":
            if b_cols[1].button("Lost", key=f"lost_{bet['id']}"):
                update_bet_status(bet["id"], "Lost"); st.rerun()
        if bet["status"] != "Push":
            if b_cols[2].button("Push", key=f"push_{bet['id']}"):
                update_bet_status(bet["id"], "Push"); st.rerun()
        if b_cols[3].button("Delete", key=f"del_{bet['id']}"):
            delete_bet(bet["id"]); st.rerun()

        # Closing line input — saves automatically when value changes
        current_cl = bet.get("closing_line", "")
        new_cl = b_cols[4].text_input(
            "Closing line",
            value=current_cl,
            key=f"cl_{bet['id']}",
            placeholder="Closing line",
            label_visibility="collapsed",
        )
        if new_cl != current_cl:
            update_bet_closing_line(bet["id"], new_cl)
            st.rerun()

        current_bettor = bet.get("bettor", BETTORS[0])
        idx = BETTORS.index(current_bettor) if current_bettor in BETTORS else 0
        new_bettor = b_cols[5].selectbox("", BETTORS, index=idx,
                                          key=f"bettor_{bet['id']}",
                                          label_visibility="collapsed")
        if new_bettor != current_bettor:
            update_bet_bettor(bet["id"], new_bettor); st.rerun()

    # ── Export ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    df_export = pd.DataFrame(bets)
    if not df_export.empty:
        df_export["pnl"] = df_export.apply(bet_pnl, axis=1)
        st.download_button(
            "Export to CSV",
            data=df_export.to_csv(index=False),
            file_name="cfb_bets.csv",
            mime="text/csv",
        )


# ─── ALL-GAMES CARD ───────────────────────────────────────────────────────────

def render_all_game_card(row, season, week):
    """One expandable card per game with Track buttons for every bet type."""
    matchup   = f"{row['home_team']} vs {row['away_team']}"
    win_p     = row.get("pred_win_p")
    spread    = row.get("spread")
    ou        = row.get("over_under")
    home_ml   = row.get("home_moneyline")
    away_ml   = row.get("away_moneyline")

    home_unrated = bool(row.get("home_unrated", False))
    away_unrated = bool(row.get("away_unrated", False))
    unrated_team = (row["away_team"] if away_unrated else
                    row["home_team"] if home_unrated else None)
    unrated_badge = "  ·  FCS" if unrated_team else ""

    pred_sp   = row.get("pred_spread")
    pred_tot  = row.get("pred_total")
    mdl_hml   = row.get("model_home_ml")
    mdl_aml   = row.get("model_away_ml")

    win_str   = f"  ·  Home {win_p:.0%}" if pd.notna(win_p) else ""
    spread_h  = f"{spread:+.1f}"  if pd.notna(spread) else None
    spread_a  = f"{-spread:+.1f}" if pd.notna(spread) else None
    ou_str    = f"{ou:.1f}"       if pd.notna(ou)     else None
    hml_str   = (f"{int(home_ml):+d}" if home_ml > 0 else str(int(home_ml))) if pd.notna(home_ml) else None
    aml_str   = (f"{int(away_ml):+d}" if away_ml > 0 else str(int(away_ml))) if pd.notna(away_ml) else None

    # pred_sp = home margin (positive = home wins); betting spread is opposite sign
    mdl_sp_h  = f"{-pred_sp:+.1f}" if pd.notna(pred_sp) else None
    mdl_sp_a  = f"{pred_sp:+.1f}"  if pd.notna(pred_sp) else None
    mdl_tot   = f"{pred_tot:.1f}"   if pd.notna(pred_tot) else None
    mdl_hml_s = (f"{int(mdl_hml):+d}" if mdl_hml > 0 else str(int(mdl_hml))) if pd.notna(mdl_hml) else None
    mdl_aml_s = (f"{int(mdl_aml):+d}" if mdl_aml > 0 else str(int(mdl_aml))) if pd.notna(mdl_aml) else None

    with st.expander(f"{matchup}{win_str}{unrated_badge}"):
        if unrated_team:
            st.warning(
                f"**{unrated_team}** has no SP+/FPI/SRS ratings (likely FCS or untracked). "
                f"Model projection is unreliable — use the Vegas line only.",
                icon=None,
            )
        if pd.notna(win_p) and not unrated_team:
            st.html("<div style='margin:-4px 0 8px 0'>" + _winprob_bar_html(row) + "</div>")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Spread**")
            if spread_h:
                st.caption(f"Vegas: {row['home_team']} {spread_h} / {row['away_team']} {spread_a}")
                if mdl_sp_h and not unrated_team:
                    st.caption(f"Model: {row['home_team']} {mdl_sp_h}")
                elif mdl_sp_h and unrated_team:
                    st.caption(f"Model: {row['home_team']} {mdl_sp_h} (unreliable)")
                track_button(f"{row['home_team']} {spread_h}", matchup, "Spread",
                             f"{row['home_team']} {spread_h}", spread_h, 1, season, week, key_prefix="ag_")
                track_button(f"{row['away_team']} {spread_a}", matchup, "Spread",
                             f"{row['away_team']} {spread_a}", spread_a, 1, season, week, key_prefix="ag_")
            elif mdl_sp_h and not unrated_team:
                st.caption(f"Model: {row['home_team']} {mdl_sp_h} (no Vegas line yet)")
                track_button(f"{row['home_team']} {mdl_sp_h} (model)", matchup, "Spread",
                             f"{row['home_team']} {mdl_sp_h}", mdl_sp_h, 1, season, week, key_prefix="ag_")
                track_button(f"{row['away_team']} {mdl_sp_a} (model)", matchup, "Spread",
                             f"{row['away_team']} {mdl_sp_a}", mdl_sp_a, 1, season, week, key_prefix="ag_")
            else:
                st.caption("No line yet")

        with c2:
            st.markdown("**Total**")
            if ou_str:
                st.caption(f"O/U: {ou_str}" + (f"  ·  Model: {mdl_tot}" if mdl_tot else ""))
                track_button(f"OVER {ou_str}", matchup, "Total",
                             f"OVER {ou_str}", ou_str, 1, season, week, key_prefix="ag_")
                track_button(f"UNDER {ou_str}", matchup, "Total",
                             f"UNDER {ou_str}", ou_str, 1, season, week, key_prefix="ag_")
            elif mdl_tot:
                st.caption(f"Model: {mdl_tot} pts (no Vegas total yet)")
                track_button(f"OVER {mdl_tot} (model)", matchup, "Total",
                             f"OVER {mdl_tot}", mdl_tot, 1, season, week, key_prefix="ag_")
                track_button(f"UNDER {mdl_tot} (model)", matchup, "Total",
                             f"UNDER {mdl_tot}", mdl_tot, 1, season, week, key_prefix="ag_")
            else:
                st.caption("No total yet")

        with c3:
            st.markdown("**Moneyline**")
            if hml_str:
                st.caption(f"{row['home_team']} {hml_str} / {row['away_team']} {aml_str or '—'}"
                           + (f"  ·  Model: {row['home_team']} {mdl_hml_s}" if mdl_hml_s else ""))
                track_button(f"{row['home_team']} {hml_str}", matchup, "Moneyline",
                             f"{row['home_team']} {hml_str}", hml_str, 1, season, week, key_prefix="ag_")
            if aml_str:
                track_button(f"{row['away_team']} {aml_str}", matchup, "Moneyline",
                             f"{row['away_team']} {aml_str}", aml_str, 1, season, week, key_prefix="ag_")
            if not hml_str and not aml_str:
                if mdl_hml_s:
                    st.caption(f"Model: {row['home_team']} {mdl_hml_s} / {row['away_team']} {mdl_aml_s or '—'}")
                else:
                    st.caption("No ML yet")


# ─── HOW IT WORKS TAB ────────────────────────────────────────────────────────

def render_guide_tab():
    """Plain-English guide: what the model does and how to use it."""

    # ── Quick-reference card row ──────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.html("""
    <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:12px;
                padding:20px 24px;margin-bottom:20px">
        <div style="color:#eab308;font-size:0.65em;font-weight:800;letter-spacing:.1em;
                    text-transform:uppercase;margin-bottom:12px">Quick Reference</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
            <div>
                <div style="color:#06b6d4;font-size:1em;font-weight:800">★★★</div>
                <div style="color:#e5e7eb;font-size:0.82em;font-weight:600;margin-top:2px">Strong edge</div>
                <div style="color:#6b7280;font-size:0.75em;margin-top:1px">Model disagrees with Vegas by 6+ pts on totals or 8+ pts on spreads</div>
            </div>
            <div>
                <div style="color:#22c55e;font-size:1em;font-weight:800">UNDER / OVER</div>
                <div style="color:#e5e7eb;font-size:0.82em;font-weight:600;margin-top:2px">Totals bet</div>
                <div style="color:#6b7280;font-size:0.75em;margin-top:1px">Primary edge. Unders win ~59% historically — the model's strongest signal</div>
            </div>
            <div>
                <div style="color:#a78bfa;font-size:1em;font-weight:800">1u / 2u / 3u</div>
                <div style="color:#e5e7eb;font-size:0.82em;font-weight:600;margin-top:2px">Kelly bet size</div>
                <div style="color:#6b7280;font-size:0.75em;margin-top:1px">How much to bet relative to your bankroll. 1 unit = 1% of total bankroll</div>
            </div>
            <div>
                <div style="color:#f97316;font-size:1em;font-weight:800">CLV</div>
                <div style="color:#e5e7eb;font-size:0.82em;font-weight:600;margin-top:2px">Closing line value</div>
                <div style="color:#6b7280;font-size:0.75em;margin-top:1px">Did you beat the closing line? More important than W/L for long-run edge</div>
            </div>
        </div>
    </div>
    """)

    # ── Section 1: What is this? ──────────────────────────────────────────────
    section_header("What Is This Model?")
    st.markdown(
        "This is a college football prediction model that analyzes hundreds of data points "
        "per game — team efficiency, recruiting quality, recent form, line movement, weather, "
        "and more — and compares its own predicted score to the Vegas line. "
        "When the model's prediction disagrees with Vegas by enough, it flags a bet."
        "\n\n"
        "The model **does not** claim to beat Vegas consistently on spreads. It does have a "
        "demonstrated edge on **totals (Unders in particular)** and occasionally on "
        "**moneylines** when there's strong expected value. Spreads are shown for reference only."
    )

    # ── Section 2: The three bet types ───────────────────────────────────────
    section_header("The Three Bet Types")
    st.html("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:4px">
        <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:16px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <span style="background:#06b6d4;color:#0f1117;font-size:0.63em;font-weight:800;
                             padding:2px 8px;border-radius:4px">TOTALS</span>
                <span style="color:#22c55e;font-size:0.7em;font-weight:700">PRIMARY EDGE</span>
            </div>
            <div style="color:#e5e7eb;font-size:0.85em;line-height:1.5">
                Bet on whether the combined score goes Over or Under the Vegas total.
                The model targets <strong>Unders</strong> — it finds games where Vegas
                has the total set too high.
            </div>
            <div style="color:#6b7280;font-size:0.75em;margin-top:8px">
                ✅ Bet these confidently (★★ or higher)
            </div>
        </div>
        <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:16px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <span style="background:#a78bfa;color:#0f1117;font-size:0.63em;font-weight:800;
                             padding:2px 8px;border-radius:4px">MONEYLINE</span>
                <span style="color:#f0b429;font-size:0.7em;font-weight:700">SECONDARY EDGE</span>
            </div>
            <div style="color:#e5e7eb;font-size:0.85em;line-height:1.5">
                Bet on which team wins outright. The model flags these when it finds
                <strong>positive expected value</strong> — the book's implied odds are worse
                than the model's predicted win probability.
            </div>
            <div style="color:#6b7280;font-size:0.75em;margin-top:8px">
                ✅ Bet at 4%+ EV. Be selective — variance is high.
            </div>
        </div>
        <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:16px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <span style="background:#4b5563;color:#e5e7eb;font-size:0.63em;font-weight:800;
                             padding:2px 8px;border-radius:4px">SPREADS</span>
                <span style="color:#6b7280;font-size:0.7em;font-weight:700">REFERENCE ONLY</span>
            </div>
            <div style="color:#e5e7eb;font-size:0.85em;line-height:1.5">
                Bet on the margin of victory. The model's spread predictions are shown
                for context, but <strong>do not bet these</strong> — the model's spread
                accuracy is near breakeven after juice.
            </div>
            <div style="color:#6b7280;font-size:0.75em;margin-top:8px">
                ⚠️ Use to understand game context only.
            </div>
        </div>
    </div>
    """)

    # ── Section 3: Reading a pick card ───────────────────────────────────────
    section_header("How to Read a Pick Card")
    st.html("""
    <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:18px 22px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <span style="background:#06b6d4;color:#0f1117;font-size:0.63em;font-weight:800;
                         padding:3px 8px;border-radius:4px">UNDER</span>
            <span style="background:#252d3d;color:#6b7280;font-size:0.63em;font-weight:700;
                         padding:3px 7px;border-radius:4px">TOTAL</span>
            <span style="background:#1e2537;color:#ef4444;font-size:0.63em;font-weight:700;
                         padding:2px 7px;border-radius:4px">💨 23 mph</span>
            <span style="flex:1"></span>
            <span style="color:#22c55e;font-size:0.82em;font-weight:700">Edge −6.2</span>
            <span style="color:#eab308;font-size:0.88em">★★★</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:#ffffff;font-size:1.1em;font-weight:700">UNDER 48.5</span>
        </div>
        <div style="color:#4b5563;font-size:0.8em;margin-top:4px">Alabama vs Auburn  ·  Iron Bowl</div>
        <div style="color:#ef4444;font-size:0.75em;margin-top:3px;font-weight:600">
            💨 23 mph — strong under lean
        </div>
        <div style="display:flex;margin-top:12px;border-top:1px solid #252d3d;padding-top:10px">
            <div style="flex:1;text-align:center">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Line</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">48.5</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Model</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">42.3</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Edge</div>
                <div style="color:#22c55e;font-size:0.88em;font-weight:700">−6.2 pts</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Kelly</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">2u</div>
            </div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
        <div style="background:#111827;border-radius:8px;padding:12px 14px">
            <span style="color:#eab308;font-weight:700">★ Stars</span>
            <span style="color:#6b7280;font-size:0.85em"> — Confidence level.</span>
            <div style="color:#9ca3af;font-size:0.8em;margin-top:4px;line-height:1.5">
                ★ = edge 4–5 pts &nbsp;·&nbsp; ★★ = 5–6 pts &nbsp;·&nbsp; ★★★ = 6+ pts<br>
                More stars = model is more confident. Still do your own homework.
            </div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:12px 14px">
            <span style="color:#a78bfa;font-weight:700">Edge</span>
            <span style="color:#6b7280;font-size:0.85em"> — Points of disagreement with Vegas.</span>
            <div style="color:#9ca3af;font-size:0.8em;margin-top:4px;line-height:1.5">
                UNDER 48.5 with Edge −6.2 means the model predicts 42.3.
                Negative edge = bet the Under. Positive = bet the Over.
            </div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:12px 14px">
            <span style="color:#a78bfa;font-weight:700">Kelly (1u / 2u / 3u)</span>
            <span style="color:#6b7280;font-size:0.85em"> — Recommended bet size.</span>
            <div style="color:#9ca3af;font-size:0.8em;margin-top:4px;line-height:1.5">
                1 unit = 1% of your total bankroll. So if you have $500 to bet with,
                1u = $5, 2u = $10, 3u = $15. Never bet more than 3u on any single game.
            </div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:12px 14px">
            <span style="color:#ef4444;font-weight:700">💨 Wind badge</span>
            <span style="color:#6b7280;font-size:0.85em"> — Weather context for totals.</span>
            <div style="color:#9ca3af;font-size:0.8em;margin-top:4px;line-height:1.5">
                High wind reduces scoring. A model Under that also has 20+ mph wind
                is doubly confirmed. Dome games show 🏟️ DOME — weather irrelevant.
            </div>
        </div>
    </div>
    """)

    # ── Section 4: Step-by-step workflow ─────────────────────────────────────
    section_header("Weekly Workflow")
    st.html("""
    <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;
                padding:18px 22px">
        <div style="display:flex;flex-direction:column;gap:14px">
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#eab308;color:#0f1117;font-size:0.72em;font-weight:800;
                             padding:3px 9px;border-radius:20px;min-width:24px;text-align:center">1</span>
                <div>
                    <div style="color:#e5e7eb;font-weight:600;font-size:0.9em">Load picks on Tuesday or Wednesday</div>
                    <div style="color:#6b7280;font-size:0.8em;margin-top:2px">
                        Select the current season and week in the sidebar, hit Load Picks.
                        Lines are still moving — earlier in the week = more time to shop.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#eab308;color:#0f1117;font-size:0.72em;font-weight:800;
                             padding:3px 9px;border-radius:20px;min-width:24px;text-align:center">2</span>
                <div>
                    <div style="color:#e5e7eb;font-weight:600;font-size:0.9em">Focus on ★★ and ★★★ totals picks</div>
                    <div style="color:#6b7280;font-size:0.8em;margin-top:2px">
                        These are the highest-confidence bets. Skip ★ picks if the card
                        shows low data coverage or the game has injury news the model can't see.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#eab308;color:#0f1117;font-size:0.72em;font-weight:800;
                             padding:3px 9px;border-radius:20px;min-width:24px;text-align:center">3</span>
                <div>
                    <div style="color:#e5e7eb;font-weight:600;font-size:0.9em">Check the coverage warning</div>
                    <div style="color:#6b7280;font-size:0.8em;margin-top:2px">
                        If the yellow "Data coverage" banner appears, some data sources are
                        missing for that week. Treat those picks with more skepticism.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#eab308;color:#0f1117;font-size:0.72em;font-weight:800;
                             padding:3px 9px;border-radius:20px;min-width:24px;text-align:center">4</span>
                <div>
                    <div style="color:#e5e7eb;font-weight:600;font-size:0.9em">Track every bet in My Bets</div>
                    <div style="color:#6b7280;font-size:0.8em;margin-top:2px">
                        Hit "Track This Bet" under any pick. Record the actual line you got
                        when you placed the bet — that's what CLV is calculated from.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#eab308;color:#0f1117;font-size:0.72em;font-weight:800;
                             padding:3px 9px;border-radius:20px;min-width:24px;text-align:center">5</span>
                <div>
                    <div style="color:#e5e7eb;font-weight:600;font-size:0.9em">Review CLV Tracker weekly, not W/L</div>
                    <div style="color:#6b7280;font-size:0.8em;margin-top:2px">
                        Week-to-week wins/losses are noisy. Positive average CLV means you're
                        getting value — stick with the process even through losing streaks.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    # ── Section 5: Bankroll basics ────────────────────────────────────────────
    section_header("Bankroll Management")
    col1, col2 = st.columns(2)
    with col1:
        st.html("""
        <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:16px">
            <div style="color:#22c55e;font-weight:700;margin-bottom:8px">✅ Do</div>
            <div style="color:#9ca3af;font-size:0.83em;line-height:1.8">
                Decide on a fixed bankroll (e.g. $500) before the season<br>
                Keep 1u = 1% of that bankroll throughout<br>
                Bet the Kelly-suggested size (1u / 2u / 3u) only<br>
                Track every bet — wins AND losses<br>
                Shop lines across multiple books<br>
                Take a week off if on a 5+ game skid
            </div>
        </div>
        """)
    with col2:
        st.html("""
        <div style="background:#1a1f2e;border:1px solid #252d3d;border-radius:10px;padding:16px">
            <div style="color:#ef4444;font-weight:700;margin-bottom:8px">❌ Don't</div>
            <div style="color:#9ca3af;font-size:0.83em;line-height:1.8">
                Chase losses by doubling up<br>
                Bet more than 3u on any single game<br>
                Bet spreads — the model has no edge there<br>
                Ignore injury news (the model can't see it)<br>
                Bet games with ⚠️ low coverage and only ★<br>
                Bet money you can't afford to lose
            </div>
        </div>
        """)

    # ── Deep-dive: Under the Hood ─────────────────────────────────────────────
    with st.expander("🔬 Under the Hood — How the Model Actually Works"):
        st.markdown("""
**Data sources (updated before each season)**

The model pulls from the CollegeFootballData API and combines:
- **SP+ ratings** — Bill Connelly's opponent-adjusted efficiency metric (offense + defense)
- **EPA (Expected Points Added)** — rolling 3-game and 5-game windows, adjusted for opponent strength
- **WEPA** — a third-party opponent-adjusted efficiency metric that captures explosiveness
- **Elo ratings** — win-probability-based power ratings updated after every game
- **Recruiting rankings** — 247Sports composite, 4-year rolling average
- **Havoc rates** — how often a defense disrupts plays (sacks, TFLs, PBUs)
- **Transfer portal** — net talent in/out each offseason
- **Line movement** — how much the spread and total have moved since opening
- **Weather** — wind speed from Open-Meteo for outdoor games

**Three models**

1. **Spread model** (Ridge 60% + LightGBM 40%) — predicts the home team's margin of victory
2. **Totals model** (same ensemble) — predicts the combined score
3. **Win probability model** (Logistic 40% + LightGBM 60%) — predicts home win probability, cross-calibrated with the spread model

**How edge is calculated**

- Totals edge = Model predicted total − Vegas total line
- Spread edge = Model predicted spread − Vegas spread (home perspective)
- Moneyline EV = (Model win prob × payout) − (Model loss prob × stake)

**Thresholds for flagging picks**

The model only surfaces picks when the edge exceeds a minimum threshold, validated on 2024 holdout data:
- Totals: **≥ 3 points** of edge (absolute value)
- Spreads: **≥ 4 points** of edge
- Moneyline: **≥ 4% expected value**

**Walk-forward validation**

The model was trained on 2017–2023 games, tuned on 2024, and the reported win rates reflect 2025 test-set performance only — games the model never saw during training.

**Why Unders?**

Vegas totals tend to be set slightly high because casual bettors like Overs (more exciting). The market corrects by Saturday but not fully. The model exploits this systematic bias, which is why Unders win at ~59% historically vs. the 52.4% needed to break even at -110 juice.
        """)


# ─── BACKTESTER TAB ──────────────────────────────────────────────────────────

def render_backtester_tab():
    """
    Backtester — simulate historical betting at configurable edge thresholds.

    Panels:
      1. Summary metrics   — total bets, win %, flat profit, ROI, max drawdown
      2. Profit curve      — cumulative units over time, flat vs Kelly
      3. Threshold sweep   — win rate + ROI at every edge cutoff (find the sweet spot)
      4. Bet type breakdown — spreads vs overs vs unders, side by side
    """
    import plotly.graph_objects as go

    wf_path      = ROOT_DIR / "outputs" / "predictions" / "walk_forward_results.csv"
    model_path   = ROOT_DIR / "outputs" / "predictions" / "model_results.csv"

    # Prefer walk-forward (OOS for all seasons) over single-season test set
    if wf_path.exists():
        df = pd.read_csv(wf_path)
        data_label = (f"Walk-forward OOS  ·  {int(df['season'].min())}–{int(df['season'].max())}"
                      f"  ·  {len(df):,} games")
        data_source = "walk_forward"
    elif model_path.exists():
        df = pd.read_csv(model_path)
        data_label = f"2025 test set only  ·  {len(df):,} games  (run walk_forward.py for full history)"
        data_source = "test_only"
    else:
        st.warning("⚠️  No prediction data found. Run `src/model.py` or `scripts/walk_forward.py` first.")
        return

    st.info(f"📊 **Data source:** {data_label}", icon=None)
    if data_source == "test_only":
        st.caption(
            "⚡ For statistically meaningful results, run the walk-forward script first:  \n"
            "`/opt/homebrew/bin/python3 scripts/walk_forward.py`  (~5 min)")

    # Compute totals edge if not already saved
    if "totals_edge" not in df.columns:
        df["totals_edge"] = (pd.to_numeric(df["pred_total"],  errors="coerce") -
                             pd.to_numeric(df["over_under"],  errors="coerce"))

    # Keep only completed games with outcomes
    df = df.dropna(subset=["covered_spread", "went_over", "spread_edge"]).copy()
    df["spread_edge"]    = pd.to_numeric(df["spread_edge"],    errors="coerce")
    df["totals_edge"]    = pd.to_numeric(df["totals_edge"],    errors="coerce")
    df["covered_spread"] = df["covered_spread"].astype(int)
    df["went_over"]      = df["went_over"].astype(int)
    df["season"]         = pd.to_numeric(df["season"], errors="coerce")
    df["week"]           = pd.to_numeric(df["week"],   errors="coerce")
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["_idx"] = range(len(df))   # chronological index for profit curve

    WIN_U  = 1.0    # win 1 unit (betting -110)
    LOSE_U = 1.1   # lose 1.1 units (juice)

    def _kelly(edge_abs: float) -> float:
        """Quarter-Kelly units, capped 0.5–3.0, rounded to nearest 0.5."""
        win_p = min(0.5238 + edge_abs * 0.005, 0.60)
        b = 100 / 110
        k = max((win_p * b - (1 - win_p)) / b, 0.0)
        u = k * 0.25 * 100
        return max(0.5, min(3.0, round(u * 2) / 2))

    def _simulate(data: pd.DataFrame, sp_min: float, tot_min: float) -> pd.DataFrame:
        """Build a row-per-bet DataFrame with flat and Kelly P&L columns."""
        rows = []
        for _, r in data.iterrows():
            sp_e  = r["spread_edge"]
            tot_e = r.get("totals_edge", float("nan"))
            base  = {"season": r["season"], "week": r["week"],
                     "_idx": r["_idx"],
                     "matchup": f"{r['home_team']} vs {r['away_team']}"}

            # Spread bet
            if pd.notna(sp_e) and abs(sp_e) >= sp_min:
                home_bet = (sp_e > 0)
                won      = (r["covered_spread"] == 1) if home_bet else (r["covered_spread"] == 0)
                ku       = _kelly(abs(sp_e))
                rows.append({**base,
                    "type": "Spread",
                    "direction": "Home" if home_bet else "Away",
                    "edge": round(sp_e, 1), "won": int(won),
                    "flat_pnl":  WIN_U if won else -LOSE_U,
                    "kelly_pnl": ku * WIN_U if won else -ku * LOSE_U,
                    "kelly_u": ku,
                })

            # Totals bet
            if pd.notna(tot_e) and abs(tot_e) >= tot_min:
                over_bet = (tot_e > 0)
                won      = (r["went_over"] == 1) if over_bet else (r["went_over"] == 0)
                ku       = _kelly(abs(tot_e))
                rows.append({**base,
                    "type": "Over" if over_bet else "Under",
                    "direction": "Over" if over_bet else "Under",
                    "edge": round(tot_e, 1), "won": int(won),
                    "flat_pnl":  WIN_U if won else -LOSE_U,
                    "kelly_pnl": ku * WIN_U if won else -ku * LOSE_U,
                    "kelly_u": ku,
                })
        return pd.DataFrame(rows)

    # ── Controls ──────────────────────────────────────────────────────────────
    section_header("Simulation Settings", "Pick edge thresholds, then watch the P&L curve")
    c1, c2, c3 = st.columns(3)
    with c1:
        sp_thresh = st.slider(
            "Spread edge minimum (pts)", 1.0, 8.0, 3.0, 0.5,
            help="Flag a spread bet only when model vs Vegas gap ≥ this value")
    with c2:
        tot_thresh = st.slider(
            "Totals edge minimum (pts)", 0.5, 6.0, 2.0, 0.5,
            help="Flag an over/under bet only when model vs O/U gap ≥ this value")
    with c3:
        season_opts = ["All seasons"] + [
            str(int(s)) for s in sorted(df["season"].dropna().unique(), reverse=True)]
        sel_s = st.selectbox("Season filter", season_opts)

    filt = df.copy()
    if sel_s != "All seasons":
        filt = filt[filt["season"] == int(sel_s)]

    bets = _simulate(filt, sp_thresh, tot_thresh)
    if bets.empty:
        st.warning("No qualifying bets at these thresholds — try lowering the edge minimums.")
        return

    bets = bets.sort_values("_idx").reset_index(drop=True)
    bets["bet_num"]   = range(1, len(bets) + 1)
    bets["cum_flat"]  = bets["flat_pnl"].cumsum()
    bets["cum_kelly"] = bets["kelly_pnl"].cumsum()

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_bets    = len(bets)
    win_pct   = bets["won"].mean() * 100
    flat_tot  = bets["flat_pnl"].sum()
    flat_roi  = flat_tot / (n_bets * LOSE_U) * 100
    kelly_tot = bets["kelly_pnl"].sum()
    max_dd    = (bets["cum_flat"] - bets["cum_flat"].cummax()).min()

    be_color = "normal" if win_pct >= 52.38 else "inverse"
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Bets",    f"{n_bets:,}")
    m2.metric("Win Rate",      f"{win_pct:.1f}%",
              f"{win_pct - 52.38:+.1f}% vs break-even", delta_color=be_color)
    m3.metric("Flat Profit",   f"{flat_tot:+.1f}u")
    m4.metric("ROI (flat)",    f"{flat_roi:+.1f}%")
    m5.metric("Max Drawdown",  f"{max_dd:.1f}u")

    # ── Profit curve ──────────────────────────────────────────────────────────
    section_header("Cumulative Profit", "Flat 1u vs quarter-Kelly sizing, chronological")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bets["bet_num"], y=bets["cum_flat"].round(2),
        mode="lines", name="Flat (1u per bet)",
        line=dict(color="#60a5fa", width=2),
        hovertemplate="Bet #%{x}<br><b>%{y:+.1f}u</b><extra>Flat</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bets["bet_num"], y=bets["cum_kelly"].round(2),
        mode="lines", name="Kelly (0.5–3u)",
        line=dict(color="#f5c518", width=2),
        hovertemplate="Bet #%{x}<br><b>%{y:+.1f}u</b><extra>Kelly</extra>",
    ))
    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.25)", line_width=1)
    fig.update_layout(
        xaxis_title="Bet # (chronological)",
        yaxis_title="Cumulative Units",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab4c5", size=12, family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#232b3d", zerolinecolor="#232b3d"),
        yaxis=dict(gridcolor="#232b3d", zerolinecolor="#232b3d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Rolling win rate ──────────────────────────────────────────────────────
    section_header("Rolling Win Rate", "20-bet moving average — is the edge holding?")
    _WINDOW = 20
    if len(bets) >= _WINDOW:
        bets["roll_wr"] = bets["won"].rolling(_WINDOW, min_periods=_WINDOW).mean() * 100
        roll_valid = bets.dropna(subset=["roll_wr"])

        fig_roll = go.Figure()
        # Breakeven reference band
        fig_roll.add_hrect(y0=0, y1=52.38, line_width=0,
                           fillcolor="rgba(239,68,68,0.06)")
        fig_roll.add_hline(y=52.38, line_dash="dash",
                           line_color="rgba(239,68,68,0.45)", line_width=1,
                           annotation_text="52.4% break-even",
                           annotation_position="top right",
                           annotation_font_color="#ef4444",
                           annotation_font_size=10)
        fig_roll.add_hline(y=50, line_dash="dot",
                           line_color="rgba(255,255,255,0.1)", line_width=1)

        # Color the line: green above breakeven, red below
        x_vals  = roll_valid["bet_num"].tolist()
        y_vals  = roll_valid["roll_wr"].tolist()
        fig_roll.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name=f"{_WINDOW}-bet rolling win rate",
            line=dict(color="#60a5fa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.07)",
            hovertemplate="Bet #%{x}<br><b>%{y:.1f}%</b> win rate (last 20)<extra></extra>",
        ))

        # Mark best / worst windows
        best_idx  = roll_valid["roll_wr"].idxmax()
        worst_idx = roll_valid["roll_wr"].idxmin()
        for idx, sym, col, label in [
            (best_idx,  "star", "#34d399", "Best run"),
            (worst_idx, "x",    "#f87171", "Worst run"),
        ]:
            r = roll_valid.loc[idx]
            fig_roll.add_trace(go.Scatter(
                x=[r["bet_num"]], y=[r["roll_wr"]],
                mode="markers+text",
                marker=dict(symbol=sym, size=12, color=col),
                text=[f"{r['roll_wr']:.0f}%"],
                textposition="top center",
                textfont=dict(size=10, color=col),
                showlegend=False,
                hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
            ))

        # Season boundary lines
        if "season" in bets.columns:
            for szn in bets["season"].dropna().unique():
                first_in_szn = bets[bets["season"] == szn]["bet_num"].min()
                if pd.notna(first_in_szn) and first_in_szn > 1:
                    fig_roll.add_vline(
                        x=first_in_szn,
                        line_color="rgba(245,197,24,0.25)", line_width=1,
                        annotation_text=str(int(szn)),
                        annotation_position="top left",
                        annotation_font_size=9,
                        annotation_font_color="#f5c518",
                    )

        fig_roll.update_layout(
            xaxis_title="Bet # (chronological)",
            yaxis_title="Win Rate (%)",
            yaxis_range=[30, 80],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aab4c5", size=12, family="Inter, sans-serif"),
            xaxis=dict(gridcolor="#232b3d", zerolinecolor="#232b3d"),
            yaxis=dict(gridcolor="#232b3d", zerolinecolor="#232b3d",
                       ticksuffix="%"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                        yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=260,
            margin=dict(l=0, r=0, t=24, b=0),
        )
        st.plotly_chart(fig_roll, width='stretch', config={"displayModeBar": False})

        # Trend indicator: compare last 20 bets to overall average
        last20_wr = bets.tail(20)["won"].mean() * 100
        overall_wr = bets["won"].mean() * 100
        trend_delta = last20_wr - overall_wr
        trend_color = "var(--green)" if trend_delta >= 0 else "var(--red)"
        trend_arrow = "↑" if trend_delta >= 0 else "↓"
        st.html(f"""
        <div style="display:flex;gap:24px;padding:6px 2px 2px 2px">
            <div>
                <span style="color:var(--ink-4);font-size:0.68em;font-weight:700;
                             text-transform:uppercase;letter-spacing:.08em">Last 20 bets</span>
                <span class="num" style="color:{trend_color};font-weight:800;
                                          font-size:0.92em;margin-left:8px">
                    {last20_wr:.1f}% {trend_arrow}</span>
            </div>
            <div>
                <span style="color:var(--ink-4);font-size:0.68em;font-weight:700;
                             text-transform:uppercase;letter-spacing:.08em">vs. overall</span>
                <span class="num" style="color:{trend_color};font-weight:800;
                                          font-size:0.92em;margin-left:8px">
                    {trend_delta:+.1f}pp</span>
            </div>
            <div>
                <span style="color:var(--ink-4);font-size:0.68em;font-weight:700;
                             text-transform:uppercase;letter-spacing:.08em">Break-even gap</span>
                <span class="num" style="color:{'var(--green)' if last20_wr >= 52.38 else 'var(--red)'};
                                          font-weight:800;font-size:0.92em;margin-left:8px">
                    {last20_wr - 52.38:+.1f}pp</span>
            </div>
        </div>
        """)
    else:
        st.caption(f"Need at least {_WINDOW} bets for rolling window — lower edge thresholds or use all seasons.")

    st.markdown("---")
    left_col, right_col = st.columns(2)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    with left_col:
        st.markdown("#### 🎯 Threshold Sweep")
        st.caption("Same threshold applied to spreads and totals. Find the edge cutoff with the best ROI.")
        sweep_rows = []
        for t in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]:
            b = _simulate(filt, t, t)
            if b.empty:
                continue
            n   = len(b)
            wp  = b["won"].mean() * 100
            fp  = b["flat_pnl"].sum()
            roi = fp / (n * LOSE_U) * 100
            kp  = b["kelly_pnl"].sum()
            is_current = (t == sp_thresh)
            sweep_rows.append({
                "Edge ≥": f"{t:.1f} pts",
                "# Bets": n,
                "Win %": f"{wp:.1f}%",
                "Flat P&L": f"{fp:+.1f}u",
                "ROI": f"{roi:+.1f}%",
                "Kelly P&L": f"{kp:+.1f}u",
                "_is_current": is_current,
            })
        if sweep_rows:
            sweep_df = pd.DataFrame(sweep_rows)
            # Highlight current threshold row
            def _hl(row):
                if row["_is_current"]:
                    return ["background-color: rgba(59,130,246,0.15)"] * len(row)
                return [""] * len(row)
            styled = (sweep_df.drop(columns=["_is_current"])
                               .style.apply(_hl, axis=1, subset=None))
            # Re-add _is_current to subset apply
            st.dataframe(
                sweep_df.drop(columns=["_is_current"]),
                width='stretch', hide_index=True)

    # ── Bet type breakdown ────────────────────────────────────────────────────
    with right_col:
        st.markdown("#### 🔍 Bet Type Breakdown")
        st.caption("Performance split by bet category at the selected thresholds above.")
        type_rows = []
        for bet_type, label in [("Spread","📏 Spread"), ("Over","⬆️ Over"), ("Under","⬇️ Under")]:
            sub = bets[bets["type"] == bet_type]
            if sub.empty:
                continue
            n   = len(sub)
            wp  = sub["won"].mean() * 100
            fp  = sub["flat_pnl"].sum()
            roi = fp / (n * LOSE_U) * 100
            type_rows.append({
                "Type": label, "Bets": n,
                "Win %": f"{wp:.1f}%",
                "Flat P&L": f"{fp:+.1f}u",
                "ROI": f"{roi:+.1f}%",
            })
        # All combined
        type_rows.append({
            "Type": "🔢 All", "Bets": n_bets,
            "Win %": f"{win_pct:.1f}%",
            "Flat P&L": f"{flat_tot:+.1f}u",
            "ROI": f"{flat_roi:+.1f}%",
        })
        if type_rows:
            st.dataframe(pd.DataFrame(type_rows),
                         width='stretch', hide_index=True)

        # Win rate context
        st.markdown("---")
        st.markdown("**Break-even reference**")
        st.caption(
            "At standard -110 juice you need **52.4%** to break even on flat betting. "
            "Kelly sizing outperforms flat when win rate is consistently above that threshold "
            "but adds variance — if the edge estimates are off, Kelly loses more."
        )

    # ── Season-by-season breakdown ────────────────────────────────────────────
    seasons_in_data = sorted(bets["season"].dropna().unique().astype(int))
    if len(seasons_in_data) > 1:
        with st.expander("📅 Season-by-season breakdown"):
            szn_rows = []
            for szn in seasons_in_data:
                sub = bets[bets["season"] == szn]
                if sub.empty:
                    continue
                n   = len(sub)
                wp  = sub["won"].mean() * 100
                fp  = sub["flat_pnl"].sum()
                roi = fp / (n * LOSE_U) * 100
                kp  = sub["kelly_pnl"].sum()
                szn_rows.append({
                    "Season": int(szn), "Bets": n,
                    "Win %": f"{wp:.1f}%",
                    "Flat P&L": f"{fp:+.1f}u",
                    "ROI": f"{roi:+.1f}%",
                    "Kelly P&L": f"{kp:+.1f}u",
                })
            if szn_rows:
                st.dataframe(pd.DataFrame(szn_rows),
                             width='stretch', hide_index=True)

    # ── All qualifying bets detail ─────────────────────────────────────────────
    with st.expander("📋 All qualifying bets (most recent first)"):
        display = bets[["season","week","matchup","type","direction",
                         "edge","won","flat_pnl","kelly_u"]].copy()
        display.columns = ["Season","Wk","Matchup","Type","Side",
                            "Edge","Won","Flat P&L","Kelly u"]
        display["Won"]     = display["Won"].map({1:"✅",0:"❌"})
        display["Flat P&L"] = display["Flat P&L"].apply(lambda x: f"{x:+.1f}u")
        display["Edge"]    = display["Edge"].apply(lambda x: f"{x:+.1f}")
        display["Kelly u"] = display["Kelly u"].apply(lambda x: f"{x:.1f}u")
        st.dataframe(display.sort_values(["Season","Wk"], ascending=False),
                     width='stretch', hide_index=True)


# ─── MODEL ANALYSIS TAB ──────────────────────────────────────────────────────

def render_analysis_tab():
    """
    Residual analysis tab — loads model_results.csv and renders four charts:
      1. Predicted vs. actual spread margin (scatter + regression)
      2. Team-level bias (which teams the model chronically misses)
      3. MAE by week (early-season noise vs. late-season fatigue)
      4. Predicted vs. actual totals (scatter + OVER/UNDER accuracy)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly is required for this tab. Add `plotly>=5.0` to requirements.txt.")
        return

    # Import analysis helpers from scripts/generate_analysis.py
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "generate_analysis", ROOT_DIR / "scripts" / "generate_analysis.py"
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_results      = _mod.load_results
    summary_stats     = _mod.summary_stats
    fig_scatter_spread = _mod.fig_scatter_spread
    fig_team_residuals = _mod.fig_team_residuals
    fig_mae_by_week    = _mod.fig_mae_by_week
    fig_scatter_totals = _mod.fig_scatter_totals

    results_path = ROOT_DIR / "outputs" / "predictions" / "model_results.csv"
    if not results_path.exists():
        st.info("No model results found. Run `python3 src/model.py` to generate predictions.")
        return

    # Season selector — only offer seasons present in the CSV
    try:
        all_seasons = sorted(pd.read_csv(results_path, usecols=["season"])["season"].unique(), reverse=True)
    except Exception:
        st.error("Could not read model_results.csv.")
        return

    col_s, col_info = st.columns([1, 3])
    with col_s:
        sel_season = st.selectbox("Season", ["All"] + [str(s) for s in all_seasons], index=1)
    season_filter = int(sel_season) if sel_season != "All" else None

    try:
        df    = load_results(season=season_filter)
        stats = summary_stats(df)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return

    if df.empty:
        st.warning(f"No games found for season {sel_season}.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games",       f"{stats['n']:,}")
    c2.metric("Spread MAE",  f"{stats['spread_mae']:.1f} pts")
    c3.metric("Spread R²",   f"{stats['spread_r2']:.3f}")
    c4.metric("Within 7 pts", f"{stats['within_7']:.0%}")
    if "ats_acc" in stats:
        c5.metric("ATS Accuracy", f"{stats['ats_acc']:.1%}")
    elif "totals_acc" in stats:
        c5.metric("Totals Acc.", f"{stats['totals_acc']:.1%}")

    plotly_cfg = {"displayModeBar": False}

    # ── Chart 1: Predicted vs. actual spread ─────────────────────────────
    section_header("Predicted vs. Actual Margin",
                   "Green = correct direction · Dashed = perfect prediction")
    st.plotly_chart(fig_scatter_spread(df), width='stretch', config=plotly_cfg)

    # ── Chart 2 + 3 side-by-side ─────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_header("Team Residuals",
                       "+pts = model overestimates team · −pts = underestimates")
        st.caption("Only teams with ≥ 3 games in the selected season. "
                   "Large bars surface systematic biases (e.g. service academies, run-heavy offenses).")
        st.plotly_chart(fig_team_residuals(df), width='stretch', config=plotly_cfg)

    with col_r:
        section_header("MAE by Week",
                       "Early = small sample noise · Late = fatigue / garbage time")
        st.plotly_chart(fig_mae_by_week(df), width='stretch', config=plotly_cfg)

    # ── Chart 4: Totals ───────────────────────────────────────────────────
    fig_tot = fig_scatter_totals(df)
    if fig_tot is not None:
        section_header("Predicted vs. Actual Total Score",
                       "Green = correct OVER/UNDER call")
        if "totals_acc" in stats and "totals_mae" in stats:
            st.caption(
                f"O/U accuracy: **{stats['totals_acc']:.1%}** · "
                f"MAE: **{stats['totals_mae']:.1f} pts**"
            )
        st.plotly_chart(fig_tot, width='stretch', config=plotly_cfg)

    # ── Residual table ────────────────────────────────────────────────────
    section_header("Raw Predictions", "Sortable — click any column header")
    disp_cols = ["season", "week", "home_team", "away_team",
                 "point_diff", "pred_spread", "residual",
                 "total_points", "pred_total"]
    disp = df[[c for c in disp_cols if c in df.columns]].copy()
    disp.columns = [c.replace("_", " ").title() for c in disp.columns]
    st.dataframe(
        disp.sort_values("Week").reset_index(drop=True),
        width='stretch',
        height=320,
    )

    st.download_button(
        "Export predictions CSV",
        data=df.to_csv(index=False),
        file_name=f"cfb_predictions_{sel_season}.csv",
        mime="text/csv",
    )


# ─── SEASON STANDINGS TAB ────────────────────────────────────────────────────

def render_standings_tab():
    """
    Season Standings — live season model record (from weekly picks archive)
    shown side-by-side with each bettor's tracked record.

    The model record reflects only picks made during the current live season —
    not historical backtest data. It starts at 0-0 and accumulates as weeks
    are played. Historical walk-forward data lives in the Research → Backtester tab.
    """
    import json as _json
    from datetime import date as _date

    current_season = _date.today().year if _date.today().month >= 7 else _date.today().year - 1
    section_header("Season Standings", f"{current_season} Season — Live Record")

    # ── Season selector (for bettor bets only — model record is always current) ─
    bets_all  = load_bets()
    bet_seasons = sorted(
        {int(b["season"]) for b in bets_all if "season" in b and str(b["season"]).isdigit()},
        reverse=True)
    all_seasons = sorted(set(bet_seasons) | {current_season}, reverse=True)
    sel_season  = st.selectbox("Season", all_seasons, index=0, key="standings_season")

    # ── Model record — from weekly picks archive (live season only) ───────────
    # Picks are saved to outputs/picks/YYYY_W{wk}.json by the newsletter generator.
    # This reflects the model's actual in-season calls, not historical backtest data.
    picks_dir  = ROOT_DIR / "outputs" / "picks"
    model_rows = []

    for picks_file in sorted(picks_dir.glob(f"{sel_season}_W*.json")):
        try:
            with open(picks_file) as f:
                week_picks = _json.load(f)
            for p in week_picks:
                outcome = p.get("outcome", "pending")
                if outcome in ("win", "loss"):
                    model_rows.append({
                        "won":    outcome == "win",
                        "status": "Won" if outcome == "win" else "Lost",
                    })
        except Exception:
            continue

    if sel_season == current_season and not model_rows:
        st.info(
            f"The {current_season} model record starts at 0–0. "
            f"Once the season begins, picks are tracked automatically each Tuesday. "
            f"Historical backtest data is in **Research → Backtester**.",
            icon="🏈"
        )

    # ── Bettor records — filtered to selected season ──────────────────────────
    bets     = load_bets()
    settled  = [b for b in bets
                if b["status"] in ("Won", "Lost", "Push")
                and str(b.get("season", "")).strip() == str(sel_season)]

    def record_for(source_bets):
        wins   = sum(1 for b in source_bets if b["status"] == "Won")
        losses = sum(1 for b in source_bets if b["status"] == "Lost")
        pushes = sum(1 for b in source_bets if b["status"] == "Push")
        pnl    = sum(bet_pnl(b) for b in source_bets)
        total  = wins + losses + pushes
        wr     = wins / (wins + losses) if (wins + losses) > 0 else None
        roi    = pnl / total if total > 0 else None
        clv_vals = [v for b in source_bets if (v := compute_clv(b)) is not None]
        avg_clv  = sum(clv_vals) / len(clv_vals) if clv_vals else None
        return dict(wins=wins, losses=losses, pushes=pushes, pnl=pnl,
                    win_rate=wr, roi=roi, avg_clv=avg_clv, n=total)

    # Build standings rows
    rows = []

    # Model row
    if model_rows:
        model_settled = [r for r in model_rows if r["status"] in ("Won", "Lost")]
        mw = sum(1 for r in model_settled if r["won"])
        ml = sum(1 for r in model_settled if not r["won"])
        mp = sum(0.909 if r["won"] else -1.0 for r in model_settled)
        wr = mw / (mw + ml) if (mw + ml) > 0 else None
        rows.append({"Who": "🤖 Model", "Record": f"{mw}–{ml}",
                     "Win Rate": f"{wr:.1%}" if wr else "—",
                     "Units P&L": f"{mp:+.1f}u",
                     "ROI": f"{mp/(mw+ml):.1%}" if (mw+ml) > 0 else "—",
                     "Avg CLV": "—", "_pnl": mp, "_wr": wr or 0})

    # Bettor rows
    all_bettors = BETTORS + ["All Bettors"]
    for bettor in all_bettors:
        if bettor == "All Bettors":
            source = settled
            label  = "📊 All Bettors"
        else:
            source = [b for b in settled if b.get("bettor") == bettor]
            label  = bettor
        if not source:
            continue
        r = record_for(source)
        rows.append({
            "Who":       label,
            "Record":    f"{r['wins']}–{r['losses']}" + (f"–{r['pushes']}P" if r["pushes"] else ""),
            "Win Rate":  f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—",
            "Units P&L": f"{r['pnl']:+.2f}u",
            "ROI":       f"{r['roi']:.1%}" if r["roi"] is not None else "—",
            "Avg CLV":   f"{r['avg_clv']:+.1f}" if r["avg_clv"] is not None else "—",
            "_pnl": r["pnl"], "_wr": r["win_rate"] or 0,
        })

    if not rows:
        st.info("No settled bets yet. Track picks from This Week's Picks and mark results in My Bets.")
        return

    # ── Render leaderboard cards ──────────────────────────────────────────────
    BREAKEVEN = 0.5238
    BAR_MAX   = 0.70   # win-rate bar scale: 0–70%
    for i, row in enumerate(sorted(rows, key=lambda x: x["_pnl"], reverse=True)):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        rank_color = ("var(--gold)" if i == 0 else "var(--ink-2)" if i == 1
                      else "#b45309" if i == 2 else "var(--ink-4)")
        pnl_val = row["_pnl"]
        pnl_color = ("var(--green)" if pnl_val > 0 else "var(--red)" if pnl_val < 0
                     else "var(--ink-3)")

        # Win-rate bar with the 52.4% breakeven line marked
        wr = row["_wr"]
        wr_pct   = min(wr / BAR_MAX, 1.0) * 100
        be_pct   = BREAKEVEN / BAR_MAX * 100
        wr_color = "var(--green)" if wr >= BREAKEVEN else "var(--orange)" if wr > 0 else "var(--line-hi)"
        wr_bar = f"""
        <div style="margin-top:10px">
          <div style="position:relative;height:5px;border-radius:999px;background:var(--line)">
            <div style="position:absolute;left:0;top:0;bottom:0;width:{wr_pct:.1f}%;
                        background:{wr_color};border-radius:999px"></div>
            <div style="position:absolute;left:{be_pct:.1f}%;top:-3px;width:2px;height:11px;
                        background:var(--ink-3);border-radius:2px" title="52.4% breakeven"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.62em;
                      color:var(--ink-4);margin-top:3px">
            <span>win rate vs <span style="color:var(--ink-3)">52.4% breakeven</span></span>
            <span class="num" style="color:{wr_color}">{row['Win Rate']}</span>
          </div>
        </div>"""

        def cell(label, value, color="var(--ink)"):
            return (f'<div style="text-align:center;min-width:64px">'
                    f'<div style="color:var(--ink-4);font-size:0.6em;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:2px">{label}</div>'
                    f'<div class="num" style="color:{color};font-size:0.92em;font-weight:800">{value}</div></div>')

        st.html(f"""
        <div class="pick-card" style="padding:15px 20px 13px 20px">
            <div class="accent" style="background:{rank_color}"></div>
            <div style="display:flex;justify-content:space-between;align-items:center;gap:16px">
                <div style="display:flex;align-items:center;gap:13px;min-width:0">
                    <span style="font-size:1.15em;min-width:30px;text-align:center;
                                 color:{rank_color};font-weight:800">{medal}</span>
                    <span style="color:var(--ink);font-size:1.05em;font-weight:800;
                                 letter-spacing:-0.01em">{row['Who']}</span>
                </div>
                <div style="display:flex;gap:22px;align-items:center">
                    {cell("Record", row['Record'])}
                    {cell("Units P&L", row['Units P&L'], pnl_color)}
                    {cell("ROI", row['ROI'])}
                    {cell("Avg CLV", row['Avg CLV'])}
                </div>
            </div>
            {wr_bar}
        </div>
        """)

    # ── Breakdown by bet type ─────────────────────────────────────────────────
    if settled:
        section_header("Breakdown by Bet Type")
        by_type = {}
        for b in settled:
            bt = b.get("bet_type", "Other")
            by_type.setdefault(bt, []).append(b)
        cols = st.columns(len(by_type))
        for i, (bt, bt_bets) in enumerate(by_type.items()):
            r = record_for(bt_bets)
            cols[i].metric(bt, f"{r['wins']}–{r['losses']}",
                           delta=f"{r['win_rate']:.1%} win rate" if r["win_rate"] else None)


# ─── CLV TRACKER TAB ─────────────────────────────────────────────────────────

def render_clv_tab():
    """Dedicated Closing Line Value tracker with chart + table."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly required. Add `plotly>=5.0` to requirements.txt.")
        return

    bets = load_bets()
    clv_bets = [b for b in bets if compute_clv(b) is not None]

    if not clv_bets:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.info("No CLV data yet. Enter closing lines on bets in My Bets to track line value.")
        st.caption("CLV (Closing Line Value) measures how much better your line was vs. the closing line. "
                   "Consistently positive CLV is the strongest indicator of long-term edge.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    all_clv   = [compute_clv(b) for b in clv_bets]
    avg_clv   = sum(all_clv) / len(all_clv)
    beat_rate = sum(1 for v in all_clv if v > 0) / len(all_clv)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bets w/ CLV", len(clv_bets))
    c2.metric("Avg CLV", f"{avg_clv:+.2f}",
              delta_color="normal" if avg_clv >= 0 else "inverse")
    c3.metric("Beat the Close", f"{beat_rate:.0%}")
    c4.metric("Total CLV", f"{sum(all_clv):+.1f}")

    # ── Per-bettor CLV chart ──────────────────────────────────────────────────
    section_header("CLV Over Time", "Cumulative closing line value per bettor")

    PANEL_BG = "rgba(0,0,0,0)"
    BORDER   = "#232b3d"
    TEXT     = "#aab4c5"
    COLORS   = ["#f5c518", "#60a5fa", "#34d399", "#fb923c", "#a78bfa"]

    fig = go.Figure()
    bettor_filter = ["All"] + BETTORS
    for i, bettor in enumerate(BETTORS):
        b_bets = sorted(
            [b for b in clv_bets if b.get("bettor") == bettor],
            key=lambda x: x.get("date", "")
        )
        if not b_bets:
            continue
        clv_vals = [compute_clv(b) for b in b_bets]
        cum_clv  = [sum(clv_vals[:j+1]) for j in range(len(clv_vals))]
        dates    = [b.get("date", f"Bet {j+1}") for j, b in enumerate(b_bets)]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum_clv)+1)), y=cum_clv,
            mode="lines+markers",
            name=bettor,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=6),
            hovertemplate=f"{bettor}<br>Bet %{{x}}<br>Cumulative CLV: %{{y:+.2f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#454f63", line_width=1, line_dash="dash")
    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, size=11, family="Inter, sans-serif"),
        xaxis=dict(title="Bet #", gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Cumulative CLV", gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor=PANEL_BG, bordercolor=BORDER),
        margin=dict(l=50, r=20, t=20, b=50),
        height=340,
    )
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    # ── CLV by bet type breakdown ─────────────────────────────────────────────
    section_header("CLV by Bet Type")
    by_type = {}
    for b in clv_bets:
        bt = b.get("bet_type", "Other")
        by_type.setdefault(bt, []).append(compute_clv(b))

    cols = st.columns(len(by_type)) if by_type else []
    for i, (bt, vals) in enumerate(by_type.items()):
        avg  = sum(vals) / len(vals)
        beat = sum(1 for v in vals if v > 0) / len(vals)
        cols[i].metric(bt, f"{avg:+.2f} avg CLV",
                       delta=f"{beat:.0%} beat close",
                       delta_color="normal" if avg >= 0 else "inverse")

    # ── Full CLV table ────────────────────────────────────────────────────────
    section_header("All Bets with CLV")
    rows = []
    for b in sorted(clv_bets, key=lambda x: compute_clv(x) or 0, reverse=True):
        clv = compute_clv(b)
        unit = "ppts" if b.get("bet_type") == "Moneyline" else "pts"
        rows.append({
            "Pick":       b.get("pick", ""),
            "Type":       b.get("bet_type", ""),
            "Bettor":     b.get("bettor", ""),
            "Line":       b.get("line", ""),
            "Close":      b.get("closing_line", ""),
            "CLV":        f"{clv:+.2f} {unit}",
            "Result":     b.get("status", ""),
            "Week":       f"Wk {b.get('week', '')}",
        })
    if rows:
        clv_df = pd.DataFrame(rows)
        st.dataframe(clv_df, width='stretch', height=300)


# ─── HISTORICAL PICKS TAB ─────────────────────────────────────────────────────

def render_history_tab():
    """Browse any prior week's model picks alongside actual results."""
    results_path = ROOT_DIR / "outputs" / "predictions" / "model_results.csv"
    if not results_path.exists():
        st.info("No historical results yet. Run `python3 src/model.py` to generate predictions.")
        return

    try:
        res = pd.read_csv(results_path)
        for col in ["spread_edge", "totals_edge", "point_diff", "total_points",
                    "spread", "over_under", "pred_spread", "pred_total",
                    "covered_spread", "went_over", "week", "season"]:
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors="coerce")
        # Compute totals_edge if not saved in the CSV
        if "totals_edge" not in res.columns and "pred_total" in res.columns and "over_under" in res.columns:
            res["totals_edge"] = (pd.to_numeric(res["pred_total"], errors="coerce")
                                  - pd.to_numeric(res["over_under"], errors="coerce"))
    except Exception as e:
        st.error(f"Could not load model results: {e}")
        return

    # ── Week / season selectors ───────────────────────────────────────────────
    seasons = sorted(res["season"].dropna().unique().astype(int), reverse=True)
    col_s, col_w, col_v, _ = st.columns([1, 1, 1, 2])
    sel_season = col_s.selectbox("Season", seasons, index=0)
    weeks      = sorted(res[res["season"] == sel_season]["week"].dropna().unique().astype(int))
    sel_week   = col_w.selectbox("Week", weeks, index=len(weeks)-1 if weeks else 0)
    view_mode  = col_v.selectbox("Show", ["Flagged Picks", "All Games"])

    week_df = res[(res["season"] == sel_season) & (res["week"] == sel_week)].copy()

    if week_df.empty:
        st.info(f"No data for {sel_season} Week {sel_week}.")
        return

    st.markdown(
        f'<div style="color:#4b5563;font-size:0.82em;margin:8px 0 16px 0">'
        f'{len(week_df)} games · {sel_season} Week {sel_week}</div>',
        unsafe_allow_html=True
    )

    # Filter to flagged picks if requested
    if view_mode == "Flagged Picks":
        mask = pd.Series(False, index=week_df.index)
        if "totals_edge" in week_df.columns:
            mask = mask | (week_df["totals_edge"].abs() >= TOTALS_EDGE_MIN)
        if "spread_edge" in week_df.columns:
            mask = mask | (week_df["spread_edge"].abs() >= SPREAD_EDGE_MIN)
        flagged = week_df[mask]
        if flagged.empty:
            st.info("No picks met the edge threshold this week.")
            return
        display_df = flagged
    else:
        display_df = week_df

    # ── Render each game ──────────────────────────────────────────────────────
    for _, row in display_df.iterrows():
        home = row["home_team"] if "home_team" in row.index else row.get("home_team", "Home")
        away = row["away_team"] if "away_team" in row.index else row.get("away_team", "Away")
        matchup    = f"{home} vs {away}"
        actual_mg  = row.get("point_diff")   # home margin
        actual_tot = row.get("total_points")
        pred_sp    = row.get("pred_spread")
        pred_tot   = row.get("pred_total")
        t_edge     = row.get("totals_edge")
        s_edge     = row.get("spread_edge")
        covered    = row.get("covered_spread")
        went_over  = row.get("went_over")

        picks_html = ""

        accent = "var(--line-hi)"   # neutral until a pick decides hit/miss

        def _pick_badge(label: str, badge_color: str, correct: bool, detail: str) -> str:
            res_color = "var(--green)" if correct else "var(--red)"
            res_label = "✓ HIT" if correct else "✗ MISS"
            return f"""
            <span style="background:{badge_color};color:#0b0e14;font-size:0.63em;font-weight:800;
                         padding:2px 8px;border-radius:5px;margin-right:6px">{label}</span>
            <span class="num" style="color:{res_color};font-size:0.76em;font-weight:800;margin-right:14px">
                {res_label} <span style="color:var(--ink-4);font-weight:600">({detail})</span></span>"""

        # Totals pick
        if pd.notna(t_edge) and abs(t_edge) >= TOTALS_EDGE_MIN:
            is_under   = t_edge < 0
            side       = "UNDER" if is_under else "OVER"
            ou_str     = f"{row['over_under']:.1f}" if pd.notna(row.get("over_under")) else "?"
            correct    = (is_under and went_over == 0) or (not is_under and went_over == 1)
            accent     = "var(--green)" if correct else "var(--red)"
            actual_str = f"{actual_tot:.0f}" if pd.notna(actual_tot) else "?"
            picks_html += _pick_badge(f"{side} {ou_str}", "var(--cyan)", correct,
                                      f"actual {actual_str}")

        # Spread pick
        if pd.notna(s_edge) and abs(s_edge) >= SPREAD_EDGE_MIN:
            bet_home   = s_edge > 0
            team       = row["home_team"] if bet_home else row["away_team"]
            vl         = row.get("spread")
            vl_str     = (f"{vl:+.1f}" if pd.notna(vl) and bet_home
                          else f"{-vl:+.1f}" if pd.notna(vl) else "?")
            correct    = (bet_home and covered == 1) or (not bet_home and covered == 0)
            accent     = "var(--green)" if correct else "var(--red)"
            actual_str = f"{actual_mg:+.0f}" if pd.notna(actual_mg) else "?"
            picks_html += _pick_badge(f"{team} {vl_str}", "var(--violet)", correct,
                                      f"margin {actual_str}")

        if not picks_html and view_mode == "All Games":
            picks_html = '<span style="color:var(--ink-4);font-size:0.78em">No flagged pick</span>'

        actual_display = f"Final: {actual_mg:+.0f} pts" if pd.notna(actual_mg) else "No result"

        st.html(f"""
        <div class="pick-card" style="padding:12px 18px 11px 18px">
            <div class="accent" style="background:{accent}"></div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:var(--ink);font-weight:800;font-size:0.95em">{matchup}</span>
                <span class="num" style="color:var(--ink-4);font-size:0.76em">{actual_display}</span>
            </div>
            <div style="margin-top:8px">{picks_html}</div>
        </div>
        """)

    # ── Weekly summary ────────────────────────────────────────────────────────
    summary_mask = pd.Series(False, index=week_df.index)
    if "totals_edge" in week_df.columns:
        summary_mask = summary_mask | (week_df["totals_edge"].abs() >= TOTALS_EDGE_MIN)
    if "spread_edge" in week_df.columns:
        summary_mask = summary_mask | (week_df["spread_edge"].abs() >= SPREAD_EDGE_MIN)
    flagged_all = week_df[summary_mask]
    if not flagged_all.empty:
        tot_picks = flagged_all[flagged_all["totals_edge"].abs() >= TOTALS_EDGE_MIN] \
            if "totals_edge" in flagged_all.columns else flagged_all.iloc[0:0]
        sp_picks  = flagged_all[flagged_all["spread_edge"].abs() >= SPREAD_EDGE_MIN] \
            if "spread_edge" in flagged_all.columns else flagged_all.iloc[0:0]

        def hit_rate(picks, col, hit_val):
            settled = picks[picks[col].notna()]
            if settled.empty: return None
            if col == "went_over" and "totals_edge" in settled.columns:
                hits = (((settled["totals_edge"] < 0) & (settled["went_over"] == 0)).sum() +
                        ((settled["totals_edge"] > 0) & (settled["went_over"] == 1)).sum())
            else:
                hits = (((settled["spread_edge"] > 0) & (settled["covered_spread"] == 1)).sum() +
                        ((settled["spread_edge"] < 0) & (settled["covered_spread"] == 0)).sum())
            return hits / len(settled)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section_header(f"Week {sel_week} Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Picks",   len(flagged_all))
        c2.metric("Totals Picks",  len(tot_picks))
        c3.metric("Spread Picks",  len(sp_picks))


# ─── RIGHT BANKROLL PANEL ────────────────────────────────────────────────────

def _render_right_panel(plays: list, n_strong: int, week: int):
    """Compact right column: bankroll card, week snapshot, top bets, season form."""
    bankroll_val = st.session_state.get("bankroll", 1000)
    unit_val     = bankroll_val / 100

    # Bankroll card
    st.html(f"""
    <div class="panel-card">
        <div class="panel-label">Bankroll</div>
        <div class="num" style="color:var(--ink);font-size:1.6em;font-weight:800;
                                line-height:1">${bankroll_val:,}</div>
        <div style="color:var(--ink-4);font-size:0.72em;margin-top:6px">
            1u = <b style="color:var(--gold)">${unit_val:,.0f}</b>
            &nbsp;·&nbsp; Quarter-Kelly sizing
        </div>
    </div>""")

    if not plays:
        return

    def _play_units(p):
        r = p["row"]
        if p["kind"] == "total":
            return kelly_units_spread(abs(r["totals_edge"]))
        if p["kind"] == "ml":
            return kelly_units_ml(r["ml_ev"])
        return 1

    top8      = plays[:8]
    t_units   = sum(_play_units(p) for p in top8)
    t_dollars = t_units * unit_val
    pct_risk  = t_dollars / bankroll_val if bankroll_val else 0
    pct_color = ("var(--red)"   if pct_risk > 0.12 else
                 "var(--gold)"  if pct_risk > 0.07 else "var(--green)")

    # Week snapshot card
    st.html(f"""
    <div class="panel-card">
        <div class="panel-label">Week {week} Snapshot</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
            <div>
                <div class="num" style="color:var(--ink);font-size:1.3em;font-weight:800">{len(plays)}</div>
                <div style="color:var(--ink-4);font-size:0.65em;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.08em">flagged</div>
            </div>
            <div>
                <div class="num" style="color:var(--green);font-size:1.3em;font-weight:800">{n_strong}</div>
                <div style="color:var(--ink-4);font-size:0.65em;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.08em">strong</div>
            </div>
        </div>
        <div style="border-top:1px solid var(--line);padding-top:8px">
            <div style="display:flex;justify-content:space-between;font-size:0.78em">
                <span style="color:var(--ink-3)">Total if all placed</span>
                <span class="num" style="color:var(--ink);font-weight:700">{t_units}u &middot; ${t_dollars:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.78em;margin-top:3px">
                <span style="color:var(--ink-3)">% at risk</span>
                <span class="num" style="color:{pct_color};font-weight:700">{pct_risk:.1%}</span>
            </div>
        </div>
    </div>""")

    # Top bets list
    rows_html = ""
    for i, p in enumerate(top8[:5], start=1):
        r  = p["row"]
        u  = _play_units(p)
        sc = p["score"]
        sc_col = ("var(--green)" if sc >= 60 else
                  "var(--gold)"  if sc >= 40 else "var(--ink-3)")
        if p["kind"] == "total":
            is_under   = r["totals_edge"] < 0
            ou_str     = f"{r['over_under']:.1f}" if pd.notna(r.get("over_under")) else "TBD"
            label_str  = f"{'UNDER' if is_under else 'OVER'} {ou_str}"
            dot_col    = "var(--cyan)" if is_under else "var(--orange)"
        elif p["kind"] == "ml":
            odds_str   = (f"+{int(r['ml_book_odds'])}" if r["ml_book_odds"] > 0
                          else str(int(r["ml_book_odds"])))
            label_str  = f"{r['ml_team']} {odds_str}"
            dot_col    = "var(--blue)"
        else:
            is_home    = r["spread_edge"] > 0
            team       = r["home_team"] if is_home else r["away_team"]
            sp         = r.get("spread")
            sp_str     = (f"{sp:+.1f}" if is_home else f"{-sp:+.1f}") if pd.notna(sp) else ""
            label_str  = f"{team} {sp_str}"
            dot_col    = "var(--violet)"
        # Sharp confirmation check for this play
        _sh = False
        if p["kind"] == "total":
            is_un  = r["totals_edge"] < 0
            _sh = bool(int(r.get("sharp_total_under", 0) or 0) if is_un
                       else int(r.get("sharp_total_over", 0) or 0))
        elif p["kind"] == "ml":
            _ml_home = (r.get("ml_team") == r.get("home_team"))
            _sh = bool(int(r.get("sharp_move_home", 0) or 0) if _ml_home
                       else int(r.get("sharp_move_away", 0) or 0))
        else:
            is_h_sp = r["spread_edge"] > 0
            _sh = bool(int(r.get("sharp_move_home", 0) or 0) if is_h_sp
                       else int(r.get("sharp_move_away", 0) or 0))
        sharp_pip = ('<span style="color:var(--gold);font-size:0.75em;margin-left:2px" '
                     'title="Sharp money confirmed">⚡</span>' if _sh else "")
        rows_html += f"""
        <div class="panel-bet-row">
            <div style="display:flex;align-items:center;gap:6px;min-width:0;overflow:hidden">
                <span style="color:var(--ink-4);font-size:0.65em;font-weight:700;min-width:14px">#{i}</span>
                <div style="width:7px;height:7px;border-radius:50%;background:{dot_col};flex-shrink:0"></div>
                <span style="color:var(--ink-2);font-size:0.75em;font-weight:600;
                             overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{label_str}</span>
                {sharp_pip}
            </div>
            <div style="flex-shrink:0;padding-left:6px;text-align:right">
                <span class="num" style="color:{sc_col};font-size:0.72em;font-weight:700">{u}u</span>
                <span style="color:var(--ink-4);font-size:0.68em;margin-left:3px">${u * unit_val:,.0f}</span>
            </div>
        </div>"""

    st.html(f"""
    <div class="panel-card">
        <div class="panel-label">Top Bets</div>
        {rows_html}
    </div>""")

    # Season form (if settled bets exist)
    all_bets = load_bets()
    settled  = [b for b in all_bets if b["status"] in ("Won", "Lost")]
    if settled:
        wins     = sum(1 for b in settled if b["status"] == "Won")
        losses   = len(settled) - wins
        wr       = wins / len(settled)
        wr_col   = "var(--green)" if wr >= 0.55 else "var(--red)" if wr < 0.45 else "var(--gold)"
        pnl      = sum(bet_pnl(b) for b in settled)
        pnl_col  = "var(--green)" if pnl > 0 else "var(--red)" if pnl < 0 else "var(--ink-3)"
        st.html(f"""
        <div class="panel-card">
            <div class="panel-label">Season Form</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>
                    <div class="num" style="color:{wr_col};font-size:1.25em;font-weight:800">{wr:.0%}</div>
                    <div style="color:var(--ink-4);font-size:0.65em;font-weight:600;
                                text-transform:uppercase">{wins}-{losses}</div>
                </div>
                <div>
                    <div class="num" style="color:{pnl_col};font-size:1.25em;font-weight:800">{pnl:+.1f}u</div>
                    <div style="color:var(--ink-4);font-size:0.65em;font-weight:600;
                                text-transform:uppercase">net units</div>
                </div>
            </div>
        </div>""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 8px 0">
            <div style="font-size:1.15em;font-weight:900;letter-spacing:-0.01em;
                        background:linear-gradient(135deg,#f3f5f9 30%,#f5c518);
                        -webkit-background-clip:text;background-clip:text;
                        -webkit-text-fill-color:transparent">CFB EDGE</div>
            <div style="color:#67738a;font-size:0.64em;font-weight:700;
                        letter-spacing:0.14em;text-transform:uppercase;margin-top:3px">
                Model vs Market
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # CFB seasons span two calendar years (e.g. fall 2026 → Jan 2027 championship).
        # The API uses the start year (2026); the UI shows "2026-27" for clarity.
        # Before August, upcoming season = current year. Aug onward = current year.
        _today = date.today()
        _current_season = _today.year if _today.month >= 8 else _today.year
        season = st.selectbox(
            "Season",
            [_current_season],
            format_func=lambda y: f"{y}–{str(y + 1)[-2:]}",
            index=0,
        )
        week   = st.slider("Week", min_value=0, max_value=15, value=1)
        bettor = st.selectbox("Betting as", BETTORS,
                              index=BETTORS.index(st.session_state.get("bettor", BETTORS[0])))
        st.session_state["bettor"] = bettor
        st.divider()

        run = st.button("Load Picks", type="primary", width='stretch')
        if run:
            st.session_state["has_run"]    = True
            st.session_state["run_season"] = season
            st.session_state["run_week"]   = week

        # ── Bankroll calculator ───────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div style="color:#67738a;font-size:0.7em;font-weight:700;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">'
            'Bankroll</div>', unsafe_allow_html=True)
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=100, max_value=1_000_000,
            value=st.session_state.get("bankroll", 1000),
            step=100,
            label_visibility="collapsed",
            help="Your total betting bankroll. 1 unit = 1% of this amount.",
        )
        st.session_state["bankroll"] = bankroll
        unit_val = bankroll / 100
        st.markdown(
            f'<div style="color:#67738a;font-size:0.72em;margin-top:2px">'
            f'1u = <b style="color:#f3f5f9">${unit_val:,.0f}</b></div>',
            unsafe_allow_html=True)

        # Pending bets badge
        pending = [b for b in load_bets() if b["status"] == "Pending"]
        if pending:
            st.divider()
            n = len(pending)
            st.markdown(f'<div style="color:#f97316;font-size:0.82em;font-weight:600">'
                        f'{n} pending bet{"s" if n != 1 else ""}</div>', unsafe_allow_html=True)
            st.caption("Go to My Bets to mark results.")

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:12px 0 10px 0;margin-bottom:4px;
                display:flex;align-items:baseline;gap:14px">
        <span style="font-size:1.5em;font-weight:900;letter-spacing:-0.02em;
                     background:linear-gradient(135deg,#f3f5f9 30%,#f5c518);
                     -webkit-background-clip:text;background-clip:text;
                     -webkit-text-fill-color:transparent">CFB EDGE</span>
        <span style="color:#67738a;font-size:0.72em;font-weight:700;
                     letter-spacing:0.16em;text-transform:uppercase">
            Model vs Market</span>
        <span style="flex:1"></span>
        <span style="color:#454f63;font-size:0.7em;font-weight:600">
            SP+ · FPI · Elo · EPA · Weather</span>
    </div>
    """, unsafe_allow_html=True)

    picks_tab, bets_tab, standings_tab, research_tab, guide_tab = st.tabs([
        "This Week's Picks", "My Bets", "Season Standings", "Research", "How It Works"
    ])

    # ── MY BETS TAB ───────────────────────────────────────────────────────
    with bets_tab:
        render_bets_tab()
        st.markdown("---")
        with st.expander("📈 Closing Line Value Tracker"):
            render_clv_tab()

    # ── SEASON STANDINGS TAB ──────────────────────────────────────────────
    with standings_tab:
        render_standings_tab()

    # ── RESEARCH TAB (Backtester + Historical Picks + Model Analysis) ──────
    with research_tab:
        r1, r2, r3 = st.tabs(["Backtester", "Historical Picks", "Model Analysis"])
        with r1:
            render_backtester_tab()
        with r2:
            render_history_tab()
        with r3:
            render_analysis_tab()

    # ── HOW IT WORKS TAB ──────────────────────────────────────────────────
    with guide_tab:
        render_guide_tab()

    # ── PICKS TAB ─────────────────────────────────────────────────────────
    def render_core_equity_curve():
        """Out-of-sample equity curve of the CORE totals portfolio.
        Data: outputs/predictions/core_history.csv (built by
        scripts/build_core_history.py after each walk-forward rerun)."""
        hist_path = Path("outputs/predictions/core_history.csv")
        if not hist_path.exists():
            return
        import plotly.graph_objects as go
        h = pd.read_csv(hist_path)
        if h.empty:
            return
        h = h.reset_index(drop=True)
        # First bet of each season anchors the x-axis ticks
        season_starts = h.groupby("season").head(1)
        fig = go.Figure(go.Scatter(
            x=h.index, y=h["cum_units"], mode="lines",
            line=dict(color="#34d399", width=2),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
            customdata=h[["season", "week", "home_team", "away_team",
                          "over_under", "pnl"]],
            hovertemplate=("%{customdata[0]} wk %{customdata[1]} · "
                           "%{customdata[3]} @ %{customdata[2]}<br>"
                           "UNDER %{customdata[4]:.1f} → %{customdata[5]:+.1f}u"
                           "<br>Total: %{y:+.1f}u<extra></extra>"),
        ))
        final = float(h["cum_units"].iloc[-1])
        fig.add_annotation(x=h.index[-1], y=final, text=f"<b>{final:+.0f}u</b>",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(color="#f3f5f9", size=13))
        fig.update_layout(
            title=dict(text="CORE unders · cumulative units, out-of-sample 2019–25",
                       font=dict(size=13, color="#9aa5b8")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=240, margin=dict(l=8, r=52, t=36, b=8),
            xaxis=dict(tickvals=season_starts.index.tolist(),
                       ticktext=[str(int(s)) for s in season_starts["season"]],
                       showgrid=False, color="#67738a"),
            yaxis=dict(gridcolor="#1a2130", zerolinecolor="#232b3d",
                       color="#67738a", ticksuffix="u"),
            showlegend=False, hoverlabel=dict(bgcolor="#151a26"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with picks_tab:
        if not st.session_state.get("has_run"):
            st.html("""
            <div style="padding:36px 8px 28px 8px;text-align:center">
                <div style="font-size:0.68em;font-weight:800;letter-spacing:0.22em;
                            color:var(--gold);text-transform:uppercase;margin-bottom:10px">
                    SP+ · FPI · Elo · EPA · Weather · Portal</div>
                <div style="font-size:2em;font-weight:900;color:var(--ink);
                            letter-spacing:-0.02em;line-height:1.15">
                    Find where the model and<br>the market disagree.</div>
                <div style="color:var(--ink-3);font-size:0.92em;max-width:540px;
                            margin:14px auto 0 auto;line-height:1.6">
                    Every game gets a predicted score, total, and win probability from an
                    ensemble trained on seven seasons. Picks appear only where the model's
                    number diverges from Vegas — each one shows <i>why</i>, and how that
                    bet type has actually performed.</div>
            </div>""")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CORE unders", "54.9%", "559 bets · '19–'25 walk-forward", delta_color="off")
            col2.metric("CORE ROI", "+4.8%", "at -110 · profitable 6 of 7 seasons", delta_color="off")
            col3.metric("Spreads ATS", "~50%", "info only · no validated edge", delta_color="off")
            col4.metric("North star", "CLV", "beat the close", delta_color="off")
            render_core_equity_curve()
            st.html("""
            <div style="background:var(--card);border:1px solid var(--line);border-radius:12px;
                        padding:12px 18px;margin-top:14px;color:var(--ink-3);font-size:0.82em">
                ⚖️ <b style="color:var(--ink-2)">Honesty note:</b> the model does not beat the
                market overall — spreads run ~50% ATS and moneylines are a paper record. The one
                validated pocket is CORE unders (edge 2–7 pts, power-conf, wind &lt; 15, total ≥ 48),
                shown above out-of-sample. Only CORE picks carry units; every other card is research.</div>""")
            st.info("Select a season and week in the sidebar, then hit **Load Picks**.")
            return

        season = st.session_state.get("run_season", season)
        week   = st.session_state.get("run_week",   week)

        # ── Load everything ───────────────────────────────────────────────
        spread_model, totals_model, win_prob_model, feature_lists = load_models()
        if spread_model is None:
            st.error("Model files not found. Run `python3 src/model.py` first.")
            return

        ratings = load_team_ratings(season)
        elo     = load_current_elo(season)
        epa     = load_recent_epa(season)
        games   = fetch_schedule(season, week)

        if games.empty:
            st.warning(f"No games found for {season} Week {week}. Check back closer to the season.")
            return

        lines     = fetch_lines(games)
        has_lines = not lines.empty and lines["spread"].notna().any()

        with st.spinner("Fetching weather forecasts..."):
            games_with_wx = attach_weather_to_games(games)
            # Extract just the weather columns as a lookup table
            weather_df = games_with_wx[
                [c for c in ["game_id", "wind_speed", "is_dome"]
                 if c in games_with_wx.columns]
            ].copy()

        with st.spinner("Running models..."):
            preds = build_and_predict(games, lines, ratings, epa, elo,
                                      spread_model, totals_model, win_prob_model,
                                      feature_lists, weather=weather_df)

        # ── QB / availability adjustments ─────────────────────────────────
        # The model can't see injury news — its biggest info gap vs the
        # market. Mark teams whose starting QB is out and every dependent
        # number (spread, win prob, EV, edges) shifts accordingly.
        all_teams = sorted(set(preds["home_team"]) | set(preds["away_team"]))
        n_qb_out  = len(st.session_state.get("qb_out", []))
        with st.expander(f"🏥 QB & availability adjustments"
                         + (f" — {n_qb_out} active" if n_qb_out else ""),
                         expanded=False):
            st.caption(
                "Mark teams whose starting QB is **out or doubtful**. The market "
                "prices a backup QB at roughly 4–7 points; the model can't see "
                "injury news, so unadjusted it produces false edges on exactly "
                "these games. Adjusted picks carry a 🏥 chip.")
            qc1, qc2 = st.columns([3, 1])
            qb_out = qc1.multiselect("Teams with QB out / doubtful",
                                     all_teams, key="qb_out")
            qb_pts = qc2.slider("Points per team", 2.0, 9.0, 5.0, 0.5,
                                key="qb_pts",
                                help="How much to move each affected line. "
                                     "Star QB on a P4 team ≈ 6–7; game-manager ≈ 3–4.")
        if qb_out:
            preds = apply_qb_adjustments(preds, qb_out, qb_pts)

        # ── Feature coverage report ───────────────────────────────────────
        # Show which data sources are actually present for this week's games
        # so users know when predictions are flying partially blind.
        # Coverage computed inside build_and_predict on the full feature frame
        # (preds itself is trimmed to prediction columns — would show 0%)
        cov = preds.attrs.get("coverage") or feature_coverage_report(preds)
        COVERAGE_WARN = {"HFA", "Talent", "WEPA", "Havoc", "Portal", "Line Move"}
        missing = [g for g, pct in cov.items() if pct < 0.5 and g in COVERAGE_WARN]
        if missing:
            with st.expander(f"⚠️  Data coverage — {len(missing)} source(s) below 50%", expanded=False):
                st.caption("Sources with low coverage may reduce prediction accuracy.")
                cols = st.columns(4)
                for i, (group, pct) in enumerate(sorted(cov.items(), key=lambda x: x[1])):
                    color = "#53d337" if pct >= 0.8 else "#f0b429" if pct >= 0.5 else "#e53e3e"
                    cols[i % 4].markdown(
                        f"<div style='font-size:0.8em;color:#8b9bb4'>{group}</div>"
                        f"<div style='font-size:1em;font-weight:700;color:{color}'>{pct:.0%}</div>",
                        unsafe_allow_html=True,
                    )

        # ── Filter picks ──────────────────────────────────────────────────
        ml_bets = preds[
            preds["ml_ev"].notna() &
            (preds["ml_ev"] >= MONEYLINE_EV_MIN) &
            (preds["ml_ev"] <  MONEYLINE_EV_MAX)
        ].sort_values("ml_ev", ascending=False)

        tot_bets = preds[
            preds["totals_edge"].notna() &
            (preds["totals_edge"].abs() >= TOTALS_EDGE_MIN) &
            (preds["totals_edge"].abs() <= TOTALS_EDGE_MAX)
        ].sort_values("totals_edge", key=abs, ascending=False)

        # High-total paper fades (total >= 60, under, not already CORE). These
        # need no model edge, so they are collected separately from tot_bets —
        # the edge filter above would drop most of them. _force_under tells the
        # card renderer to show UNDER regardless of the model's direction.
        ht_bets = preds[
            preds["over_under"].notna() & (preds["over_under"] >= 60)
        ].copy()
        if not ht_bets.empty:
            ht_bets = ht_bets[~ht_bets.apply(_core_total, axis=1)]
        if not ht_bets.empty:
            ht_bets = ht_bets[~ht_bets["game_id"].isin(tot_bets["game_id"])]
            ht_bets["_force_under"] = True

        sp_bets = preds[
            preds["spread_edge"].notna() &
            (preds["spread_edge"].abs() >= SPREAD_EDGE_MIN) &
            (preds["spread_edge"].abs() <= SPREAD_EDGE_MAX)
        ]

        # ── Unified play ranking ──────────────────────────────────────────
        # Score every flagged pick on one 0–100 scale, weighted by how each
        # bet type actually performed on the 2025 holdout (unders > selective
        # ML > overs > spreads). The board surfaces the strongest leads first.
        plays: list[dict] = []
        for _, r in tot_bets.iterrows():
            # CORE unders (edge 2-7, power-conf, wind<15, total>=48) are the one
            # validated pocket — surface them at the top regardless of edge size,
            # since hit rate is flat-to-decreasing with edge (don't reward
            # magnitude, and high wind is excluded, not rewarded). Everything else
            # (overs, low-total/G5/big-edge unders) is research-only.
            if _core_total(r):
                score = 90.0
            else:
                base = min(abs(r["totals_edge"]) / TOTALS_EDGE_MAX, 1.0)
                score = 100 * base * 0.4
            plays.append({"kind": "total", "row": r, "score": score})
        # ht_bets are deliberately NOT added to `plays` — they are 0u paper
        # picks and get their own section, so they never occupy a ranked slot
        # or appear in the bankroll sizing table.
        for _, r in ml_bets.iterrows():
            base = min(float(r["ml_ev"]) / MONEYLINE_EV_MAX, 1.0)
            reliability = 0.8 if r["ml_book_odds"] <= 0 else 0.5
            plays.append({"kind": "ml", "row": r, "score": 100 * base * reliability})
        for _, r in sp_bets.iterrows():
            base = min(abs(r["spread_edge"]) / SPREAD_EDGE_MAX, 1.0)
            plays.append({"kind": "spread", "row": r, "score": 100 * base * 0.35})
        plays.sort(key=lambda p: p["score"], reverse=True)
        n_strong = sum(1 for p in plays if p["score"] >= 60)

        # ── Week header + summary tiles ───────────────────────────────────
        st.html(f"""
        <div style="display:flex;align-items:baseline;gap:12px;padding:14px 0 6px 0">
            <span style="color:var(--ink);font-size:1.3em;font-weight:900;letter-spacing:-0.02em">
                Week {week}</span>
            <span style="color:var(--gold);font-size:0.7em;font-weight:800;
                         letter-spacing:0.14em;text-transform:uppercase">{season} season</span>
            <span style="color:var(--ink-4);font-size:0.82em">{len(preds)} games ·
                {len(plays)} flagged · {n_strong} strong</span>
        </div>""")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strong Plays", n_strong, "score ≥ 60")
        col2.metric("Totals", len(tot_bets),
                    f"{(tot_bets['totals_edge'] < 0).sum()} unders" if not tot_bets.empty else "—",
                    delta_color="off")
        col3.metric("Moneylines", len(ml_bets))
        col4.metric("Spreads", len(sp_bets), "info only", delta_color="off")

        if not has_lines:
            st.warning("No Vegas lines yet — lines usually appear 7–10 days before kickoff.")

        # ── Two-column: picks feed (3/4) + bankroll panel (1/4) ──────────
        _feed_col, _panel_col = st.columns([3, 1], gap="large")
        with _panel_col:
            _render_right_panel(plays, n_strong, week)
        with _feed_col:
            view = st.radio(
                "View",
                ["Top Plays", "Totals", "Spreads", "Moneylines", "All Games"],
                horizontal=True,
                label_visibility="collapsed",
            )

            # ── Top Plays: one ranked board across all bet types ──────────
            if view == "Top Plays":
                if not plays:
                    st.info("No picks clear the edge thresholds this week — that's the model "
                            "telling you to keep your bankroll. Check All Games for projections.")
                else:
                    section_header("Top Plays",
                                   "Ranked by edge × how this bet type performs historically")
                    for rank, play in enumerate(plays[:8], start=1):
                        score = play["score"]
                        score_color = ("var(--green)" if score >= 60
                                       else "var(--gold)" if score >= 40 else "var(--ink-3)")
                        st.html(f"""
                        <div style="display:flex;align-items:center;gap:10px;margin:14px 0 4px 2px">
                            <span class="num" style="color:var(--ink-4);font-size:0.95em;
                                  font-weight:800">#{rank}</span>
                            <div style="flex:1;height:4px;border-radius:999px;background:var(--line);
                                        position:relative;max-width:160px">
                                <div style="position:absolute;left:0;top:0;bottom:0;border-radius:999px;
                                            width:{min(score, 100):.0f}%;background:{score_color}"></div>
                            </div>
                            <span class="num" style="color:{score_color};font-size:0.72em;
                                  font-weight:800">{score:.0f}</span>
                        </div>""")
                        if play["kind"] == "total":
                            render_totals_card(play["row"], season, week)
                        elif play["kind"] == "ml":
                            render_moneyline_card(play["row"], season, week)
                        else:
                            render_spread_card(play["row"], season, week)

                    # Paper experiment: kept out of the ranked board so it never
                    # competes with real plays, but visible so the tracked record
                    # is auditable week to week.
                    if not ht_bets.empty:
                        with st.expander(
                            f"🧪 Paper watchlist — {len(ht_bets)} high-total fade"
                            f"{'s' if len(ht_bets) != 1 else ''} (0u, tracking only)",
                            expanded=False,
                        ):
                            st.caption(
                                "Bet UNDER whenever the market total is ≥ 60, independent "
                                "of the model: 54.3% over 978 walk-forward bets (+3.6% ROI, "
                                "z=2.69), profitable 4 of 7 seasons. Thin and unproven live, "
                                "so it carries no units until a season corroborates it."
                            )
                            for _, row in ht_bets.iterrows():
                                render_totals_card(row, season, week)

            # ── Totals ────────────────────────────────────────────────────────
            elif view == "Totals":
                section_header("Totals", "Unders are the proven pocket — treat overs with caution")
                if tot_bets.empty:
                    st.info("No totals bets meet the threshold this week.")
                else:
                    under_bets = tot_bets[tot_bets["totals_edge"] < 0]
                    over_bets  = tot_bets[tot_bets["totals_edge"] > 0]
                    if not under_bets.empty:
                        st.markdown(
                            '<span style="color:#06b6d4;font-size:0.75em;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.1em">Unders</span>',
                            unsafe_allow_html=True)
                        for _, row in under_bets.iterrows():
                            render_totals_card(row, season, week)
                    if not over_bets.empty:
                        st.markdown(
                            '<span style="color:#f97316;font-size:0.75em;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.1em">Overs</span>',
                            unsafe_allow_html=True)
                        for _, row in over_bets.iterrows():
                            render_totals_card(row, season, week)

            # ── Spreads ───────────────────────────────────────────────────────
            elif view == "Spreads":
                section_header("Spreads", "Informational only · near breakeven")
                if sp_bets.empty:
                    st.info("No spread bets meet the threshold this week.")
                else:
                    for _, row in sp_bets.iterrows():
                        render_spread_card(row, season, week)

            # ── Moneylines ────────────────────────────────────────────────────
            elif view == "Moneylines":
                section_header("Moneylines", "Selective favorites held up in '25 — dogs are high variance")
                if not has_lines or preds["home_moneyline"].isna().all():
                    st.info("No moneyline data yet — appears closer to kickoff.")
                elif ml_bets.empty:
                    st.info("No +EV moneyline bets this week.")
                else:
                    dog_bets = ml_bets[ml_bets["ml_book_odds"] > 0]
                    fav_bets = ml_bets[ml_bets["ml_book_odds"] <= 0]
                    if not dog_bets.empty:
                        st.markdown(
                            '<span style="color:#3b82f6;font-size:0.75em;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.1em">Underdogs</span>',
                            unsafe_allow_html=True)
                        for _, row in dog_bets.iterrows():
                            render_moneyline_card(row, season, week)
                    if not fav_bets.empty:
                        st.markdown(
                            '<span style="color:#6b7280;font-size:0.75em;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.1em">Favorites</span>',
                            unsafe_allow_html=True)
                        for _, row in fav_bets.iterrows():
                            render_moneyline_card(row, season, week)

            # ── All Games ─────────────────────────────────────────────────────
            if view == "All Games":
                section_header(f"All Games — Week {week}",
                               "Expand any game to track a spread, total, or moneyline bet")

                search_col, clear_col = st.columns([4, 1])
                with search_col:
                    team_search = st.text_input(
                        "Search teams",
                        placeholder="e.g. Ohio State, Michigan, Alabama…",
                        label_visibility="collapsed",
                        key="team_search",
                    )
                with clear_col:
                    if st.button("Clear", key="clear_search", width='stretch'):
                        st.session_state["team_search"] = ""
                        st.rerun()

                query = team_search.strip().lower()
                if query:
                    filtered_preds = preds[
                        preds["home_team"].str.lower().str.contains(query, na=False) |
                        preds["away_team"].str.lower().str.contains(query, na=False)
                    ]
                    if filtered_preds.empty:
                        st.info(f'No games found matching "{team_search}" this week.')
                    else:
                        match_word = "game" if len(filtered_preds) == 1 else "games"
                        st.markdown(
                            f'<div style="color:#4b5563;font-size:0.8em;margin-bottom:6px">'
                            f'{len(filtered_preds)} {match_word} matching '
                            f'<span style="color:#ffffff">"{team_search}"</span></div>',
                            unsafe_allow_html=True,
                        )
                        for _, row in filtered_preds.iterrows():
                            render_all_game_card(row, season, week)
                else:
                    for _, row in preds.iterrows():
                        render_all_game_card(row, season, week)

        # ── Bankroll summary for this week ────────────────────────────────
        if plays:
            bankroll_val = st.session_state.get("bankroll", 1000)
            unit_val     = bankroll_val / 100
            with st.expander("💰 This week's bet sizes", expanded=False):
                st.caption(
                    f"Based on your bankroll of **${bankroll_val:,}** · 1u = ${unit_val:.0f} · "
                    f"Quarter-Kelly sizing · Max 3u per game"
                )
                bk_rows = []
                for play in plays[:8]:
                    r = play["row"]
                    if play["kind"] == "total":
                        is_under = r["totals_edge"] < 0
                        edge_abs = abs(r["totals_edge"])
                        u = kelly_units_spread(edge_abs)
                        side = "UNDER" if is_under else "OVER"
                        line_str = f"{r['over_under']:.1f}" if pd.notna(r.get("over_under")) else "TBD"
                        pick_str = f"{side} {line_str}"
                        bet_type = "Total"
                        edge_str = f"{r['totals_edge']:+.1f} pts"
                    elif play["kind"] == "ml":
                        u = kelly_units_ml(r["ml_ev"])
                        ml_odds = r["ml_book_odds"]
                        label   = f"+{int(ml_odds)}" if ml_odds > 0 else str(int(ml_odds))
                        pick_str = f"{r['ml_team']} ML {label}"
                        bet_type = "ML"
                        edge_str = f"EV {r['ml_ev']:+.1%}"
                    else:
                        u = 1
                        is_home = r["spread_edge"] > 0
                        team    = r["home_team"] if is_home else r["away_team"]
                        sp      = r["spread"]
                        sp_str  = (f"{sp:+.1f}" if is_home else f"{-sp:+.1f}") if pd.notna(sp) else ""
                        pick_str = f"{team} {sp_str}"
                        bet_type = "Spread"
                        edge_str = f"{r['spread_edge']:+.1f} pts"

                    bk_rows.append({
                        "Matchup": f"{r['away_team']} @ {r['home_team']}",
                        "Pick":    pick_str,
                        "Type":    bet_type,
                        "Edge":    edge_str,
                        "Kelly":   f"{u}u",
                        "$ Amount": f"${u * unit_val:,.0f}",
                    })

                if bk_rows:
                    bk_df = pd.DataFrame(bk_rows)
                    total_exposure = sum(int(r["Kelly"][0]) * unit_val for r in bk_rows)
                    st.dataframe(bk_df, width="stretch", hide_index=True)
                    st.markdown(
                        f'<div style="color:var(--ink-4);font-size:0.76em;margin-top:6px">'
                        f'Total exposure if all placed: '
                        f'<b style="color:var(--ink-2)">${total_exposure:,.0f}</b> '
                        f'({total_exposure / bankroll_val:.1%} of bankroll)</div>',
                        unsafe_allow_html=True
                    )

        st.markdown(
            '<div style="color:#4b5563;font-size:0.78em;padding:16px 0 8px 0">'
            'Always verify before betting — check injuries, weather, and current lines. '
            'This model is a tool, not a guarantee.'
            '</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
