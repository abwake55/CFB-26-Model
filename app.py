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
from model import EnsembleRegressor, EnsembleClassifier  # required so joblib can unpickle saved models
import __main__
__main__.EnsembleRegressor  = EnsembleRegressor   # joblib looks in __main__ when model was trained via python3 src/model.py
__main__.EnsembleClassifier = EnsembleClassifier
from feature_builder import (
    load_rating_sources,
    load_recent_epa    as _fb_load_recent_epa,
    load_current_elo   as _fb_load_current_elo,
    attach_team_features,
    feature_coverage_report,
)

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from difflib import SequenceMatcher
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
def _odds_api_key()     -> str: return get_secret("ODDSBLAZE_KEY", "")

CFB_BASE_URL  = "https://api.collegefootballdata.com"
ODDSBLAZE_BASE  = "https://data.oddsblaze.com/v1/odds"
ODDSBLAZE_BOOKS = ["draftkings_ncaaf", "fanduel_ncaaf", "betmgm_ncaaf", "caesars_ncaaf"]

ODDS_TO_CFBD = {
    "Louisiana State": "LSU",
    "Mississippi": "Ole Miss",
    "Southern California": "USC",
    "Central Florida": "UCF",
    "Southern Methodist": "SMU",
    "Texas Christian": "TCU",
    "Brigham Young": "BYU",
    "Nevada Las Vegas": "UNLV",
    "Massachusetts": "UMass",
    "Florida International": "FIU",
    "Middle Tennessee State": "Middle Tennessee",
    "North Carolina State": "NC State",
}

SPREAD_EDGE_MIN, SPREAD_EDGE_MAX = 4.0, 7.0
TOTALS_EDGE_MIN, TOTALS_EDGE_MAX = 3.0, 6.0
MONEYLINE_EV_MIN = 0.04
MONEYLINE_EV_MAX = 0.08

BETTORS = ["Alex", "Joe", "Zou", "Pat"]


# ─── KELLY CRITERION SIZING ───────────────────────────────────────────────────

def kelly_units_spread(edge_abs: float, fraction: float = 0.25) -> int:
    """
    Quarter-Kelly bet sizing for ATS bets at standard -110 juice.

    Empirical calibration: each point of spread edge ≈ 2% improvement
    in ATS cover probability beyond the 50% baseline.

    Full Kelly formula at -110:
        b = 100/110 ≈ 0.909 (net payout per unit)
        f = (p·b − q) / b  where q = 1 − p

    Uses quarter-Kelly (25%) as a conservative default. Capped at 4 units.
    """
    win_prob = min(0.50 + edge_abs * 0.02, 0.70)
    b = 100 / 110  # -110 payout
    kelly_f = max((win_prob * b - (1 - win_prob)) / b, 0.0)
    units = kelly_f * fraction * 100  # bankroll assumed = 100 units
    return max(1, min(4, round(units)))


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
    # Ensure both ensemble classes are findable in __main__ before joblib unpickling
    import sys
    sys.modules["__main__"].__dict__.setdefault("EnsembleRegressor",  EnsembleRegressor)
    sys.modules["__main__"].__dict__.setdefault("EnsembleClassifier", EnsembleClassifier)
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

    return df


@st.cache_data(show_spinner="Fetching odds...", ttl=1800)
def fetch_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches live lines from OddsBlaze (DraftKings primary, others as fallback).
    Falls back to CFBD lines API if OddsBlaze key is missing or returns no data.

    OddsBlaze response shape:
      { "sportsbook": {...}, "events": [
          { "teams": {"home": {"name": "..."}, "away": {"name": "..."}},
            "odds": [{"market": "Moneyline"|"Point Spread"|"Total Points",
                      "name": "<team or Over/Under>", "price": <int>, "line": <float>}]
          }, ...
      ]}
    """
    key = _odds_api_key()
    if key:
        try:
            data = None
            book_name = None
            for book_id in ODDSBLAZE_BOOKS:
                resp = requests.get(
                    f"{ODDSBLAZE_BASE}/{book_id}.json",
                    params={"key": key,
                            "market": "Moneyline,Point Spread,Total Points",
                            "main": "true", "price": "american"},
                    timeout=15)
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("events"):
                        data = payload
                        book_name = (payload.get("sportsbook") or {}).get("name", book_id)
                        break

            if data and data.get("events"):
                odds_rows = []
                for event in data["events"]:
                    teams    = event.get("teams", {})
                    home_raw = teams.get("home", {}).get("name", "")
                    away_raw = teams.get("away", {}).get("name", "")
                    home = ODDS_TO_CFBD.get(home_raw, home_raw)
                    away = ODDS_TO_CFBD.get(away_raw, away_raw)
                    spread = over_under = home_ml = away_ml = None
                    for odd in event.get("odds", []):
                        market = odd.get("market", "")
                        name   = odd.get("name", "")
                        price  = odd.get("price")
                        line   = odd.get("line")
                        if market == "Moneyline":
                            if name == home_raw:  home_ml = price
                            elif name == away_raw: away_ml = price
                        elif market == "Point Spread":
                            if name == home_raw and line is not None:
                                spread = line
                        elif market == "Total Points":
                            if "Over" in name and line is not None:
                                over_under = line
                    odds_rows.append({"odds_home": home, "odds_away": away,
                                      "spread": spread, "over_under": over_under,
                                      "home_moneyline": home_ml, "away_moneyline": away_ml,
                                      "provider": book_name})

                if odds_rows:
                    odds_df = pd.DataFrame(odds_rows)
                    def sim(a, b):
                        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
                    matched = []
                    cfbd = list(zip(games_df["game_id"], games_df["home_team"], games_df["away_team"]))
                    for _, r in odds_df.iterrows():
                        best_id, best_score = None, 0.0
                        for gid, ch, ca in cfbd:
                            score = (sim(r["odds_home"], ch) + sim(r["odds_away"], ca)) / 2
                            if score > best_score: best_score, best_id = score, gid
                        if best_score >= 0.70:
                            matched.append({"game_id": best_id,
                                            "spread": r["spread"],
                                            "over_under": r["over_under"],
                                            "home_moneyline": r["home_moneyline"],
                                            "away_moneyline": r["away_moneyline"],
                                            "spread_open": None,
                                            "provider": r["provider"]})
                    if matched:
                        return pd.DataFrame(matched).drop_duplicates("game_id")
        except Exception as exc:
            st.toast(f"OddsBlaze unavailable — using CFBD lines ({exc})", icon="⚠️")

    # ── Fallback: CFBD lines API ──────────────────────────────────────────────
    season = int(games_df["season"].iloc[0])
    week   = int(games_df["week"].iloc[0])
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception:
        return pd.DataFrame()
    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
    rank_map = {p: i for i, p in enumerate(priority)}
    rows = []
    for game in data:
        for line in game.get("lines", []):
            rows.append({"game_id": game.get("id"),
                         "spread": line.get("spread"),
                         "over_under": line.get("overUnder"),
                         "spread_open": line.get("spreadOpen"),
                         "provider": line.get("provider"),
                         "_rank": rank_map.get(line.get("provider", ""), 99)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.sort_values("_rank")
              .drop_duplicates("game_id", keep="first")
              .drop(columns=["_rank"]))


# ─── FEATURE BUILDING & PREDICTION ───────────────────────────────────────────

def build_and_predict(games, lines, ratings, epa, elo,
                      spread_model, totals_model, win_prob_model, feature_lists):
    """
    Merge lines onto games, build feature vectors via feature_builder, run
    the three models, and return a predictions DataFrame.
    """
    # ── Merge lines ───────────────────────────────────────────────────────
    if not lines.empty:
        ml_avail = [c for c in ["home_moneyline", "away_moneyline"] if c in lines.columns]
        line_cols = ["game_id", "spread", "over_under", "spread_open"] + ml_avail
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
    out_cols = ["game_id", "season", "week", "home_team", "away_team",
                "neutral_site", "conference_game", "spread", "over_under",
                "spread_open", "home_moneyline", "away_moneyline",
                "home_unrated", "away_unrated", "has_unrated_opponent"]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    if "provider" in df.columns:
        out["provider"] = df["provider"]

    out["pred_spread"]     = spread_model.predict(feat_sp)
    out["pred_total"]      = totals_model.predict(feat_tot)
    out["pred_win_p"]      = win_prob_model.predict_proba(feat_win)[:, 1]
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
    return out


# ─── UI HELPERS ───────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background-color: #0f1117;
        color: #ffffff;
    }
    [data-testid="stHeader"] {
        background-color: #0f1117;
        border-bottom: 1px solid #1e2537;
    }
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0b0e14;
        border-right: 1px solid #1e2537;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #6b7280 !important;
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0b0e14;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #1e2537;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        background: transparent;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88em;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1a1f2e;
        color: #ffffff;
    }
    /* ── Sub-nav radio as pills ── */
    .stRadio > label { display: none !important; }
    .stRadio > div {
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        padding: 6px 0 10px 0 !important;
    }
    .stRadio > div > label {
        background: #1a1f2e !important;
        border: 1px solid #252d3d !important;
        border-radius: 20px !important;
        padding: 5px 16px !important;
        cursor: pointer !important;
        font-size: 0.82em !important;
        font-weight: 600 !important;
        color: #6b7280 !important;
        margin: 0 !important;
        transition: all 0.15s !important;
    }
    .stRadio > div > label:has(input:checked) {
        background: #eab308 !important;
        border-color: #eab308 !important;
        color: #0f1117 !important;
    }
    .stRadio > div > label > div:first-child { display: none !important; }
    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #252d3d;
    }
    [data-testid="metric-container"] label {
        color: #6b7280 !important;
        font-size: 0.72em !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5em !important;
        font-weight: 700 !important;
    }
    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #1a1f2e;
        border: 1px solid #252d3d !important;
        border-radius: 10px !important;
        margin-bottom: 6px;
    }
    [data-testid="stExpander"] summary {
        color: #d1d5db;
        font-weight: 600;
        font-size: 0.92em;
    }
    [data-testid="stExpander"] summary:hover { color: #eab308; }
    /* ── Buttons ── */
    [data-testid="stButton"] > button {
        background: #1a1f2e;
        color: #6b7280;
        border: 1px solid #252d3d;
        border-radius: 6px;
        font-size: 0.8em;
        padding: 4px 12px;
        transition: all 0.15s;
    }
    [data-testid="stButton"] > button:hover {
        background: #252d3d;
        color: #ffffff;
        border-color: #374151;
    }
    [data-testid="stButton"] > button[kind="primary"] {
        background: #eab308;
        color: #0f1117;
        border: none;
        font-weight: 700;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #ca8a04;
    }
    /* ── Selects / Slider ── */
    [data-baseweb="select"] > div {
        background: #1a1f2e !important;
        border-color: #252d3d !important;
        color: #ffffff !important;
    }
    [data-testid="stSlider"] > div > div > div {
        background: #eab308 !important;
    }
    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: #1a1f2e;
        border-radius: 10px;
    }
    /* ── Dividers ── */
    hr { border-color: #252d3d !important; margin: 16px 0 !important; }
    /* ── Typography ── */
    h1, h2, h3, h4 { color: #ffffff !important; }
    p, li, .stMarkdown { color: #d1d5db; }
    [data-testid="stCaptionContainer"] { color: #4b5563 !important; }
    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: #1a1f2e;
        color: #6b7280;
        border: 1px solid #252d3d;
        border-radius: 6px;
    }
    /* ── Code blocks ── */
    code { background: #1a1f2e; color: #eab308; border-radius: 4px; padding: 1px 5px; }
    /* ── Text inputs ── */
    [data-testid="stTextInput"] input {
        background: #1a1f2e !important;
        border-color: #252d3d !important;
        color: #ffffff !important;
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

def ev_stars(ev: float) -> str:
    if ev >= 0.07: return "★★★"
    if ev >= 0.05: return "★★"
    return "★"

def track_button(label: str, game: str, bet_type: str, pick: str,
                 line: str, units: int, season: int, week: int, edge: str = "", key_prefix: str = ""):
    """Render a small Track button. Returns True if clicked."""
    key = f"{key_prefix}track_{game}_{bet_type}_{pick}".replace(" ", "_")
    bettor = st.session_state.get("bettor", BETTORS[0])
    if st.button(f"+ Track  {label}", key=key, use_container_width=False):
        add_bet(game, bet_type, pick, line, units, season, week, edge, bettor)
        st.toast(f"Added: {pick} — {bettor}", icon="✅")
        return True
    return False

def render_moneyline_card(row, season, week):
    ev      = row["ml_ev"]
    team    = row["ml_team"]
    book_ml = row["ml_book_odds"]
    mdl_ml  = row["ml_model_odds"]
    stars   = ev_stars(ev)
    is_dog  = book_ml > 0
    label   = f"+{int(book_ml)}" if is_dog else str(int(book_ml))
    model_label = (f"+{int(mdl_ml)}" if (not pd.isna(mdl_ml) and mdl_ml > 0)
                   else str(int(mdl_ml)) if not pd.isna(mdl_ml) else "—")
    dog_tag = "DOG" if is_dog else "FAV"
    matchup = f"{row['home_team']} vs {row['away_team']}"
    units   = kelly_units_ml(ev)
    ev_str  = f"{ev:+.1%}"

    # Yellow border = high-EV hot pick, blue = standard
    left_color = "#eab308" if ev >= 0.07 else "#3b82f6"
    ev_color   = "#22c55e" if ev >= 0.05 else "#9ca3af"

    st.html(f"""
    <div style="background:#1a1f2e;border-left:4px solid {left_color};
                border-top:1px solid #252d3d;border-right:1px solid #252d3d;
                border-bottom:1px solid #252d3d;
                border-radius:10px;padding:14px 18px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="background:#3b82f6;color:#ffffff;font-size:0.63em;
                             font-weight:800;letter-spacing:0.1em;padding:3px 8px;
                             border-radius:4px">MONEYLINE</span>
                <span style="background:#252d3d;color:#6b7280;font-size:0.63em;
                             font-weight:700;letter-spacing:0.08em;padding:3px 7px;
                             border-radius:4px">{dog_tag}</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="color:{ev_color};font-size:0.82em;font-weight:700">EV {ev_str}</span>
                <span style="color:#eab308;font-size:0.88em">{stars}</span>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:flex-center">
            <span style="color:#ffffff;font-size:1.1em;font-weight:700">{team}</span>
            <span style="color:{left_color};font-size:1.3em;font-weight:800;
                         font-variant-numeric:tabular-nums">{label}</span>
        </div>
        <div style="color:#4b5563;font-size:0.8em;margin-top:4px">{matchup}</div>
        <div style="display:flex;margin-top:12px;border-top:1px solid #252d3d;padding-top:10px">
            <div style="flex:1;text-align:center">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Book</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{label}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Model</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{model_label}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">EV</div>
                <div style="color:{ev_color};font-size:0.88em;font-weight:700">{ev_str}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Kelly</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{units}u</div>
            </div>
        </div>
    </div>
    """)
    track_button(f"{team} ML {label}", matchup, "Moneyline", f"{team} ML {label}",
                 label, units, season, week, f"EV {ev:+.1%}")

def render_totals_card(row, season, week):
    is_under = row["totals_edge"] < 0
    side_str = "UNDER" if is_under else "OVER"
    edge_abs = abs(row["totals_edge"])
    stars    = confidence_stars(edge_abs)
    units    = kelly_units_spread(edge_abs)
    matchup  = f"{row['home_team']} vs {row['away_team']}"
    ou_str   = f"{row['over_under']:.1f}" if pd.notna(row["over_under"]) else "TBD"
    neutral_tag = "  ·  Neutral" if row.get("neutral_site") else ""
    edge_str = f"{row['totals_edge']:+.1f}"

    # Under = cyan, Over = orange
    left_color = "#06b6d4" if is_under else "#f97316"
    edge_color = "#22c55e" if edge_abs >= 4.5 else "#9ca3af"

    st.html(f"""
    <div style="background:#1a1f2e;border-left:4px solid {left_color};
                border-top:1px solid #252d3d;border-right:1px solid #252d3d;
                border-bottom:1px solid #252d3d;
                border-radius:10px;padding:14px 18px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="background:{left_color};color:#0f1117;font-size:0.63em;
                             font-weight:800;letter-spacing:0.1em;padding:3px 8px;
                             border-radius:4px">{side_str}</span>
                <span style="background:#252d3d;color:#6b7280;font-size:0.63em;
                             font-weight:700;letter-spacing:0.08em;padding:3px 7px;
                             border-radius:4px">TOTAL</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="color:{edge_color};font-size:0.82em;font-weight:700">Edge {edge_str}</span>
                <span style="color:#eab308;font-size:0.88em">{stars}</span>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:#ffffff;font-size:1.1em;font-weight:700">{side_str} {ou_str}</span>
            <span style="color:{left_color};font-size:1.3em;font-weight:800;
                         font-variant-numeric:tabular-nums">{ou_str}</span>
        </div>
        <div style="color:#4b5563;font-size:0.8em;margin-top:4px">{matchup}{neutral_tag}</div>
        <div style="display:flex;margin-top:12px;border-top:1px solid #252d3d;padding-top:10px">
            <div style="flex:1;text-align:center">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Line</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{ou_str}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Model</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{row['pred_total']:.1f}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Edge</div>
                <div style="color:{edge_color};font-size:0.88em;font-weight:700">{edge_str} pts</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Kelly</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{units}u</div>
            </div>
        </div>
    </div>
    """)
    track_button(f"{side_str} {ou_str}", matchup, "Total",
                 f"{side_str} {ou_str}", ou_str, units, season, week,
                 f"{row['totals_edge']:+.1f} pts")


def render_spread_card(row, season, week):
    """Sportsbook-style spread card (reference only)."""
    is_home  = row["spread_edge"] > 0
    bet_on   = row["home_team"] if is_home else row["away_team"]
    edge     = row["spread_edge"]
    spread   = row["spread"]
    pred_sp  = row["pred_spread"]
    matchup  = f"{row['home_team']} vs {row['away_team']}"
    stars    = confidence_stars(abs(edge))
    edge_str = f"{edge:+.1f}"

    # Vegas line from bet_on's perspective
    if pd.notna(spread):
        vl_bet = f"{spread:+.1f}" if is_home else f"{-spread:+.1f}"
    else:
        vl_bet = "N/A"

    # Model line from bet_on's perspective
    if pd.notna(pred_sp):
        mdl_str = f"{-pred_sp:+.1f}" if is_home else f"{pred_sp:+.1f}"
    else:
        mdl_str = "—"

    edge_color = "#22c55e" if abs(edge) >= 5.5 else "#9ca3af"

    st.html(f"""
    <div style="background:#1a1f2e;border-left:4px solid #8b5cf6;
                border-top:1px solid #252d3d;border-right:1px solid #252d3d;
                border-bottom:1px solid #252d3d;
                border-radius:10px;padding:14px 18px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="background:#8b5cf6;color:#ffffff;font-size:0.63em;
                             font-weight:800;letter-spacing:0.1em;padding:3px 8px;
                             border-radius:4px">SPREAD</span>
                <span style="background:#252d3d;color:#6b7280;font-size:0.63em;
                             font-weight:700;letter-spacing:0.08em;padding:3px 7px;
                             border-radius:4px">REF ONLY</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="color:{edge_color};font-size:0.82em;font-weight:700">Edge {edge_str}</span>
                <span style="color:#eab308;font-size:0.88em">{stars}</span>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:#ffffff;font-size:1.1em;font-weight:700">{bet_on}</span>
            <span style="color:#8b5cf6;font-size:1.3em;font-weight:800;
                         font-variant-numeric:tabular-nums">{vl_bet}</span>
        </div>
        <div style="color:#4b5563;font-size:0.8em;margin-top:4px">{matchup}</div>
        <div style="display:flex;margin-top:12px;border-top:1px solid #252d3d;padding-top:10px">
            <div style="flex:1;text-align:center">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Vegas</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{vl_bet}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Model</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">{mdl_str}</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Edge</div>
                <div style="color:{edge_color};font-size:0.88em;font-weight:700">{edge_str} pts</div>
            </div>
            <div style="flex:1;text-align:center;border-left:1px solid #252d3d">
                <div style="color:#4b5563;font-size:0.63em;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px">Kelly</div>
                <div style="color:#e5e7eb;font-size:0.88em;font-weight:700">1u</div>
            </div>
        </div>
    </div>
    """)
    track_button(f"{bet_on} {vl_bet}", matchup, "Spread",
                 f"{bet_on} {vl_bet}", vl_bet, 1, season, week, edge_str)


# ─── MY BETS TAB ──────────────────────────────────────────────────────────────

def render_bets_tab():
    bets = load_bets()

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

    # ── Filters ──────────────────────────────────────────────────────────
    section_header("Bet History")
    col_f1, col_f2, _ = st.columns([1, 1, 2])
    status_filter = col_f1.selectbox("Status", ["All", "Pending", "Won", "Lost", "Push"])
    bettor_filter = col_f2.selectbox("Bettor",  ["All"] + BETTORS)

    filtered = bets
    if status_filter != "All":
        filtered = [b for b in filtered if b["status"] == status_filter]
    if bettor_filter != "All":
        filtered = [b for b in filtered if b.get("bettor", "") == bettor_filter]

    if not filtered:
        st.info("No bets match the selected filters.")
        return

    # ── Bet rows ─────────────────────────────────────────────────────────
    status_left = {"Pending": "#252d3d", "Won": "#22c55e", "Lost": "#ef4444", "Push": "#f97316"}
    status_bg   = {"Pending": "#1a1f2e", "Won": "#0f1f14", "Lost": "#1f0f0f", "Push": "#1f1608"}
    status_label = {"Pending": "PENDING", "Won": "WON", "Lost": "LOST", "Push": "PUSH"}

    for bet in reversed(filtered):
        left    = status_left.get(bet["status"], "#2d3340")
        bg      = status_bg.get(bet["status"],   "#23272b")
        slabel  = status_label.get(bet["status"], bet["status"])
        pnl     = bet_pnl(bet)
        pnl_str = f"{pnl:+.2f}u" if bet["status"] != "Pending" else "—"
        pnl_col = "#53d337" if pnl > 0 else "#e74c3c" if pnl < 0 else "#8b9bb4"
        bettor  = bet.get("bettor", "—")
        edge_tag = f" · {bet['edge']}" if bet.get("edge") else ""

        clv      = compute_clv(bet)
        clv_col  = "#53d337" if (clv or 0) > 0 else "#e74c3c" if (clv or 0) < 0 else "#8b9bb4"
        clv_unit = "ppts" if bet.get("bet_type") == "Moneyline" else "pts"
        clv_html = (
            f'<span style="color:{clv_col};font-size:0.78em;font-weight:700;'
            f'margin-left:10px">CLV {clv:+.1f}{clv_unit}</span>'
            if clv is not None else ""
        )

        st.html(f"""
        <div style="background:{bg};border-left:3px solid {left};
                    border-top:1px solid #2d3340;border-right:1px solid #2d3340;
                    border-bottom:1px solid #2d3340;
                    border-radius:8px;padding:12px 16px;margin-bottom:4px;">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span style="color:#ffffff;font-weight:700;font-size:0.98em">{bet['pick']}</span>
                    <span style="background:#2d3340;color:#8b9bb4;font-size:0.68em;
                                 font-weight:700;letter-spacing:0.07em;padding:2px 6px;
                                 border-radius:4px;margin-left:8px">{bet['bet_type'].upper()}</span>
                    <span style="color:#8b9bb4;font-size:0.82em;margin-left:8px">{bet['units']}u</span>
                    <span style="color:#5c6680;font-size:0.8em;margin-left:8px">{bettor}</span>
                    {clv_html}
                </div>
                <div style="text-align:right">
                    <span style="color:{pnl_col};font-weight:700;font-size:0.95em">{pnl_str}</span>
                    <span style="background:{left};color:#1a1d21;font-size:0.68em;
                                 font-weight:700;letter-spacing:0.07em;padding:2px 7px;
                                 border-radius:4px;margin-left:8px">{slabel}</span>
                </div>
            </div>
            <div style="color:#8b9bb4;font-size:0.82em;margin-top:5px">{bet['game']}</div>
            <div style="color:#5c6680;font-size:0.76em;margin-top:2px">
                Wk {bet['week']} · {bet['date']} · Line: {bet['line']}{edge_tag}
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 8px 0">
            <div style="color:#ffffff;font-size:1.1em;font-weight:800;
                        letter-spacing:0.02em">CFB Picks</div>
            <div style="color:#eab308;font-size:0.68em;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">
                SP+ · FPI · Elo · EPA
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

        run = st.button("Load Picks", type="primary", use_container_width=True)
        if run:
            st.session_state["has_run"]    = True
            st.session_state["run_season"] = season
            st.session_state["run_week"]   = week

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
    <div style="padding:10px 0 6px 0;border-bottom:1px solid #1e2537;margin-bottom:6px;
                display:flex;align-items:baseline;gap:14px">
        <span style="color:#ffffff;font-size:1.4em;font-weight:800;
                     letter-spacing:0.01em">CFB Picks</span>
        <span style="color:#eab308;font-size:0.75em;font-weight:700;
                     letter-spacing:0.08em;text-transform:uppercase">Model-Powered</span>
        <span style="color:#252d3d;flex:1;height:1px;display:inline-block;
                     vertical-align:middle;margin-left:4px"></span>
    </div>
    """, unsafe_allow_html=True)

    picks_tab, bets_tab = st.tabs(["This Week's Picks", "My Bets"])

    # ── MY BETS TAB ───────────────────────────────────────────────────────
    with bets_tab:
        render_bets_tab()

    # ── PICKS TAB ─────────────────────────────────────────────────────────
    with picks_tab:
        if not st.session_state.get("has_run"):
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Totals Win Rate", "54.7%", "+2.3% above breakeven")
            col2.metric("Under Win Rate",  "59.0%", "Primary edge")
            col3.metric("Spread Win Rate", "50.4%", "Informational only")
            col4.metric("ML EV Min",       "4%",    "Threshold")

            section_header("How It Works")
            st.markdown(
                "The model predicts a final score and compares it to the Vegas line. "
                "When they disagree by 3–6 points on a total, or 4–8% on a moneyline, "
                "it flags a bet. **Unders win 59% historically** — the primary edge. "
                "Spreads are near breakeven and shown for reference only. "
                "Always check injuries and current lines before placing a bet."
            )
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

        with st.spinner("Running models..."):
            preds = build_and_predict(games, lines, ratings, epa, elo,
                                      spread_model, totals_model, win_prob_model,
                                      feature_lists)

        # ── Feature coverage report ───────────────────────────────────────
        # Show which data sources are actually present for this week's games
        # so users know when predictions are flying partially blind.
        cov = feature_coverage_report(preds)
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

        sp_bets = preds[
            preds["spread_edge"].notna() &
            (preds["spread_edge"].abs() >= SPREAD_EDGE_MIN) &
            (preds["spread_edge"].abs() <= SPREAD_EDGE_MAX)
        ]

        # ── Week header + summary tiles ───────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;align-items:baseline;gap:10px;padding:12px 0 4px 0">
            <span style="color:#ffffff;font-size:1.05em;font-weight:700">
                {season} · Week {week}
            </span>
            <span style="color:#4b5563;font-size:0.82em">{len(preds)} games</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Moneyline Bets", len(ml_bets))
        col2.metric("Totals Bets",    len(tot_bets))
        col3.metric("Spread Bets",    len(sp_bets))
        col4.metric("Model Accuracy", "54.7%", "Totals win rate")

        if not has_lines:
            st.warning("No Vegas lines yet — lines usually appear 7–10 days before kickoff.")

        # ── Sub-navigation pill tabs ──────────────────────────────────────
        best_tot = tot_bets.head(3) if not tot_bets.empty else pd.DataFrame()
        best_ml  = ml_bets.head(2)  if not ml_bets.empty  else pd.DataFrame()
        has_best = not best_tot.empty or not best_ml.empty

        view = st.radio(
            "View",
            ["Best Bets", "Totals", "Spreads", "Moneylines", "All Games"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # ── Best Bets ─────────────────────────────────────────────────────
        if view == "Best Bets":
            if not has_best:
                st.info("No high-confidence picks this week.")
            else:
                section_header("Best Bets", "Highest-confidence picks this week")
                if not best_tot.empty:
                    for _, row in best_tot.iterrows():
                        render_totals_card(row, season, week)
                if not best_ml.empty:
                    for _, row in best_ml.iterrows():
                        render_moneyline_card(row, season, week)

        # ── Totals ────────────────────────────────────────────────────────
        elif view == "Totals":
            section_header("Totals", "Unders win 59% historically")
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
            section_header("Moneylines", "Underdogs drive +52.7% historical ROI")
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
                if st.button("Clear", key="clear_search", use_container_width=True):
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

        st.markdown(
            '<div style="color:#4b5563;font-size:0.78em;padding:16px 0 8px 0">'
            'Always verify before betting — check injuries, weather, and current lines. '
            'This model is a tool, not a guarantee.'
            '</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
