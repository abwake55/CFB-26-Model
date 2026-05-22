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
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from difflib import SequenceMatcher

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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

# ─── API KEYS (from Streamlit secrets, fall back to env vars) ─────────────────

def get_secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, fallback)

CFB_API_KEY       = get_secret("CFB_API_KEY",
    "uxvnvwwBh6dQBE/hxA+GK+srmnfZ1mkRSr8E7gOg/BuIL/TeNHw5aHbbZDbi4TMt")
ODDS_API_KEY      = get_secret("ODDS_API_KEY", "97fefeb9de733240ae640967ed5c1427")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", "")

CFB_BASE_URL  = "https://api.collegefootballdata.com"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NCAAF_SPORT   = "americanfootball_ncaaf"
BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

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

SPREAD_EDGE_MIN, SPREAD_EDGE_MAX = 2.0, 5.0
TOTALS_EDGE_MIN, TOTALS_EDGE_MAX = 3.0, 6.0
MONEYLINE_EV_MIN = 0.04   # 4% minimum EV — below this is noise
MONEYLINE_EV_MAX = 0.08   # 8% max EV — above this model is likely overconfident


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
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    resp = requests.get(f"{CFB_BASE_URL}/{endpoint}",
                        headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─── CACHED LOADERS ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_models():
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
    import ast
    ratings = {}

    sp_path = DATA_DIR / "master_sp_ratings.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)

        def safe_parse(val):
            if pd.isna(val): return {}
            if isinstance(val, dict): return val
            try: return ast.literal_eval(val)
            except: return {}

        sp["off_dict"]  = sp["offense"].apply(safe_parse)
        sp["def_dict"]  = sp["defense"].apply(safe_parse)
        sp["sp_offense"] = sp["off_dict"].apply(lambda d: d.get("rating"))
        sp["sp_defense"] = sp["def_dict"].apply(lambda d: d.get("rating"))
        year_col = "year" if "year" in sp.columns else "season"
        sp = sp.rename(columns={year_col: "season", "rating": "sp_rating"})
        sp["season"] = sp["season"] + 1
        ratings["sp"] = sp[sp["season"] == pred_season][
            ["team","sp_rating","sp_offense","sp_defense"]].set_index("team")

    for key, fname, col in [
        ("fpi", "master_fpi_ratings.csv", "fpi"),
        ("srs", "master_srs_ratings.csv", "srs"),
    ]:
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            if "school" in df.columns: df = df.rename(columns={"school": "team"})
            if "year"   in df.columns: df = df.rename(columns={"year": "season"})
            if "rating" in df.columns and col == "srs":
                df = df.rename(columns={"rating": "srs"})
            df["season"] = pd.to_numeric(df["season"], errors="coerce") + 1
            if col in df.columns:
                ratings[key] = df[df["season"] == pred_season][
                    ["team", col]].set_index("team")

    rec_path = DATA_DIR / "master_recruiting.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rec.columns = [c.lower() for c in rec.columns]
        if "points" not in rec.columns and "total" in rec.columns:
            rec = rec.rename(columns={"total": "points"})
        rec = rec.sort_values(["team","year"])
        rec["recruiting_4yr"] = rec.groupby("team")["points"].transform(
            lambda x: x.rolling(4, min_periods=1).mean())
        ratings["recruiting"] = rec[rec["year"] == pred_season - 1][
            ["team","recruiting_4yr"]].set_index("team")

    fm_path = DATA_DIR / "feature_matrix.csv"
    if fm_path.exists():
        try:
            fm = pd.read_csv(fm_path, usecols=["season","home_team","home_hfa"])
            fm = fm[fm["season"] == pred_season - 1]
            if not fm.empty:
                hfa = fm.groupby("home_team")["home_hfa"].last().reset_index()
                hfa.columns = ["team","hfa_estimate"]
                ratings["hfa"] = hfa.set_index("team")
        except Exception:
            pass

    return ratings


@st.cache_data(show_spinner="Computing Elo ratings...", ttl=86400)
def load_current_elo(pred_season: int) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT_DIR / "src"))
    try:
        from elo_ratings import EloSystem
    except ImportError:
        return pd.DataFrame(columns=["elo"])

    games_path = DATA_DIR / "master_games.csv"
    sp_path    = DATA_DIR / "master_sp_ratings.csv"
    if not games_path.exists():
        return pd.DataFrame(columns=["elo"])

    games = pd.read_csv(games_path)
    if sp_path.exists():
        sp  = pd.read_csv(sp_path)
        fbs = set(sp["team"].unique())
        games = games[games["home_team"].isin(fbs) & games["away_team"].isin(fbs)]
    games = games[games["season"] <= pred_season - 1].dropna(
        subset=["home_points","away_points"])

    elo = EloSystem()
    elo.run(games)
    return elo.current_ratings_df().set_index("team")[["elo"]]


@st.cache_data(show_spinner="Loading recent form...", ttl=86400)
def load_recent_epa(pred_season: int) -> pd.DataFrame:
    ppa_path = DATA_DIR / "master_ppa_games.csv"
    if not ppa_path.exists():
        return pd.DataFrame()
    ppa = pd.read_csv(ppa_path)
    last = ppa[ppa["season"] == pred_season - 1].sort_values(["team","week"])
    if last.empty:
        return pd.DataFrame()
    cols = ["off_epa","def_epa","off_epa_pass","off_epa_rush"]
    last3 = last.groupby("team").tail(3).groupby("team")[cols].mean()
    last3.columns = ["off_epa_roll3","def_epa_roll3","off_epa_pass_roll3","off_epa_rush_roll3"]
    last5 = last.groupby("team").tail(5).groupby("team")[cols].mean()
    last5.columns = ["off_epa_roll5","def_epa_roll5","off_epa_pass_roll5","off_epa_rush_roll5"]
    return last3.join(last5, how="outer")


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
    return df


@st.cache_data(show_spinner="Fetching odds...", ttl=1800)
def fetch_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """Try Odds API first, fall back to CFBD."""
    # ── Odds API ──────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{NCAAF_SPORT}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american", "dateFormat": "iso"},
            timeout=15)
        resp.raise_for_status()
        data = resp.json()

        odds_rows = []
        for game in data:
            home_raw = game["home_team"]
            away_raw = game["away_team"]
            home = ODDS_TO_CFBD.get(home_raw, home_raw)
            away = ODDS_TO_CFBD.get(away_raw, away_raw)
            bookmakers = {b["key"]: b for b in game.get("bookmakers", [])}
            book = next((bookmakers[k] for k in BOOK_PRIORITY if k in bookmakers),
                        next(iter(bookmakers.values()), None) if bookmakers else None)
            spread = over_under = home_ml = away_ml = None
            book_name = None
            if book:
                book_name = book.get("title", book.get("key"))
                for mkt in book.get("markets", []):
                    if mkt["key"] == "spreads":
                        for o in mkt["outcomes"]:
                            if o["name"] == home_raw: spread = o["point"]
                    elif mkt["key"] == "totals":
                        if mkt["outcomes"]: over_under = mkt["outcomes"][0]["point"]
                    elif mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == home_raw: home_ml = o["price"]
                            elif o["name"] == away_raw: away_ml = o["price"]
            odds_rows.append({"odds_home": home, "odds_away": away,
                               "spread": spread, "over_under": over_under,
                               "home_moneyline": home_ml, "away_moneyline": away_ml,
                               "provider": book_name})

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
                matched.append({"game_id": best_id, "spread": r["spread"],
                                 "over_under": r["over_under"],
                                 "home_moneyline": r["home_moneyline"],
                                 "away_moneyline": r["away_moneyline"],
                                 "spread_open": None,
                                 "provider": r["provider"]})
        if matched:
            return pd.DataFrame(matched).drop_duplicates("game_id")
    except Exception:
        pass

    # ── CFBD fallback ─────────────────────────────────────────────────────
    season = int(games_df["season"].iloc[0])
    week   = int(games_df["week"].iloc[0])
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception:
        return pd.DataFrame()
    priority = ["consensus","Bovada","DraftKings","ESPN Bet"]
    rank_map = {p: i for i, p in enumerate(priority)}
    rows = []
    for game in data:
        for line in game.get("lines", []):
            rows.append({"game_id": game.get("id"),
                         "spread": line.get("spread"),
                         "over_under": line.get("overUnder"),
                         "spread_open": line.get("spreadOpen"),
                         "provider": line.get("provider"),
                         "_rank": rank_map.get(line.get("provider",""), 99)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.sort_values("_rank")
              .drop_duplicates("game_id", keep="first")
              .drop(columns=["_rank"]))


# ─── FEATURE BUILDING ─────────────────────────────────────────────────────────

def build_and_predict(games, lines, ratings, epa, elo,
                      spread_model, totals_model, win_prob_model, feature_lists):
    if not lines.empty:
        ml_avail = [c for c in ["home_moneyline","away_moneyline"] if c in lines.columns]
        line_cols = ["game_id","spread","over_under","spread_open"] + ml_avail
        if "provider" in lines.columns:
            line_cols.append("provider")
        df = games.merge(lines[[c for c in line_cols if c in lines.columns]],
                         on="game_id", how="left")
    else:
        df = games.copy()
        df["spread"] = df["over_under"] = df["spread_open"] = np.nan

    if "home_moneyline" not in df.columns: df["home_moneyline"] = np.nan
    if "away_moneyline" not in df.columns: df["away_moneyline"] = np.nan

    def get_r(team, src_key, col):
        src = ratings.get(src_key)
        if src is None or team not in src.index: return np.nan
        if col not in src.columns: return np.nan
        val = src.loc[team, col]
        if isinstance(val, pd.Series): val = val.iloc[0]
        try: return np.nan if pd.isna(val) else float(val)
        except: return np.nan

    for side, tcol in [("home","home_team"), ("away","away_team")]:
        ts = df[tcol]
        df[f"{side}_sp_rating"]      = ts.map(lambda t: get_r(t,"sp","sp_rating"))
        df[f"{side}_sp_offense"]     = ts.map(lambda t: get_r(t,"sp","sp_offense"))
        df[f"{side}_sp_defense"]     = ts.map(lambda t: get_r(t,"sp","sp_defense"))
        df[f"{side}_fpi"]            = ts.map(lambda t: get_r(t,"fpi","fpi"))
        df[f"{side}_srs"]            = ts.map(lambda t: get_r(t,"srs","srs"))
        df[f"{side}_recruiting_4yr"] = ts.map(lambda t: get_r(t,"recruiting","recruiting_4yr"))
        df[f"{side}_hfa"]            = ts.map(lambda t: get_r(t,"hfa","hfa_estimate"))
        df[f"{side}_pregame_elo"]    = ts.map(
            lambda t: float(elo.loc[t,"elo"]) if (not elo.empty and t in elo.index) else np.nan)
        if not epa.empty:
            for col in epa.columns:
                df[f"{side}_{col}"] = ts.map(
                    lambda t, c=col: float(epa.loc[t,c]) if t in epa.index else np.nan)

    df["sp_diff"]      = df["home_sp_rating"]   - df["away_sp_rating"]
    df["sp_off_diff"]  = df["home_sp_offense"]  - df["away_sp_offense"]
    df["sp_def_diff"]  = df["home_sp_defense"]  - df["away_sp_defense"]
    df["elo_diff"]     = df["home_pregame_elo"] - df["away_pregame_elo"]
    df["fpi_diff"]     = df["home_fpi"]         - df["away_fpi"]
    df["srs_diff"]     = df["home_srs"]         - df["away_srs"]
    df["recruiting_diff"] = df["home_recruiting_4yr"] - df["away_recruiting_4yr"]
    df["hfa_diff"]     = df["home_hfa"].fillna(0) - df["away_hfa"].fillna(0)
    if "home_off_epa_roll3" in df.columns:
        df["epa_off_diff_roll3"] = df["home_off_epa_roll3"] - df["away_off_epa_roll3"]
        df["epa_def_diff_roll3"] = df["home_def_epa_roll3"] - df["away_def_epa_roll3"]
    df["home_rest_days"] = df["away_rest_days"] = 14
    df["rest_diff"]      = 0
    df["spread"]         = pd.to_numeric(df["spread"], errors="coerce")
    df["over_under"]     = pd.to_numeric(df["over_under"], errors="coerce")
    df["spread_open"]    = pd.to_numeric(df.get("spread_open", pd.Series(dtype=float)), errors="coerce")
    df["line_movement"]  = df["spread"] - df["spread_open"]
    df["line_moved_home"] = (df["line_movement"] < -1.0).astype(int)
    df["line_moved_away"] = (df["line_movement"] >  1.0).astype(int)
    df["vegas_home_margin"] = -df["spread"].fillna(0)

    def make_feat(feat_names):
        out = pd.DataFrame(index=df.index)
        for f in feat_names:
            out[f] = df[f] if f in df.columns else np.nan
        return out

    feat_sp  = make_feat(feature_lists["spread"])
    feat_tot = make_feat(feature_lists["totals"])
    feat_win = make_feat(feature_lists["win_prob"])

    out = df[["game_id","season","week","home_team","away_team",
              "neutral_site","conference_game","spread","over_under",
              "spread_open","home_moneyline","away_moneyline"]].copy()
    if "provider" in df.columns:
        out["provider"] = df["provider"]

    out["pred_spread"]    = spread_model.predict(feat_sp)
    out["pred_total"]     = totals_model.predict(feat_tot)
    out["pred_win_p"]     = win_prob_model.predict_proba(feat_win)[:, 1]
    out["pred_away_win_p"] = 1 - out["pred_win_p"]
    out["spread_edge"]    = out["pred_spread"] - (-out["spread"])
    out["totals_edge"]    = out["pred_total"]  - out["over_under"]

    # ── Moneyline EV ─────────────────────────────────────────────────────
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

    out[["ml_team","ml_ev","ml_book_odds","ml_model_odds"]] = out.apply(best_ml, axis=1)

    return out


# ─── UI ──────────────────────────────────────────────────────────────────────

def confidence_stars(edge_abs: float) -> str:
    if edge_abs >= 5.5: return "⭐⭐⭐"
    if edge_abs >= 4.5: return "⭐⭐"
    return "⭐"

def ev_stars(ev: float) -> str:
    if ev >= 0.07: return "⭐⭐⭐"
    if ev >= 0.05: return "⭐⭐"
    return "⭐"

def render_moneyline_card(row):
    ev      = row["ml_ev"]
    team    = row["ml_team"]
    book_ml = row["ml_book_odds"]
    mdl_ml  = row["ml_model_odds"]
    stars   = ev_stars(ev)
    is_dog  = book_ml > 0
    label   = f"+{int(book_ml)}" if is_dog else str(int(book_ml))
    model_label = f"+{int(mdl_ml)}" if (not pd.isna(mdl_ml) and mdl_ml > 0) else str(int(mdl_ml)) if not pd.isna(mdl_ml) else "—"
    color   = "#1a3a5c" if ev >= 0.07 else "#1c2b3a"
    border  = "#3498db" if ev >= 0.07 else "#5b7fa6"
    dog_tag = " 🐶 Underdog" if is_dog else " 🏆 Favorite"

    matchup = f"{row['home_team']} vs {row['away_team']}"

    st.markdown(f"""
    <div style="background:{color};border-left:4px solid {border};
                border-radius:8px;padding:14px 18px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="color:#3498db;font-size:1.1em;font-weight:700">{team}</span>
                <span style="color:#aaa;font-size:0.82em;margin-left:8px">{label}{dog_tag}</span>
                <span style="color:#aaa;font-size:0.85em;margin-left:8px">{stars}</span>
            </div>
            <div style="color:#ccc;font-size:0.9em">EV: <b style="color:#3498db">{ev:+.1%}</b></div>
        </div>
        <div style="color:#ddd;margin-top:6px;font-size:1em">{matchup}</div>
        <div style="color:#aaa;font-size:0.82em;margin-top:4px">
            Model implied: <b>{model_label}</b>
            &nbsp;·&nbsp; Book: <b>{label}</b>
            &nbsp;·&nbsp; Edge: model gives {team} a higher win% than the book
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_totals_card(row):
    side     = "UNDER 🔽" if row["totals_edge"] < 0 else "OVER 🔼"
    edge_abs = abs(row["totals_edge"])
    stars    = confidence_stars(edge_abs)
    color    = "#1a4d2e" if edge_abs >= 5.0 else "#2d3a1e"
    border   = "#2ecc71" if edge_abs >= 5.0 else "#a8d08d"

    st.markdown(f"""
    <div style="background:{color};border-left:4px solid {border};
                border-radius:8px;padding:14px 18px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="color:#2ecc71;font-size:1.1em;font-weight:700">{side} {row['over_under']:.1f}</span>
                <span style="color:#aaa;font-size:0.85em;margin-left:10px">{stars}</span>
            </div>
            <div style="color:#ccc;font-size:0.9em">Edge: <b style="color:#2ecc71">{row['totals_edge']:+.1f} pts</b></div>
        </div>
        <div style="color:#ddd;margin-top:6px;font-size:1em">
            {row['home_team']} <span style="color:#888">vs</span> {row['away_team']}
        </div>
        <div style="color:#aaa;font-size:0.82em;margin-top:4px">
            Model: {row['pred_total']:.1f} pts total
            {"&nbsp;·&nbsp;Neutral site" if row.get("neutral_site") else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── CHAT HELPERS ─────────────────────────────────────────────────────────────

def build_chat_context(preds: pd.DataFrame, season: int, week: int) -> str:
    """Convert predictions DataFrame into a readable context string for the LLM."""
    lines = [
        f"Season {season}, Week {week} — {len(preds)} games analysed by the CFB Betting Model.\n",
        "MODEL RECORD (2023–2025 test set):",
        "  • Totals: 54.7% win rate, +4.3% ROI  ← primary bet type",
        "  • Spreads: 50.4% win rate, +3.2% ROI  (informational only)",
        "  • Moneyline 4–8% EV: 63.1% win rate, +15.9% ROI",
        "  • ML Underdogs specifically: 44.4% win rate, +52.7% ROI\n",
        "UNIT SIZING GUIDE:",
        "  • 1 unit (1u) = 1–2% of your total bankroll",
        "  • 2 units (2u) = 2–3% of your total bankroll",
        "  • Moneyline dogs = 1u max (lotto plays — most won't win but EV is positive)\n",
    ]

    # Totals picks
    tot = preds[
        preds["totals_edge"].notna() &
        (preds["totals_edge"].abs() >= TOTALS_EDGE_MIN) &
        (preds["totals_edge"].abs() <= TOTALS_EDGE_MAX)
    ].sort_values("totals_edge", key=abs, ascending=False)

    if not tot.empty:
        lines.append("TOTALS PICKS (strongest signal):")
        for _, r in tot.iterrows():
            side = "UNDER" if r["totals_edge"] < 0 else "OVER"
            stars = "⭐⭐⭐" if abs(r["totals_edge"]) >= 5.5 else "⭐⭐" if abs(r["totals_edge"]) >= 4.5 else "⭐"
            ou = f"{r['over_under']:.1f}" if pd.notna(r["over_under"]) else "line TBD"
            units = "2u" if abs(r["totals_edge"]) >= 5.0 else "1u"
            lines.append(
                f"  • {r['home_team']} vs {r['away_team']}: "
                f"{side} {ou} {stars} — model projects {r['pred_total']:.1f} total, "
                f"edge {r['totals_edge']:+.1f} pts vs Vegas — bet {units}"
            )
        lines.append("")

    # Moneyline picks
    ml = preds[
        preds["ml_ev"].notna() &
        (preds["ml_ev"] >= MONEYLINE_EV_MIN) &
        (preds["ml_ev"] <  MONEYLINE_EV_MAX)
    ].sort_values("ml_ev", ascending=False)

    if not ml.empty:
        lines.append("MONEYLINE PICKS (+EV bets):")
        for _, r in ml.iterrows():
            odds = r["ml_book_odds"]
            odds_str = f"+{int(odds)}" if odds > 0 else str(int(odds))
            dog_fav = "underdog" if odds > 0 else "favorite"
            units = "1u"
            lines.append(
                f"  • {r['ml_team']} ML {odds_str} ({dog_fav}) — "
                f"EV {r['ml_ev']:+.1%}, model win prob {r['pred_win_p'] if r['ml_team'] == r['home_team'] else 1-r['pred_win_p']:.0%} — bet {units}"
            )
        lines.append("")

    # Spread picks
    sp = preds[
        preds["spread_edge"].notna() &
        (preds["spread_edge"].abs() >= SPREAD_EDGE_MIN) &
        (preds["spread_edge"].abs() <= SPREAD_EDGE_MAX)
    ].sort_values("spread_edge", key=abs, ascending=False)

    if not sp.empty:
        lines.append("SPREAD PICKS (informational — model near breakeven, use as secondary signal):")
        for _, r in sp.iterrows():
            team = r["home_team"] if r["spread_edge"] > 0 else r["away_team"]
            sp_str = f"{r['spread']:+.1f}" if pd.notna(r["spread"]) else "TBD"
            lines.append(
                f"  • {team} (vs {r['away_team'] if team == r['home_team'] else r['home_team']}) "
                f"— Vegas {sp_str}, model {r['pred_spread']:+.1f}, edge {r['spread_edge']:+.1f} pts"
            )
        lines.append("")

    if tot.empty and ml.empty and sp.empty:
        lines.append("No bets meet the threshold this week — the model sees no clear edges. "
                     "This happens — skip the week and wait for better spots.")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a friendly, knowledgeable CFB betting assistant. You help a user (a regular football fan, not a statistician) understand and use their son's college football betting model.

Your personality: warm, clear, direct. You talk like a knowledgeable friend who knows football and betting — not a robot or a professor. Keep answers short and conversational unless detail is asked for.

Your role:
- Explain what picks the model is flagging and why, in plain English
- Help the user decide how much to bet (units) and on what
- Answer questions about how the model works, what EV means, how to read the app
- Always remind the user to verify injuries, check current lines, and bet responsibly

Rules:
- Never recommend betting more than 3% of bankroll on any single bet
- If the model has no picks this week, say so clearly and explain why that's fine
- If asked about a specific team not in the picks, explain the model didn't flag that game
- Keep explanations jargon-free — "EV" means "expected value" which means "the math says you'll profit over time even if you lose this one"
- Remind the user this is a tool to inform decisions, not a guarantee

Context about how the model works:
- It compares its predicted score to the Vegas line. When they disagree enough, it flags a bet.
- Totals (OVER/UNDER) are the strongest part of the model — 54.7% win rate historically
- Unders win more than overs (59% historically) — the model leans toward unders
- Moneyline bets use win probability vs. book odds — a 4–8% edge means the math is in your favor
- Spreads are informational only — near 50/50 historically"""


def chat_tab(preds_context: str | None):
    """Render the Ask the Model chat tab."""

    if not ANTHROPIC_AVAILABLE:
        st.warning("The `anthropic` package is not installed. Run `pip install anthropic` and restart.")
        return

    if not ANTHROPIC_API_KEY:
        st.info(
            "💬 **Chat tab requires an Anthropic API key.**\n\n"
            "1. Get a free key at [console.anthropic.com](https://console.anthropic.com)\n"
            "2. Add it to your Streamlit secrets: `ANTHROPIC_API_KEY = \"sk-ant-...\"`\n"
            "3. Redeploy or restart the app."
        )
        return

    st.markdown("### 💬 Ask the Model")
    st.caption(
        "Ask anything in plain English — *'What should I bet this week?'* "
        "*'Explain the Ohio State pick'* *'How much should I put on it?'* *'What does EV mean?'*"
    )

    # Build system prompt with this week's context if available
    system = SYSTEM_PROMPT
    if preds_context:
        system += f"\n\n--- THIS WEEK'S MODEL OUTPUT ---\n{preds_context}"
    else:
        system += (
            "\n\n--- NO PICKS LOADED YET ---\n"
            "The user hasn't loaded picks for a specific week yet. "
            "Tell them to select a season/week in the sidebar and click 'Get This Week's Picks', "
            "then come back here for context-aware answers. "
            "You can still answer general questions about how the model works."
        )

    # Init session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Suggested openers
    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        starters = [
            "What should I bet this week?",
            "Explain the best pick",
            "How much should I put on each?",
        ]
        for col, q in zip(cols, starters):
            if col.button(q, use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                st.rerun()

    # Render history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="🏈" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about this week's picks..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call Claude
        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        with st.chat_message("assistant", avatar="🏈"):
            with st.spinner("Thinking..."):
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=600,
                    system=system,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                    ],
                )
                reply = response.content[0].text
            st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    # Clear button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear chat", use_container_width=False):
            st.session_state.chat_messages = []
            st.rerun()


def main():
    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/emoji/96/american-football-emoji.png", width=64)
        st.title("CFB Picks")
        st.caption("Built with ❤️ — powered by real data")
        st.divider()

        season = st.selectbox("Season", [2026, 2025], index=0)
        week   = st.slider("Week", min_value=0, max_value=15, value=1)
        st.divider()

        run = st.button("🔍 Get This Week's Picks", type="primary", use_container_width=True)

        st.divider()
        st.markdown("**How it works**")
        st.caption(
            "The model compares its predicted score to the Vegas line. "
            "When they disagree by 3–6 points, that's a flagged bet. "
            "Historically the totals model wins **54.7%** of bets — "
            "above the 52.4% needed to profit."
        )
        st.caption("🔽 **Unders win 59%** — the model tends to lean toward unders.")

    # ── Main area ─────────────────────────────────────────────────────────
    st.title("🏈 CFB Bet Recommendations")

    picks_tab, ask_tab = st.tabs(["📊 This Week's Picks", "💬 Ask the Model"])

    # ── CHAT TAB (always available) ───────────────────────────────────────
    with ask_tab:
        chat_tab(st.session_state.get("chat_context"))

    # ── PICKS TAB ─────────────────────────────────────────────────────────
    with picks_tab:
        if not run:
            st.info("👈 Select a season and week in the sidebar, then hit **Get This Week's Picks**.")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Totals Win Rate", "54.7%", "+2.3% above breakeven")
            col2.metric("Under Win Rate", "59.0%", "Primary edge")
            col3.metric("Spread Win Rate", "50.4%", "Informational only")
            col4.metric("Moneyline EV Min", "4%", "Per $1 bet threshold")
            st.divider()
            st.subheader("📋 How to read the picks")
            st.markdown("""
**Totals (most reliable)**
- ⭐⭐⭐ Three stars = edge ≥ 5.5 pts vs. Vegas total
- ⭐⭐ Two stars = edge 4.5–5.5 pts
- ⭐ One star = edge 3–4.5 pts
- Unders win 59% historically — lean under when in doubt

**Moneylines (use model win probability vs. book odds)**
- ⭐⭐⭐ = EV ≥ 7% (strong) &nbsp;&nbsp; ⭐⭐ = EV 5–7% &nbsp;&nbsp; ⭐ = EV 4–5%
- EV = expected return per $1 bet — positive means the book is mispricing the game
- Underdog +EV bets tend to be more valuable than favorite +EV bets

**Spreads (informational only)** — model near breakeven; use as secondary confirmation

*Always check: injuries, weather forecast, and whether the line has moved since this loaded.*
            """)
            st.info("💬 You can also switch to the **Ask the Model** tab to chat about picks in plain English.")
            return

        # ── Load models ───────────────────────────────────────────────────
        spread_model, totals_model, win_prob_model, feature_lists = load_models()
        if spread_model is None:
            st.error("❌ Model files not found in the `models/` folder. "
                     "Run `python3 src/model.py` first, then redeploy.")
            return

        # ── Load data ─────────────────────────────────────────────────────
        ratings = load_team_ratings(season)
        elo     = load_current_elo(season)
        epa     = load_recent_epa(season)

        games = fetch_schedule(season, week)
        if games.empty:
            st.warning(f"No games found for {season} Week {week}. "
                       "The schedule may not be posted yet.")
            return

        lines     = fetch_lines(games)
        has_lines = not lines.empty and lines["spread"].notna().any()

        with st.spinner("Running models..."):
            preds = build_and_predict(games, lines, ratings, epa, elo,
                                      spread_model, totals_model, win_prob_model,
                                      feature_lists)

        # Save context so chat tab can reference this week's picks
        st.session_state["chat_context"] = build_chat_context(preds, season, week)

        # ── Header metrics ────────────────────────────────────────────────
        st.subheader(f"Season {season} — Week {week}")
        st.caption("💬 Picks loaded — switch to **Ask the Model** for plain-English explanations.")

        ml_bets = preds[
            preds["ml_ev"].notna() &
            (preds["ml_ev"] >= MONEYLINE_EV_MIN) &
            (preds["ml_ev"] <  MONEYLINE_EV_MAX)
        ].sort_values("ml_ev", ascending=False)

        tot_bets = preds[
            preds["totals_edge"].notna() &
            (preds["totals_edge"].abs() >= TOTALS_EDGE_MIN) &
            (preds["totals_edge"].abs() <= TOTALS_EDGE_MAX)
        ]
        sp_bets = preds[
            preds["spread_edge"].notna() &
            (preds["spread_edge"].abs() >= SPREAD_EDGE_MIN) &
            (preds["spread_edge"].abs() <= SPREAD_EDGE_MAX)
        ]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Moneyline +EV Bets", len(ml_bets))
        col2.metric("🎯 Totals Bets", len(tot_bets))
        col3.metric("📊 Spread Bets", len(sp_bets), help="Informational — spread model near breakeven")
        col4.metric("🏈 Games This Week", len(preds))

        if not has_lines:
            st.warning("⚠️ No Vegas lines found yet. Showing model projections only — "
                       "lines usually appear 7–10 days before kickoff.")

        # ── Moneyline section ─────────────────────────────────────────────
        st.divider()
        st.subheader("💰 Moneyline Picks  *(+EV bets)*")
        st.caption(
            f"EV window: {MONEYLINE_EV_MIN:.0%}–{MONEYLINE_EV_MAX:.0%}. "
            "🐶 **Underdogs drive the edge** — +52.7% historical ROI. "
            "⭐⭐⭐ = EV ≥ 7%  ·  ⭐⭐ = 5–7%  ·  ⭐ = 4–5%"
        )
        if not has_lines or preds["home_moneyline"].isna().all():
            st.info("No moneyline data yet — Odds API moneylines appear closer to kickoff.")
        elif ml_bets.empty:
            st.info("No +EV moneyline bets this week. The market is efficiently priced — skip.")
        else:
            dog_bets = ml_bets[ml_bets["ml_book_odds"] > 0]
            fav_bets = ml_bets[ml_bets["ml_book_odds"] <= 0]
            if not dog_bets.empty:
                st.markdown("**🐶 Underdogs with value**")
                for _, row in dog_bets.iterrows():
                    render_moneyline_card(row)
            if not fav_bets.empty:
                st.markdown("**🏆 Favorites with value**")
                for _, row in fav_bets.iterrows():
                    render_moneyline_card(row)

        # ── Totals section ────────────────────────────────────────────────
        st.divider()
        st.subheader("🎯 Totals Picks  *(your stronger model)*")
        st.caption("Under bets win 59% historically. Flags games where model disagrees with Vegas total by 3–6 pts.")
        if tot_bets.empty:
            st.info("No totals bets meet the threshold this week. Check back after lines sharpen.")
        else:
            tot_sorted  = tot_bets.sort_values("totals_edge", key=abs, ascending=False)
            under_bets  = tot_sorted[tot_sorted["totals_edge"] < 0]
            over_bets   = tot_sorted[tot_sorted["totals_edge"] > 0]
            if not under_bets.empty:
                st.markdown("**Unders 🔽**")
                for _, row in under_bets.iterrows():
                    render_totals_card(row)
            if not over_bets.empty:
                st.markdown("**Overs 🔼**")
                for _, row in over_bets.iterrows():
                    render_totals_card(row)

        # ── Spreads section ───────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Spread Picks  *(informational)*")
        st.caption("Near breakeven historically — use as a secondary signal only.")
        if sp_bets.empty:
            st.info("No spread bets meet the threshold this week.")
        else:
            sp_display = sp_bets.copy()
            sp_display["Bet on"]     = sp_display.apply(
                lambda r: r["home_team"] if r["spread_edge"] > 0 else r["away_team"], axis=1)
            sp_display["Vegas line"] = sp_display["spread"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
            sp_display["Model"]      = sp_display["pred_spread"].apply(lambda x: f"{x:+.1f}")
            sp_display["Edge"]       = sp_display["spread_edge"].apply(lambda x: f"{x:+.1f}")
            sp_display["Matchup"]    = sp_display["home_team"] + " vs " + sp_display["away_team"]
            sp_display["⭐"]          = sp_display["spread_edge"].abs().apply(confidence_stars)
            st.dataframe(
                sp_display[["⭐","Bet on","Vegas line","Model","Edge","Matchup"]],
                use_container_width=True, hide_index=True,
            )

        # ── All games expander ────────────────────────────────────────────
        st.divider()
        with st.expander(f"📋 All {len(preds)} games this week"):
            all_display = preds.copy()
            all_display["Spread"]       = all_display["spread"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            all_display["Model Spread"] = all_display["pred_spread"].apply(lambda x: f"{x:+.1f}")
            all_display["S.Edge"]       = all_display["spread_edge"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            all_display["O/U"]          = all_display["over_under"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            all_display["Model Total"]  = all_display["pred_total"].apply(lambda x: f"{x:.1f}")
            all_display["T.Edge"]       = all_display["totals_edge"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            all_display["Home Win%"]    = all_display["pred_win_p"].apply(lambda x: f"{x:.0%}")
            all_display["Home ML"]      = all_display["home_moneyline"].apply(
                lambda x: f"{int(x):+d}" if pd.notna(x) else "—")
            all_display["ML EV"]        = all_display.apply(
                lambda r: f"{r['ml_ev']:+.1%}" if pd.notna(r.get("ml_ev")) else "—", axis=1)
            st.dataframe(
                all_display[["home_team","away_team",
                             "Spread","Model Spread","S.Edge",
                             "O/U","Model Total","T.Edge",
                             "Home Win%","Home ML","ML EV"]].rename(columns={
                    "home_team": "Home", "away_team": "Away"}),
                use_container_width=True, hide_index=True,
            )

        # ── Footer ────────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "📌 **Always verify before betting:** check injury reports, weather forecast, "
            "and whether the line has moved. Betting involves risk — this model is a tool, not a guarantee."
        )


if __name__ == "__main__":
    main()
