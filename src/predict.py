"""
CFB Weekly Predictor — Phase 7
================================
Pulls the upcoming week's schedule and current Vegas lines from the CFBD API,
builds feature vectors from the latest available team data, runs the saved
spread / totals / win-probability models, and prints ranked bet recommendations.

Usage:
    python3 src/predict.py                          # auto-detect next week
    python3 src/predict.py --season 2026 --week 1
    python3 src/predict.py --season 2026 --week 1 --show-all   # include no-bet games

How features are built for upcoming games
------------------------------------------
Pre-season (week 1 or before any 2026 games exist):
  • Ratings  (SP+, FPI, SRS, recruiting, HFA) come from the 2025 final season
    values — already shifted +1 year in features.py, so they slot in as 2026
    pre-season baseline automatically.
  • Elo      is recomputed through the end of 2025 using EloSystem.run().
  • EPA      falls back to each team's last-5-game rolling average from 2025.
    Once 2026 games are played, re-run data_collection.py + features.py and
    the in-season rolling EPA will replace these.
  • Rest     defaults to 14 days for week 1 (season opener).
  • Weather  is not available for future games — totals predictions will be
    slightly less calibrated until the game date approaches.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# ── Shared feature builder ────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from model import patch_sklearn_compat
from feature_builder import (
    load_rating_sources,
    load_recent_epa    as _fb_load_recent_epa,
    load_current_elo   as _fb_load_current_elo,
    attach_team_features,
)

# ─── PATHS ───────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).parent.parent
DATA_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RAW_DIR   = ROOT_DIR / "data" / "raw"

CFB_API_KEY  = os.getenv("CFB_API_KEY", "")
CFB_BASE_URL = "https://api.collegefootballdata.com"

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NCAAF_SPORT   = "americanfootball_ncaaf"

# Book priority for line selection (best liquidity / sharpest lines first)
BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "caesars",
                 "pointsbetus", "betrivers", "bovada"]

# Team name differences between Odds API and CFBD
ODDS_TO_CFBD = {
    "Louisiana State":   "LSU",
    "Mississippi":       "Ole Miss",
    "Southern California": "USC",
    "Central Florida":   "UCF",
    "Southern Methodist": "SMU",
    "Texas Christian":   "TCU",
    "Brigham Young":     "BYU",
    "Nevada Las Vegas":  "UNLV",
    "Pittsburgh":        "Pittsburgh",
    "Massachusetts":     "UMass",
    "Florida International": "FIU",
    "Middle Tennessee State": "Middle Tennessee",
    "North Carolina State": "NC State",
    "Miami (OH)":        "Miami (OH)",
    "Miami":             "Miami",
}

# ─── BETTING THRESHOLDS (from backtesting) ───────────────────────────────────

SPREAD_EDGE_MIN = 2.0   # minimum model-vs-Vegas disagreement to flag
SPREAD_EDGE_MAX = 5.0   # above this, Vegas probably has info you don't
TOTALS_EDGE_MIN = 3.0
TOTALS_EDGE_MAX = 6.0
MONEYLINE_EV_MIN = 0.04  # minimum expected value per $1 bet (4%)
MONEYLINE_EV_MAX = 0.08  # above this, model likely overconfident vs. market


# ─── MONEYLINE MATH HELPERS ──────────────────────────────────────────────────

def american_to_implied_prob(odds: float) -> float:
    """Convert American moneyline to raw implied probability (includes vig)."""
    if pd.isna(odds):
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(home_prob: float, away_prob: float):
    """
    Divide out the bookmaker's juice so home + away sum to 1.0.
    Returns (fair_home_prob, fair_away_prob).
    """
    if pd.isna(home_prob) or pd.isna(away_prob):
        return np.nan, np.nan
    total = home_prob + away_prob
    if total <= 0:
        return np.nan, np.nan
    return home_prob / total, away_prob / total


def prob_to_american(prob: float) -> float:
    """Convert a win probability to American odds (no vig)."""
    if pd.isna(prob) or prob <= 0 or prob >= 1:
        return np.nan
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round(((1 - prob) / prob) * 100)


def moneyline_ev(model_prob: float, american_odds: float) -> float:
    """
    Expected value per $1 wagered at the given American odds.
    Positive EV means your model's probability implies you're being
    offered better-than-fair odds.

    EV = (model_prob × win_payout) − (1 − model_prob)
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
    if american_odds < 0:
        win_payout = 100 / abs(american_odds)
    else:
        win_payout = american_odds / 100
    return model_prob * win_payout - (1 - model_prob)


# ─── API HELPER ──────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    url     = f"{CFB_BASE_URL}/{endpoint}"
    resp    = requests.get(url, headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─── ODDS API HELPERS ────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Map Odds API team names to CFBD team names where they differ."""
    return ODDS_TO_CFBD.get(name, name)


def fetch_odds_api_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch current NCAAF spread + total odds from The Odds API and match
    them to the CFBD schedule by team name.

    Returns a DataFrame with columns:
      game_id, spread, over_under, home_moneyline, away_moneyline, book
    ready to drop into build_features() as the lines argument.
    """
    from difflib import SequenceMatcher

    url = f"{ODDS_API_BASE}/sports/{NCAAF_SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  Odds API error: {e}")
        return pd.DataFrame()

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"  Odds API: {remaining} requests remaining this month (used {used} total)")

    data = resp.json()
    if not data:
        return pd.DataFrame()

    # ── Parse each game from the API response ─────────────────────────────
    odds_records = []
    for game in data:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home     = _normalize_name(home_raw)
        away     = _normalize_name(away_raw)

        # Pick the sharpest available book
        bookmakers = {b["key"]: b for b in game.get("bookmakers", [])}
        book = next((bookmakers[k] for k in BOOK_PRIORITY if k in bookmakers),
                    next(iter(bookmakers.values()), None) if bookmakers else None)

        spread = over_under = home_ml = away_ml = None
        book_name = None

        if book:
            book_name = book.get("title", book.get("key"))
            for market in book.get("markets", []):
                if market["key"] == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:   # home team's point = spread
                            spread = o["point"]
                elif market["key"] == "totals":
                    if market["outcomes"]:
                        over_under = market["outcomes"][0]["point"]
                elif market["key"] == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_raw:
                            home_ml = o["price"]
                        elif o["name"] == away_raw:
                            away_ml = o["price"]

        odds_records.append({
            "odds_home": home, "odds_away": away,
            "spread": spread, "over_under": over_under,
            "home_moneyline": home_ml, "away_moneyline": away_ml,
            "book": book_name,
            "spread_open": None,   # not available on free tier
        })

    if not odds_records:
        return pd.DataFrame()

    odds_df = pd.DataFrame(odds_records)

    # ── Match Odds API games to CFBD game_ids ─────────────────────────────
    def sim(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    matched = []
    cfbd_teams = list(zip(games_df["game_id"], games_df["home_team"], games_df["away_team"]))

    for _, odds_row in odds_df.iterrows():
        best_id, best_score = None, 0.0
        for gid, cfbd_home, cfbd_away in cfbd_teams:
            score = (sim(odds_row["odds_home"], cfbd_home) +
                     sim(odds_row["odds_away"], cfbd_away)) / 2
            if score > best_score:
                best_score, best_id = score, gid

        if best_score >= 0.70:   # require a reasonable match
            matched.append({
                "game_id":        best_id,
                "spread":         odds_row["spread"],
                "over_under":     odds_row["over_under"],
                "home_moneyline": odds_row["home_moneyline"],
                "away_moneyline": odds_row["away_moneyline"],
                "spread_open":    None,
                "provider":       odds_row["book"],
            })

    result = pd.DataFrame(matched)
    if not result.empty:
        result = result.drop_duplicates("game_id", keep="first")

    print(f"  Odds API matched {len(result)} of {len(games_df)} games")
    return result


# ─── 1. LOAD SAVED MODELS ────────────────────────────────────────────────────

def load_models():
    """Load the three saved models and their feature lists."""
    for f in ["spread_model.pkl", "totals_model.pkl",
              "win_prob_model.pkl", "feature_lists.json"]:
        if not (MODEL_DIR / f).exists():
            print(f"❌ Missing {f} — run python3 src/model.py first.")
            sys.exit(1)

    spread_model   = patch_sklearn_compat(joblib.load(MODEL_DIR / "spread_model.pkl"))
    totals_model   = patch_sklearn_compat(joblib.load(MODEL_DIR / "totals_model.pkl"))
    win_prob_model = patch_sklearn_compat(joblib.load(MODEL_DIR / "win_prob_model.pkl"))

    with open(MODEL_DIR / "feature_lists.json") as f:
        feature_lists = json.load(f)

    return spread_model, totals_model, win_prob_model, feature_lists


# ─── 2. LOAD TEAM DATA ───────────────────────────────────────────────────────
# All three loaders now delegate to feature_builder — single source of truth.

def load_team_ratings(pred_season: int) -> dict:
    """Load all per-team rating data for pred_season via feature_builder."""
    return load_rating_sources(pred_season, DATA_DIR)


def load_current_elo(pred_season: int) -> pd.DataFrame:
    """Recompute Elo through pred_season−1 via feature_builder."""
    return _fb_load_current_elo(pred_season, DATA_DIR)


def load_recent_epa(pred_season: int) -> pd.DataFrame:
    """Load last-season rolling EPA per team via feature_builder."""
    return _fb_load_recent_epa(pred_season, DATA_DIR)


# ─── 3. FETCH SCHEDULE & LINES ───────────────────────────────────────────────

def fetch_schedule(season: int, week: int) -> pd.DataFrame:
    """Pull scheduled games for a given season and week from the CFBD API."""
    print(f"  Fetching schedule: {season} week {week}...")
    try:
        data = cfb_get("games", params={"year": season, "week": week,
                                         "seasonType": "regular"})
    except Exception as e:
        print(f"  ⚠️  Could not fetch schedule: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    records = []
    for g in data:
        records.append({
            "game_id":        g.get("id"),
            "season":         g.get("season"),
            "week":           g.get("week"),
            "home_team":      g.get("homeTeam"),
            "away_team":      g.get("awayTeam"),
            "home_conference": g.get("homeConference"),
            "away_conference": g.get("awayConference"),
            "neutral_site":   g.get("neutralSite", False),
            "start_date":     g.get("startDate"),
            "completed":      g.get("completed", False),
            # If the game is already played:
            "home_points":    g.get("homePoints"),
            "away_points":    g.get("awayPoints"),
            "home_pregame_elo": g.get("homePregameElo"),
            "away_pregame_elo": g.get("awayPregameElo"),
        })

    df = pd.DataFrame(records)
    df["neutral_site"]   = df["neutral_site"].fillna(False).astype(int)
    df["conference_game"] = (
        df["home_conference"].notna() &
        df["away_conference"].notna() &
        (df["home_conference"] == df["away_conference"])
    ).astype(int)

    return df


def fetch_lines(season: int, week: int) -> pd.DataFrame:
    """Pull betting lines for a given season and week."""
    print(f"  Fetching lines: {season} week {week}...")
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception as e:
        print(f"  ⚠️  Could not fetch lines: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet",
                "William Hill (New Jersey)", "FanDuel"]
    rank_map = {p: i for i, p in enumerate(priority)}

    records = []
    for game in data:
        game_id = game.get("id")
        for line in game.get("lines", []):
            records.append({
                "game_id":     game_id,
                "provider":    line.get("provider"),
                "spread":      line.get("spread"),
                "over_under":  line.get("overUnder"),
                "spread_open": line.get("spreadOpen"),
                "_rank": rank_map.get(line.get("provider", ""), len(priority)),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    best = (
        df.sort_values("_rank")
          .drop_duplicates("game_id", keep="first")
          .drop(columns=["_rank"])
          .reset_index(drop=True)
    )

    best["spread"]     = pd.to_numeric(best["spread"],     errors="coerce")
    best["over_under"] = pd.to_numeric(best["over_under"], errors="coerce")
    best["spread_open"]= pd.to_numeric(best["spread_open"],errors="coerce")

    return best[["game_id", "provider", "spread", "over_under", "spread_open"]]


# ─── 4. BUILD FEATURES FOR UPCOMING GAMES ────────────────────────────────────

def build_features(
    games:         pd.DataFrame,
    lines:         pd.DataFrame,
    ratings:       dict,
    epa:           pd.DataFrame,
    elo:           pd.DataFrame,
    feature_names: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge lines onto games, build all team features via feature_builder, and
    extract the feature matrix columns that the saved model expects.

    Returns (games_with_features, feature_matrix).
    """
    # ── Merge lines ────────────────────────────────────────────────────────
    if not lines.empty:
        line_cols = ["game_id", "spread", "over_under", "spread_open"]
        for ml_col in ["home_moneyline", "away_moneyline"]:
            if ml_col in lines.columns:
                line_cols.append(ml_col)
        df = games.merge(lines[line_cols], on="game_id", how="left")
    else:
        df = games.copy()
        df["spread"] = df["over_under"] = df["spread_open"] = np.nan

    if "home_moneyline" not in df.columns:
        df["home_moneyline"] = np.nan
    if "away_moneyline" not in df.columns:
        df["away_moneyline"] = np.nan

    # ── Build all team features via shared feature_builder ─────────────────
    df = attach_team_features(
        df, ratings, epa,
        elo if (elo is not None and not elo.empty) else None,
    )

    # Situational spot features (lookahead / sandwich / altitude).
    try:
        from situational import add_situational_features
        df = add_situational_features(df)
    except Exception as e:
        print(f"  ⚠️  situational features unavailable: {e}")

    # ── Extract the exact columns the model expects ────────────────────────
    feature_df = pd.DataFrame(index=df.index)
    for f in feature_names:
        feature_df[f] = df[f] if f in df.columns else np.nan

    return df, feature_df


# ─── 5. GENERATE PREDICTIONS ─────────────────────────────────────────────────

def generate_predictions(
    df:            pd.DataFrame,
    feature_df_sp: pd.DataFrame,
    feature_df_tot:pd.DataFrame,
    feature_df_win:pd.DataFrame,
    spread_model, totals_model, win_prob_model,
    feature_lists: dict | None = None,
) -> pd.DataFrame:
    """Run all three models and attach predictions to the games DataFrame."""
    base_cols = ["game_id","season","week","home_team","away_team",
                 "neutral_site","conference_game",
                 "home_conference","away_conference","wind_speed","is_dome",
                 "spread","over_under","spread_open",
                 "home_moneyline","away_moneyline"]
    out = df[[c for c in base_cols if c in df.columns]].copy()

    # Ensure moneyline columns exist
    if "home_moneyline" not in out.columns:
        out["home_moneyline"] = np.nan
    if "away_moneyline" not in out.columns:
        out["away_moneyline"] = np.nan

    # Spread model: current models predict the residual vs the Vegas line
    # (feature_lists.json: spread_target=margin_residual) — add the line back.
    # Legacy models predicted the raw home margin directly.
    sp_raw = spread_model.predict(feature_df_sp)
    if (feature_lists or {}).get("spread_target") == "margin_residual":
        vm = -pd.to_numeric(out["spread"], errors="coerce")
        out["pred_spread"] = vm + sp_raw     # NaN when no line posted yet
    else:
        out["pred_spread"] = sp_raw
    # Totals model predicts deviation from the O/U line, not the raw total.
    # Add the line back so pred_total is the expected combined score
    # (same handling as weekly_pipeline.py; NaN when no line is posted yet).
    ou_vals = pd.to_numeric(out["over_under"], errors="coerce")
    out["pred_total"]  = ou_vals + totals_model.predict(feature_df_tot)
    out["pred_win_p"]  = win_prob_model.predict_proba(feature_df_win)[:, 1]

    from inference import adjust_predictions
    calib_path = MODEL_DIR / "win_prob_calibration.json"
    calibration = json.loads(calib_path.read_text()) if calib_path.exists() else None
    out = adjust_predictions(out, calibration)

    # ── Moneyline Expected Value ───────────────────────────────────────────────
    # Step 1: implied probs from book (with vig baked in)
    out["implied_home_prob_raw"] = out["home_moneyline"].apply(american_to_implied_prob)
    out["implied_away_prob_raw"] = out["away_moneyline"].apply(american_to_implied_prob)

    # Step 2: remove vig → fair market probabilities
    out[["fair_home_prob", "fair_away_prob"]] = out.apply(
        lambda r: pd.Series(remove_vig(r["implied_home_prob_raw"], r["implied_away_prob_raw"])),
        axis=1,
    )

    # Step 3: model-implied American odds (no vig)
    out["model_home_ml"] = out["pred_win_p"].apply(prob_to_american)
    out["model_away_ml"] = out["pred_away_win_p"].apply(prob_to_american)

    # Step 4: EV for betting home vs. away
    out["home_ev"] = out.apply(
        lambda r: moneyline_ev(r["pred_win_p"], r["home_moneyline"]), axis=1)
    out["away_ev"] = out.apply(
        lambda r: moneyline_ev(r["pred_away_win_p"], r["away_moneyline"]), axis=1)

    # Step 5: best side and its EV
    def best_ml_bet(row):
        h, a = row["home_ev"], row["away_ev"]
        if pd.isna(h) and pd.isna(a):
            return pd.Series({"ml_bet_side": None, "ml_ev": np.nan, "ml_odds": np.nan})
        if pd.isna(a) or (not pd.isna(h) and h >= a):
            return pd.Series({"ml_bet_side": row["home_team"], "ml_ev": h, "ml_odds": row["home_moneyline"]})
        return pd.Series({"ml_bet_side": row["away_team"], "ml_ev": a, "ml_odds": row["away_moneyline"]})

    out[["ml_bet_side", "ml_ev", "ml_odds"]] = out.apply(best_ml_bet, axis=1)

    return out


# ─── 6. PRINT RECOMMENDATIONS ────────────────────────────────────────────────

def print_recommendations(preds: pd.DataFrame, show_all: bool = False):
    """Use the same CORE gate as the app and weekly pipeline."""
    import gates
    print("\nCORE unders (flat 1u; prospective validation required)")
    core = preds[preds.apply(gates.core_total, axis=1)]
    if core.empty:
        print("  No qualifying CORE unders.")
    for _, row in core.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}: UNDER {row['over_under']:.1f} "
              f"| model {row['pred_total']:.1f} | edge {row['totals_edge']:+.1f} | 1u")
    print("\nPaper signals (no units)")
    for _, row in preds.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        edge = row.get("spread_edge")
        if pd.notna(edge) and abs(edge) >= 3:
            home = edge > 0
            team = row['home_team'] if home else row['away_team']
            line = row['spread'] if home else -row['spread']
            print(f"  {matchup}: {team} {line:+.1f} | spread edge {edge:+.1f}")
        edge = row.get("totals_edge")
        if pd.notna(edge) and abs(edge) >= 2 and not gates.core_total(row):
            side = "OVER" if edge > 0 else "UNDER"
            print(f"  {matchup}: {side} {row['over_under']:.1f} | total edge {edge:+.1f}")
        if pd.notna(row.get("ml_ev")) and row['ml_ev'] >= .04:
            print(f"  {matchup}: {row['ml_bet_side']} ML | estimated EV {row['ml_ev']:+.1%}")
    if show_all:
        columns = ['away_team', 'home_team', 'spread', 'pred_spread',
                   'over_under', 'pred_total', 'pred_win_p']
        print("\n" + preds[[c for c in columns if c in preds]].to_string(index=False))


# ─── 7. MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CFB bet recommendations for an upcoming week.")
    parser.add_argument("--season", type=int, default=2026,
        help="Season year (default: 2026)")
    parser.add_argument("--week", type=int, default=1,
        help="Week number (default: 1)")
    parser.add_argument("--show-all", action="store_true",
        help="Show all games, not just flagged bets")
    args = parser.parse_args()

    season = args.season
    week   = args.week

    print(f"\n{'='*55}")
    print(f"CFB Predictor — {season} Season, Week {week}")
    print(f"{'='*55}")

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading models...")
    spread_model, totals_model, win_prob_model, feature_lists = load_models()
    print("  ✓ Models loaded")

    # ── Load team data ────────────────────────────────────────────────────
    print("\nLoading team data...")
    ratings = load_team_ratings(season)
    print(f"  ✓ Ratings loaded "
          f"(SP+: {'✓' if 'sp' in ratings else '✗'}  "
          f"FPI: {'✓' if 'fpi' in ratings else '✗'}  "
          f"SRS: {'✓' if 'srs' in ratings else '✗'})")

    print("  Computing current Elo ratings...")
    elo = load_current_elo(season)
    print(f"  ✓ Elo ratings: {len(elo)} teams")

    print("  Loading recent EPA (end of prior season)...")
    epa = load_recent_epa(season)
    print(f"  ✓ EPA data: {len(epa)} teams")

    # ── Fetch schedule (CFBD) ────────────────────────────────────────────
    print(f"\nFetching {season} week {week} schedule from CFBD...")
    games = fetch_schedule(season, week)

    if games.empty:
        print(f"\n⚠️  No games found for {season} week {week}.")
        print("   The schedule may not be posted yet, or all games are listed")
        print("   under a different week number. Try --week 0 for week 0 games.")
        return

    print(f"  ✓ {len(games)} games on schedule")

    # ── Fetch lines: try Odds API first, fall back to CFBD ───────────────
    print("Fetching lines from The Odds API...")
    lines = fetch_odds_api_lines(games)

    if lines.empty:
        print("  Falling back to CFBD lines...")
        lines = fetch_lines(season, week)

    n_lines = lines["game_id"].nunique() if not lines.empty else 0
    source  = lines["provider"].iloc[0] if (not lines.empty and "provider" in lines.columns) else "unknown"
    print(f"  ✓ {n_lines} games with lines (source: {source})")

    # ── Build features ────────────────────────────────────────────────────
    print("\nBuilding feature vectors...")
    games_df, feat_sp  = build_features(
        games, lines, ratings, epa, elo, feature_lists["spread"])
    _,         feat_tot = build_features(
        games, lines, ratings, epa, elo, feature_lists["totals"])
    _,         feat_win = build_features(
        games, lines, ratings, epa, elo, feature_lists["win_prob"])

    # ── Run predictions ───────────────────────────────────────────────────
    print("Running models...")
    preds = generate_predictions(
        games_df, feat_sp, feat_tot, feat_win,
        spread_model, totals_model, win_prob_model, feature_lists)

    # ── Print recommendations ─────────────────────────────────────────────
    print_recommendations(preds, show_all=args.show_all)

    # ── Save to CSV ───────────────────────────────────────────────────────
    out_dir = ROOT_DIR / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"week_{season}_w{week:02d}_predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"Full predictions saved → {out_path}\n")


if __name__ == "__main__":
    main()
