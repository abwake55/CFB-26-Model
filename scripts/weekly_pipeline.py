#!/usr/bin/env python3
"""
CFB Betting Model — Weekly Automated Pipeline
==============================================
Runs every Tuesday morning via scheduled task.

Steps
-----
1. Retrain all three models (spread, totals, win-prob) on the latest data
2. Fetch the upcoming week's schedule from CFBD
3. Fetch live lines from OddsBlaze (falls back to CFBD lines)
4. Run predictions and apply edge/EV thresholds
5. Write picks summary to scripts/pipeline_output.json
6. Print a compact iMessage-ready text block to stdout

Usage (standalone):
    cd /Users/alex/Desktop/CFB-Betting-Model
    python3 scripts/weekly_pipeline.py

    # Skip retrain (data refresh only, faster):
    python3 scripts/weekly_pipeline.py --skip-retrain
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from difflib import SequenceMatcher
from math import erf, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).parent.parent
SRC_DIR    = ROOT_DIR / "src"
MODEL_DIR  = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = ROOT_DIR / "scripts" / "pipeline_output.json"

sys.path.insert(0, str(SRC_DIR))

# ── joblib unpickling requires the ensemble classes in __main__ ───────────────
from model import EnsembleRegressor, EnsembleClassifier  # noqa: E402
import __main__
__main__.EnsembleRegressor  = EnsembleRegressor
__main__.EnsembleClassifier = EnsembleClassifier

from feature_builder import (         # noqa: E402
    load_rating_sources,
    load_recent_epa,
    load_current_elo,
    attach_team_features,
)

# ─── Secrets ─────────────────────────────────────────────────────────────────

def load_secrets() -> dict:
    """Read .streamlit/secrets.toml without importing Streamlit."""
    secrets: dict = {}
    path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                secrets[k.strip()] = v.strip().strip('"').strip("'")
    # Also check environment variables (for CI / cron)
    for key in ["CFB_API_KEY", "ODDSBLAZE_KEY", "ODDS_API_KEY"]:
        if key not in secrets and os.getenv(key):
            secrets[key] = os.getenv(key)
    return secrets

SECRETS = load_secrets()

def _cfb_key()  -> str: return SECRETS.get("CFB_API_KEY",   "")
def _odds_key() -> str: return SECRETS.get("ODDSBLAZE_KEY", "")

# ─── Constants ────────────────────────────────────────────────────────────────

CFB_BASE      = "https://api.collegefootballdata.com"
OB_BASE       = "https://data.oddsblaze.com/v1/odds"
OB_BOOKS      = ["draftkings_ncaaf", "fanduel_ncaaf", "betmgm_ncaaf", "caesars_ncaaf"]

ODDS_TO_CFBD  = {
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

SPREAD_EDGE_MIN   = 4.0
SPREAD_EDGE_MAX   = 7.0
TOTALS_EDGE_MIN   = 3.0
MONEYLINE_EV_MIN  = 0.04

# ─── Season / week helpers ────────────────────────────────────────────────────

def current_cfb_week() -> tuple[int, int]:
    """
    Return (season, week) for the upcoming game week.
    CFB regular season runs roughly weeks 1–15, September–November.
    Preseason (off-season) returns week 1 of the upcoming season.
    """
    today = date.today()
    year  = today.year
    # Regular season typically starts Labor Day weekend (first Saturday in Sep)
    # Approximate: season weeks start ~Aug 24 each year
    season_start = date(year, 8, 24)
    if today < season_start:
        return year, 1  # pre-season — return week 1 of upcoming season
    day_offset = (today - season_start).days
    week = min(day_offset // 7 + 1, 15)
    return year, week


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def cfb_get(endpoint: str, params: dict = None) -> list:
    headers = {"Authorization": f"Bearer {_cfb_key()}"}
    resp = requests.get(f"{CFB_BASE}/{endpoint}",
                        headers=headers, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ─── Math helpers ─────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(float(x) / sqrt(2)))

def american_to_implied(odds: float) -> float:
    if pd.isna(odds): return float("nan")
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def prob_to_american(p: float) -> float:
    if pd.isna(p) or p <= 0 or p >= 1: return float("nan")
    return round(-(p / (1 - p)) * 100) if p >= 0.5 else round(((1 - p) / p) * 100)

def ml_ev(model_prob: float, american_odds: float) -> float:
    if pd.isna(model_prob) or pd.isna(american_odds): return float("nan")
    payout = 100 / abs(american_odds) if american_odds < 0 else american_odds / 100
    return model_prob * payout - (1 - model_prob)

def kelly_spread(edge: float, fraction: float = 0.25) -> int:
    win_prob = min(0.50 + abs(edge) * 0.02, 0.70)
    b = 100 / 110
    f = max((win_prob * b - (1 - win_prob)) / b, 0.0)
    return max(1, min(4, round(f * fraction * 100)))

def kelly_ml(ev: float) -> int:
    if ev >= 0.08: return 3
    if ev >= 0.06: return 2
    return 1


# ─── Step 1: Retrain ─────────────────────────────────────────────────────────

def retrain() -> bool:
    print("=" * 60)
    print("STEP 1 — RETRAINING MODELS")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "model.py")],
        cwd=str(ROOT_DIR),
        capture_output=False,   # let training output print live
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] model.py exited with code {result.returncode}")
        return False
    print("\n✅ Retrain complete")
    return True


# ─── Step 2: Load models ──────────────────────────────────────────────────────

def load_models():
    import sys as _sys
    _sys.modules["__main__"].__dict__.setdefault("EnsembleRegressor",  EnsembleRegressor)
    _sys.modules["__main__"].__dict__.setdefault("EnsembleClassifier", EnsembleClassifier)
    required = ["spread_model.pkl", "totals_model.pkl",
                "win_prob_model.pkl", "feature_lists.json"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    spread   = joblib.load(MODEL_DIR / "spread_model.pkl")
    totals   = joblib.load(MODEL_DIR / "totals_model.pkl")
    win_prob = joblib.load(MODEL_DIR / "win_prob_model.pkl")
    with open(MODEL_DIR / "feature_lists.json") as f:
        feat_lists = json.load(f)
    print("✅ Models loaded")
    return spread, totals, win_prob, feat_lists


# ─── Step 3: Fetch schedule ───────────────────────────────────────────────────

def fetch_schedule(season: int, week: int) -> pd.DataFrame:
    print(f"\n📅 Fetching schedule: Season {season}, Week {week}")
    data = cfb_get("games", params={"year": season, "week": week,
                                     "seasonType": "regular"})
    if not data:
        print("[WARN] No games returned from CFBD")
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
    df = df.drop_duplicates(subset=["home_team", "away_team"])
    seen: set = set()
    clean = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in seen and a not in seen:
            clean.append(row)
            seen.update([h, a])
    df = pd.DataFrame(clean).reset_index(drop=True)
    print(f"✅ {len(df)} games found")
    return df


# ─── Step 4: Fetch lines ──────────────────────────────────────────────────────

def fetch_lines(games_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    print("\n💰 Fetching lines...")
    key = _odds_key()
    if key:
        try:
            for book_id in OB_BOOKS:
                resp = requests.get(
                    f"{OB_BASE}/{book_id}.json",
                    params={"key": key,
                            "market": "Moneyline,Point Spread,Total Points",
                            "main": "true", "price": "american"},
                    timeout=15)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                if not payload.get("events"):
                    continue
                book_name = (payload.get("sportsbook") or {}).get("name", book_id)
                odds_rows = []
                for event in payload["events"]:
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
                                      "home_moneyline": home_ml,
                                      "away_moneyline": away_ml,
                                      "provider": book_name})
                if not odds_rows:
                    continue

                def sim(a, b):
                    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

                odds_df = pd.DataFrame(odds_rows)
                matched = []
                cfbd = list(zip(games_df["game_id"], games_df["home_team"], games_df["away_team"]))
                for _, r in odds_df.iterrows():
                    best_id, best_score = None, 0.0
                    for gid, ch, ca in cfbd:
                        score = (sim(r["odds_home"], ch) + sim(r["odds_away"], ca)) / 2
                        if score > best_score:
                            best_score, best_id = score, gid
                    if best_score >= 0.70:
                        matched.append({
                            "game_id": best_id,
                            "spread": r["spread"], "over_under": r["over_under"],
                            "home_moneyline": r["home_moneyline"],
                            "away_moneyline": r["away_moneyline"],
                            "spread_open": None, "provider": r["provider"],
                        })
                if matched:
                    df = pd.DataFrame(matched).drop_duplicates("game_id")
                    print(f"✅ Lines from OddsBlaze ({book_name}): {len(df)} games matched")
                    return df
        except Exception as e:
            print(f"[WARN] OddsBlaze failed: {e} — falling back to CFBD")

    # Fallback: CFBD lines API
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception as e:
        print(f"[ERROR] CFBD lines also failed: {e}")
        return pd.DataFrame()
    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
    rank_map  = {p: i for i, p in enumerate(priority)}
    rows = []
    for game in data:
        for line in game.get("lines", []):
            rows.append({
                "game_id": game.get("id"),
                "spread": line.get("spread"),
                "over_under": line.get("overUnder"),
                "spread_open": line.get("spreadOpen"),
                "provider": line.get("provider"),
                "_rank": rank_map.get(line.get("provider", ""), 99),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("_rank").drop_duplicates("game_id", keep="first").drop(columns=["_rank"])
    print(f"✅ Lines from CFBD fallback: {len(df)} games")
    return df


# ─── Step 5: Build predictions ────────────────────────────────────────────────

def build_predictions(games, lines, spread_model, totals_model,
                      win_prob_model, feature_lists,
                      pred_season: int) -> pd.DataFrame:
    print("\n🤖 Building predictions...")

    # Merge lines
    if not lines.empty:
        ml_cols  = [c for c in ["home_moneyline", "away_moneyline"] if c in lines.columns]
        keep     = ["game_id", "spread", "over_under", "spread_open"] + ml_cols
        if "provider" in lines.columns:
            keep.append("provider")
        df = games.merge(lines[[c for c in keep if c in lines.columns]],
                         on="game_id", how="left")
    else:
        df = games.copy()
        df["spread"] = df["over_under"] = df["spread_open"] = np.nan

    for col in ["home_moneyline", "away_moneyline"]:
        if col not in df.columns:
            df[col] = np.nan

    # Load ratings/EPA/Elo (uses cached data files — no API calls)
    ratings = load_rating_sources(pred_season, DATA_DIR)
    epa     = load_recent_epa(pred_season, DATA_DIR)
    elo     = load_current_elo(pred_season, DATA_DIR)

    df = attach_team_features(df, ratings, epa, elo if not elo.empty else None)

    def make_feat(feat_names):
        out = pd.DataFrame(index=df.index)
        for f in feat_names:
            out[f] = df[f] if f in df.columns else np.nan
        return out

    out_cols = ["game_id", "season", "week", "home_team", "away_team",
                "neutral_site", "conference_game", "spread", "over_under",
                "spread_open", "home_moneyline", "away_moneyline"]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    if "provider" in df.columns:
        out["provider"] = df["provider"]

    out["pred_spread"] = spread_model.predict(make_feat(feature_lists["spread"]))
    out["pred_total"]  = totals_model.predict(make_feat(feature_lists["totals"]))
    out["pred_win_p"]  = win_prob_model.predict_proba(make_feat(feature_lists["win_prob"]))[:, 1]

    # Cross-calibration: spread-implied win probability
    calib_path = MODEL_DIR / "win_prob_calibration.json"
    if calib_path.exists():
        calib  = json.loads(calib_path.read_text())
        sigma  = calib["spread_sigma"]
        alpha  = calib["blend_alpha"]
        s_impl = out["pred_spread"].apply(lambda s: norm_cdf(s / sigma))
        out["pred_win_p"] = (alpha * s_impl + (1 - alpha) * out["pred_win_p"]).clip(0.01, 0.99)
    out["pred_away_win_p"] = 1 - out["pred_win_p"]

    # Edge calculations
    out["spread_edge"] = out["pred_spread"] - (-out["spread"])
    out["totals_edge"] = out["pred_total"]  - out["over_under"]

    out["home_ml_ev"] = out.apply(lambda r: ml_ev(r["pred_win_p"],      r["home_moneyline"]), axis=1)
    out["away_ml_ev"] = out.apply(lambda r: ml_ev(r["pred_away_win_p"], r["away_moneyline"]), axis=1)
    out["model_home_ml"] = out["pred_win_p"].apply(prob_to_american)
    out["model_away_ml"] = out["pred_away_win_p"].apply(prob_to_american)

    print(f"✅ Predictions generated for {len(out)} games")
    return out


# ─── Step 6: Filter picks & format summary ────────────────────────────────────

def filter_picks(predictions: pd.DataFrame) -> dict:
    """Apply edge thresholds and return categorised picks."""
    picks: dict = {"spreads": [], "totals": [], "moneylines": [], "all_games": []}

    for _, r in predictions.iterrows():
        game_label = f"{r['away_team']} @ {r['home_team']}"
        spread_val = r.get("spread")
        ou_val     = r.get("over_under")

        # All games summary (for context)
        picks["all_games"].append({
            "game": game_label,
            "pred_spread":  round(float(r["pred_spread"]), 1),
            "vegas_spread": round(float(spread_val), 1) if pd.notna(spread_val) else None,
            "pred_total":   round(float(r["pred_total"]),  1),
            "vegas_total":  round(float(ou_val), 1) if pd.notna(ou_val) else None,
            "home_win_pct": round(float(r["pred_win_p"]) * 100, 1),
        })

        # Spread picks
        edge = r.get("spread_edge")
        if pd.notna(edge) and pd.notna(spread_val):
            if SPREAD_EDGE_MIN <= abs(float(edge)) <= SPREAD_EDGE_MAX:
                team = r["home_team"] if float(edge) > 0 else r["away_team"]
                vegas_line = (-float(spread_val)) if float(edge) > 0 else float(spread_val)
                picks["spreads"].append({
                    "game": game_label,
                    "pick": f"{team} {vegas_line:+.1f}",
                    "edge_pts": round(float(edge), 1),
                    "units": kelly_spread(float(edge)),
                    "model_spread": round(float(r["pred_spread"]), 1),
                    "vegas_spread": round(float(spread_val), 1),
                })

        # Totals picks
        t_edge = r.get("totals_edge")
        if pd.notna(t_edge) and pd.notna(ou_val):
            if abs(float(t_edge)) >= TOTALS_EDGE_MIN:
                direction = "OVER" if float(t_edge) > 0 else "UNDER"
                picks["totals"].append({
                    "game": game_label,
                    "pick": f"{direction} {float(ou_val):.1f}",
                    "edge_pts": round(float(t_edge), 1),
                    "units": kelly_spread(float(t_edge)),
                    "model_total": round(float(r["pred_total"]), 1),
                    "vegas_total": round(float(ou_val), 1),
                })

        # Moneyline picks
        for side, ev_col, odds_col, pct_col in [
            ("home", "home_ml_ev", "home_moneyline", "pred_win_p"),
            ("away", "away_ml_ev", "away_moneyline",  "pred_away_win_p"),
        ]:
            ev   = r.get(ev_col)
            odds = r.get(odds_col)
            if pd.notna(ev) and float(ev) >= MONEYLINE_EV_MIN:
                team = r["home_team"] if side == "home" else r["away_team"]
                picks["moneylines"].append({
                    "game": game_label,
                    "pick": team,
                    "ev_pct": round(float(ev) * 100, 1),
                    "book_odds": int(odds) if pd.notna(odds) else None,
                    "model_win_pct": round(float(r[pct_col]) * 100, 1),
                    "units": kelly_ml(float(ev)),
                })

    # Sort by edge desc
    picks["spreads"].sort(key=lambda x: abs(x["edge_pts"]), reverse=True)
    picks["totals"].sort(key=lambda x: abs(x["edge_pts"]), reverse=True)
    picks["moneylines"].sort(key=lambda x: x["ev_pct"], reverse=True)

    return picks


def format_imessage(picks: dict, season: int, week: int) -> str:
    """
    Format a compact iMessage text with all actionable picks for the week.
    Keeps it readable on a phone screen.
    """
    lines = [
        f"🏈 CFB Model — Week {week}, {season}",
        f"Generated {datetime.now().strftime('%a %b %-d @ %-I:%M %p')}",
        "",
    ]

    total = len(picks["spreads"]) + len(picks["totals"]) + len(picks["moneylines"])
    if total == 0:
        lines.append("No picks above threshold this week.")
        return "\n".join(lines)

    if picks["spreads"]:
        lines.append(f"📊 SPREAD ({len(picks['spreads'])})")
        for p in picks["spreads"][:5]:
            lines.append(f"  {p['pick']}  |  edge {p['edge_pts']:+.1f}  |  {p['units']}u")

    if picks["totals"]:
        lines.append(f"\n🔢 TOTALS ({len(picks['totals'])})")
        for p in picks["totals"][:5]:
            lines.append(f"  {p['pick']} ({p['game'].split('@')[0].strip()} @ {p['game'].split('@')[1].strip()})  |  edge {p['edge_pts']:+.1f}  |  {p['units']}u")

    if picks["moneylines"]:
        lines.append(f"\n💵 MONEYLINE ({len(picks['moneylines'])})")
        for p in picks["moneylines"][:5]:
            odds_str = f"{p['book_odds']:+d}" if p["book_odds"] else "N/A"
            lines.append(f"  {p['pick']} ({odds_str})  |  EV {p['ev_pct']:+.1f}%  |  {p['units']}u")

    lines.append(f"\n{total} total picks. Check the app for full details.")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CFB Weekly Pipeline")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip model retraining (data refresh + picks only)")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week",   type=int, default=None)
    args = parser.parse_args()

    season, week = current_cfb_week()
    if args.season: season = args.season
    if args.week:   week   = args.week

    print(f"\n🚀 CFB Weekly Pipeline — Season {season}, Week {week}")
    print(f"   Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Retrain
    if not args.skip_retrain:
        ok = retrain()
        if not ok:
            print("[ABORT] Retrain failed. Picks not generated.")
            sys.exit(1)

    # Step 2: Load models
    spread_m, totals_m, win_prob_m, feat_lists = load_models()

    # Step 3: Schedule
    games = fetch_schedule(season, week)
    if games.empty:
        print("[ABORT] No games found for this week.")
        sys.exit(0)

    # Step 4: Lines
    lines = fetch_lines(games, season, week)

    # Step 5: Predictions
    preds = build_predictions(games, lines, spread_m, totals_m,
                               win_prob_m, feat_lists, season)

    # Step 6: Filter & format
    picks   = filter_picks(preds)
    message = format_imessage(picks, season, week)

    # Write output JSON for the scheduled task to read
    output = {
        "run_at": datetime.now().isoformat(),
        "season": season,
        "week":   week,
        "picks":  picks,
        "imessage_text": message,
        "games_count": len(games),
        "lines_count":  len(lines) if not lines.empty else 0,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    # Print iMessage text to stdout (picked up by scheduled task)
    print("\n" + "=" * 60)
    print("iMESSAGE SUMMARY")
    print("=" * 60)
    print(message)
    print("=" * 60)
    print(f"\n✅ Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
