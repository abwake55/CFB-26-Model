#!/usr/bin/env python3
"""
CFB Picks Model — Weekly Newsletter Generator
==============================================
Generates and emails a weekly picks newsletter every Tuesday containing:
  • This week's top model picks (spread + totals, with edge/confidence/Kelly)
  • Last week's results (how each flagged pick did)
  • Season ATS record and unit P&L
  • Context notes (weather flags, new-P4 road teams, notable situations)

Picks are saved to outputs/picks/YYYY_W{wk}.json each week so results
can be checked the following Tuesday.

Usage:
  # Send newsletter (requires SENDER_EMAIL + SENDER_APP_PASSWORD env vars):
  /opt/homebrew/bin/python3 scripts/generate_newsletter.py

  # Preview HTML in browser without sending:
  /opt/homebrew/bin/python3 scripts/generate_newsletter.py --preview

  # Test with dummy data (no API calls):
  /opt/homebrew/bin/python3 scripts/generate_newsletter.py --dry-run

Email secrets (add to GitHub Actions secrets):
  SENDER_EMAIL        — Gmail address used to send (e.g. cfbmodel2026@gmail.com)
  SENDER_APP_PASSWORD — Gmail app password (not your main password)
  RECIPIENT_EMAIL     — Where to send it (defaults to alexwaked@me.com)
"""

import os, sys, json, smtplib, argparse, time, math
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import requests
import pandas as pd
import numpy as np

ROOT      = Path(__file__).parent.parent
PICKS_DIR = ROOT / "outputs" / "picks"
PICKS_DIR.mkdir(parents=True, exist_ok=True)

# ── API key ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    key = os.getenv("CFB_API_KEY", "")
    if key:
        return key
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        for line in path.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

CFB_KEY      = load_api_key()
CFB_BASE     = "https://api.collegefootballdata.com"
EDGE_MIN_SP  = 3.0   # minimum spread edge to flag as a pick
EDGE_MIN_TOT = 2.0   # minimum totals edge (CORE gate: unders, power-conf)
ML_EV_MIN    = 0.04  # minimum moneyline EV (matches app MONEYLINE_EV_MIN)
SPREAD_MAX_WEEK = 9  # spreads hit 44.7% ATS wk10+ in walk-forward — no picks
RECIPIENT    = os.getenv("RECIPIENT_EMAIL", "alexwaked@me.com")

# Walk-forward 2019-25: totals edge only exists on unders in games with a
# power-conference team (55.8%, n=868). Same gate as the app's CORE PLAY.
POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
               "Big East", "FBS Independents"}

def cfb_get(endpoint, params=None):
    headers = {"Authorization": f"Bearer {CFB_KEY}"}
    r = requests.get(f"{CFB_BASE}/{endpoint}", headers=headers,
                     params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

# ── Week / season detection ───────────────────────────────────────────────────

def current_cfb_season() -> int:
    today = dt.date.today()
    return today.year if today.month >= 7 else today.year - 1

def current_cfb_week(season: int) -> int:
    """Estimate the upcoming game week based on today's date."""
    today = dt.date.today()
    # CFB Week 1 typically starts last week of August
    # Use a rough mapping: week = floor((today - Aug 24) / 7) + 1
    season_start = dt.date(season, 8, 24)
    if today < season_start:
        return 1
    weeks_elapsed = (today - season_start).days // 7
    return min(weeks_elapsed + 1, 16)

def last_cfb_week(season: int) -> int:
    return max(1, current_cfb_week(season) - 1)

# ── Model prediction ─────────────────────────────────────────────────────────
# Predictions come from weekly_pipeline's full feature stack (ratings, EPA,
# Elo, travel, portal, etc.). The old inline path here fed mostly-NaN feature
# frames to the models and produced absurd edges on nearly every game.

def predict_games(season: int, week: int) -> list:
    """
    Fetch schedule + lines and run the full weekly_pipeline prediction path.
    Returns list of pick dicts for games meeting the gated thresholds.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import weekly_pipeline as wp

    games = wp.fetch_schedule(season, week)
    if games.empty:
        return []
    lines = wp.fetch_lines(games, season, week)
    spread_model, totals_model, win_prob_model, feature_lists = wp.load_models()
    df = wp.build_predictions(games, lines, spread_model, totals_model,
                              win_prob_model, feature_lists,
                              pred_season=season)
    df["vegas_home_margin"] = -pd.to_numeric(df["spread"], errors="coerce")

    # Build picks list
    picks = []
    for _, row in df.iterrows():
        start = row.get("start_date", "")
        kickoff = _format_kickoff(start)

        week_n = int(row["week"]) if pd.notna(row["week"]) else 0
        power_involved = (row.get("home_conference") in POWER_CONFS
                          or row.get("away_conference") in POWER_CONFS)

        # Spread pick — weeks 1-9 only (wk10+ hit 44.7% ATS in walk-forward)
        sp_edge = row["spread_edge"]
        if (pd.notna(sp_edge) and abs(sp_edge) >= EDGE_MIN_SP
                and 1 <= week_n <= SPREAD_MAX_WEEK):
            home_bet = sp_edge > 0
            bet_team = row["home_team"] if home_bet else row["away_team"]
            vl = row["spread"] if home_bet else -row["spread"]
            picks.append({
                "type":      "SPREAD",
                "tier":      "WATCH" if week_n <= 3 else "INFO",
                "game_id":   row["game_id"],
                "week":      int(row["week"]) if pd.notna(row["week"]) else 0,
                "matchup":   f"{row['home_team']} vs {row['away_team']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on":    bet_team,
                "line":      f"{vl:+.1f}",
                "edge":      round(float(sp_edge), 1),
                "pred":      round(float(row["pred_spread"]), 1),
                "kickoff":   kickoff,
                "stars":     _stars(abs(sp_edge)),
                "kelly":     _kelly(abs(sp_edge)),
                "start_date": str(start),
                "wind_speed": row.get("wind_speed"),
                "is_dome":    row.get("is_dome", 0),
                "neutral":    bool(row.get("neutral_site", 0)),
            })

        # Totals pick — CORE gate only: unders, edge>=2, power-conf involved.
        # Overs (49.4%) and G5vG5 (48.2%) had no walk-forward edge.
        tot_edge = row["totals_edge"]
        ou_str   = f"{row['over_under']:.1f}" if pd.notna(row["over_under"]) else "TBD"
        # CORE gate: edge 2-7 (hit rate FALLS with edge — 59.9% at 2-3 vs
        # 50.7% at 7+), power-conf, market total >= 48 (low totals have no
        # over-bias to fade, ~49%). Wind>=15 is gated app-side at kickoff.
        _ou = pd.to_numeric(row.get("over_under"), errors="coerce")
        both_power = (row.get("home_conference") in POWER_CONFS
                      and row.get("away_conference") in POWER_CONFS)
        is_core = (pd.notna(tot_edge) and -7.0 <= tot_edge <= -EDGE_MIN_TOT
                   and power_involved and pd.notna(_ou) and _ou >= 48)
        if is_core:
            picks.append({
                "type":      "TOTAL",
                # Both-power CORE measures 59.8% (n=413) vs 55.0% (n=151) for
                # power-vs-G5 — marquee games draw the most public over money.
                "tier":      "CORE+" if both_power else "CORE",
                "game_id":   row["game_id"],
                "week":      int(row["week"]) if pd.notna(row["week"]) else 0,
                "matchup":   f"{row['home_team']} vs {row['away_team']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on":    "UNDER",
                "line":      ou_str,
                "edge":      round(float(tot_edge), 1),
                "pred":      round(float(row["pred_total"]), 1),
                "kickoff":   kickoff,
                "stars":     _stars(abs(tot_edge)),
                # 2u portfolio base (quarter-Kelly on 58.5% ≈ 3.2%); the
                # both-power slice earns 3u (its own quarter-Kelly is 3.9%).
                "kelly":     3 if both_power else 2,
                "start_date": str(start),
                "wind_speed": row.get("wind_speed"),
                "is_dome":    row.get("is_dome", 0),
                "neutral":    bool(row.get("neutral_site", 0)),
            })

        # High-total fade — PAPER only. Bet UNDER whenever the market total is
        # >= 60, independent of the model: walk-forward 2019-25 (excluding CORE)
        # 53.8% over 974 bets, +2.7% ROI, p=0.019, 5/7 seasons. Edge is uniform
        # whether the model leans under (53.5%) or over (54.0%), so this is
        # market over-inflation on shootout games, not model skill. Thin, so 0u
        # until a live season corroborates it.
        elif pd.notna(_ou) and _ou >= 60:
            picks.append({
                "type":      "TOTAL_HT",
                "tier":      "PAPER",
                "game_id":   row["game_id"],
                "week":      int(row["week"]) if pd.notna(row["week"]) else 0,
                "matchup":   f"{row['home_team']} vs {row['away_team']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on":    "UNDER",
                "line":      ou_str,
                "edge":      round(float(tot_edge), 1) if pd.notna(tot_edge) else 0.0,
                "pred":      round(float(row["pred_total"]), 1) if pd.notna(row["pred_total"]) else 0.0,
                "kickoff":   kickoff,
                "stars":     "",
                "kelly":     0,   # paper — graded at nominal 1u for the record
                "start_date": str(start),
                "wind_speed": row.get("wind_speed"),
                "is_dome":    row.get("is_dome", 0),
                "neutral":    bool(row.get("neutral_site", 0)),
            })

        # Moneyline pick — best-EV side, threshold matches the app
        win_p = row["pred_win_p"]
        if pd.notna(win_p):
            home_ml = pd.to_numeric(row.get("home_moneyline"), errors="coerce")
            away_ml = pd.to_numeric(row.get("away_moneyline"), errors="coerce")
            h_ev = _ml_ev(win_p, home_ml)
            a_ev = _ml_ev(1 - win_p, away_ml)
            best = max([(h_ev, row["home_team"], home_ml, win_p),
                        (a_ev, row["away_team"], away_ml, 1 - win_p)],
                       key=lambda t: (t[0] if not pd.isna(t[0]) else -9))
            ev, ml_team, ml_odds, ml_p = best
            # Odds band guard: tail miscalibration dominates apparent EV on
            # long odds, so only quote prices a subscriber can sanely bet.
            # Walk-forward 2023-25 (n=868): +13.5% ROI in '23, +0.6% '24,
            # -3.7% '25, z=0.77 — unvalidated. Tracked as a PAPER record
            # (graded at flat 1u to keep the ROI history) but never sized.
            if (not pd.isna(ev) and ev >= ML_EV_MIN
                    and -300 <= ml_odds <= 300):
                odds_str = f"+{int(ml_odds)}" if ml_odds > 0 else str(int(ml_odds))
                picks.append({
                    "type":      "MONEYLINE",
                    "tier":      "PAPER",
                    "game_id":   row["game_id"],
                    "week":      week_n,
                    "matchup":   f"{row['home_team']} vs {row['away_team']}",
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_on":    ml_team,
                    "line":      odds_str,
                    "odds":      float(ml_odds),
                    "edge":      round(float(ev) * 100, 1),   # EV in % for display/sort
                    "pred":      round(float(ml_p) * 100, 1), # model win prob %
                    "kickoff":   kickoff,
                    "stars":     "★★★" if ev >= 0.07 else ("★★" if ev >= 0.05 else "★"),
                    "kelly":     0,   # paper — displayed as "paper", graded at 1u nominal
                    "start_date": str(start),
                    "wind_speed": row.get("wind_speed"),
                    "is_dome":    row.get("is_dome", 0),
                    "neutral":    bool(row.get("neutral_site", 0)),
                })

    # CORE totals lead, then spreads, then moneylines; edge desc within each
    # (edge is points for spread/total but EV% for ML — never sort across types)
    type_rank = {"TOTAL": 0, "SPREAD": 1, "MONEYLINE": 2, "TOTAL_HT": 3}
    picks.sort(key=lambda p: (type_rank.get(p["type"], 9), -abs(p["edge"])))
    return picks


def _ml_ev(model_prob, american_odds):
    """Expected value of a 1u moneyline bet (same formula as app.py)."""
    if pd.isna(model_prob) or pd.isna(american_odds) or american_odds == 0:
        return np.nan
    payout = 100 / abs(american_odds) if american_odds < 0 else american_odds / 100
    return model_prob * payout - (1 - model_prob)

def _format_kickoff(start_date) -> str:
    if not start_date or pd.isna(start_date):
        return ""
    try:
        ts = pd.to_datetime(start_date, utc=True)
        et = ts - dt.timedelta(hours=4)
        day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][et.weekday()]
        h, m = et.hour % 12 or 12, et.minute
        ap = "AM" if et.hour < 12 else "PM"
        return f"{day} {et.month}/{et.day} · {h}:{m:02d} {ap} ET"
    except Exception:
        return ""

def _stars(edge_abs: float) -> str:
    if edge_abs >= 5.5: return "★★★"
    if edge_abs >= 4.0: return "★★"
    return "★"

def _kelly(edge_abs: float) -> int:
    win_p = min(0.5238 + edge_abs * 0.005, 0.60)
    b = 100 / 110
    k = max((win_p * b - (1 - win_p)) / b, 0.0)
    u = k * 0.25 * 100
    return max(1, min(3, round(u)))

# ── Last week results ─────────────────────────────────────────────────────────

def _fetch_closing_lines(season: int, week: int) -> dict:
    """
    Closing lines keyed by game_id. CFBD lines stop updating at kickoff, so
    fetched after the games they ARE the closing numbers. Same provider
    priority as pick generation.
    """
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
    except Exception as e:
        print(f"  Warning: could not fetch closing lines: {e}")
        return {}
    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
    out = {}
    for game in data:
        best, best_rank = None, 999
        for line in game.get("lines", []):
            rank = next((i for i, p in enumerate(priority)
                         if p.lower() in (line.get("provider") or "").lower()), 999)
            if rank < best_rank and line.get("spread") is not None:
                best_rank, best = rank, line
        if best:
            out[game.get("id")] = {
                "spread":     best.get("spread"),
                "over_under": best.get("overUnder"),
                "home_ml":    best.get("homeMoneyline"),
                "away_ml":    best.get("awayMoneyline"),
            }
    return out


def _attach_clv(pick: dict, close: dict) -> None:
    """
    Set clv (points; positive = published number beat the close) and
    clv_beat on the pick, in place. ML uses payout comparison instead of
    points. No-ops when the closing number is missing.
    """
    if not close:
        return
    try:
        if pick["type"] == "SPREAD":
            if close.get("spread") is None:
                return
            home_bet = pick["edge"] > 0
            pick_vl  = float(str(pick["line"]).replace("+", ""))
            close_vl = float(close["spread"]) if home_bet else -float(close["spread"])
            pick["closing_line"] = f"{close_vl:+.1f}"
            pick["clv"] = round(pick_vl - close_vl, 1)   # -7 vs close -8.5 → +1.5
            pick["clv_beat"] = pick["clv"] > 0
        elif pick["type"] in ("TOTAL", "TOTAL_HT"):
            if close.get("over_under") is None or pick["line"] == "TBD":
                return
            close_ou = float(close["over_under"])
            pick_ou  = float(pick["line"])
            pick["closing_line"] = f"{close_ou:.1f}"
            clv = (pick_ou - close_ou) if pick["bet_on"] == "UNDER" else (close_ou - pick_ou)
            pick["clv"] = round(clv, 1)
            pick["clv_beat"] = clv > 0
        elif pick["type"] == "MONEYLINE":
            key = "home_ml" if pick["bet_on"] == pick.get("home_team") else "away_ml"
            if close.get(key) is None:
                return
            close_ml = float(close[key])
            pick_ml  = float(pick.get("odds") or 0)
            if not pick_ml or not close_ml:
                return
            pay = lambda o: 100 / abs(o) if o < 0 else o / 100
            pick["closing_line"] = f"+{int(close_ml)}" if close_ml > 0 else str(int(close_ml))
            pick["clv"] = round((pay(pick_ml) - pay(close_ml)) * 100, 1)  # payout cents
            pick["clv_beat"] = pick["clv"] > 0
    except (TypeError, ValueError):
        return


def load_last_week_results(season: int, week: int) -> tuple[list, dict]:
    """
    Load saved picks from last week and fetch actual results.
    Returns (picks_with_outcomes, summary_stats).
    """
    picks_file = PICKS_DIR / f"{season}_W{week:02d}.json"
    if not picks_file.exists():
        return [], {}

    with open(picks_file) as f:
        saved_picks = json.load(f)

    if not saved_picks:
        return [], {}

    # Capture closing lines once, persist into the picks file so re-grades
    # (season record recomputes every week) never refetch or drift.
    if any("closing_line" not in p for p in saved_picks):
        closing = _fetch_closing_lines(season, week)
        if closing:
            for p in saved_picks:
                if "closing_line" not in p:
                    _attach_clv(p, closing.get(p.get("game_id"), {}))
            try:
                with open(picks_file, "w") as f:
                    json.dump(saved_picks, f, indent=2)
            except OSError as e:
                print(f"  Warning: could not persist closing lines: {e}")

    # Fetch actual results from CFBD
    try:
        games_data = cfb_get("games", params={
            "year": season, "week": week, "seasonType": "regular",
            "division": "fbs"
        })
        results = {g["id"]: g for g in games_data}
    except Exception:
        return saved_picks, {}

    stats = {t: {"wins": 0, "losses": 0, "pushes": 0, "units": 0.0}
             for t in ("SPREAD", "TOTAL", "TOTAL_HT", "MONEYLINE")}
    annotated = []

    def grade(pick, outcome, pnl=0.0):
        s = stats.get(pick["type"])
        pick["outcome"] = outcome
        pick["pnl"] = round(pnl, 2)
        if s is not None and outcome in ("win", "loss", "push"):
            key = {"win": "wins", "loss": "losses", "push": "pushes"}[outcome]
            s[key] += 1
            s["units"] += pnl

    for pick in saved_picks:
        gid  = pick.get("game_id")
        game = results.get(gid, {})
        if not game or not game.get("completed"):
            pick["outcome"] = "pending"
            annotated.append(pick)
            continue

        hp = game.get("homePoints", 0) or 0
        ap = game.get("awayPoints", 0) or 0
        diff = hp - ap
        total = hp + ap
        score_str = f"{game.get('homeTeam','?')} {int(hp)} – {int(ap)} {game.get('awayTeam','?')}"

        pick["score"] = score_str
        pick["actual_diff"]  = diff
        pick["actual_total"] = total

        if pick["type"] == "SPREAD":
            spread_val = float(str(pick.get("line", 0)).replace("+", "") or 0)
            home_bet = pick["edge"] > 0
            covered = (diff + spread_val > 0) if home_bet else (diff + spread_val < 0)
            if diff + spread_val == 0:
                grade(pick, "push")
            elif covered:
                grade(pick, "win", 1.0)
            else:
                grade(pick, "loss", -1.1)
        elif pick["type"] in ("TOTAL", "TOTAL_HT"):
            ou = float(pick["line"]) if pick["line"] != "TBD" else None
            if ou is None:
                pick["outcome"] = "pending"
            elif total == ou:
                grade(pick, "push")
            elif (total > ou) == (pick["bet_on"] == "OVER"):
                grade(pick, "win", 1.0)
            else:
                grade(pick, "loss", -1.1)
        elif pick["type"] == "MONEYLINE":
            odds = float(pick.get("odds") or
                         str(pick.get("line", "0")).replace("+", "") or 0)
            bet_home = pick["bet_on"] == pick.get("home_team")
            if diff == 0:
                grade(pick, "push")
            elif (diff > 0) == bet_home:
                payout = 100 / abs(odds) if odds < 0 else odds / 100
                grade(pick, "win", payout)
            else:
                grade(pick, "loss", -1.0)

        annotated.append(pick)

    for s in stats.values():
        s["units"] = round(s["units"], 1)
    stats["all"] = {
        k: round(sum(stats[t][k] for t in ("SPREAD", "TOTAL", "TOTAL_HT",
                                           "MONEYLINE")), 1)
        for k in ("wins", "losses", "pushes", "units")
    }
    clv_graded = [p for p in annotated if p.get("clv_beat") is not None]
    stats["clv"] = {"beat": sum(1 for p in clv_graded if p["clv_beat"]),
                    "n": len(clv_graded)}
    return annotated, stats

# ── Season record ─────────────────────────────────────────────────────────────

def compute_season_record(season: int) -> dict:
    """
    Cumulative season record from saved weekly picks files, kept separate
    per market (SPREAD / TOTAL / MONEYLINE) — each market's record reflects
    its own validated edge and is never blended into one number.
    """
    markets = ("SPREAD", "TOTAL", "TOTAL_HT", "MONEYLINE")
    agg = {t: {"wins": 0, "losses": 0, "pushes": 0, "units": 0.0} for t in markets}
    weeks_counted = 0
    clv_beat = clv_total = 0

    for f in sorted(PICKS_DIR.glob(f"{season}_W*.json")):
        try:
            week_num = int(f.stem.split("_W")[1])
        except Exception:
            continue
        picks, stats = load_last_week_results(season, week_num)
        if not stats:
            continue
        weeks_counted += 1
        for t in markets:
            for k in ("wins", "losses", "pushes"):
                agg[t][k] += stats.get(t, {}).get(k, 0)
            agg[t]["units"] += stats.get(t, {}).get("units", 0.0)
        clv_beat  += stats.get("clv", {}).get("beat", 0)
        clv_total += stats.get("clv", {}).get("n", 0)

    for t in markets:
        bets = agg[t]["wins"] + agg[t]["losses"]
        agg[t]["bets"]    = bets
        agg[t]["win_pct"] = round(agg[t]["wins"] / bets * 100, 1) if bets else 0.0
        agg[t]["units"]   = round(agg[t]["units"], 1)

    agg["weeks"] = weeks_counted
    agg["bets"]  = sum(agg[t]["bets"] for t in markets)
    agg["units"] = round(sum(agg[t]["units"] for t in markets), 1)
    agg["clv"]   = {"beat": clv_beat, "n": clv_total}
    return agg

# ── Context notes ─────────────────────────────────────────────────────────────

NEW_P4_2026 = {
    "Oregon","Washington","USC","UCLA","Texas","Oklahoma",
    "Arizona","Arizona State","Colorado","Utah","California","Stanford","SMU"
}

def generate_context_notes(picks: list) -> list:
    """Auto-generate one-line context notes for notable situations."""
    notes = []
    seen_games = set()

    for p in picks:
        key = p["matchup"]
        if key in seen_games:
            continue
        seen_games.add(key)

        wind = p.get("wind_speed")
        is_dome = p.get("is_dome", 0)

        # Wind warning
        if wind and not is_dome and float(wind) >= 15:
            mph = float(wind)
            notes.append(f"💨 {p['matchup']}: {mph:.0f} mph wind forecast — "
                         f"{'strong' if mph >= 20 else 'mild'} under lean")

        # New-P4 road team
        away = p.get("away_team", "")
        home = p.get("home_team", "")
        if away in NEW_P4_2026 and not p.get("neutral", False):
            notes.append(f"🔄 {away} is in Year 2 of the {_get_new_conf(away)} — "
                         f"new-conference road games historically under-perform spreads")

        # Dome note
        if is_dome:
            notes.append(f"🏟️ {p['matchup']} played in a dome — weather not a factor")

        # Postseason
        if p.get("type") == "TOTAL" and p.get("week", 0) >= 15:
            notes.append(f"🏆 Late-season/bowl game: rolling EPA features may be "
                         f"stale — treat totals edge with extra caution")

    return notes[:6]  # cap at 6 notes

def _get_new_conf(team: str) -> str:
    conf_map = {
        "Oregon":"Big Ten","Washington":"Big Ten","USC":"Big Ten","UCLA":"Big Ten",
        "Texas":"SEC","Oklahoma":"SEC",
        "Arizona":"Big 12","Arizona State":"Big 12","Colorado":"Big 12","Utah":"Big 12",
        "California":"ACC","Stanford":"ACC","SMU":"ACC",
    }
    return conf_map.get(team, "new conference")

# ── Save weekly picks ─────────────────────────────────────────────────────────

def save_weekly_picks(picks: list, season: int, week: int):
    """Save this week's picks so we can grade them next Tuesday."""
    path = PICKS_DIR / f"{season}_W{week:02d}.json"
    # Strip non-serializable fields
    clean = []
    for p in picks:
        c = {k: v for k, v in p.items()
             if isinstance(v, (str, int, float, bool, type(None)))}
        clean.append(c)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  Saved {len(clean)} picks → {path.name}")

# ── HTML generation ────────────────────────────────────────────────────────────

def build_html(season: int, this_week: int, picks: list,
               last_results: list, last_stats: dict,
               season_record: dict, notes: list) -> str:

    last_week = this_week - 1
    has_results = bool(last_results and last_stats)
    has_picks   = bool(picks)

    # ── Picks table rows ──────────────────────────────────────────────────────
    def pick_row(p: dict) -> str:
        is_ml = p["type"] == "MONEYLINE"
        is_ht = p["type"] == "TOTAL_HT"
        edge_color = "#22c55e" if abs(p["edge"]) >= 4.5 else "#94a3b8"
        type_color = ("#eab308" if is_ml else
                      "#8b5cf6" if p["type"] == "SPREAD" else
                      "#06b6d4" if p["bet_on"] == "UNDER" else "#f97316")
        # The high-total fade ignores the model, so an edge number would imply
        # model support it does not have.
        edge_str = ("market fade" if is_ht else
                    f"EV +{p['edge']:.1f}%" if is_ml else f"{p['edge']:+.1f}")
        wind_note = ""
        ws = p.get("wind_speed")
        if ws and not p.get("is_dome") and float(ws) >= 15:
            wind_note = f' &nbsp;<span style="color:#f97316;font-size:0.8em">💨 {float(ws):.0f} mph</span>'
        dome_note = ' &nbsp;<span style="color:#6b7280;font-size:0.8em">🏟️ DOME</span>' if p.get("is_dome") else ""
        neutral = ' &nbsp;<span style="color:#6b7280;font-size:0.8em">Neutral</span>' if p.get("neutral") else ""

        return f"""
        <tr>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0">
            <span style="background:{type_color};color:#fff;font-size:0.7em;
                         font-weight:700;padding:2px 7px;border-radius:4px;
                         letter-spacing:0.08em">{p['type']}</span>
            &nbsp;<strong>{p['bet_on']}</strong> {p['line']}
            {wind_note}{dome_note}{neutral}
            <br><span style="color:#64748b;font-size:0.85em">{p['matchup']}</span>
            &nbsp;&nbsp;<span style="color:#94a3b8;font-size:0.8em">{p['kickoff']}</span>
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;
                     text-align:center;color:{edge_color};font-weight:700">
            {edge_str}
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;
                     text-align:center;color:#eab308">{p['stars']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;
                     text-align:center;font-weight:700">{'paper' if (is_ml or is_ht) else f"{p['kelly']}u"}</td>
        </tr>"""

    picks_rows = "".join(pick_row(p) for p in picks[:12]) if has_picks else \
        '<tr><td colspan="4" style="padding:16px;color:#94a3b8;text-align:center">No picks meet the edge threshold this week</td></tr>'

    # ── Results rows ──────────────────────────────────────────────────────────
    def result_row(p: dict) -> str:
        outcome = p.get("outcome", "pending")
        icon = {"win":"✅","loss":"❌","push":"➖","pending":"⏳"}.get(outcome, "⏳")
        score = p.get("score", "Pending")
        pnl   = f"{p['pnl']:+.1f}u" if outcome in ("win", "loss") and "pnl" in p else "—"
        pnl_color = "#22c55e" if outcome == "win" else ("#ef4444" if outcome == "loss" else "#94a3b8")
        if p.get("clv_beat") is None:
            clv_str, clv_color = "—", "#94a3b8"
        else:
            unit = "¢" if p["type"] == "MONEYLINE" else ""
            clv_str = f"{p['clv']:+.1f}{unit}"
            clv_color = "#22c55e" if p["clv_beat"] else "#ef4444"
        return f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0">{icon} <strong>{p['bet_on']}</strong> {p['line']} &nbsp;<span style="color:#64748b;font-size:0.85em">{p['matchup']}</span></td>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0;color:#64748b;font-size:0.85em">{score}</td>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0;color:{clv_color};font-weight:600;text-align:center;font-size:0.85em">{clv_str}</td>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0;color:{pnl_color};font-weight:700;text-align:center">{pnl}</td>
        </tr>"""

    MARKET_LABELS = [("TOTAL", "O/U"), ("SPREAD", "Spread"),
                     ("TOTAL_HT", "High-tot"), ("MONEYLINE", "ML")]

    def market_summary(stats: dict) -> str:
        parts = []
        for key, label in MARKET_LABELS:
            s = stats.get(key, {})
            if (s.get("wins", 0) + s.get("losses", 0) + s.get("pushes", 0)) == 0:
                continue
            parts.append(f"{label} {s['wins']}–{s['losses']}"
                         + (f"–{s['pushes']}" if s.get("pushes") else ""))
        return " &nbsp;·&nbsp; ".join(parts)

    results_section = ""
    if has_results:
        results_rows = "".join(result_row(p) for p in last_results)
        all_units = last_stats.get("all", {}).get("units", 0.0)
        units_color = "#22c55e" if all_units >= 0 else "#ef4444"
        clv = last_stats.get("clv", {})
        clv_str = (f' &nbsp;|&nbsp; Beat close: <strong>{clv["beat"]}/{clv["n"]}</strong>'
                   if clv.get("n") else "")
        results_section = f"""
        <h2 style="color:#1e293b;font-size:1.1em;margin:28px 0 4px 0">
          📊 Last Week's Results — Week {last_week}
        </h2>
        <p style="color:#64748b;margin:0 0 12px 0;font-size:0.9em">
          {market_summary(last_stats)}
          &nbsp;|&nbsp; <span style="color:{units_color};font-weight:700">{all_units:+.1f}u</span>{clv_str}
        </p>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:0.9em">
          <thead>
            <tr style="background:#f8fafc">
              <th style="padding:8px;text-align:left;color:#64748b;font-weight:600">Pick</th>
              <th style="padding:8px;text-align:left;color:#64748b;font-weight:600">Result</th>
              <th style="padding:8px;text-align:center;color:#64748b;font-weight:600">CLV</th>
              <th style="padding:8px;text-align:center;color:#64748b;font-weight:600">P&L</th>
            </tr>
          </thead>
          <tbody>{results_rows}</tbody>
        </table>"""

    # ── Season record — one line per market, never blended ───────────────────
    sr = season_record
    record_section = ""
    if sr.get("bets", 0) > 0:
        market_rows = []
        for key, label in [("TOTAL", "Over/Under (CORE)"), ("SPREAD", "Spread"),
                           ("TOTAL_HT", "High-total unders (paper)"),
                           ("MONEYLINE", "Moneyline (paper)")]:
            m = sr.get(key, {})
            if m.get("bets", 0) == 0:
                continue
            push_str = f"–{m['pushes']}" if m.get("pushes") else ""
            u_color = "#22c55e" if m["units"] >= 0 else "#ef4444"
            market_rows.append(f"""
            <tr>
              <td style="padding:6px 8px;color:#64748b;font-weight:600">{label}</td>
              <td style="padding:6px 8px;text-align:center">
                <strong>{m['wins']}–{m['losses']}{push_str}</strong>
                <span style="color:#94a3b8;font-size:0.85em"> ({m['win_pct']:.1f}%)</span></td>
              <td style="padding:6px 8px;text-align:right;font-weight:700;
                         color:{u_color}">{m['units']:+.1f}u</td>
            </tr>""")
        sclv = sr.get("clv", {})
        clv_line = ""
        if sclv.get("n"):
            pct = sclv["beat"] / sclv["n"] * 100
            clv_color = "#22c55e" if pct >= 50 else "#ef4444"
            clv_line = f"""
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;
                      color:#64748b;font-size:0.85em">
            Closing line value: beat the close on
            <strong style="color:{clv_color}">{sclv['beat']}/{sclv['n']} ({pct:.0f}%)</strong>
            of picks — the strongest long-run indicator of real edge
          </div>"""
        record_section = f"""
        <div style="background:#f8fafc;border-radius:8px;padding:16px;margin:24px 0">
          <h2 style="color:#1e293b;font-size:1.0em;margin:0 0 10px 0">
            📈 {season} Season Record
            <span style="color:#94a3b8;font-size:0.75em;font-weight:400">
              · {sr['weeks']} week{'s' if sr['weeks'] != 1 else ''} tracked · graded at published lines</span>
          </h2>
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-collapse:collapse;font-size:0.92em">
            {''.join(market_rows)}
          </table>{clv_line}
        </div>"""

    # ── Context notes ─────────────────────────────────────────────────────────
    notes_html = ""
    if notes:
        notes_items = "".join(f'<li style="margin-bottom:6px;color:#374151">{n}</li>' for n in notes)
        notes_html = f"""
        <h2 style="color:#1e293b;font-size:1.1em;margin:28px 0 8px 0">⚠️ Context Notes</h2>
        <ul style="margin:0;padding-left:20px;font-size:0.9em">{notes_items}</ul>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CFB Picks — Week {this_week}, {season}</title></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:600px;margin:24px auto;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">

  <!-- Header -->
  <div style="background:#0f172a;padding:24px 28px">
    <div style="color:#eab308;font-size:0.75em;font-weight:700;letter-spacing:0.12em;text-transform:uppercase">
      College Football Picks Model
    </div>
    <div style="color:#ffffff;font-size:1.6em;font-weight:800;margin:4px 0">
      Week {this_week} · {season} Season
    </div>
    <div style="color:#94a3b8;font-size:0.85em">
      Generated {dt.datetime.now().strftime('%A, %B %-d, %Y')}
    </div>
  </div>

  <div style="padding:24px 28px">

    <!-- This week's picks -->
    <h2 style="color:#1e293b;font-size:1.1em;margin:0 0 4px 0">
      🏈 This Week's Top Picks — Week {this_week}
    </h2>
    <p style="color:#64748b;margin:0 0 12px 0;font-size:0.85em">
      O/U CORE: unders, {EDGE_MIN_TOT:.0f}–7 pt edge, power-conf, total ≥ 48 (58.5% in '19–'25;
      both-power 59.8% = 3u) · High-total unders ≥ 60: paper only (53.8%) ·
      Spread: {EDGE_MIN_SP:.0f}+ pts, weeks 1–{SPREAD_MAX_WEEK} only ·
      ML: {ML_EV_MIN:.0%}+ EV, paper record only · -110 juice unless noted
    </p>
    <table width="100%" cellpadding="0" cellspacing="0"
           style="border-collapse:collapse;font-size:0.9em">
      <thead>
        <tr style="background:#f8fafc">
          <th style="padding:10px 8px;text-align:left;color:#64748b;font-weight:600">Pick</th>
          <th style="padding:10px 8px;text-align:center;color:#64748b;font-weight:600">Edge</th>
          <th style="padding:10px 8px;text-align:center;color:#64748b;font-weight:600">Conf</th>
          <th style="padding:10px 8px;text-align:center;color:#64748b;font-weight:600">Kelly</th>
        </tr>
      </thead>
      <tbody>{picks_rows}</tbody>
    </table>

    {results_section}
    {record_section}
    {notes_html}

    <!-- Footer -->
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #e2e8f0;
                color:#94a3b8;font-size:0.8em;text-align:center">
      CFB Picks Model · Automated weekly report<br>
      Break-even at -110: 52.4% ATS · Model direction accuracy: ~72%<br>
      <span style="color:#cbd5e1">Use as research, not financial advice. Bet responsibly.</span>
    </div>

  </div>
</div>
</body></html>"""

# ── Email sending ─────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    """Send the newsletter via Gmail SMTP."""
    sender   = os.getenv("SENDER_EMAIL", "")
    password = os.getenv("SENDER_APP_PASSWORD", "")

    if not sender or not password:
        print("⚠️  SENDER_EMAIL and SENDER_APP_PASSWORD not set — skipping email send")
        print("   Add them as GitHub Actions secrets or env vars to enable email delivery")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"CFB Picks Model <{sender}>"
    msg["To"]      = RECIPIENT
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, RECIPIENT, msg.as_string())
        print(f"✅ Newsletter sent to {RECIPIENT}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Save HTML to /tmp/newsletter_preview.html and open in browser")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls, use placeholder data")
    parser.add_argument("--week", type=int, default=None,
                        help="Override week number")
    parser.add_argument("--season", type=int, default=None,
                        help="Override season year")
    args = parser.parse_args()

    season    = args.season or current_cfb_season()
    this_week = args.week   or current_cfb_week(season)
    last_week = this_week - 1

    print(f"CFB Newsletter Generator — {season} Season, Week {this_week}")
    print(f"Recipient: {RECIPIENT}")

    if args.dry_run:
        picks        = []
        last_results = []
        last_stats   = {}
        notes        = ["🔄 Oregon (Big Ten year 2) plays road game at Wisconsin — new-conference road caution",
                        "💨 Columbus, OH forecast: 22 mph — strong under lean for OSU game"]
    else:
        if not CFB_KEY:
            print("❌ CFB_API_KEY not found")
            sys.exit(1)

        print(f"Running predictions (full weekly_pipeline feature stack)...")
        picks = predict_games(season, this_week)
        print(f"  {len(picks)} picks meet edge threshold")

        if picks:
            save_weekly_picks(picks, season, this_week)

        print(f"\nLoading last week's results (week {last_week})...")
        last_results, last_stats = load_last_week_results(season, last_week)
        _all = last_stats.get("all", {})
        print(f"  {_all.get('wins',0)}W – {_all.get('losses',0)}L "
              f"({_all.get('units',0):+.1f}u)")

        notes = generate_context_notes(picks)

    print(f"\nComputing {season} season record...")
    season_record = compute_season_record(season)
    for mkt, label in [("TOTAL", "O/U"), ("SPREAD", "Spread"),
                       ("TOTAL_HT", "High-tot"), ("MONEYLINE", "ML")]:
        m = season_record.get(mkt, {})
        if m.get("bets", 0):
            print(f"  {label}: {m['wins']}–{m['losses']} "
                  f"({m['win_pct']:.1f}%, {m['units']:+.1f}u)")
    print(f"  {season_record['weeks']} weeks tracked")

    html = build_html(season, this_week, picks, last_results, last_stats,
                      season_record, notes)

    subject = f"CFB Picks — Week {this_week} ({season})"

    if args.preview:
        out = Path("/tmp/newsletter_preview.html")
        out.write_text(html)
        print(f"\n📄 Preview saved to {out}")
        import subprocess
        subprocess.run(["open", str(out)])
        return

    send_email(subject, html)

    print("\nDone.")

if __name__ == "__main__":
    main()
