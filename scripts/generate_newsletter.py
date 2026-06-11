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
EDGE_MIN_TOT = 2.5   # minimum totals edge to flag
RECIPIENT    = os.getenv("RECIPIENT_EMAIL", "alexwaked@me.com")

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

# ── Fetch schedule and lines ──────────────────────────────────────────────────

def fetch_week_games(season: int, week: int, season_type: str = "regular") -> list:
    """Fetch scheduled games for a given week."""
    try:
        data = cfb_get("games", params={
            "year": season, "week": week, "seasonType": season_type,
            "division": "fbs"
        })
        return data or []
    except Exception as e:
        print(f"  Warning: could not fetch games for week {week}: {e}")
        return []

def fetch_week_lines(season: int, week: int) -> dict:
    """Fetch betting lines keyed by game_id."""
    try:
        data = cfb_get("lines", params={"year": season, "week": week})
        lines = {}
        priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
        for game in data:
            gid = game.get("id")
            best = None
            best_rank = 999
            for line in game.get("lines", []):
                rank = next((i for i, p in enumerate(priority)
                             if p.lower() in (line.get("provider") or "").lower()), 999)
                if rank < best_rank and line.get("spread") is not None:
                    best_rank = rank
                    best = line
            if best:
                lines[gid] = {
                    "spread":       best.get("spread"),
                    "over_under":   best.get("overUnder"),
                    "home_ml":      best.get("homeMoneyline"),
                    "away_ml":      best.get("awayMoneyline"),
                }
        return lines
    except Exception as e:
        print(f"  Warning: could not fetch lines: {e}")
        return {}

# ── Model prediction ─────────────────────────────────────────────────────────

def load_models():
    """Load trained models and feature lists."""
    import joblib
    models_dir = ROOT / "models"
    spread_model   = joblib.load(models_dir / "spread_model.pkl")
    totals_model   = joblib.load(models_dir / "totals_model.pkl")
    win_prob_model = joblib.load(models_dir / "win_prob_model.pkl")
    with open(models_dir / "feature_lists.json") as f:
        feature_lists = json.load(f)
    return spread_model, totals_model, win_prob_model, feature_lists

def make_simple_features(games_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Build feature matrix from available columns, filling NaN for missing."""
    out = pd.DataFrame(index=games_df.index)
    for f in feature_names:
        out[f] = games_df[f] if f in games_df.columns else np.nan
    return out

def predict_games(games: list, lines: dict) -> list:
    """
    Run model predictions on a list of game dicts.
    Returns list of pick dicts for games meeting edge threshold.
    """
    if not games or not lines:
        return []

    try:
        sys.path.insert(0, str(ROOT / "src"))
        from features import (load_sp_ratings, load_wepa, load_talent, load_havoc,
                               load_fpi_ratings, load_srs_ratings,
                               load_conference_familiarity)
        sp     = load_sp_ratings()
        wepa   = load_wepa()
        talent = load_talent()
        havoc  = load_havoc()
    except Exception as e:
        print(f"  Warning: could not load features: {e}")
        sp = wepa = talent = havoc = pd.DataFrame()

    spread_model, totals_model, win_prob_model, feature_lists = load_models()

    # Build minimal game DataFrame
    rows = []
    for g in games:
        gid  = g.get("id")
        line = lines.get(gid, {})
        spread   = line.get("spread")
        over_under = line.get("over_under")
        if spread is None and over_under is None:
            continue   # no line data yet
        rows.append({
            "game_id":        gid,
            "season":         g.get("season"),
            "week":           g.get("week"),
            "home_team":      g.get("homeTeam"),
            "away_team":      g.get("awayTeam"),
            "neutral_site":   int(g.get("neutralSite") or False),
            "conference_game": int(g.get("conferenceGame") or False),
            "start_date":     g.get("startDate"),
            "spread":         spread,
            "over_under":     over_under,
            "home_moneyline": line.get("home_ml"),
            "away_moneyline": line.get("away_ml"),
            "home_pregame_elo": g.get("homePregameElo"),
            "away_pregame_elo": g.get("awayPregameElo"),
        })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["spread"]     = pd.to_numeric(df["spread"],     errors="coerce")
    df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")
    df["season"]     = pd.to_numeric(df["season"],     errors="coerce")
    df["week"]       = pd.to_numeric(df["week"],       errors="coerce")

    # Add week/postseason/magnitude context features
    df["week_num"]        = df["week"].fillna(0)
    df["is_postseason"]   = 0
    df["late_season"]     = (df["week_num"] >= 11).astype(int)
    df["spread_magnitude"] = df["spread"].abs()
    df["is_big_favorite"]  = (df["spread_magnitude"] >= 14).astype(float)
    df["vegas_home_margin"] = -df["spread"]
    df["elo_diff"] = (pd.to_numeric(df["home_pregame_elo"], errors="coerce") -
                      pd.to_numeric(df["away_pregame_elo"], errors="coerce"))

    # Merge SP+ ratings
    season = int(df["season"].iloc[0]) if len(df) > 0 else 2026
    if len(sp) > 0:
        for side in ["home", "away"]:
            team_col = f"{side}_team"
            sp_sub = sp[sp["season"] == season][["team", "sp_rating", "sp_offense", "sp_defense"]]
            df = df.merge(sp_sub.rename(columns={
                "team": team_col,
                "sp_rating": f"{side}_sp_rating",
                "sp_offense": f"{side}_sp_offense",
                "sp_defense": f"{side}_sp_defense",
            }), on=team_col, how="left")

    if "home_sp_rating" in df.columns and "away_sp_rating" in df.columns:
        df["sp_diff"] = df["home_sp_rating"] - df["away_sp_rating"]

    # Generate predictions
    sp_feats  = make_simple_features(df, feature_lists.get("spread", []))
    tot_feats = make_simple_features(df, feature_lists.get("totals", []))
    win_feats = make_simple_features(df, feature_lists.get("win_prob", []))

    # Spread model: current models predict the residual vs the Vegas line
    # (feature_lists.json: spread_target=margin_residual) — add the line back.
    sp_raw = spread_model.predict(sp_feats)
    if feature_lists.get("spread_target") == "margin_residual":
        df["pred_spread"] = df["vegas_home_margin"] + sp_raw
    else:
        df["pred_spread"] = sp_raw
    df["pred_total_dev"] = totals_model.predict(tot_feats)
    df["pred_total"]     = df["over_under"] + df["pred_total_dev"]
    df["pred_win_p"]     = win_prob_model.predict_proba(win_feats)[:, 1]

    df["spread_edge"] = df["pred_spread"] - df["vegas_home_margin"]
    df["totals_edge"] = df["pred_total"]  - df["over_under"]

    # Build picks list
    picks = []
    for _, row in df.iterrows():
        start = row.get("start_date", "")
        kickoff = _format_kickoff(start)

        # Spread pick
        sp_edge = row["spread_edge"]
        if pd.notna(sp_edge) and abs(sp_edge) >= EDGE_MIN_SP:
            home_bet = sp_edge > 0
            bet_team = row["home_team"] if home_bet else row["away_team"]
            vl = row["spread"] if home_bet else -row["spread"]
            picks.append({
                "type":      "SPREAD",
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

        # Totals pick
        tot_edge = row["totals_edge"]
        ou_str   = f"{row['over_under']:.1f}" if pd.notna(row["over_under"]) else "TBD"
        if pd.notna(tot_edge) and abs(tot_edge) >= EDGE_MIN_TOT:
            over_bet = tot_edge > 0
            picks.append({
                "type":      "TOTAL",
                "game_id":   row["game_id"],
                "week":      int(row["week"]) if pd.notna(row["week"]) else 0,
                "matchup":   f"{row['home_team']} vs {row['away_team']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on":    "OVER" if over_bet else "UNDER",
                "line":      ou_str,
                "edge":      round(float(tot_edge), 1),
                "pred":      round(float(row["pred_total"]), 1),
                "kickoff":   kickoff,
                "stars":     _stars(abs(tot_edge)),
                "kelly":     _kelly(abs(tot_edge)),
                "start_date": str(start),
                "wind_speed": row.get("wind_speed"),
                "is_dome":    row.get("is_dome", 0),
                "neutral":    bool(row.get("neutral_site", 0)),
            })

    picks.sort(key=lambda p: (-abs(p["edge"])))
    return picks

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

    # Fetch actual results from CFBD
    try:
        games_data = cfb_get("games", params={
            "year": season, "week": week, "seasonType": "regular",
            "division": "fbs"
        })
        results = {g["id"]: g for g in games_data}
    except Exception:
        return saved_picks, {}

    wins = losses = pushes = 0
    units = 0.0
    annotated = []

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

        spread_val = float(pick.get("line", 0).replace("+","") or 0)

        if pick["type"] == "SPREAD":
            home_bet = pick["edge"] > 0
            covered = (diff + spread_val > 0) if home_bet else (diff + spread_val < 0)
            if diff + spread_val == 0:
                pick["outcome"] = "push"; pushes += 1
            elif covered:
                pick["outcome"] = "win"; wins += 1; units += 1.0
            else:
                pick["outcome"] = "loss"; losses += 1; units -= 1.1
        elif pick["type"] == "TOTAL":
            ou = float(pick["line"]) if pick["line"] != "TBD" else None
            if ou is None:
                pick["outcome"] = "pending"
            elif total > ou and pick["bet_on"] == "OVER":
                pick["outcome"] = "win"; wins += 1; units += 1.0
            elif total < ou and pick["bet_on"] == "UNDER":
                pick["outcome"] = "win"; wins += 1; units += 1.0
            elif total == ou:
                pick["outcome"] = "push"; pushes += 1
            else:
                pick["outcome"] = "loss"; losses += 1; units -= 1.1

        annotated.append(pick)

    stats = {"wins": wins, "losses": losses, "pushes": pushes,
             "units": round(units, 1)}
    return annotated, stats

# ── Season record ─────────────────────────────────────────────────────────────

def compute_season_record(season: int) -> dict:
    """
    Compute cumulative season record from saved weekly picks files.
    """
    total_wins = total_losses = total_pushes = 0
    total_units = 0.0
    weeks_counted = 0

    for f in sorted(PICKS_DIR.glob(f"{season}_W*.json")):
        try:
            week_num = int(f.stem.split("_W")[1])
        except Exception:
            continue
        picks, stats = load_last_week_results(season, week_num)
        if stats:
            total_wins   += stats["wins"]
            total_losses += stats["losses"]
            total_pushes += stats["pushes"]
            total_units  += stats["units"]
            weeks_counted += 1

    total_bets = total_wins + total_losses
    win_pct = (total_wins / total_bets * 100) if total_bets > 0 else 0

    return {
        "wins": total_wins, "losses": total_losses, "pushes": total_pushes,
        "units": round(total_units, 1), "win_pct": round(win_pct, 1),
        "bets": total_bets, "weeks": weeks_counted,
    }

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
        edge_color = "#22c55e" if abs(p["edge"]) >= 4.5 else "#94a3b8"
        type_color = "#8b5cf6" if p["type"] == "SPREAD" else (
                     "#06b6d4" if p["bet_on"] == "UNDER" else "#f97316")
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
            {p['edge']:+.1f}
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;
                     text-align:center;color:#eab308">{p['stars']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;
                     text-align:center;font-weight:700">{p['kelly']}u</td>
        </tr>"""

    picks_rows = "".join(pick_row(p) for p in picks[:12]) if has_picks else \
        '<tr><td colspan="4" style="padding:16px;color:#94a3b8;text-align:center">No picks meet the edge threshold this week</td></tr>'

    # ── Results rows ──────────────────────────────────────────────────────────
    def result_row(p: dict) -> str:
        outcome = p.get("outcome", "pending")
        icon = {"win":"✅","loss":"❌","push":"➖","pending":"⏳"}.get(outcome, "⏳")
        score = p.get("score", "Pending")
        pnl   = "+1.0u" if outcome == "win" else ("-1.1u" if outcome == "loss" else "—")
        pnl_color = "#22c55e" if outcome == "win" else ("#ef4444" if outcome == "loss" else "#94a3b8")
        return f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0">{icon} <strong>{p['bet_on']}</strong> {p['line']} &nbsp;<span style="color:#64748b;font-size:0.85em">{p['matchup']}</span></td>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0;color:#64748b;font-size:0.85em">{score}</td>
          <td style="padding:8px;border-bottom:1px solid #e2e8f0;color:{pnl_color};font-weight:700;text-align:center">{pnl}</td>
        </tr>"""

    results_section = ""
    if has_results:
        results_rows = "".join(result_row(p) for p in last_results)
        units_color = "#22c55e" if last_stats["units"] >= 0 else "#ef4444"
        results_section = f"""
        <h2 style="color:#1e293b;font-size:1.1em;margin:28px 0 4px 0">
          📊 Last Week's Results — Week {last_week}
        </h2>
        <p style="color:#64748b;margin:0 0 12px 0;font-size:0.9em">
          {last_stats['wins']}W – {last_stats['losses']}L
          &nbsp;|&nbsp; <span style="color:{units_color};font-weight:700">{last_stats['units']:+.1f}u</span>
        </p>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:0.9em">
          <thead>
            <tr style="background:#f8fafc">
              <th style="padding:8px;text-align:left;color:#64748b;font-weight:600">Pick</th>
              <th style="padding:8px;text-align:left;color:#64748b;font-weight:600">Result</th>
              <th style="padding:8px;text-align:center;color:#64748b;font-weight:600">P&L</th>
            </tr>
          </thead>
          <tbody>{results_rows}</tbody>
        </table>"""

    # ── Season record ─────────────────────────────────────────────────────────
    sr = season_record
    sr_color = "#22c55e" if sr.get("units", 0) >= 0 else "#ef4444"
    record_section = ""
    if sr.get("bets", 0) > 0:
        record_section = f"""
        <div style="background:#f8fafc;border-radius:8px;padding:16px;margin:24px 0">
          <h2 style="color:#1e293b;font-size:1.0em;margin:0 0 12px 0">📈 {season} Season Record</h2>
          <div style="display:flex;gap:24px;flex-wrap:wrap">
            <div><span style="color:#64748b;font-size:0.8em">RECORD</span><br>
              <strong style="font-size:1.2em">{sr['wins']}–{sr['losses']}</strong>
              <span style="color:#94a3b8;font-size:0.85em"> ({sr['win_pct']:.1f}%)</span></div>
            <div><span style="color:#64748b;font-size:0.8em">UNITS</span><br>
              <strong style="font-size:1.2em;color:{sr_color}">{sr['units']:+.1f}u</strong></div>
            <div><span style="color:#64748b;font-size:0.8em">WEEKS TRACKED</span><br>
              <strong style="font-size:1.2em">{sr['weeks']}</strong></div>
          </div>
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
      Minimum edge: {EDGE_MIN_SP:.1f} pts (spread) / {EDGE_MIN_TOT:.1f} pts (totals) · Standard -110 juice
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

        print(f"\nFetching week {this_week} schedule...")
        games = fetch_week_games(season, this_week)
        print(f"  {len(games)} games found")

        print(f"Fetching betting lines...")
        lines = fetch_week_lines(season, this_week)
        print(f"  {len(lines)} games with lines")

        print(f"Running predictions...")
        picks = predict_games(games, lines)
        print(f"  {len(picks)} picks meet edge threshold")

        if picks:
            save_weekly_picks(picks, season, this_week)

        print(f"\nLoading last week's results (week {last_week})...")
        last_results, last_stats = load_last_week_results(season, last_week)
        print(f"  {last_stats.get('wins',0)}W – {last_stats.get('losses',0)}L "
              f"({last_stats.get('units',0):+.1f}u)")

        notes = generate_context_notes(picks)

    print(f"\nComputing {season} season record...")
    season_record = compute_season_record(season)
    print(f"  {season_record['wins']}W – {season_record['losses']}L "
          f"({season_record['units']:+.1f}u) over {season_record['weeks']} weeks")

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
