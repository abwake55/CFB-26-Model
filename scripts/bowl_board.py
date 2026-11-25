#!/usr/bin/env python3
"""
Bowl board (paper signal)
=========================
Projects every upcoming FBS bowl / CFP game, adds the bowl adjustment
(models/bowl_adjust.json, from scripts/research/bowl_lab.py) and flags games
where |adjustment| >= the frozen threshold. Status: PAPER. The adjustment is
built from QB changes (likely/confirmed starter vs the season's most-started
QB), regular-season attempt share of the expected QB, transfer-portal exits
before kickoff, and interim / changed head coaches.

    python3 scripts/bowl_board.py                 # auto: current season, upcoming bowls
    python3 scripts/bowl_board.py --season 2026 --no-odds

Outputs (append-only history, the board file is the latest view):
  outputs/predictions/bowl_projections_{season}.csv  every run, timestamped
  outputs/predictions/ledger_{season}_bowls.csv      one row per bowl: first
                    flag line + time (the bet line), frozen at kickoff, graded
  outputs/picks/bowls_{season}.json                  latest board

Odds: CFBD lines (free). One Odds API call (3 credits) only with --odds and
only when the free quota check shows >= BOWL_ODDS_MIN (default 150) credits
left and a bowl kicks off within 48 hours.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bowl_context as B  # noqa: E402
import qb_watch  # noqa: E402

ADJ = ROOT / "models" / "bowl_adjust.json"
PRED_DIR = ROOT / "outputs" / "predictions"
PICKS = ROOT / "outputs" / "picks"


def _now():
    return pd.Timestamp.now(tz="UTC")


def fetch_bowls(season: int) -> pd.DataFrame:
    rows = []
    for g in qb_watch.cfbd("games", year=season, seasonType="postseason") or []:
        if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs":
            continue
        if not g.get("homeTeam") or not g.get("awayTeam"):
            continue
        rows.append({"game_id": g.get("id"), "season": season, "week": g.get("week"),
                     "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                     "home_conference": g.get("homeConference"),
                     "away_conference": g.get("awayConference"),
                     "neutral_site": int(bool(g.get("neutralSite"))),
                     "conference_game": 0, "start_date": g.get("startDate"),
                     "notes": g.get("notes"),
                     "home_points": g.get("homePoints"), "away_points": g.get("awayPoints")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce", format="mixed")
    return df


def fetch_cfbd_lines(season: int) -> pd.DataFrame:
    prio = {"consensus": 0, "DraftKings": 1, "ESPN Bet": 2, "Bovada": 3}
    rows = []
    for g in qb_watch.cfbd("lines", year=season, seasonType="postseason") or []:
        for ln in g.get("lines", []) or []:
            rows.append({"game_id": g.get("id"), "spread": ln.get("spread"),
                         "over_under": ln.get("overUnder"), "spread_open": ln.get("spreadOpen"),
                         "home_moneyline": ln.get("homeMoneyline"),
                         "away_moneyline": ln.get("awayMoneyline"),
                         "provider": ln.get("provider"), "_r": prio.get(ln.get("provider"), 9)})
    if not rows:
        return pd.DataFrame(columns=["game_id", "spread", "over_under", "spread_open"])
    df = pd.DataFrame(rows)
    opener = df.groupby("game_id")["spread_open"].median()
    df = df.sort_values("_r").drop_duplicates("game_id").drop(columns="_r")
    df["spread_open"] = df["game_id"].map(opener)
    return df


def maybe_odds(games: pd.DataFrame, use_odds: bool) -> pd.DataFrame:
    key = os.environ.get("ODDS_API_KEY", "")
    if not (use_odds and key):
        return pd.DataFrame()
    soon = games["t"].notna() & (games["t"] <= _now() + pd.Timedelta(hours=48)) & (games["t"] > _now())
    if not soon.any():
        print("[bowls] no bowl within 48h — skipping the Odds API")
        return pd.DataFrame()
    import odds_api
    try:
        q = odds_api.check_quota(key)        # free endpoint
        left = int(float(q.get("remaining") or 0))
    except Exception as exc:
        print(f"[bowls] quota check failed ({type(exc).__name__}) — skipping the Odds API")
        return pd.DataFrame()
    floor = int(os.environ.get("BOWL_ODDS_MIN", "150"))
    if left < floor:
        print(f"[bowls] {left} Odds API credits left (< {floor}) — CFBD lines only")
        return pd.DataFrame()
    try:
        return odds_api.fetch_lines(key, games)
    except Exception as exc:
        print(f"[bowls] Odds API unavailable ({type(exc).__name__})")
        return pd.DataFrame()


def model_predictions(games: pd.DataFrame, lines: pd.DataFrame, season: int) -> pd.DataFrame:
    """Main model + opener-gap + PT + pricing + edge ranges, via the weekly
    pipeline's builder. Returns an empty frame if the models can't run."""
    try:
        import weekly_pipeline as wp
        spread_m, totals_m, win_m, feats = wp.load_models()
        return wp.build_predictions(games.drop(columns=["t", "notes", "home_points", "away_points"],
                                               errors="ignore"),
                                    lines, spread_m, totals_m, win_m, feats, season,
                                    archive_decisions=False)
    except Exception as exc:
        print(f"[bowls] model predictions unavailable ({type(exc).__name__}: {exc}) — "
              "bowl adjustment still computed against the line")
        return pd.DataFrame()


def bowl_adjustment(games: pd.DataFrame, season: int, passing: pd.DataFrame,
                    portal: pd.DataFrame, coaches: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    adj = json.loads(ADJ.read_text())
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    status = qb_watch.team_status(passing, teams, season, portal=portal)
    ov = pd.concat([games[["game_id"]].assign(team=games["home_team"]),
                    games[["game_id"]].assign(team=games["away_team"])])
    ov = ov.merge(status[["team", "likely_starter_id"]].rename(columns={"likely_starter_id": "starter_id"}),
                  on="team", how="left")
    ov = ov[ov["starter_id"].notna()]
    # portal entries count only up to NOW for games not yet played
    g = games.copy()
    g["t"] = g["t"].where(g["t"] < _now(), _now())
    feats = B.game_features(g, passing=passing.assign(season=season), portal=portal,
                            coaches=coaches, starters_override=ov, research=False)
    out = games[["game_id"]].merge(feats, on="game_id", how="left")
    out["bowl_adj"] = sum(out[k].fillna(0) * v for k, v in adj["beta"].items() if k in out)
    out["flag_bowl"] = out["bowl_adj"].abs() >= float(adj["threshold"])
    rng = adj.get("range")
    if rng:
        import edge_ranges
        # ranges are only defined inside the backtest's adjustment sizes
        e = out["bowl_adj"].abs().clip(upper=float(rng.get("e_max", 6.0)))
        mu, lo, hi = edge_ranges.expected(rng, e)
        out["bowl_edge_exp"], out["bowl_edge_lo80"], out["bowl_edge_hi80"] = mu.round(2), lo.round(2), hi.round(2)
    return out, status


def coaches_live(season: int) -> pd.DataFrame | None:
    try:
        rows = qb_watch.cfbd("coaches", year=season) or []
    except Exception as exc:
        print(f"[bowls] coaches unavailable ({type(exc).__name__})")
        return None
    out = []
    for c in rows:
        for cs in c.get("seasons", []) or []:
            if cs.get("year") == season:
                out.append({"season": season, "school": cs.get("school"),
                            "coach": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
                            "games": cs.get("games")})
    return pd.DataFrame(out)


LEDGER_KEEP = ["game_id", "season", "notes", "start_date", "home_team", "away_team",
               "first_seen", "bowl_adj", "flag_bowl", "bet_side", "bet_line", "bet_ts",
               "bowl_edge_exp", "bowl_edge_lo80", "bowl_edge_hi80",
               "home_qb", "away_qb", "home_qb_source", "away_qb_source",
               "last_line", "last_line_ts", "home_points", "away_points",
               "cover_margin", "result", "clv"]


def update_ledger(board: pd.DataFrame, season: int, now: str) -> pd.DataFrame:
    path = PRED_DIR / f"ledger_{season}_bowls.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LEDGER_KEEP)
    old = old.set_index("game_id") if not old.empty else old
    rows = []
    for r in board.to_dict("records"):
        gid = r["game_id"]
        prev = old.loc[gid].to_dict() if (not old.empty and gid in old.index) else None
        started = pd.notna(r.get("t")) and r["t"] <= _now()
        row = dict(prev) if prev else {"first_seen": now}
        row["game_id"] = gid
        if not started:   # pre-kickoff: refresh; frozen once the game starts
            for k in ("season", "notes", "start_date", "home_team", "away_team", "bowl_adj",
                      "flag_bowl", "bowl_edge_exp", "bowl_edge_lo80", "bowl_edge_hi80",
                      "home_qb", "away_qb", "home_qb_source", "away_qb_source"):
                row[k] = r.get(k)
            if pd.notna(r.get("spread")):
                row["last_line"], row["last_line_ts"] = r["spread"], now
            if bool(r.get("flag_bowl")) and pd.isna(row.get("bet_line")) and pd.notna(r.get("spread")):
                row["bet_side"] = "home" if r["bowl_adj"] > 0 else "away"
                row["bet_line"], row["bet_ts"] = r["spread"], now
        hp, ap = r.get("home_points"), r.get("away_points")
        if pd.notna(hp) and pd.notna(ap):
            row["home_points"], row["away_points"] = hp, ap
            if pd.notna(row.get("bet_line")):
                s = 1 if row["bet_side"] == "home" else -1
                c = s * (hp - ap + float(row["bet_line"]))
                row["cover_margin"] = c
                row["result"] = "W" if c > 0 else ("L" if c < 0 else "P")
                if pd.notna(row.get("last_line")):
                    row["clv"] = s * (float(row["bet_line"]) - float(row["last_line"]))
        rows.append(row)
    new = pd.DataFrame(rows)
    if not old.empty:
        rest = old[~old.index.isin(new["game_id"])].reset_index()
        new = pd.concat([rest, new], ignore_index=True)
    new = new.reindex(columns=LEDGER_KEEP)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def run(season: int | None = None, use_odds: bool = False, only_upcoming_days: float | None = None) -> dict | None:
    season = season or (_now().year if _now().month >= 8 else _now().year - 1)
    games = fetch_bowls(season)
    if games.empty:
        print(f"[bowls] no {season} FBS postseason games published yet — nothing to do")
        return None
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = fetch_cfbd_lines(season)
    odds = maybe_odds(games, use_odds)
    if not odds.empty:
        import odds_api
        lines = odds_api.merge_lines(odds, lines)
    passing = qb_watch.refresh_passing(season, qb_watch.last_regular_week(season))
    portal = qb_watch.portal_entries(season)
    coaches = coaches_live(season)
    upcoming = games[games["t"].isna() | (games["t"] > _now() - pd.Timedelta(hours=6))]
    adj, status = bowl_adjustment(games, season, passing, portal, coaches)
    preds = model_predictions(upcoming, lines, season) if not upcoming.empty else pd.DataFrame()
    board = games.merge(lines[[c for c in ("game_id", "spread", "spread_open", "over_under")
                               if c in lines.columns]], on="game_id", how="left") \
                 .merge(adj, on="game_id", how="left")
    keep = [c for c in ("game_id", "pred_spread", "spread_edge", "pred_indep_margin", "opener_gap_edge",
                        "flag_opener_gap", "pt_fpi", "pt_edge", "flag_pt", "pred_margin_lo80",
                        "pred_margin_hi80") if c in preds.columns]
    if keep:
        board = board.merge(preds[keep], on="game_id", how="left")
    st = status.set_index("team")
    for side in ("home", "away"):
        board[f"{side}_qb"] = board[f"{side}_team"].map(st["likely_starter"])
        board[f"{side}_qb_primary"] = board[f"{side}_team"].map(st["primary_qb"])
        board[f"{side}_qb_source"] = board[f"{side}_team"].map(st["source"])
    board["recorded_at"] = now
    hist = PRED_DIR / f"bowl_projections_{season}.csv"
    cols = [c for c in board.columns if c != "t"]
    import history
    history.append(board[cols], hist)
    led = update_ledger(board, season, now)
    flags = []
    for r in board[board["flag_bowl"] & (board["t"] > _now())].to_dict("records"):
        home = r["bowl_adj"] > 0
        team = r["home_team"] if home else r["away_team"]
        line = r.get("spread")
        flags.append({"game": f"{r['away_team']} vs {r['home_team']}", "bowl": r.get("notes"),
                      "kickoff": r.get("start_date"),
                      "pick": f"{team} {(line if home else -line):+.1f}" if pd.notna(line) else team,
                      "bowl_adj_pts": round(float(r["bowl_adj"]), 1),
                      "expected_cover_pts": r.get("bowl_edge_exp"),
                      "expected_range80": [r.get("bowl_edge_lo80"), r.get("bowl_edge_hi80")],
                      "qb": {"home": [r.get("home_qb"), r.get("home_qb_source")],
                             "away": [r.get("away_qb"), r.get("away_qb_source")]},
                      "status": "paper"})
    out = {"season": season, "generated_at": now, "status": "paper",
           "rule": json.loads(ADJ.read_text()).get("selected"),
           "n_games": int(len(board)), "flags": flags,
           "qb_watch": status[status["qb_change_likely"] == 1][
               ["team", "primary_qb", "likely_starter", "source", "status"]].to_dict("records"),
           "cfbd_calls": qb_watch.CALLS["n"]}
    PICKS.mkdir(parents=True, exist_ok=True)
    (PICKS / f"bowls_{season}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[bowls] {len(board)} bowls, {len(flags)} paper flags, ledger {len(led)} rows, "
          f"{qb_watch.CALLS['n']} CFBD calls")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--odds", action="store_true", help="allow one guarded Odds API call")
    a = ap.parse_args()
    run(a.season, use_odds=a.odds)


if __name__ == "__main__":
    main()
