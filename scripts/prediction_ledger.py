#!/usr/bin/env python3
"""
2026 live prediction ledger
===========================
tracked_bets.json stores only the bets Alex selected. This ledger stores
EVERY weekly prediction — probability, market lines at record time, closing
lines, and the final result — so the 2026 season can be graded honestly:
live Brier score, CLV, and per-bucket performance (CORE vs everything else).

    python3 scripts/prediction_ledger.py record   # snapshot this week's predictions
    python3 scripts/prediction_ledger.py grade    # fill results + closing lines, rebuild summary

Files
-----
outputs/predictions/ledger_2026.csv          one row per game per week (append/update)
outputs/predictions/ledger_2026_summary.json aggregate live metrics the app displays

Wired into .github/workflows/weekly_refresh.yml (record + grade each Tuesday).
Grading never deletes or rewrites history: rows are only updated with
closing lines and final scores once games are completed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import gates  # noqa: E402  unified recommendation gate

LEDGER   = ROOT_DIR / "outputs" / "predictions" / "ledger_2026.csv"
SUMMARY  = ROOT_DIR / "outputs" / "predictions" / "ledger_2026_summary.json"

LEDGER_COLS = [
    "game_id", "season", "week", "home_team", "away_team",
    "home_conference", "away_conference", "neutral_site", "start_date",
    "recorded_at",
    # model outputs
    "pred_spread", "pred_total", "pred_win_p",
    # market at record time (home-perspective spread, O/U, MLs)
    "spread", "over_under", "spread_open", "home_moneyline", "away_moneyline",
    "spread_edge", "totals_edge",
    # gate flags at record time
    "flag_core", "flag_spread", "flag_ml",
    # filled by grade
    "closing_spread", "closing_over_under",
    "home_score", "away_score", "completed",
    "home_win", "actual_margin", "actual_total",
]


# ─── record ──────────────────────────────────────────────────────────────────

def record(season: int, week: int) -> None:
    import weekly_pipeline as wp  # reuse schedule/line/prediction builders

    games = wp.fetch_schedule(season, week)
    if games.empty:
        print(f"[record] no games for {season} wk{week} — nothing to do")
        return
    lines = wp.fetch_lines(games, season, week)
    spread_m, totals_m, win_prob_m, feat_lists = wp.load_models()
    preds = wp.build_predictions(games, lines, spread_m, totals_m,
                                 win_prob_m, feat_lists, season)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    for _, r in preds.iterrows():
        row = r.to_dict()
        rows.append({
            "game_id": r["game_id"], "season": season, "week": week,
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_conference": row.get("home_conference"),
            "away_conference": row.get("away_conference"),
            "neutral_site": row.get("neutral_site"),
            "start_date": row.get("start_date"),
            "recorded_at": now,
            "pred_spread": row.get("pred_spread"),
            "pred_total": row.get("pred_total"),
            "pred_win_p": row.get("pred_win_p"),
            "spread": row.get("spread"),
            "over_under": row.get("over_under"),
            "spread_open": row.get("spread_open"),
            "home_moneyline": row.get("home_moneyline"),
            "away_moneyline": row.get("away_moneyline"),
            "spread_edge": row.get("spread_edge"),
            "totals_edge": row.get("totals_edge"),
            "flag_core": bool(gates.core_total(row)),
            "flag_spread": bool(pd.notna(row.get("spread_edge"))
                                and abs(float(row["spread_edge"])) >= 3),
            "flag_ml": bool((pd.notna(row.get("home_ml_ev")) and row["home_ml_ev"] >= 0.04)
                            or (pd.notna(row.get("away_ml_ev")) and row["away_ml_ev"] >= 0.04)),
        })
    new = pd.DataFrame(rows, columns=LEDGER_COLS)

    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        # keep historical rows; drop stale ungraded rows for this season/week
        # only when they were never completed (re-record replaces the snapshot)
        keep = old[~((old["season"] == season) & (old["week"] == week)
                     & (old["completed"] != True))]  # noqa: E712
        merged = pd.concat([keep, new], ignore_index=True)
        # a game already graded stays graded: restore graded columns
        graded = old[(old["season"] == season) & (old["week"] == week)
                     & (old["completed"] == True)]  # noqa: E712
        if not graded.empty:
            merged = merged[~merged["game_id"].isin(graded["game_id"])]
            merged = pd.concat([merged, graded], ignore_index=True)
    else:
        merged = new

    merged = merged.sort_values(["season", "week", "game_id"])
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(LEDGER, index=False)
    print(f"[record] {len(new)} predictions recorded for {season} wk{week} "
          f"({int(new['flag_core'].sum())} CORE flags) -> {LEDGER.name}")


# ─── grade ───────────────────────────────────────────────────────────────────

def _fetch_week_results(season: int, week: int, key: str) -> dict:
    """{game_id: (home_score, away_score, completed)} from CFBD."""
    import requests
    out = {}
    for stype in ("regular", "postseason"):
        try:
            resp = requests.get(
                "https://api.collegefootballdata.com/games",
                headers={"Authorization": f"Bearer {key}"},
                params={"year": season, "week": week, "seasonType": stype},
                timeout=20)
            resp.raise_for_status()
            for g in resp.json():
                if g.get("completed"):
                    out[g["id"]] = (g.get("homePoints"), g.get("awayPoints"), True)
        except Exception as e:
            print(f"[grade] CFBD games {season} wk{week} {stype} failed: {e}")
    return out


def _fetch_week_closing(season: int, week: int, key: str) -> dict:
    """{game_id: (spread, over_under)} — CFBD lines freeze at kickoff, so a
    post-game fetch returns the closing line."""
    import requests
    out = {}
    try:
        resp = requests.get(
            "https://api.collegefootballdata.com/lines",
            headers={"Authorization": f"Bearer {key}"},
            params={"year": season, "week": week}, timeout=20)
        resp.raise_for_status()
        priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet"]
        for game in resp.json():
            best = None
            for want in priority:
                for line in game.get("lines", []):
                    if line.get("provider") == want:
                        best = line
                        break
                if best:
                    break
            if not best and game.get("lines"):
                best = game["lines"][0]
            if best:
                out[game["id"]] = (best.get("spread"), best.get("overUnder"))
    except Exception as e:
        print(f"[grade] CFBD lines {season} wk{week} failed: {e}")
    return out


def grade(season: int) -> None:
    if not LEDGER.exists():
        print("[grade] no ledger yet — run record first")
        return
    df = pd.read_csv(LEDGER)

    key = ""
    sec = ROOT_DIR / ".streamlit" / "secrets.toml"
    import os
    key = os.getenv("CFB_API_KEY", "")
    if not key and sec.exists():
        for line in sec.read_text().splitlines():
            if "CFB_API_KEY" in line and "=" in line:
                key = line.split("=", 1)[1].strip().strip('"').strip("'")

    pending = df[df["completed"] != True]  # noqa: E712
    for (s, w), grp in pending.groupby(["season", "week"]):
        results = _fetch_week_results(int(s), int(w), key)
        closing = _fetch_week_closing(int(s), int(w), key)
        for gid, (hs, as_, done) in results.items():
            m = df["game_id"] == gid
            if not m.any():
                continue
            df.loc[m, "home_score"] = hs
            df.loc[m, "away_score"] = as_
            df.loc[m, "completed"] = True
        for gid, (sp, ou) in closing.items():
            m = (df["game_id"] == gid) & (df["completed"] == True)
            if not m.any():
                continue
            if pd.notna(sp):
                df.loc[m, "closing_spread"] = sp
            if pd.notna(ou):
                df.loc[m, "closing_over_under"] = ou

    done = df["completed"] == True  # noqa: E712
    df.loc[done, "home_win"] = (df.loc[done, "home_score"]
                                > df.loc[done, "away_score"]).astype(float)
    df.loc[done, "actual_margin"] = df.loc[done, "home_score"] - df.loc[done, "away_score"]
    df.loc[done, "actual_total"] = df.loc[done, "home_score"] + df.loc[done, "away_score"]
    df.to_csv(LEDGER, index=False)
    print(f"[grade] {int(done.sum())}/{len(df)} ledger rows completed")

    _build_summary(df)


def _bucket_stats(g: pd.DataFrame, kind: str) -> dict:
    """Grade one bucket of flagged bets at -110 against the recorded line."""
    g = g[g["completed"] == True]  # noqa: E712
    n = wins = losses = pushes = 0
    units = 0.0
    clvs = []
    for _, r in g.iterrows():
        if kind == "under":
            if pd.isna(r.get("over_under")) or pd.isna(r.get("actual_total")):
                continue
            n += 1
            if r["actual_total"] == r["over_under"]:
                pushes += 1
            elif r["actual_total"] < r["over_under"]:
                wins += 1; units += 1.0
            else:
                losses += 1; units -= 1.1
            if pd.notna(r.get("closing_over_under")):
                clvs.append(r["over_under"] - r["closing_over_under"])
        elif kind == "spread":
            if pd.isna(r.get("spread")) or pd.isna(r.get("actual_margin")):
                continue
            bet_home = r["spread_edge"] > 0          # model likes home vs line
            line = r["spread"]                       # home perspective
            cover_margin = r["actual_margin"] + line  # home covers if > 0
            if not bet_home:
                cover_margin = -cover_margin
            n += 1
            if cover_margin == 0:
                pushes += 1
            elif cover_margin > 0:
                wins += 1; units += 1.0
            else:
                losses += 1; units -= 1.1
            if pd.notna(r.get("closing_spread")):
                clv = (line - r["closing_spread"]) * (1 if bet_home else -1)
                clvs.append(clv)
    return {"n": n, "wins": wins, "losses": losses, "pushes": pushes,
            "units": round(units, 2),
            "hit_rate": round(wins / (wins + losses), 4) if wins + losses else None,
            "avg_clv_pts": round(float(np.mean(clvs)), 2) if clvs else None}


def _build_summary(df: pd.DataFrame) -> None:
    done = df[df["completed"] == True]  # noqa: E712
    summary = {"season": int(df["season"].max()) if len(df) else None,
               "games_tracked": int(len(df)),
               "games_completed": int(len(done))}

    # Live Brier on win probabilities (honest, every game — not just bets)
    b = done.dropna(subset=["pred_win_p", "home_win"])
    summary["live_brier"] = (round(float(((b["pred_win_p"] - b["home_win"]) ** 2).mean()), 4)
                             if len(b) else None)
    summary["live_brier_n"] = int(len(b))

    core = df[df["flag_core"] == True]   # noqa: E712
    spreads = df[(df["flag_spread"] == True)]  # noqa: E712
    summary["core_unders"] = _bucket_stats(core, "under")
    summary["spread_flags"] = _bucket_stats(spreads, "spread")
    # every-game context: how all totals leans did vs the gate
    all_under = done[done["totals_edge"] <= -2]
    summary["all_under_leans"] = _bucket_stats(all_under, "under")

    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"[grade] summary -> {SUMMARY.name}: brier={summary['live_brier']} "
          f"core={summary['core_unders']}")


def seed(season: int, weeks: list) -> None:
    """One-time backfill from the saved week CSVs
    (outputs/predictions/week_{season}_w{ww}_predictions.csv) so weeks played
    before this script existed still enter the ledger. Predictions are the
    honest record of what the model said; market lines were largely absent
    from those files, so gate flags are computed where lines exist and CLV
    will be sparse for seeded rows. Grading fills results automatically."""
    frames = []
    for w in weeks:
        f = (ROOT_DIR / "outputs" / "predictions"
             / f"week_{season}_w{w:02d}_predictions.csv")
        if not f.exists():
            print(f"[seed] {f.name} missing — skipped")
            continue
        d = pd.read_csv(f)
        rows = []
        for _, r in d.iterrows():
            row = r.to_dict()
            rows.append({
                "game_id": r["game_id"], "season": season, "week": w,
                "home_team": r["home_team"], "away_team": r["away_team"],
                "home_conference": row.get("home_conference"),
                "away_conference": row.get("away_conference"),
                "neutral_site": row.get("neutral_site"),
                "start_date": None,
                "recorded_at": "seeded-from-week-csv",
                "pred_spread": row.get("pred_spread"),
                "pred_total": row.get("pred_total"),
                "pred_win_p": row.get("pred_win_p"),
                "spread": row.get("spread"),
                "over_under": row.get("over_under"),
                "spread_open": row.get("spread_open"),
                "home_moneyline": None, "away_moneyline": None,
                "spread_edge": row.get("spread_edge"),
                "totals_edge": row.get("totals_edge"),
                "flag_core": bool(gates.core_total(row)),
                "flag_spread": bool(pd.notna(row.get("spread_edge"))
                                    and abs(float(row["spread_edge"])) >= 3),
                "flag_ml": False,
            })
        frames.append(pd.DataFrame(rows, columns=LEDGER_COLS))
        print(f"[seed] {f.name}: {len(rows)} rows")
    if not frames:
        return
    new = pd.concat(frames, ignore_index=True)
    if LEDGER.exists():
        old_df = pd.read_csv(LEDGER)
        new = new[~new["game_id"].isin(old_df["game_id"])]
        new = pd.concat([old_df, new], ignore_index=True)
    new = new.sort_values(["season", "week", "game_id"])
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(LEDGER, index=False)
    print(f"[seed] ledger now {len(new)} rows -> {LEDGER.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "grade", "seed"])
    ap.add_argument("--weeks", type=int, nargs="*", default=None,
                    help="seed mode: weeks to backfill (default 1 2)")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    if args.season and args.week:
        season, week = args.season, args.week
    elif args.mode == "seed" and args.season:
        season, week = args.season, None
    else:
        import weekly_pipeline as wp
        season, week = wp.current_cfb_week()

    if args.mode == "record":
        record(season, week)
    elif args.mode == "seed":
        seed(season, args.weeks or [1, 2])
    else:
        grade(season)


if __name__ == "__main__":
    main()
