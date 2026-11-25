#!/usr/bin/env python3
"""
Game-day update
===============
Saturday mornings (and daily through bowl season) re-check the likely / confirmed
starting QB for each remaining game, re-project, and flag picks whose edge
changed since the ledger recorded them. Nothing recorded earlier is changed:
every run APPENDS timestamped rows.

    python3 scripts/game_day_update.py                 # auto (regular week and/or bowls)
    python3 scripts/game_day_update.py --mode regular  # force the regular-season check
    python3 scripts/game_day_update.py --mode bowls --odds

Regular season (no Odds API calls): lines come from the newest daily
snapshot (data/lines_snapshots, already paid for by line_snapshots.yml).
Re-projection: the market-free opener-gap margin moves by the dev-fitted
QB effect in models/qb_adjust.json when a team's likely starter is not its
most-started QB. The main model (anchored to the market line) and PT's FPI
(external) are not adjusted; their edges are re-priced at the current line.

Outputs
  outputs/predictions/game_day_projections_{season}.csv  append-only history
  outputs/picks/gameday_{season}_w{week:02d}.json        latest summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qb_watch  # noqa: E402

LEDGER = ROOT / "outputs" / "predictions" / "ledger_2026.csv"
SNAP_DIR = ROOT / "data" / "lines_snapshots"
PRED_DIR = ROOT / "outputs" / "predictions"
PICKS = ROOT / "outputs" / "picks"
CHANGE_PTS = 1.5        # |edge now - edge recorded| that counts as "changed"
THRESH = {"spread": 3.0, "opener_gap": 5.0}


def _thresholds() -> dict:
    t = dict(THRESH)
    try:
        t["pt"] = float(json.loads((ROOT / "models/pt_rule.json").read_text())["thr"])
    except Exception:
        t["pt"] = 7.0
    return t


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def current_lines(season: int, week: int) -> dict:
    p = SNAP_DIR / f"lines_{season}_W{week:02d}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text()).get("games", {})
    out = {}
    for gid, g in d.items():
        cur = g.get("current") or {}
        if cur.get("spread") is not None:
            out[int(gid)] = (float(cur["spread"]), g.get("updated"))
    return out


def regular(season: int, week: int) -> dict | None:
    if not LEDGER.exists():
        print("[gameday] no ledger yet")
        return None
    led = pd.read_csv(LEDGER, low_memory=False)
    led = led[(led["season"] == season) & (led["week"] == week)].copy()
    led["t"] = pd.to_datetime(led["start_date"], utc=True, errors="coerce", format="mixed")
    now = pd.Timestamp.now(tz="UTC")
    led = led[led["t"].isna() | (led["t"] > now)]
    if led.empty:
        print(f"[gameday] {season} wk{week}: no games left to kick off")
        return None
    qa = json.loads((ROOT / "models/qb_adjust.json").read_text())
    beta = float(qa.get("per_qb_out_vs_model_pts") or 0.0)
    passing = qb_watch.refresh_passing(season, max(week - 1, 1))
    teams = sorted(set(led["home_team"]) | set(led["away_team"]))
    status = qb_watch.team_status(passing, teams, season).set_index("team")
    lines = current_lines(season, week)
    thr = _thresholds()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    import edge_ranges
    params = edge_ranges.load()
    rows, changed = [], []
    for r in led.to_dict("records"):
        gid = int(r["game_id"])
        line_now, line_ts = lines.get(gid, (np.nan, None))
        rec_line = pd.to_numeric(r.get("spread"), errors="coerce")
        if pd.isna(line_now):
            line_now = rec_line
        hs, as_ = (status.loc[r["home_team"]] if r["home_team"] in status.index else None,
                   status.loc[r["away_team"]] if r["away_team"] in status.index else None)
        hq = float(hs["qb_change_likely"]) if hs is not None and pd.notna(hs["qb_change_likely"]) else 0.0
        aq = float(as_["qb_change_likely"]) if as_ is not None and pd.notna(as_["qb_change_likely"]) else 0.0
        qb_adj = beta * (hq - aq)
        indep = pd.to_numeric(r.get("pred_indep_margin"), errors="coerce")
        margins = {"spread": pd.to_numeric(r.get("pred_spread"), errors="coerce"),
                   "opener_gap": indep + qb_adj if pd.notna(indep) else np.nan,
                   "pt": -pd.to_numeric(r.get("pt_fpi"), errors="coerce")}
        # edge at the line the bet was (or would have been) placed: the
        # signal's first-flag bet line, else the line at record time
        recorded = {}
        for sig, m in margins.items():
            bl = pd.to_numeric(r.get(f"{sig}_bet_line"), errors="coerce")
            base = bl if pd.notna(bl) else rec_line
            recorded[sig] = (m + base) if pd.notna(m) and pd.notna(base) else np.nan
        flags = {"spread": r.get("flag_spread"), "opener_gap": r.get("flag_opener_gap"),
                 "pt": r.get("flag_pt")}
        row = {"recorded_at": ts, "season": season, "week": week, "game_id": gid,
               "home_team": r["home_team"], "away_team": r["away_team"], "start_date": r.get("start_date"),
               "line_recorded": rec_line, "line_recorded_at": r.get("recorded_at"),
               "line_now": line_now, "line_now_at": line_ts,
               "home_qb_likely": hs["likely_starter"] if hs is not None else None,
               "home_qb_primary": hs["primary_qb"] if hs is not None else None,
               "home_qb_source": hs["source"] if hs is not None else None,
               "away_qb_likely": as_["likely_starter"] if as_ is not None else None,
               "away_qb_primary": as_["primary_qb"] if as_ is not None else None,
               "away_qb_source": as_["source"] if as_ is not None else None,
               "home_qb_starts": hs["likely_starts"] if hs is not None else np.nan,
               "away_qb_starts": as_["likely_starts"] if as_ is not None else np.nan,
               "home_qb_change": hq, "away_qb_change": aq, "qb_adj_home_pts": round(qb_adj, 2),
               "pred_indep_margin_gd": round(float(margins["opener_gap"]), 2)
               if pd.notna(margins["opener_gap"]) else np.nan}
        reasons = []
        for sig, m in margins.items():
            rec_e = pd.to_numeric(recorded[sig], errors="coerce")
            # spread convention: pt edge = line - FPI spread = line + FPI margin
            now_e = (m + line_now) if pd.notna(m) and pd.notna(line_now) else np.nan
            row[f"{sig}_edge_recorded"] = rec_e
            row[f"{sig}_edge_now"] = round(float(now_e), 2) if pd.notna(now_e) else np.nan
            row[f"flag_{sig}_recorded"] = _truthy(flags[sig])
            row[f"flag_{sig}_now"] = bool(pd.notna(now_e) and abs(now_e) >= thr[sig])
            if params and sig in params["signals"] and pd.notna(now_e):
                mu, lo, hi = edge_ranges.expected(params["signals"][sig], abs(now_e))
                row[f"{sig}_edge_now_exp"], row[f"{sig}_edge_now_lo80"], row[f"{sig}_edge_now_hi80"] = (
                    round(float(mu), 2), round(float(lo), 2), round(float(hi), 2))
            if sig == "spread":
                continue        # main-model spreads are paper only; tracked, not alerted
            if row[f"flag_{sig}_recorded"]:
                if pd.notna(now_e) and pd.notna(rec_e) and abs(now_e - rec_e) >= CHANGE_PTS:
                    reasons.append(f"{sig} edge {rec_e:+.1f} at the bet line -> {now_e:+.1f} now")
            if row[f"flag_{sig}_recorded"]:
                if pd.notna(now_e) and abs(now_e) < thr[sig]:
                    reasons.append(f"{sig} edge below {thr[sig]:g} at the current line")
                for side, q, s in (("home", hq, hs), ("away", aq, as_)):
                    # alert only on a FRESH change (<= 2 starts): older changes
                    # are already in the ratings and the line
                    if q == 1 and s is not None and int(s.get("likely_starts", 0)) <= 2:
                        reasons.append(f"{r[side + '_team']} new QB: {s['likely_starter']} "
                                       f"({int(s['likely_starts'])} starts; most-started "
                                       f"{s['primary_qb']}; {s['source']}, {s['status']})")
        row["edge_changed"] = bool(reasons)
        row["change_reasons"] = "; ".join(dict.fromkeys(reasons))
        rows.append(row)
        if reasons:
            changed.append({"game": f"{r['away_team']} @ {r['home_team']}", "kickoff": r.get("start_date"),
                            "line_recorded": rec_line, "line_now": line_now,
                            "reasons": list(dict.fromkeys(reasons))})
    df = pd.DataFrame(rows)
    import history
    history.append(df, PRED_DIR / f"game_day_projections_{season}.csv")
    out = {"season": season, "week": week, "generated_at": ts, "games_checked": len(df),
           "qb_effect_pts": beta, "changed": changed,
           "qb_changes": status[status["qb_change_likely"] == 1].reset_index()[
               ["team", "primary_qb", "likely_starter", "likely_starts", "source", "status"]].to_dict("records"),
           "note": "Likely starters come from last game's box score unless "
                   "data/context/qb_overrides.csv has confirmed news.",
           "cfbd_calls": qb_watch.CALLS["n"]}
    PICKS.mkdir(parents=True, exist_ok=True)
    (PICKS / f"gameday_{season}_w{week:02d}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[gameday] {season} wk{week}: {len(df)} games, {len(changed)} flagged picks changed, "
          f"{qb_watch.CALLS['n']} CFBD calls")
    return out


def bowls_upcoming(season: int, hours: float = 24 * 45) -> bool:
    try:
        g = qb_watch.cfbd("games", year=season, seasonType="postseason") or []
    except Exception as exc:
        print(f"[gameday] postseason schedule unavailable ({type(exc).__name__})")
        return False
    now = pd.Timestamp.now(tz="UTC")
    t = pd.to_datetime(pd.Series([x.get("startDate") for x in g]), utc=True, errors="coerce", format="mixed")
    return bool(((t > now - pd.Timedelta(hours=6)) & (t <= now + pd.Timedelta(hours=hours))).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "regular", "bowls"], default="auto")
    ap.add_argument("--odds", action="store_true", help="bowls: allow one guarded Odds API call")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    a = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    season = a.season or (now.year if now.month >= 8 else now.year - 1)
    if a.mode in ("auto", "regular"):
        import weekly_pipeline as wp
        s, w = (season, a.week) if a.week else wp.current_cfb_week()
        if a.mode == "regular" or now.tz_convert("America/Los_Angeles").dayofweek == 5:
            if s == season:
                regular(s, w)
    if a.mode == "bowls" or (a.mode == "auto" and bowls_upcoming(season)):
        import bowl_board
        bowl_board.run(season, use_odds=a.odds)


if __name__ == "__main__":
    main()
