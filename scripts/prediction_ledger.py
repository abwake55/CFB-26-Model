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
    # opener-gap signal (src/opener_gap.py), graded at opener_gap_line
    "pred_indep_margin", "opener_gap_line", "opener_gap_edge", "flag_opener_gap",
    # Prediction Tracker signal (src/prediction_tracker.py, models/pt_rule.json)
    "pt_fpi", "pt_sys_median", "pt_sys_std", "pt_opener", "pt_edge", "flag_pt",
    # live = recorded before kickoff by the scheduled job; replay_not_live =
    # weeks 1-3 reseeded after the fact; backfill_not_live = weeks missed by
    # the Oct 29 - Nov 17 CI outage, re-predicted later with a date cutoff.
    "provenance",
]

# Per signal: the line and time a bet would have been placed = the current
# consensus spread the FIRST time the flag appeared. Kept across re-records.
# signal -> (flag column, edge column)
BET_LINE_SIGNALS = {"pt": ("flag_pt", "pt_edge"),
                    "opener_gap": ("flag_opener_gap", "opener_gap_edge"),
                    "spread": ("flag_spread", "spread_edge")}
for _sig in BET_LINE_SIGNALS:
    LEDGER_COLS += [f"{_sig}_bet_line", f"{_sig}_bet_ts", f"{_sig}_bet_edge",
                    f"{_sig}_clv", f"{_sig}_clv_sharp"]
# filled by grade from data/lines_snapshots: the first captured line (true
# opener when the early-week snapshot saw it) and the sharp book's last
# pre-kickoff line (LowVig/BetOnline; Pinnacle only if the eu region is on)
LEDGER_COLS += ["snap_open_spread", "snap_open_ts", "closing_sharp_spread",
                "closing_sharp_book", "closing_sharp_ts"]
SNAP_DIR = ROOT_DIR / "data" / "lines_snapshots"


def _week_snapshot(season: int, week: int) -> dict:
    path = SNAP_DIR / f"lines_{season}_W{int(week):02d}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("games", {})
    except Exception:
        return {}


def _sharp_close(rec: dict):
    """(spread, book, ts) of the last pre-kickoff snapshot carrying a sharp line."""
    hist = rec.get("history") or []
    for h in reversed(hist):
        if h.get("sharp_spread") is not None:
            return h["sharp_spread"], h.get("sharp_book"), h.get("ts")
    cur = rec.get("current") or {}
    if cur.get("sharp_spread") is not None:
        return cur["sharp_spread"], cur.get("sharp_book"), rec.get("updated")
    return None, None, None


def _sharp_close_from_quotes(game_ids) -> dict:
    """{game_id: (spread, book, observed_at)} — last sharp-book home spread
    quote observed before kickoff, from the raw quote archive
    (data/lines_snapshots/quotes/*.jsonl). Fallback when a week's snapshot
    predates sharp fields."""
    import glob
    want = {int(g) for g in game_ids}
    pref = {"pinnacle": 0, "lowvig": 1, "betonlineag": 2}
    best: dict = {}
    for f in sorted(glob.glob(str(SNAP_DIR / "quotes" / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if '"spreads"' not in line or not any(b in line for b in pref):
                    continue
                try:
                    q = json.loads(line)
                except ValueError:
                    continue
                gid = q.get("game_id")
                if gid not in want or q.get("book") not in pref or q.get("market") != "spreads" \
                        or q.get("selection") != "home" or q.get("line") is None:
                    continue
                obs, kick = q.get("observed_at"), q.get("kickoff")
                if not obs or not kick or pd.Timestamp(obs) >= pd.Timestamp(kick):
                    continue
                key = (pd.Timestamp(obs), -pref[q["book"]])
                if gid not in best or key > best[gid][0]:
                    best[gid] = (key, float(q["line"]), q["book"], obs)
    return {g: v[1:] for g, v in best.items()}


def _fill_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row CLV (points, bet side) for every placed signal bet:
    vs consensus close (CFBD) and vs the sharp book's last pre-kick line."""
    for sig in BET_LINE_SIGNALS:
        lc, ec = f"{sig}_bet_line", f"{sig}_bet_edge"
        if lc not in df.columns:
            continue
        line = pd.to_numeric(df[lc], errors="coerce")
        side = np.sign(pd.to_numeric(df[ec], errors="coerce"))
        for close_col, out_col in (("closing_spread", f"{sig}_clv"),
                                   ("closing_sharp_spread", f"{sig}_clv_sharp")):
            close = pd.to_numeric(df.get(close_col), errors="coerce")
            df[out_col] = (side * (line - close)).round(2)
    return df


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def _carry_bet_lines(new: pd.DataFrame, old: pd.DataFrame | None) -> pd.DataFrame:
    """Set <sig>_bet_line/_bet_ts for newly flagged games; keep the first
    flagged line/time from earlier records of the same game."""
    new = new.copy()
    prev = (old.drop_duplicates("game_id", keep="last").set_index("game_id")
            if old is not None and len(old) else pd.DataFrame())
    for sig, (flag, edge) in BET_LINE_SIGNALS.items():
        cols = [f"{sig}_bet_line", f"{sig}_bet_ts", f"{sig}_bet_edge"]
        for c in cols:
            new[c] = new[c].astype(object)
        for i, r in new.iterrows():
            gid = r["game_id"]
            if gid in prev.index and cols[0] in prev.columns and pd.notna(prev.at[gid, cols[0]]):
                for c in cols:
                    new.at[i, c] = prev.at[gid, c]
            elif _is_true(r.get(flag)) and pd.notna(r.get("spread")):
                new.at[i, cols[0]] = r["spread"]
                new.at[i, cols[1]] = r["recorded_at"]
                new.at[i, cols[2]] = r[edge]
    return new


def _provenance(df: pd.DataFrame) -> pd.Series:
    """Existing rows without a provenance: recorded_at present -> live."""
    prov = df.get("provenance")
    if prov is None:
        prov = pd.Series(np.nan, index=df.index, dtype=object)
    rec = pd.to_datetime(df.get("recorded_at"), errors="coerce", utc=True, format="mixed")
    default = np.where(rec.notna(), "live", "replay_not_live")
    return prov.where(prov.notna(), pd.Series(default, index=df.index))


# ─── record ──────────────────────────────────────────────────────────────────

def record(season: int, week: int) -> None:
    import weekly_pipeline as wp  # reuse schedule/line/prediction builders

    games = wp.fetch_schedule(season, week)
    if games.empty:
        print(f"[record] no games for {season} wk{week} — nothing to do")
        return

    # Once a week has kicked off, keep the existing snapshot: the recorded
    # lines are the dealt lines CLV is measured against — re-recording with
    # post-kickoff (frozen/closing) lines would zero out the CLV signal.
    first_kick = pd.to_datetime(games["start_date"], errors="coerce", utc=True).min()
    if LEDGER.exists() and pd.notna(first_kick) \
            and pd.Timestamp.now(tz="UTC") > first_kick:
        print(f"[record] {season} wk{week} has kicked off — keeping existing snapshot")
        return

    lines = wp.fetch_lines(games, season, week)
    spread_m, totals_m, win_prob_m, feat_lists = wp.load_models()
    preds = wp.build_predictions(games, lines, spread_m, totals_m,
                                 win_prob_m, feat_lists, season, archive_decisions=True)

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
            "pred_indep_margin": row.get("pred_indep_margin"),
            "opener_gap_line": row.get("opener_gap_line"),
            "opener_gap_edge": row.get("opener_gap_edge"),
            "flag_opener_gap": bool(row.get("flag_opener_gap") is True
                                    or row.get("flag_opener_gap") == 1),
            "pt_fpi": row.get("pt_fpi"),
            "pt_sys_median": row.get("pt_sys_median"),
            "pt_sys_std": row.get("pt_sys_std"),
            "pt_opener": row.get("pt_opener"),
            "pt_edge": row.get("pt_edge"),
            "flag_pt": bool(row.get("flag_pt") is True or row.get("flag_pt") == 1),
            "provenance": "live",
        })
    new = pd.DataFrame(rows, columns=LEDGER_COLS)
    old = pd.read_csv(LEDGER) if LEDGER.exists() else None
    new = _carry_bet_lines(new, old)

    if old is not None:
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

    merged["provenance"] = _provenance(merged)
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
    for col in LEDGER_COLS:          # ledgers written before newer columns
        if col not in df.columns:
            df[col] = np.nan
    for col in ("closing_sharp_book", "closing_sharp_ts", "snap_open_ts"):
        df[col] = df[col].astype(object)
    # An ungraded ledger round-trips `completed` as all-NaN float64, and newer
    # pandas raises TypeError ("Invalid value 'True' for dtype 'float64'")
    # instead of warning when we later set True into it. Normalize to real
    # bools up front, tolerating bool / float / string round-trips.
    c = df["completed"]
    df["completed"] = c.notna() & c.astype(str).str.strip().str.lower().isin(
        ["true", "1", "1.0"])

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

    for (s_, w_), grp in df.groupby(["season", "week"]):
        snap = _week_snapshot(int(s_), int(w_))
        if not snap:
            continue
        for i, r in grp.iterrows():
            rec = snap.get(str(int(r["game_id"])))
            if not rec:
                continue
            if pd.isna(r.get("snap_open_spread")) and (rec.get("first") or {}).get("spread") is not None:
                df.at[i, "snap_open_spread"] = rec["first"]["spread"]
                df.at[i, "snap_open_ts"] = rec.get("first_seen")
            if r.get("completed") is True or r.get("completed") == True:  # noqa: E712
                sp, bk, ts = _sharp_close(rec)
                if sp is not None:
                    df.at[i, "closing_sharp_spread"] = sp
                    df.at[i, "closing_sharp_book"] = bk
                    df.at[i, "closing_sharp_ts"] = ts
    need = df[(df["completed"] == True) & df["closing_sharp_spread"].isna()]  # noqa: E712
    if len(need):
        q = _sharp_close_from_quotes(need["game_id"])
        for i, r in need.iterrows():
            v = q.get(int(r["game_id"]))
            if v:
                df.at[i, "closing_sharp_spread"], df.at[i, "closing_sharp_book"], \
                    df.at[i, "closing_sharp_ts"] = v
    df = _fill_clv(df)

    done = df["completed"] == True  # noqa: E712
    df.loc[done, "home_win"] = (df.loc[done, "home_score"]
                                > df.loc[done, "away_score"]).astype(float)
    df.loc[done, "actual_margin"] = df.loc[done, "home_score"] - df.loc[done, "away_score"]
    df.loc[done, "actual_total"] = df.loc[done, "home_score"] + df.loc[done, "away_score"]
    df["provenance"] = _provenance(df)
    df.to_csv(LEDGER, index=False)
    print(f"[grade] {int(done.sum())}/{len(df)} ledger rows completed")

    _build_summary(df)


def _bucket_stats(g: pd.DataFrame, kind: str, line_col: str = "spread",
                  edge_col: str = "spread_edge") -> dict:
    """Grade one bucket of flagged bets at -110 against the recorded line."""
    g = g[g["completed"] == True]  # noqa: E712
    n = wins = losses = pushes = 0
    units = 0.0
    clvs, sharp_clvs = [], []
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
            if pd.isna(r.get(line_col)) or pd.isna(r.get("actual_margin")):
                continue
            bet_home = r[edge_col] > 0               # model likes home vs line
            line = r[line_col]                       # home perspective
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
            sharp = pd.to_numeric(r.get("closing_sharp_spread"), errors="coerce")
            if pd.notna(sharp):
                sharp_clvs.append((line - sharp) * (1 if bet_home else -1))
    return {"n": n, "wins": wins, "losses": losses, "pushes": pushes,
            "units": round(units, 2),
            "hit_rate": round(wins / (wins + losses), 4) if wins + losses else None,
            "avg_clv_pts": round(float(np.mean(clvs)), 2) if clvs else None,
            "clv_n": len(clvs),
            "avg_clv_sharp_pts": round(float(np.mean(sharp_clvs)), 2) if sharp_clvs else None,
            "clv_sharp_n": len(sharp_clvs)}


def _bucket_block(df: pd.DataFrame) -> dict:
    done = df[df["completed"] == True]  # noqa: E712
    out = {"games_tracked": int(len(df)), "games_completed": int(len(done))}
    b = done.dropna(subset=["pred_win_p", "home_win"])
    out["brier"] = round(float(((b["pred_win_p"] - b["home_win"]) ** 2).mean()), 4) if len(b) else None
    out["brier_n"] = int(len(b))
    # Lined games only, against the market's spread-implied probability:
    # unlined (mostly FCS) games inflate the all-games Brier.
    bl = b.dropna(subset=["spread"])
    if len(bl):
        from math import erf, sqrt
        sigma = 15.29
        cal = ROOT_DIR / "models" / "win_prob_calibration.json"
        if cal.exists():
            sigma = float(json.loads(cal.read_text()).get("spread_sigma", sigma))
        mkt = (-bl["spread"] / sigma).map(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
        out["brier_lined"] = round(float(((bl["pred_win_p"] - bl["home_win"]) ** 2).mean()), 4)
        out["market_brier_lined"] = round(float(((mkt - bl["home_win"]) ** 2).mean()), 4)
        out["brier_lined_n"] = int(len(bl))
    out["core_unders"] = _bucket_stats(df[df["flag_core"] == True], "under")  # noqa: E712
    out["spread_flags"] = _bucket_stats(df[df["flag_spread"] == True], "spread")  # noqa: E712
    out["all_under_leans"] = _bucket_stats(done[done["totals_edge"] <= -2], "under")
    if "flag_opener_gap" in df.columns:
        fl = df["flag_opener_gap"].astype(str).str.lower().isin(["true", "1", "1.0"])
        out["opener_gap_flags"] = _bucket_stats(df[fl], "spread", "opener_gap_line",
                                                "opener_gap_edge")
    for sig in BET_LINE_SIGNALS:
        lc = f"{sig}_bet_line"
        if lc not in df.columns:
            continue
        placed = df[pd.to_numeric(df[lc], errors="coerce").notna()].copy()
        placed[lc] = pd.to_numeric(placed[lc], errors="coerce")
        placed[f"{sig}_bet_edge"] = pd.to_numeric(placed[f"{sig}_bet_edge"], errors="coerce")
        # graded at the line the bet would have been placed at (first flag)
        out[f"{sig}_bets"] = _bucket_stats(placed, "spread", lc, f"{sig}_bet_edge")
    if "pt_opener" in df.columns and "pt_bet_line" in df.columns:
        placed = df[pd.to_numeric(df["pt_bet_line"], errors="coerce").notna()].copy()
        placed["pt_bet_edge"] = pd.to_numeric(placed["pt_bet_edge"], errors="coerce")
        out["pt_bets_at_opener"] = _bucket_stats(placed, "spread", "pt_opener", "pt_bet_edge")
    return out


def _build_summary(df: pd.DataFrame) -> None:
    df = df.copy()
    df["provenance"] = _provenance(df)
    live = df[df["provenance"] == "live"]
    blk = _bucket_block(live)
    summary = {"season": int(df["season"].max()) if len(df) else None,
               "basis": "live rows only (recorded before kickoff)",
               "games_tracked": blk["games_tracked"],
               "games_completed": blk["games_completed"],
               "live_brier": blk["brier"], "live_brier_n": blk["brier_n"],
               "live_brier_lined": blk.get("brier_lined"),
               "market_brier_lined": blk.get("market_brier_lined"),
               "live_brier_lined_n": blk.get("brier_lined_n"),
               "core_unders": blk["core_unders"],
               "spread_flags": blk["spread_flags"],
               "opener_gap_flags": blk.get("opener_gap_flags"),
               "pt_bets": blk.get("pt_bets"),
               "opener_gap_bets": blk.get("opener_gap_bets"),
               "spread_bets": blk.get("spread_bets"),
               "pt_bets_at_opener": blk.get("pt_bets_at_opener"),
               "all_under_leans": blk["all_under_leans"],
               "weeks_live": sorted(int(w) for w in live["week"].unique()),
               "not_live": {}}
    for prov, g in df[df["provenance"] != "live"].groupby("provenance"):
        summary["not_live"][prov] = {"weeks": sorted(int(w) for w in g["week"].unique()),
                                     **_bucket_block(g)}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"[grade] summary -> {SUMMARY.name}: brier={summary['live_brier']} "
          f"core={summary['core_unders']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "grade", "summary"])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    if args.mode != "summary":
        if args.season and args.week:
            season, week = args.season, args.week
        else:
            import weekly_pipeline as wp
            season, week = wp.current_cfb_week()

    if args.mode == "record":
        record(season, week)
    elif args.mode == "summary":   # rebuild summary offline (no API calls)
        df = pd.read_csv(LEDGER)
        c = df["completed"]
        df["completed"] = c.notna() & c.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        _build_summary(df)
    else:
        grade(season)


if __name__ == "__main__":
    main()
