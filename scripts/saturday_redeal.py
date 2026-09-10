#!/usr/bin/env python3
"""
Saturday re-deal
================
Lines move between Tuesday's card and Saturday kickoff. This re-runs every
flagged bet at the CURRENT line and auto-drops anything that no longer passes
the unified gate (src/gates.py) — selection discipline instead of hoping the
edge survived.

    python3 scripts/saturday_redeal.py                    # auto season/week
    python3 scripts/saturday_redeal.py --season 2026 --week 3

Output: outputs/picks/redeal_{season}_w{week:02d}.json — the app reads the
latest file for the selected week and banners any drops. Tracked bets are
NEVER deleted or rewritten; a drop is a recommendation recorded in the
re-deal file, and the bet's history stays intact.

Intended schedule: Saturdays 8 AM Pacific (see saturday_redeal.workflow.yml
template — the automation token cannot push into .github/workflows/).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import gates  # noqa: E402

LEDGER = ROOT_DIR / "outputs" / "predictions" / "ledger_2026.csv"
BETS   = ROOT_DIR / "tracked_bets.json"
OUT_DIR = ROOT_DIR / "outputs" / "picks"


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _gate_at_current(row: dict) -> tuple[bool, str]:
    """Re-evaluate the unified gate at the CURRENT line. Returns
    (passes, reason). Model numbers are unchanged; only the market line moves,
    so totals_edge is recomputed against the current O/U."""
    ou = row.get("current_over_under")
    if pd.isna(ou) or pd.isna(row.get("pred_total")):
        return False, "no current total posted"
    edge = float(row["pred_total"]) - float(ou)
    probe = dict(row)
    probe["totals_edge"] = edge
    probe["over_under"] = float(ou)
    if gates.core_total(probe):
        return True, f"still CORE at {ou:.1f} (edge {edge:+.1f})"
    reasons = []
    if not (edge <= -2):
        reasons.append(f"edge shrunk to {edge:+.1f}")
    elif edge < -7:
        reasons.append(f"edge blew past 7 ({edge:+.1f}) — winner's-curse zone")
    if gates.wind15(probe):
        reasons.append("wind >= 15 mph")
    if gates.low_total(probe):
        reasons.append("total < 48")
    if not gates.power_involved(probe):
        reasons.append("no power-conference team")
    return False, "; ".join(reasons) or "outside CORE gate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    import weekly_pipeline as wp
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = wp.current_cfb_week()
    print(f"Saturday re-deal — {season} week {week}")

    # ── Current lines ─────────────────────────────────────────────────────
    games = wp.fetch_schedule(season, week)
    if games.empty:
        print("No games this week — nothing to re-deal.")
        return
    lines = wp.fetch_lines(games, season, week)

    # ── Predictions: prefer the recorded ledger snapshot, else run live ───
    preds = None
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        wk = led[(led["season"] == season) & (led["week"] == week)]
        if len(wk):
            preds = wk.copy()
            print(f"Using {len(preds)} recorded ledger predictions")
    if preds is None:
        spread_m, totals_m, win_prob_m, feat_lists = wp.load_models()
        preds = wp.build_predictions(games, lines, spread_m, totals_m,
                                     win_prob_m, feat_lists, season)
        print(f"Built {len(preds)} live predictions")

    # Join current lines onto predictions
    cur = lines.rename(columns={"spread": "current_spread",
                                "over_under": "current_over_under"})
    keep_cols = ["game_id", "current_spread", "current_over_under"]
    if "provider" in cur.columns:
        keep_cols.append("provider")
        cur = cur.rename(columns={"provider": "current_provider"})
    df = preds.merge(cur[[c for c in keep_cols if c in cur.columns]],
                     on="game_id", how="left")
    if "home_conference" not in df.columns and "home_conference" in games.columns:
        df = df.merge(games[["game_id", "home_conference", "away_conference"]],
                      on="game_id", how="left")

    decisions = []

    # ── 1. Re-evaluate CORE flags from the original card ──────────────────
    flagged = df[df.get("flag_core", False) == True] if "flag_core" in df.columns else df[df["totals_edge"].between(-7, -2)]  # noqa: E712
    for _, r in flagged.iterrows():
        row = r.to_dict()
        passes, reason = _gate_at_current(row)
        decisions.append({
            "source": "model-flag",
            "game": f"{r['away_team']} @ {r['home_team']}",
            "bet_type": "Total",
            "pick": f"UNDER {r.get('over_under')}",
            "recorded_line": r.get("over_under"),
            "current_line": row.get("current_over_under"),
            "decision": "keep" if passes else "drop",
            "reason": reason,
        })

    # ── 2. Re-evaluate Alex's pending tracked bets for this week ──────────
    if BETS.exists():
        bets = json.loads(BETS.read_text())
        for b in bets:
            if b.get("status") not in (None, "Pending"):
                continue
            if b.get("season") != season or b.get("week") != week:
                continue
            label = b.get("game", "")
            parts = None
            if " @ " in label:
                away, home = label.split(" @ ", 1)
                parts = (_norm(home), _norm(away))
            if not parts:
                continue
            m = df[(df["home_team"].map(_norm) == parts[0])
                   & (df["away_team"].map(_norm) == parts[1])]
            if m.empty:
                decisions.append({"source": "tracked_bet", "game": label,
                                  "bet_type": b.get("bet_type"),
                                  "pick": b.get("pick"),
                                  "decision": "review",
                                  "reason": "game not found in this week's predictions"})
                continue
            row = m.iloc[0].to_dict()
            if str(b.get("bet_type", "")).lower().startswith("total"):
                passes, reason = _gate_at_current(row)
                decisions.append({
                    "source": "tracked_bet", "game": label,
                    "bet_type": b.get("bet_type"), "pick": b.get("pick"),
                    "recorded_line": b.get("line"),
                    "current_line": row.get("current_over_under"),
                    "decision": "keep" if passes else "drop",
                    "reason": reason if passes else
                              f"no longer CORE at current line — {reason}",
                })
            else:
                se = row.get("spread_edge")
                edge_now = (None if pd.isna(se) or pd.isna(row.get("current_spread"))
                            else float(row["pred_spread"]) - (-float(row["current_spread"])))
                decisions.append({
                    "source": "tracked_bet", "game": label,
                    "bet_type": b.get("bet_type"), "pick": b.get("pick"),
                    "recorded_line": b.get("line"),
                    "current_line": row.get("current_spread"),
                    "decision": "drop",
                    "reason": ("spreads are paper-only under the unified gate "
                               f"(edge at current line: {edge_now:+.1f})"
                               if edge_now is not None else
                               "spreads are paper-only under the unified gate"),
                })

    # ── 3. Newly-qualified CORE plays at current lines (not on the card) ──
    new_qualifiers = []
    flagged_ids = set(flagged["game_id"]) if len(flagged) else set()
    for _, r in df.iterrows():
        if r["game_id"] in flagged_ids:
            continue
        row = r.to_dict()
        passes, reason = _gate_at_current(row)
        if passes:
            new_qualifiers.append({
                "game": f"{r['away_team']} @ {r['home_team']}",
                "pick": f"UNDER {row['current_over_under']:.1f}",
                "reason": reason,
            })

    drops = [d for d in decisions if d["decision"] == "drop"]
    out = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "week": week,
        "decisions": decisions,
        "new_qualifiers": new_qualifiers,
        "summary": f"{len(decisions) - len(drops)} keep / {len(drops)} drop / "
                   f"{len(new_qualifiers)} new",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"redeal_{season}_w{week:02d}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(out["summary"])
    for d in drops:
        print(f"  DROP {d['pick']} ({d['game']}): {d['reason']}")
    for q in new_qualifiers:
        print(f"  NEW  {q['pick']} ({q['game']}): {q['reason']}")
    print(f"-> {dest.name}")


if __name__ == "__main__":
    main()
