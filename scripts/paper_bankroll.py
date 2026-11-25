#!/usr/bin/env python3
"""Paper bankroll CLI.

    python3 scripts/paper_bankroll.py seed-week13
    python3 scripts/paper_bankroll.py sync
    python3 scripts/paper_bankroll.py grade
    python3 scripts/paper_bankroll.py slip [--week N]
    python3 scripts/paper_bankroll.py update   # seed (if needed) + sync + grade + slip

Wired into weekly_refresh.yml and game_day_update.yml. Never sends the
newsletter, never calls The Odds API, never invents prices or scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import paper_bankroll as pb


def _load_ledger() -> pd.DataFrame:
    if not pb.LEDGER_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(pb.LEDGER_CSV)


def cmd_seed_week13() -> None:
    df = pb.load_bankroll()
    n_before = len(df)
    df = pb.seed_week13(df)
    pb.save_bankroll(df)
    pb.write_summary(df)
    path = pb.write_slip(df, week=13)
    print(f"[paper] seeded week 13: {len(df) - n_before} new bets, "
          f"{len(df)} total -> {pb.BANKROLL_CSV.relative_to(ROOT)}")
    print(f"[paper] slip -> {path.relative_to(ROOT)}")


def cmd_sync() -> None:
    df = pb.load_bankroll()
    n_before = len(df)
    df = pb.sync_from_ledger(df, _load_ledger())
    pb.save_bankroll(df)
    print(f"[paper] sync: {len(df) - n_before} new bets, {len(df)} total")


def cmd_grade() -> None:
    df = pb.load_bankroll()
    df = pb.grade_bankroll(df, _load_ledger())
    pb.save_bankroll(df)
    summary = pb.write_summary(df)
    rec = summary["record_str"]
    print(f"[paper] grade: balance=${summary['current_balance']:.2f} "
          f"record={rec} roi={summary['roi']} clv={summary['avg_clv_pts']}")


def cmd_slip(week: int | None) -> None:
    df = pb.load_bankroll()
    if week is None:
        week = pb.current_week_from_df(df) or 13
    path = pb.write_slip(df, week=int(week))
    print(f"[paper] slip week {week} -> {path.relative_to(ROOT)}")
    print(path.read_text())


def cmd_update() -> None:
    df = pb.load_bankroll()
    n_before = len(df)
    # Seed week 13 once, as long as those games have not kicked off.
    have_w13 = bool(len(df) and (pd.to_numeric(df.get("week"), errors="coerce") == 13).any())
    if not have_w13:
        try:
            df = pb.seed_week13(df)
        except ValueError as e:
            print(f"[paper] week 13 seed skipped: {e}")
    df = pb.sync_from_ledger(df, _load_ledger())
    df = pb.grade_bankroll(df, _load_ledger())
    pb.save_bankroll(df)
    summary = pb.write_summary(df)
    week = pb.current_week_from_df(df) or 13
    path = pb.write_slip(df, week=int(week))
    print(f"[paper] update: +{len(df) - n_before} bets, {len(df)} total, "
          f"balance=${summary['current_balance']:.2f} record={summary['record_str']} "
          f"slip={path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seed-week13", "sync", "grade", "slip", "update"])
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "seed-week13":
        cmd_seed_week13()
    elif args.mode == "sync":
        cmd_sync()
    elif args.mode == "grade":
        cmd_grade()
    elif args.mode == "slip":
        cmd_slip(args.week)
    else:
        cmd_update()


if __name__ == "__main__":
    main()
