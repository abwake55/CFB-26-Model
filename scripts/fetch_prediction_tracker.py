#!/usr/bin/env python3
"""Fetch Prediction Tracker CFB system lines (free public CSVs).

    python scripts/fetch_prediction_tracker.py            # current season + live week
    python scripts/fetch_prediction_tracker.py --all      # 2015..current (backfill)
    python scripts/fetch_prediction_tracker.py --live     # live week file only

Writes data/external/prediction_tracker/ and data/processed/pt_lines.csv.
Never fails the calling workflow: PT being down only means no PT signal.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prediction_tracker as pt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1
    try:
        if not a.live:
            seasons = list(range(pt.FIRST_SEASON, season + 1)) if a.all else [season]
            h = pt.build_history(seasons=seasons, fetch=True)
            print(f"[pt] pt_lines.csv: {len(h)} games, seasons {h.season.min()}-{h.season.max()}")
        live = pt.fetch_live(save=True)
        print(f"[pt] live predictions: {len(live)} games")
    except Exception as exc:
        print(f"[pt] WARNING fetch failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
