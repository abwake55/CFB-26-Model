#!/usr/bin/env python3
"""Log The Odds API quota (used / remaining this month) at zero credit cost
via /v4/sports, and print the month's logged spend by workflow.

    python scripts/odds_usage.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import odds_api  # noqa: E402


def main() -> None:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("ODDS_API_KEY not set — skipping quota check")
        return
    try:
        rec = odds_api.check_quota(key)
        print(f"Odds API quota: used={rec['used']} remaining={rec['remaining']}")
    except Exception as exc:
        print(f"quota check failed: {type(exc).__name__}")
        return
    files = sorted(odds_api.USAGE_DIR.glob("*.csv"))
    if files:
        u = pd.concat([pd.read_csv(f, on_bad_lines="skip") for f in files], ignore_index=True)
        u["ts_utc"] = pd.to_datetime(u["ts_utc"], utc=True, errors="coerce", format="ISO8601")
        m = u[(u.endpoint == "odds") & (u.ts_utc.dt.strftime("%Y-%m")
                                        == pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))]
        if len(m):
            spend = pd.to_numeric(m.credits_last, errors="coerce").groupby(m.source).agg(["count", "sum"])
            print("Logged odds calls this month (calls, credits) by workflow:")
            print(spend.to_string())


if __name__ == "__main__":
    main()
