#!/usr/bin/env python3
"""Record future games independently, even when an earlier game has kicked off."""
import json
from pathlib import Path
import pandas as pd
import weekly_pipeline as wp
from snapshot_lines import current_week
from collect_context import collect_weather


def record():
    now=pd.Timestamp.now(tz='UTC');season=now.year if now.month>=2 else now.year-1
    # weekly_pipeline already resolves the existing credential from env/config.
    from snapshot_lines import _secret
    week=current_week(_secret('CFB_API_KEY'),season)
    if week is None:
        print('No upcoming regular-season week');return
    games=wp.fetch_schedule(season,week)
    if games.empty:
        print('No scheduled games');return
    games=games[pd.to_datetime(games.start_date,utc=True,errors='coerce')>now]
    if games.empty:
        print('No unstarted games');return
    collect_weather(games,now)
    lines=wp.fetch_lines(games,season,week)
    sp,tot,win,features=wp.load_models()
    predictions=wp.build_predictions(games,lines,sp,tot,win,features,season,archive_decisions=True)
    from market_edges import board_rows
    board = {"generated_at": now.isoformat(), "season": season, "week": week,
             "mode": "research_not_validated_bets", "candidates": board_rows(predictions)}
    output = Path(__file__).resolve().parents[1]/'outputs/predictions/current_market_board.json'
    output.write_text(json.dumps(board, indent=2, allow_nan=False)+'\n')
    print('Archived:' ,len(predictions),'games;',predictions.decision_status.value_counts().to_dict())


if __name__=='__main__':record()
