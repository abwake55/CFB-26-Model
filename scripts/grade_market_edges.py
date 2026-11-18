#!/usr/bin/env python3
"""Grade the first eligible paper candidate per game/market at recorded odds."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from decision_quality import utc
from decision_report import grade


def grade_records(records, games):
    selected={}
    # Skip blocked reviews; missing context is not a placed or eligible paper bet.
    for record in sorted(records,key=lambda r:r['decision']['decision_at']):
        p,d=record['prediction'],record['decision'];gid=p['game_id']
        if gid not in games.index:continue
        g=games.loc[gid];kickoff=utc(g.get('start_date'));decision_at=utc(d['decision_at'])
        if kickoff is None or decision_at is None or decision_at>=kickoff:continue
        for c in sorted(d.get('candidates',[]),key=lambda c:c['ev'],reverse=True):
            if c['status']!='PAPER' or c['tier']=='PASS':continue
            key=(gid,c['market'])
            if key not in selected:selected[key]=(p,d,c,g)
    rows=[]
    for (gid,market),(p,d,c,g) in selected.items():
        if pd.isna(g.home_points) or pd.isna(g.away_points):continue
        q=c['quote'];side=g[c['selection']+'_team'] if c['selection'] in ('home','away') else c['selection']
        outcome,profit=grade(market,side,q.get('line'),q['odds'],g)
        rows.append({'game_id':gid,'market':market,'tier':c['tier'],
                     'result':outcome,'paper_profit':profit,'decision_at':d['decision_at'],
                     'book':q['book'],'line':q.get('line'),'odds':q['odds']})
    return rows


def run():
    games=pd.read_csv(ROOT/'data/processed/master_games.csv').drop_duplicates('game_id').set_index('game_id')
    records=[json.loads(p.read_text()) for p in (ROOT/'outputs/predictions/market_decisions').glob('*.json')]
    rows=grade_records(records,games)
    report={'mode':'prospective_paper_not_placed_bets','selection':'first eligible candidate per game/market',
            'completed':len(rows),'by_market':[], 'records':rows}
    if rows:
        for (market,tier),d in pd.DataFrame(rows).groupby(['market','tier']):
            report['by_market'].append({'market':market,'tier':tier,'n':len(d),
                'wins':int(d.result.eq('win').sum()),'losses':int(d.result.eq('loss').sum()),
                'pushes':int(d.result.eq('push').sum()),'paper_roi':float(d.paper_profit.mean())})
    (ROOT/'outputs/predictions/market_record.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Graded',len(rows),'eligible all-market paper candidates')

if __name__=='__main__':run()
