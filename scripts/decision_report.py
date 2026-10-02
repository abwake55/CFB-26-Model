#!/usr/bin/env python3
"""Grade immutable paper decisions and analyst picks at their recorded prices."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from decision_quality import utc,payout
from pregame_context import asof


def grade(market, side, line, odds, game):
    margin=float(game['home_points'])-float(game['away_points'])
    if market=='totals':
        delta=(float(game['home_points'])+float(game['away_points'])-float(line))*(1 if side.lower()=='over' else -1)
    elif market=='spreads':
        delta=(margin if side==game['home_team'] else -margin)+float(line)
    elif market=='h2h':
        delta=margin if side==game['home_team'] else -margin
    else:raise ValueError('Unknown market')
    return ('win',payout(odds)) if delta>0 else ('loss',-1.) if delta<0 else ('push',0.)


def closing_comparison(quote, snapshots, kickoff):
    """Same book/market/side, last observed pre-kickoff quote; no postgame fill."""
    eligible=[q for q in snapshots if q.get('book')==quote.get('book') and q.get('market')=='totals'
              and q.get('side')=='Under' and utc(q.get('observed_at')) is not None
              and utc(q.get('updated_at')) is not None
              and utc(q['updated_at']) <= utc(q['observed_at']) < kickoff
              and 0 <= (kickoff-utc(q['updated_at'])).total_seconds() <= 3600]
    if not eligible:return {'clv_status':'missing_fresh_pre_kickoff_quote'}
    close=max(eligible,key=lambda q:q['observed_at'])
    result={'clv_status':'observed','closing_line':close['line'],'closing_odds':close['odds'],
            'point_clv':float(quote['line'])-float(close['line']),
            'price_clv_same_line':None}
    if quote['line']==close['line']:
        result['price_clv_same_line']=1/(1+payout(close['odds']))-1/(1+payout(quote['odds']))
    return result


def evaluate():
    games=pd.read_csv(ROOT/'data/processed/master_games.csv').drop_duplicates('game_id').set_index('game_id')
    records=[]
    quotes={}
    for path in (ROOT/'data/lines_snapshots/quotes').glob('*.json'):
        q=json.loads(path.read_text());quotes.setdefault(q['game_id'],[]).append(q)
    for f in (ROOT/'outputs/predictions/decisions').glob('*.json'):
        r=json.loads(f.read_text());p=r['prediction'];d=r['decision'];gid=p['game_id']
        if gid not in games.index or 'quote' not in d:continue
        g=games.loc[gid];cutoff=utc(d['decision_at']);kickoff=utc(g.start_date)
        if kickoff is None or cutoff>=kickoff or pd.isna(g.home_points) or pd.isna(g.away_points):continue
        q=d['quote'];outcome,profit=grade('totals','Under',q['line'],q['odds'],g)
        analysts=[r for r in asof(gid,cutoff) if r['kind']=='analyst']
        # Latest eligible publication per analyst/game/market: repeated articles don't vote twice.
        latest={}
        for a in analysts:
            ap=a['payload'];latest[(ap['analyst'],ap['market'])]=ap
        total_analysts=[a for a in latest.values() if a['market']=='totals']
        records.append({'game_id':gid,'decision_at':d['decision_at'],'status':d['decision_status'],
                        'line':q['line'],'odds':q['odds'],'book':q['book'],'result':outcome,'paper_profit':profit,
                        'model_probability':d['probabilities']['win'],
                        **closing_comparison(q,quotes.get(gid,[]),kickoff),
                        'analyst_coverage':len(total_analysts),
                        'analyst_under_share':sum(a['side'].lower()=='under' for a in total_analysts)/len(total_analysts) if total_analysts else None})
    rows=pd.DataFrame(records)
    report={'mode':'paper_simulation_not_placed_bets','selection':'first priced decision per game',
            'completed_decisions':0,'analyst_records':0,'limitations':[
                'Consensus movement is not executable CLV; exact same-market book snapshots required.',
                'Analyst agreement comparison is observational, not a causal improvement estimate.',
                'Prospective promotion remains manual after independent review.']}
    if not rows.empty:
        rows=rows.sort_values('decision_at').drop_duplicates('game_id')
        # REVIEW rows were explicitly blocked; do not count as a recommended strategy.
        paper=rows[rows.status=='PAPER'].copy()
        report['completed_decisions']=len(rows);report['eligible_paper_decisions']=len(paper)
        report['clv_coverage']=int(rows.clv_status.eq('observed').sum())
        report['mean_point_clv']=float(rows.point_clv.mean()) if 'point_clv' in rows and rows.point_clv.notna().any() else None
        if len(paper):
            returns=paper.paper_profit.to_numpy();equity=np.r_[0,np.cumsum(returns)]
            report.update(paper_profit=float(returns.sum()),paper_roi=float(returns.mean()),
                          max_drawdown=float(np.max(np.maximum.accumulate(equity)-equity)))
        report['agreement_groups']=[]
        for label,subset in [('no_coverage',rows[rows.analyst_coverage==0]),
                             ('agree',rows[rows.analyst_under_share>=.5]),
                             ('disagree',rows[rows.analyst_under_share<.5])]:
            report['agreement_groups'].append({'group':label,'n':len(subset),
                'paper_roi':float(subset.paper_profit.mean()) if len(subset) else None})
    # Grade each analyst's first reported pick, only if collected before kickoff.
    analyst=[];seen=set()
    reports=[]
    for f in (ROOT/'data/context').glob('*.json'):
        r=json.loads(f.read_text())
        if r['kind']=='analyst':reports.append(r)
    for r in sorted(reports,key=lambda x:x['observed_at']):
        gid=r['game_id'];p=r['payload'];key=(gid,p['analyst'],p['market'])
        if key in seen or gid not in games.index:continue
        g=games.loc[gid];kickoff=utc(g.start_date)
        if kickoff is None or utc(r['observed_at'])>=kickoff or utc(r['published_at'])>=kickoff:continue
        seen.add(key)
        if pd.isna(g.home_points) or pd.isna(g.away_points):continue
        result,profit=grade(p['market'],p['side'],p.get('line'),p['odds'],g)
        analyst.append({'analyst':p['analyst'],'game_id':gid,'market':p['market'],'result':result,'paper_profit':profit})
    report['analyst_records']=len(analyst)
    report['analysts']=pd.DataFrame(analyst).groupby('analyst').agg(n=('game_id','size'),paper_profit=('paper_profit','sum')).reset_index().to_dict('records') if analyst else []
    (ROOT/'outputs/predictions/decision_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':evaluate()
