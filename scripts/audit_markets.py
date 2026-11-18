#!/usr/bin/env python3
"""Season-forward screening audit. Assumed -110 spread/total prices, not live ROI."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from market_edges import calibration_history, estimate, win_lower_bound, MIN_SAMPLES, VERSION
from decision_quality import ev, finite, payout


def run():
    wf=pd.read_csv(ROOT/'outputs/predictions/walk_forward_results.csv')
    records=[]
    for season in sorted(wf.season.unique()):
        hist=calibration_history(wf,season)
        if len(hist)<MIN_SAMPLES:continue
        test=wf[(wf.season==season)&wf.evaluation_version.eq('season_holdout_v2')].drop_duplicates('game_id')
        for _,g in test.iterrows():
            choices=[('spreads','home',g.spread,-110),('spreads','away',-g.spread,-110),
                     ('totals','under',g.over_under,-110),('totals','over',g.over_under,-110),
                     ('h2h','home',None,g.home_moneyline),('h2h','away',None,g.away_moneyline)]
            best={}
            for market,side,line,odds in choices:
                if not finite(odds):continue
                try:
                    p=estimate(g,hist,market,side,line)
                    expected=ev(p,odds)
                except (ValueError,TypeError,KeyError,AttributeError):continue
                if p['samples']<MIN_SAMPLES or expected<.02:continue
                lower=win_lower_bound(p['win'],p['samples'])
                tier='RESEARCH_A' if lower*payout(odds)-(1-lower-p['push'])>0 else 'RESEARCH_B'
                if market=='spreads':delta=(g.point_diff if side=='home' else -g.point_diff)+line
                elif market=='totals':delta=(g.total_points-line)*(1 if side=='over' else -1)
                else:delta=g.point_diff*(1 if side=='home' else -1)
                if not finite(delta):continue
                result='win' if delta>0 else 'loss' if delta<0 else 'push'
                r={'game_id':int(g.game_id),'season':int(season),'market':market,'side':side,
                   'tier':tier,'ev':expected,'result':result,
                   'profit':payout(odds) if result=='win' else -1 if result=='loss' else 0}
                if market not in best or expected>best[market]['ev']:best[market]=r
            records.extend(best.values())
    rows=pd.DataFrame(records)
    report={'version':VERSION,'live_approved':False,'assumptions':[
        'Spreads and totals priced at assumed -110; historical moneylines only where recorded.',
        'Prior-season holdout errors only; no same-season calibration or threshold optimization.',
        'One highest-EV side per game/market; no cross-market independence assumption.',
        'Research A sampling bounds do not measure model misspecification or selection bias.',
        'These are historical screening diagnostics, not executable betting returns.'],
        'seasons':[], 'overall':[]}
    for keys,d in rows.groupby(['season','market','tier']):
        report['seasons'].append(dict(zip(['season','market','tier'],[int(keys[0]),*keys[1:] ])))
        report['seasons'][-1].update(summary(d))
    for keys,d in rows.groupby(['market','tier']):
        report['overall'].append(dict(zip(['market','tier'],keys)))
        report['overall'][-1].update(summary(d))
    out=ROOT/'outputs/predictions/market_audit.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['overall'],indent=2))


def summary(d):
    w=int(d.result.eq('win').sum());l=int(d.result.eq('loss').sum())
    return {'n':len(d),'wins':w,'losses':l,'pushes':int(d.result.eq('push').sum()),
            'hit_rate':w/(w+l) if w+l else None,'assumed_price_roi':float(d.profit.mean())}

if __name__=='__main__':run()
