"""Evaluate walk-forward CSVs vs closing (and opening) lines. Usage: evaluate_wf.py file1 [file2...]"""
import sys, math, json
import numpy as np, pandas as pd
from pathlib import Path
FM=Path(__file__).resolve().parents[1]/'data/processed/feature_matrix.csv'
fm=pd.read_csv(FM,usecols=['game_id','spread_open','over_under_open','home_conference','away_conference','wind_speed','is_dome']).drop_duplicates('game_id')
POWER={"SEC","Big Ten","Big 12","ACC","Pac-12","Pac-10","Big East","FBS Independents"}
def rec(sig, mar):
    s=sig*mar; w=int((s>0).sum()); l=int((s<0).sum()); p=int((s==0).sum()); u=w-1.1*l; n=w+l+p
    return dict(n=n,W=w,L=l,P=p,pct=round(100*w/(w+l),2) if w+l else None,units=round(u,1),roi=round(100*u/(1.1*n),2) if n else None)
def evaluate(path):
    d=pd.read_csv(path).merge(fm,on='game_id',how='left')
    d['ats']=d.point_diff-d.vegas_home_margin; d['ou']=d.total_points-d.over_under
    r={'file':path.split('/')[-1],'n_games':len(d),
       'spread_mae':round((d.pred_spread-d.point_diff).abs().mean(),3),'close_spread_mae':round((d.vegas_home_margin-d.point_diff).abs().mean(),3),
       'total_mae':round((d.pred_total-d.total_points).abs().mean(),3),'close_total_mae':round((d.over_under-d.total_points).abs().mean(),3),
       'brier':round(((d.pred_home_win_p-d.home_win)**2).mean(),4)}
    for t in (2,3,4,5):
        g=d[d.spread_edge.abs()>=t]; r[f'spread_close_e{t}']=rec(np.sign(g.spread_edge),g.ats)
    for t in (2,3):
        g=d[d.totals_edge.abs()>=t]; r[f'total_close_e{t}']=rec(np.sign(g.totals_edge),g.ou)
        g=d[d.totals_edge<=-t]; r[f'under_close_e{t}']=rec(-np.ones(len(g)),g.ou)
    # CORE gate (same as gates.core_candidate, historical observed wind)
    c=d[(d.totals_edge<=-2)&(d.totals_edge>=-7)&(d.home_conference.isin(POWER)|d.away_conference.isin(POWER))
        &~((d.is_dome.fillna(0)==0)&(d.wind_speed.fillna(0)>=15))&(d.over_under>=48)]
    r['core']=rec(-np.ones(len(c)),c.ou)
    # decision at OPENING line (games with an opener): residual edge, graded at open, CLV = move toward us
    o=d.dropna(subset=['spread_open']).copy(); o['open_margin']=-o.spread_open
    o['ats_open']=o.point_diff-o.open_margin
    for t in (2,3):
        g=o[o.spread_edge.abs()>=t]; sg=np.sign(g.spread_edge)
        x=rec(sg,g.ats_open); clv=sg*(g.vegas_home_margin-g.open_margin)
        x.update(mean_clv_pts=round(clv.mean(),3),clv_pos_share=round((clv>0).mean(),3),clv_neg_share=round((clv<0).mean(),3)); r[f'spread_open_e{t}']=x
    r['open_n_games']=len(o); r['open_seasons']=sorted(o.season.unique().tolist())
    r['by_season_spread_close_e3']={int(s):rec(np.sign(g[g.spread_edge.abs()>=3].spread_edge),g[g.spread_edge.abs()>=3].ats) for s,g in d.groupby('season')}
    r['by_season_spread_mae']={int(s):[round((g.pred_spread-g.point_diff).abs().mean(),3),round((g.vegas_home_margin-g.point_diff).abs().mean(),3)] for s,g in d.groupby('season')}
    return r
if __name__=='__main__':
    res=[evaluate(p) for p in sys.argv[1:]]
    print(json.dumps(res,indent=1))
