#!/usr/bin/env python3
"""Season-forward benchmark and empirical probability audit; no auto-promotion."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from challengers import PossessionModel,select_blend,matchup_frame,matchup_model
from decision_quality import probabilities, POLICY


def evaluate():
    wf=pd.read_csv(ROOT/'outputs/predictions/walk_forward_results.csv')
    games=pd.read_csv(ROOT/'data/processed/master_games.csv')
    wf=wf.merge(games[['game_id','start_date']].drop_duplicates(),on='game_id',validate='one_to_one')
    report={'policy':POLICY,'live_approved':False,'probability_folds':[],
            'limitations':['Retrospectively developed models; prospective evidence required.',
                          'Historical quotes lack decision-time book prices; ROI is not executable evidence.']}
    # Earlier OOS seasons calibrate the next season's probabilities.
    probability_rows=[]
    for season in sorted(wf.season.unique()):
        history=wf[wf.season<season]; test=wf[wf.season==season]
        if len(history)<300:continue
        for _,g in test.iterrows():
            for market,actual,pred,line,side in [('totals','total_points','pred_total','over_under','under'),
                                               ('spreads','point_diff','pred_spread','vegas_home_margin','home')]:
                p=probabilities(history[actual],history[pred],g[pred],g[line],side)
                delta=(g[actual]-g[line])*(-1 if side=='under' else 1)
                outcome='win' if delta>0 else 'loss' if delta<0 else 'push'
                probability_rows.append({'game_id':g.game_id,'season':season,'market':market,
                    **p,'outcome':outcome,'calibration_through':int(history.season.max()),
                    'brier':sum((p[k]-int(outcome==k))**2 for k in ['win','loss','push']),
                    'log_loss':-np.log(max(p[outcome],1e-6))})
    pr=pd.DataFrame(probability_rows)
    report['probability_folds']=pr.groupby(['season','market']).agg(n=('game_id','size'),brier=('brier','mean'),log_loss=('log_loss','mean')).reset_index().to_dict('records')
    bins=[]
    for market,d in pr.groupby('market'):
        d=d.copy();d['bin']=pd.cut(d.win,bins=np.linspace(0,1,11),include_lowest=True)
        for interval,group in d.groupby('bin',observed=True):
            bins.append({'market':market,'bin':str(interval),'n':len(group),
                         'forecast':float(group.win.mean()),'observed':float(group.outcome.eq('win').mean())})
    report['reliability']=bins
    features=pd.read_csv(ROOT/'data/processed/feature_matrix.csv').drop_duplicates('game_id')
    if features.game_id.duplicated().any():
        raise ValueError('Conflicting feature rows')
    joined=wf.merge(features.drop(columns=[c for c in features if c in wf and c!='game_id']),on='game_id',validate='one_to_one')
    report['matchup_folds']=[]
    for season in sorted(joined.season.unique()):
        previous=sorted(y for y in joined.season.unique() if y<season and y!=2020)
        if len(previous)<2:continue
        validation=joined[joined.season==previous[-1]]
        train=joined[(joined.season<previous[-1]) & (joined.season!=2020)]
        test=joined[joined.season==season]
        for target,market in [('total_points','over_under'),('point_diff','vegas_home_margin')]:
            fitted=[]
            for alpha in [10,100,1000]:
                model=matchup_model(alpha).fit(matchup_frame(train),train[target]-train[market])
                prediction=validation[market]+model.predict(matchup_frame(validation))
                fitted.append((float((prediction-validation[target]).abs().mean()),model,alpha))
            _,model,alpha=min(fitted,key=lambda item:item[0])
            prediction=test[market]+model.predict(matchup_frame(test))
            report['matchup_folds'].append({'season':int(season),'target':target,
                'training_through':int(train.season.max()),'validation_season':int(previous[-1]),
                'alpha':alpha,'n':len(test),'mae':float((prediction-test[target]).abs().mean()),
                'market_mae':float((test[market]-test[target]).abs().mean())})
    files=list((ROOT/'data/raw/drives').glob('*.json'))
    if files:
        drives=pd.DataFrame([r for path in files for r in json.loads(path.read_text())])
        folds=[]
        for season in sorted(wf.season.unique()):
            test=wf[wf.season==season].copy()
            if test.empty:continue
            cutoff=pd.to_datetime(test.start_date,utc=True).min()
            try:
                model=PossessionModel().fit(drives,games,cutoff)
            except ValueError as exc:
                folds.append({'season':int(season),'status':str(exc)});continue
            pred=model.predict(test)
            folds.append({'season':int(season),'status':'evaluated','games':len(pred),
                'possession_total_mae':float((pred.possession_total-pred.total_points).abs().mean()),
                'market_total_mae':float((pred.over_under-pred.total_points).abs().mean())})
            if 'all_predictions' not in locals():all_predictions=[]
            all_predictions.append(pred)
        report['possession_folds']=folds
        if 'all_predictions' in locals():
            combined=pd.concat(all_predictions,ignore_index=True); blended=[]
            for season in sorted(combined.season.unique()):
                prior=combined[combined.season<season]
                if prior.empty:continue
                validation=prior[prior.season==prior.season.max()]
                weights=select_blend(validation)
                test=combined[combined.season==season].copy()
                # Correct the scoring scale only using previous season outcomes.
                # Keep raw challenger above; blend is deliberately constrained.
                test['ensemble_total']=test[['pred_total','possession_total','over_under']].to_numpy()@weights
                blended.append({'season':int(season),'validation_season':int(validation.season.max()),
                    'weights':weights.tolist(),'mae':float((test.ensemble_total-test.total_points).abs().mean())})
            report['ensemble_folds']=blended
            report['live_ensemble_weights']=select_blend(combined[combined.season==combined.season.max()]).tolist()
            report['live_ensemble_validation_season']=int(combined.season.max())
            combined.to_csv(ROOT/'outputs/predictions/possession_shadow.csv',index=False)
    else:
        report['possession_status']='blocked_missing_drive_data: run scripts/refresh_drives.py with CFB_API_KEY'
    path=ROOT/'outputs/predictions/challenger_report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pr.to_csv(ROOT/'outputs/predictions/probability_shadow.csv',index=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':evaluate()
