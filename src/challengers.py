"""Opponent-adjusted possession challenger. Always shadow; never sizes bets."""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


def unique_drives(drives):
    # CFBD clock fields are nested JSON objects; pandas cannot hash them.
    keys = drives.copy()
    for column in keys.select_dtypes(include=['object']).columns:
        keys[column] = keys[column].map(
            lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
    unique = drives.loc[~keys.duplicated()].copy()
    if unique.duplicated(['gameId', 'id']).any():
        raise ValueError('Conflicting drive IDs')
    return unique


def non_garbage(drives):
    required = ['gameId', 'id', 'offense', 'defense', 'startPeriod',
                'startOffenseScore', 'startDefenseScore', 'endOffenseScore', 'driveResult']
    missing = set(required) - set(drives)
    if missing:
        raise ValueError(f'Missing drive fields: {sorted(missing)}')
    d = unique_drives(drives)
    if d.duplicated(['gameId', 'id']).any():
        raise ValueError('Conflicting drive IDs')
    margin = (d.startOffenseScore - d.startDefenseScore).abs()
    # Fixed research definition, selected before evaluation, not a copy of FEI.
    keep = ((d.startPeriod <= 2) | ((d.startPeriod == 3) & (margin <= 28)) |
            ((d.startPeriod == 4) & (margin <= 21)))
    keep &= ~d.driveResult.str.lower().str.contains('end of half|end of game|kneel', na=False)
    d = d[keep].copy()
    d['points'] = d.endOffenseScore - d.startOffenseScore
    return d[d.points.between(0, 8)].dropna(subset=required)


class PossessionModel:
    """Ridge attack/defense effects plus team/opponent regulation possession pace."""
    def fit(self, drives, games, cutoff):
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError('UTC-aware training cutoff required')
        dates = games[['game_id', 'start_date']].drop_duplicates()
        dates['start_date'] = pd.to_datetime(dates.start_date, utc=True)
        # No same-day or in-progress scores, including staggered kickoffs.
        dates = dates[dates.start_date + pd.Timedelta(hours=12) < cutoff]
        raw = drives.merge(dates, left_on='gameId', right_on='game_id', validate='many_to_one')
        raw = unique_drives(raw)
        d = non_garbage(raw)
        if len(d) < 100:
            raise ValueError('At least 100 eligible prior drives required')
        self.vectorizer = DictVectorizer()
        x = self.vectorizer.fit_transform([{'offense':o, 'defense':de} for o,de in zip(d.offense,d.defense)])
        self.model = Ridge(alpha=30, solver='lsqr').fit(x,d.points)
        pace = raw[raw.startPeriod.le(4)].groupby(['gameId','offense','defense']).size().rename('drives').reset_index()
        self.pace_vectorizer = DictVectorizer()
        px = self.pace_vectorizer.fit_transform([{'team':o, 'opponent':de} for o,de in zip(pace.offense,pace.defense)])
        self.pace_model = Ridge(alpha=20,solver='lsqr').fit(px,pace.drives)
        self.cutoff = cutoff.isoformat()
        self.training_drives = len(d)
        return self

    def predict(self, games):
        out = games.copy()
        scores=[]
        for _,g in games.iterrows():
            match=[{'offense':g.home_team,'defense':g.away_team},
                   {'offense':g.away_team,'defense':g.home_team}]
            ppd=np.clip(self.model.predict(self.vectorizer.transform(match)),0,8)
            pace=np.clip(self.pace_model.predict(self.pace_vectorizer.transform([
                {'team':g.home_team,'opponent':g.away_team},
                {'team':g.away_team,'opponent':g.home_team}])),1,30)
            scores.append(ppd*pace.mean())
        scores=np.asarray(scores)
        out['possession_total']=scores.sum(axis=1)
        out['possession_margin']=scores[:,0]-scores[:,1]
        out['possession_training_cutoff']=self.cutoff
        return out


def select_blend(validation, target='total_points'):
    """Choose weights on a completed prior season, never the evaluation season."""
    columns=['pred_total','possession_total','over_under']
    d=validation.dropna(subset=columns+[target])
    if len(d)<100:
        return np.array([0.,0.,1.])
    candidates=[np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a)]
    return min(candidates,key=lambda w: np.mean(np.abs(d[columns].to_numpy()@w-d[target])))


def matchup_frame(frame):
    """Pregame-only inputs; never use unlagged EPA or full-season scoring columns."""
    names=['sp_diff','elo_diff','talent_diff','havoc_diff','tempo_combined','rush_rate_combined',
           'home_off_epa_roll3','away_off_epa_roll3','home_def_epa_roll3','away_def_epa_roll3',
           'home_off_epa_pass_roll3','away_off_epa_pass_roll3',
           'home_off_epa_rush_roll3','away_off_epa_rush_roll3',
           'home_ret_pass_ppa_pct','away_ret_pass_ppa_pct']
    x=pd.DataFrame({c:pd.to_numeric(frame[c],errors='coerce') if c in frame else np.nan for c in names},index=frame.index)
    x['home_attack_vs_defense']=x.home_off_epa_roll3-x.away_def_epa_roll3
    x['away_attack_vs_defense']=x.away_off_epa_roll3-x.home_def_epa_roll3
    x['passing_matchup']=x.home_off_epa_pass_roll3-x.away_off_epa_pass_roll3
    x['rushing_matchup']=x.home_off_epa_rush_roll3-x.away_off_epa_rush_roll3
    return x.replace([np.inf,-np.inf],np.nan)


def matchup_model(alpha):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         StandardScaler(),Ridge(alpha=alpha))


def live_shadow(frame, root):
    """Archive current possession and ensemble projections without promotion."""
    import json
    files=list((root/'data/raw/drives').glob('*.json'))
    if not files or frame.empty:
        return [{'status':'missing_drive_data'} for _ in range(len(frame))]
    drives=pd.DataFrame([r for f in files for r in json.loads(f.read_text())])
    games=pd.read_csv(root/'data/processed/master_games.csv')
    now=pd.Timestamp.now(tz='UTC')
    try:
        model=PossessionModel().fit(drives,games,now)
        pred=model.predict(frame)
    except ValueError as exc:
        return [{'status':str(exc)} for _ in range(len(frame))]
    report_path=root/'outputs/predictions/challenger_report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    weights=report.get('live_ensemble_weights',[0,0,1])
    validation=report.get('live_ensemble_validation_season')
    results=[]
    for _,row in pred.iterrows():
        eligible=validation is not None and validation<int(row['season'])
        result={'status':'paper','possession_total':float(row.possession_total),
                'possession_margin':float(row.possession_margin),'trained_before':model.cutoff,
                'training_drives':model.training_drives}
        if eligible and np.isfinite([row.pred_total,row.possession_total,row.over_under]).all():
            result['ensemble_total']=float(np.dot(weights,[row.pred_total,row.possession_total,row.over_under]))
            result['weights']=weights;result['validation_season']=validation
        results.append(result)
    return results
