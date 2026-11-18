import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from market_edges import evaluate_markets, estimate, calibration_history, board_rows

NOW = '2026-11-18T10:00:00Z'


def history():
    return pd.DataFrame({'game_id': range(600), 'season': [2025]*600,
        'evaluation_version': ['season_holdout_v2']*600,
        'point_diff': [3, 7, 10]*200, 'pred_spread': [7]*600,
        'total_points': [40, 50, 60]*200, 'pred_total': [50]*600,
        'pred_home_win_p': [.6]*600, 'home_win': [1, 1, 0]*200})


def row():
    quotes = []
    for market, side, line in [('spreads','home',-7), ('spreads','away',7),
                               ('totals','over',45.5), ('totals','under',55.5),
                               ('h2h','home',None), ('h2h','away',None)]:
        quotes.append({'market':market,'selection':side,'line':line,
                       'odds':-110 if market!='h2h' else 150,'book':'test',
                       'updated_at':NOW,'observed_at':NOW})
    return {'game_id':1000,'season':2026,'start_date':'2026-11-19T00:00:00Z',
            'home_team':'A','away_team':'B','pred_spread':7,'pred_total':50,
            'pred_win_p':.6,'book_quotes':quotes,'model_data_asof':NOW,
            'availability_status':'reviewed','availability_observed_at':NOW,
            'weather_source':'test','weather_observed_at':NOW,'is_dome':1}


class MarketTests(unittest.TestCase):
    def test_spread_signs_and_pushes(self):
        h=history();r=row()
        home=estimate(r,h,'spreads','home',-7)
        away=estimate(r,h,'spreads','away',7)
        self.assertAlmostEqual(home['push'],1/3)
        self.assertEqual(home['loss'],away['win'])
        self.assertEqual(estimate(r,h,'spreads','away',7.5)['push'],0)
        self.assertGreater(estimate(r,h,'spreads','away',7.5)['win'],away['win'])

    def test_all_markets_and_zero_units(self):
        d=evaluate_markets(row(),history(),NOW)
        self.assertEqual(len(d['candidates']),6)
        self.assertEqual({x['selection'] for x in d['candidates']},{'home','away','over','under'})
        self.assertTrue(all(x['units']==0 and x['status']=='PAPER' for x in d['candidates']))
        self.assertTrue(all(x['sampling_lower_ev'] < x['ev'] for x in d['candidates']))

    def test_prior_seasons_only_and_deduplicated(self):
        h=history();future=h.assign(season=2026,total_points=100)
        self.assertEqual(len(calibration_history(pd.concat([h,h,future]),2026)),600)
        self.assertEqual(evaluate_markets(row(),h,NOW),evaluate_markets(row(),pd.concat([h,h,future]),NOW))
        h['evaluation_version']='in_sample'
        self.assertFalse(evaluate_markets(row(),h,NOW)['candidates'])

    def test_stale_started_and_unknown_side(self):
        r=row();r['start_date']=NOW
        self.assertFalse(evaluate_markets(r,history(),NOW)['candidates'])
        r=row()
        for q in r['book_quotes']:q['updated_at']='2026-10-01T00:00:00Z'
        self.assertFalse(evaluate_markets(r,history(),NOW)['candidates'])
        r=row();r['book_quotes']=[{**r['book_quotes'][0],'selection':'unknown'}]
        self.assertFalse(evaluate_markets(r,history(),NOW)['candidates'])

    def test_best_line_price_pair_and_missing_context(self):
        r=row();r.pop('model_data_asof');q=r['book_quotes'][2]
        r['book_quotes'].append({**q,'book':'better','odds':100})
        d=evaluate_markets(r,history(),NOW)
        over=next(x for x in d['candidates'] if x['selection']=='over')
        self.assertEqual(over['quote']['book'],'better')
        self.assertEqual(over['status'],'REVIEW')
        self.assertIn('model_data_freshness_unverified',over['reasons'])
        self.assertTrue(board_rows(pd.DataFrame([{**r,'market_decisions':d}])))

    def test_moneyline_calibration_and_samples(self):
        p=estimate(row(),history(),'h2h','home',None)
        self.assertAlmostEqual(p['win'],2/3)
        r=row();r['pred_win_p']=.95
        d=evaluate_markets(r,history(),NOW)
        self.assertFalse(any(c['market']=='h2h' for c in d['candidates']))
        self.assertFalse(evaluate_markets(row(),history().iloc[:20],NOW)['candidates'])


class ProspectiveTests(unittest.TestCase):
    def test_first_eligible_quote_once_per_market_and_actual_price(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from grade_market_edges import grade_records
        r=row();decision=evaluate_markets(r,history(),NOW)
        games=pd.DataFrame([{'game_id':1000,'home_team':'A','away_team':'B',
            'home_points':31,'away_points':21,'start_date':r['start_date']}]).set_index('game_id')
        records=[{'prediction':r,'decision':decision}]*2
        graded=grade_records(records,games)
        self.assertEqual(len({g['market'] for g in graded}),len(graded))
        ml=next(g for g in graded if g['market']=='h2h')
        self.assertEqual(ml['paper_profit'],1.5)
        for c in decision['candidates']:c['status']='REVIEW'
        self.assertEqual(grade_records(records,games),[])

    def test_integration_attaches_both_paths_without_live_promotion(self):
        import tempfile
        import json
        from decision_quality import attach_decisions
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'feature_lists.json').write_text(json.dumps({'model_data_asof':NOW}))
            out=attach_decisions(pd.DataFrame([row()]),directory,history(),NOW)
        self.assertEqual(len(out.iloc[0].market_decisions['candidates']),6)
        self.assertEqual(out.iloc[0].recommended_units,0)

if __name__=='__main__':unittest.main()
