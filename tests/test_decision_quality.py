import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from decision_quality import probabilities,ev,total_decision,append_record
from pregame_context import ingest,asof
from challengers import non_garbage,PossessionModel,select_blend
import gates


class DecisionTests(unittest.TestCase):
    def test_push_mass_and_exact_odds(self):
        p=probabilities([49,50,51],[50,50,50],50,50,'under')
        self.assertAlmostEqual(p['push'],1/3)
        self.assertAlmostEqual(sum(p[k] for k in ['win','loss','push']),1)
        self.assertEqual(probabilities([49,50,51],[50]*3,50,50.5,'under')['push'],0)
        self.assertGreater(ev(p,-105),ev(p,-120))
        with self.assertRaises(ValueError):ev(p,0)

    def test_missing_context_never_live(self):
        row={'totals_edge':-3,'over_under':53,'home_conference':'SEC'}
        self.assertTrue(gates.core_candidate(row))
        self.assertFalse(gates.core_total(row))
        d=total_decision(row,pd.DataFrame())
        self.assertEqual(d['units'],0)
        self.assertIn('weather_missing_stale_or_high_wind',d['decision_reasons'])

    def test_price_selection_and_future_calibration_excluded(self):
        now='2026-10-01T10:00:00Z'
        history=pd.DataFrame({'season':[2025]*300+[2026]*500,'total_points':[48]*300+[99]*500,
                              'pred_total':[50]*800,'evaluation_version':['season_holdout_v2']*800})
        row={'season':2026,'totals_edge':-3,'over_under':53,'pred_total':50,'home_conference':'SEC',
             'start_date':'2026-10-03T16:00:00Z','is_dome':1,'weather_source':'https://example.com',
             'weather_observed_at':now,'availability_status':'reviewed','availability_observed_at':now,
             'book_quotes':[{'book':'a','market':'totals','side':'Under','line':53,'odds':-200,'updated_at':now,'observed_at':now},
                            {'book':'b','market':'totals','side':'Under','line':52.5,'odds':-105,'updated_at':now,'observed_at':now}]}
        d=total_decision(row,history,now)
        self.assertEqual(d['quote']['book'],'b')
        self.assertEqual(d['probabilities']['samples'],300)
        self.assertEqual(d['decision_status'],'PAPER')
        self.assertEqual(d['units'],0)
        stale={**row,'book_quotes':[{**q,'updated_at':'2026-09-29T00:00:00Z'} for q in row['book_quotes']]}
        self.assertIn('no_fresh_priced_book_quote',total_decision(stale,history,now)['decision_reasons'])
        started={**row,'start_date':'2026-10-01T09:00:00Z'}
        self.assertIn('missing_or_started_kickoff',total_decision(started,history,now)['decision_reasons'])

    def test_archive_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            a=append_record(directory,{'line':50});b=append_record(directory,{'line':51})
            self.assertNotEqual(a,b)
            self.assertEqual(a,append_record(directory,{'line':50}))
            self.assertEqual(len(list(Path(directory).glob('*.json'))),2)

    def test_collected_later_news_unavailable_in_backtest(self):
        with tempfile.TemporaryDirectory() as directory:
            r={'game_id':1,'kind':'weather','source_url':'https://example.com',
               'published_at':'2026-09-01T00:00:00Z','payload':{'wind_speed':5}}
            ingest(r,directory,now='2026-09-03T00:00:00Z')
            self.assertEqual(asof(1,'2026-09-02T00:00:00Z',directory),[])
            self.assertEqual(len(asof(1,'2026-09-04T00:00:00Z',directory)),1)

    def test_raw_book_quotes_preserve_line_price_pair(self):
        from odds_api import events_to_consensus,match_to_games
        events=[{'id':'a','home_team':'Home','away_team':'Away','commence_time':'2026-10-03T00:00:00Z',
                 'bookmakers':[{'key':'book','last_update':'2026-10-01T00:00:00Z','markets':[
                     {'key':'totals','outcomes':[{'name':'Under','point':52,'price':-125},
                                                {'name':'Over','point':52,'price':105}]}]}]}]
        c=events_to_consensus(events)
        q=c.iloc[0].book_quotes[0]
        self.assertEqual((q['line'],q['odds'],q['book']),(52,-125,'book'))
        games=pd.DataFrame({'game_id':[1],'home_team':['Home'],'away_team':['Away'],'start_date':['2026-10-03T00:00:00Z']})
        self.assertEqual(len(match_to_games(c,games).iloc[0].book_quotes),2)

    def test_epa_no_prior_season_tail(self):
        from feature_builder import load_recent_epa
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            pd.DataFrame({'game_id':[1,2],'season':[2025,2026],'team':['A','A'],
                          'off_epa':[99,.2],'def_epa':[99,.1]}).to_csv(d/'master_ppa_games.csv',index=False)
            pd.DataFrame({'game_id':[1,2],'start_date':['2025-12-30','2026-09-01']}).to_csv(d/'master_games.csv',index=False)
            r=load_recent_epa(2026,d)
            self.assertAlmostEqual(r.loc['A','off_epa_roll3'],.2)
            self.assertTrue(load_recent_epa(2027,d).empty)

    def test_garbage_and_conflicting_drives(self):
        drives=pd.DataFrame({'gameId':[1,1,2],'id':['a','b','c'],'offense':['A']*3,'defense':['B']*3,
            'startPeriod':[1,4,5],'startOffenseScore':[0,42,0],'startDefenseScore':[0,0,0],
            'endOffenseScore':[7,49,7],'driveResult':['TD']*3})
        drives['startTime'] = [{'minutes': 15, 'seconds': 0} for _ in range(len(drives))]
        self.assertEqual(non_garbage(pd.concat([drives, drives.iloc[:1]])).id.tolist(),['a'])
        with self.assertRaises(ValueError):
            non_garbage(pd.concat([drives,drives.iloc[:1].assign(endOffenseScore=3)]))

    def test_possession_fit_ignores_future_game(self):
        rows=[]
        for gid,score in [(1,3),(2,7)]:
            for n in range(120):rows.append({'gameId':gid,'id':f'{gid}-{n}',
                'offense':'A' if n%2 else 'B','defense':'B' if n%2 else 'A','startPeriod':1,
                'startOffenseScore':0,'startDefenseScore':0,'endOffenseScore':score,'driveResult':'score'})
        games=pd.DataFrame({'game_id':[1,2],'start_date':['2025-09-01T00:00:00Z','2026-09-01T00:00:00Z']})
        model=PossessionModel().fit(pd.DataFrame(rows),games,pd.Timestamp('2026-01-01',tz='UTC'))
        self.assertEqual(model.training_drives,120)
        pred=model.predict(pd.DataFrame({'home_team':['A'],'away_team':['B']}))
        self.assertAlmostEqual(pred.possession_total.iloc[0],180) # capped at 30 drives/side * 3

    def test_blend_no_data_defaults_market(self):
        d=pd.DataFrame(columns=['pred_total','possession_total','over_under','total_points'])
        np.testing.assert_equal(select_blend(d),[0,0,1])

    def test_newsletter_paper_has_zero_units(self):
        import generate_newsletter as gn
        import weekly_pipeline as wp
        row={'game_id':1,'season':2026,'week':5,'home_team':'A','away_team':'B','home_conference':'SEC',
             'start_date':'2026-10-03T00:00:00Z','spread':-7,'over_under':53,'pred_spread':11,'pred_total':50,
             'spread_edge':4,'totals_edge':-3,'home_moneyline':np.nan,'away_moneyline':np.nan,
             'pred_win_p':.7,'pred_away_win_p':.3}
        with patch.object(wp,'fetch_schedule',return_value=pd.DataFrame([row])),patch.object(wp,'fetch_lines',return_value=pd.DataFrame()),patch.object(wp,'load_models',return_value=(None,None,None,{})),patch.object(wp,'build_predictions',return_value=pd.DataFrame([row])):
            picks=gn.predict_games(2026,5)
        self.assertTrue(picks)
        self.assertTrue(all(p['kelly']==0 and p['tier']!='CORE' for p in picks))

    def test_closing_quote_cannot_use_postgame_or_stale_prices(self):
        from decision_report import closing_comparison
        start=pd.Timestamp('2026-10-03T16:00:00Z')
        quote={'book':'a','line':54,'odds':-110}
        rows=[{'book':'a','market':'totals','side':'Under','line':52,'odds':-105,
               'observed_at':'2026-10-03T15:55:00Z','updated_at':'2026-10-03T15:54:00Z'},
              {'book':'a','market':'totals','side':'Under','line':49,'odds':-105,
               'observed_at':'2026-10-03T18:00:00Z','updated_at':'2026-10-03T18:00:00Z'}]
        result=closing_comparison(quote,rows,start)
        self.assertEqual(result['point_clv'],2)
        self.assertIsNone(result['price_clv_same_line'])
        self.assertEqual(closing_comparison(quote,rows,pd.Timestamp('2026-10-04T16:00:00Z'))['clv_status'],'missing_fresh_pre_kickoff_quote')

    def test_analyst_spread_grading_signs(self):
        from decision_report import grade
        game={'home_team':'A','away_team':'B','home_points':30,'away_points':27}
        self.assertEqual(grade('spreads','B',3,-110,game),('push',0))
        self.assertEqual(grade('spreads','B',3.5,-110,game)[0],'win')
