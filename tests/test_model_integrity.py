import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from betting import settle
from inference import adjust_predictions
from validation import clean_games, split_seasons, temporal_calibrator
import gates


class IntegrityTests(unittest.TestCase):
    def games(self):
        return pd.DataFrame({"game_id": [1, 2, 3], "season": [2022, 2023, 2024],
                             "point_diff": [7, -3, 0], "total_points": [48, 49, 50],
                             "spread": [-7, 3.5, 0], "over_under": [48, 48.5, 51]})

    def test_duplicate_join_rows_count_once(self):
        g = self.games()
        self.assertEqual(len(clean_games(pd.concat([g, g.iloc[:1]]))), 3)

    def test_bowl_week_one_cannot_leak_into_september(self):
        from features import build_rolling_epa
        ppa = pd.DataFrame({"game_id": [3, 1, 2], "season": [2025] * 3,
                            "team": ["A"] * 3, "week": [1, 1, 2],
                            "start_date": ["2025-12-20", "2025-09-01", "2025-09-08"],
                            "off_epa": [99., .1, .2], "def_epa": [99., .2, .3]})
        result = build_rolling_epa(ppa).set_index("game_id")
        self.assertTrue(np.isnan(result.loc[1, "off_epa_roll3"]))
        self.assertAlmostEqual(result.loc[2, "off_epa_roll3"], .1)
        self.assertAlmostEqual(result.loc[3, "off_epa_roll3"], .15)

    def test_epa_without_kickoff_dates_fails(self):
        from features import build_rolling_epa
        with self.assertRaises(ValueError):
            build_rolling_epa(pd.DataFrame({"game_id": [1], "season": [2025],
                                           "team": ["A"], "week": [1]}))

    def test_conflicting_join_rows_rejected(self):
        g = self.games()
        extra = g.iloc[:1].assign(spread=-8)
        with self.assertRaises(ValueError):
            clean_games(pd.concat([g, extra]))

    def test_invalid_targets_removed(self):
        g = self.games().astype({"total_points": float})
        g.loc[0, "total_points"] = np.inf
        self.assertEqual(len(clean_games(g)), 2)

    def test_stale_feature_matrix_cannot_be_trained(self):
        import model
        with tempfile.TemporaryDirectory() as tmp:
            self.games().to_csv(Path(tmp) / "feature_matrix.csv", index=False)
            with patch.object(model, "DATA_DIR", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "Stale EPA chronology"):
                    model.load_data()

    def test_push_targets_and_signs(self):
        g = clean_games(self.games())
        self.assertEqual(g.covered_spread.tolist(), [.5, 1., .5])
        self.assertEqual(g.went_over.tolist(), [.5, 1., 0.])
        self.assertEqual(g.vegas_home_margin.tolist(), [7, -3.5, 0])

    def test_season_holdout_disjoint(self):
        tr, val, te = split_seasons(self.games(), 2024)
        self.assertEqual(tr.season.tolist(), [2022])
        self.assertEqual(val.season.tolist(), [2023])
        self.assertEqual(te.season.tolist(), [2024])

    def test_covid_not_training_or_validation(self):
        g = self.games(); g.loc[0, "season"] = 2020
        with self.assertRaises(ValueError):
            split_seasons(g, 2024)

    def test_calibration_never_trains_on_future(self):
        years = np.repeat([2021, 2022, 2023, 2024], 4)
        cal = temporal_calibrator(LogisticRegression(), years)
        for tr, va in cal.cv:
            self.assertLess(years[tr].max(), years[va].min())
            self.assertFalse(set(tr) & set(va))

    def test_temporal_calibration_fits(self):
        years = np.repeat([2022, 2023], 10)
        x = pd.DataFrame({"x": np.tile([-1., 1.], 10)})
        cal = temporal_calibrator(LogisticRegression(), years)
        cal.fit(x, np.tile([0, 1], 10))
        self.assertTrue(np.isfinite(cal.predict_proba(x)).all())

    def test_single_season_does_not_use_random_calibration(self):
        base = LogisticRegression()
        self.assertIs(temporal_calibrator(base, [2022, 2022]), base)

    def test_push_returns_stake_on_both_sides(self):
        for direction in [-1, 1]:
            self.assertEqual(settle(7, 7, direction), ("push", 0., 1.1))

    def test_spread_direction(self):
        self.assertEqual(settle(10, 7, 1)[0], "win")
        self.assertEqual(settle(10, 7, -1)[0], "loss")
        self.assertEqual(settle(-2, -3, 1)[0], "win")

    def test_actual_price_changes_return(self):
        self.assertEqual(settle(40, 48, -1, -120), ("win", 1., 1.2))
        self.assertEqual(settle(50, 48, -1, 150), ("loss", -1., 1.))
        self.assertEqual(settle(40, 48, -1, 150), ("win", 1.5, 1.))

    def test_missing_line_cannot_be_core(self):
        row = {"totals_edge": -3, "home_conference": "SEC", "over_under": np.nan}
        self.assertFalse(gates.core_candidate(row))

    def test_core_boundaries(self):
        row = {"totals_edge": -2, "home_conference": "SEC", "over_under": 48}
        self.assertTrue(gates.core_candidate(row))
        self.assertFalse(gates.core_candidate({**row, "wind_speed": 15}))
        self.assertFalse(gates.core_candidate({**row, "over_under": 47.5}))
        self.assertFalse(gates.core_candidate({**row, "totals_edge": -7.01}))

    def predictions(self):
        return pd.DataFrame({"week": [1, 2, 3, 4], "spread": [-7.] * 4,
                             "over_under": [50.] * 4, "pred_spread": [10.] * 4,
                             "pred_total": [46.] * 4, "pred_win_p": [.75] * 4})

    def test_early_season_edges_same_adjustment(self):
        out = adjust_predictions(self.predictions())
        np.testing.assert_allclose(out.totals_edge, [-2.4, -3, -3.6, -4])
        np.testing.assert_allclose(out.spread_edge, [1.8, 2.25, 2.7, 3])
        np.testing.assert_allclose(out.pred_away_win_p, 1 - out.pred_win_p)

    def test_no_market_preserves_classifier_late_season(self):
        d = self.predictions().iloc[[3]].copy()
        d["spread"] = np.nan; d["pred_spread"] = np.nan
        out = adjust_predictions(d, {"spread_sigma": 15, "blend_alpha": 1})
        self.assertEqual(out.pred_win_p.iloc[0], .75)
        self.assertTrue(np.isnan(out.spread_edge.iloc[0]))

    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):
            adjust_predictions(self.predictions(), {"spread_sigma": 0})

    def test_cli_and_weekly_pipeline_match(self):
        import predict
        import weekly_pipeline as wp
        class Regressor:
            def __init__(self, value): self.value = value
            def predict(self, x): return np.full(len(x), self.value)
        class Classifier:
            def predict_proba(self, x): return np.tile([.25, .75], (len(x), 1))
        games = pd.DataFrame({"game_id": [1], "season": [2026], "week": [1],
                              "home_team": ["A"], "away_team": ["B"]})
        lines = pd.DataFrame({"game_id": [1], "spread": [-7.], "over_under": [50.],
                             "spread_open": [-6.5]})
        data = games.merge(lines, on="game_id")
        features = {"spread": ["f"], "totals": ["f"], "win_prob": ["f"],
                    "spread_target": "margin_residual"}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "win_prob_calibration.json").write_text(json.dumps(
                {"spread_sigma": 15, "blend_alpha": .4}))
            with patch.object(predict, "MODEL_DIR", Path(tmp)), patch.object(wp, "MODEL_DIR", Path(tmp)), \
                 patch.object(wp, "load_rating_sources", return_value={}), \
                 patch.object(wp, "load_recent_epa", return_value=pd.DataFrame()), \
                 patch.object(wp, "load_current_elo", return_value=pd.DataFrame()), \
                 patch.object(wp, "attach_team_features", side_effect=lambda d, *args: d):
                feat = pd.DataFrame({"f": [0]})
                a = predict.generate_predictions(data, feat, feat, feat,
                    Regressor(3), Regressor(-4), Classifier(), features)
                b = wp.build_predictions(games, lines, Regressor(3), Regressor(-4),
                    Classifier(), features, 2026)
                for col in ["pred_spread", "pred_total", "pred_win_p", "spread_edge", "totals_edge"]:
                    np.testing.assert_allclose(a[col], b[col])

    def test_weekly_paper_spread_uses_selected_teams_line(self):
        import weekly_pipeline as wp
        # Edges sit inside the unified 4-7 flag range (src/gates.py); the test's
        # intent is line selection per picked team, not the threshold itself.
        d = pd.DataFrame({"home_team": ["A", "A"], "away_team": ["B", "B"],
                          "spread": [-7., -7.], "over_under": [50., 50.],
                          "pred_spread": [12., 1.], "pred_total": [50., 50.],
                          "pred_win_p": [.7, .6], "spread_edge": [5., -6.],
                          "totals_edge": [0., 0.]})
        picks = wp.filter_picks(d)
        self.assertEqual([r["pick"] for r in picks["paper"]], ["A -7.0", "B +7.0"])


if __name__ == "__main__":
    unittest.main()
