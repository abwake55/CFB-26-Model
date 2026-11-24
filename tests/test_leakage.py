"""Guards against same-season (post-game) information entering features."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import situational  # noqa: E402


class SituationalRatingVintage(unittest.TestCase):
    def test_lookahead_ratings_are_prior_season(self):
        sp = pd.read_csv(ROOT / "data" / "processed" / "master_sp_ratings.csv",
                         usecols=["year", "team", "rating"])
        ratings = situational._load_ratings()
        row = sp.iloc[0]
        # season Y final SP+ may only describe season Y+1 games
        self.assertEqual(ratings[(int(row.year) + 1, row.team)], float(row.rating))
        latest = int(sp["year"].max())
        self.assertFalse(any(k[0] == int(sp["year"].min()) for k in ratings),
                         "earliest season must have no same-season ratings")
        self.assertTrue(any(k[0] == latest + 1 for k in ratings))

    def test_no_closing_line_features(self):
        import model
        for feats in (model.SPREAD_FEATURES, model.TOTALS_FEATURES):
            for banned in ("spread_magnitude", "spread", "over_under",
                           "spread_open_val", "line_movement"):
                self.assertNotIn(banned, feats)


if __name__ == "__main__":
    unittest.main()
