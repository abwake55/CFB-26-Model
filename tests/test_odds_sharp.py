import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from odds_api import events_to_consensus


def _bk(key, spread, total):
    return {"key": key, "markets": [
        {"key": "spreads", "outcomes": [{"name": "H", "point": spread, "price": -110},
                                        {"name": "A", "point": -spread, "price": -110}]},
        {"key": "totals", "outcomes": [{"name": "Over", "point": total, "price": -110},
                                       {"name": "Under", "point": total, "price": -110}]}]}


class SharpReferenceTest(unittest.TestCase):
    def test_pinnacle_preferred_and_excluded_from_consensus(self):
        ev = [{"id": "e1", "home_team": "H", "away_team": "A", "commence_time": None,
               "bookmakers": [_bk("draftkings", -3.0, 50.0), _bk("fanduel", -3.0, 50.0),
                              _bk("lowvig", -3.5, 49.5), _bk("pinnacle", -10.0, 40.0)]}]
        r = events_to_consensus(ev).iloc[0]
        self.assertEqual(r.sharp_book, "pinnacle")
        self.assertEqual(r.sharp_spread, -10.0)
        self.assertEqual(r.sharp_total, 40.0)
        self.assertEqual(r.spread, -3.0)      # US books only
        self.assertEqual(r.n_books, 3)

    def test_lowvig_fallback(self):
        ev = [{"id": "e1", "home_team": "H", "away_team": "A", "commence_time": None,
               "bookmakers": [_bk("draftkings", -3.0, 50.0), _bk("lowvig", -3.5, 49.5)]}]
        r = events_to_consensus(ev).iloc[0]
        self.assertEqual((r.sharp_book, r.sharp_spread), ("lowvig", -3.5))


if __name__ == "__main__":
    unittest.main()
