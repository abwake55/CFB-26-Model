import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prediction_tracker as pt  # noqa: E402


def raw_rows():
    # PT convention: line = predicted HOME margin
    return pd.DataFrame({
        "Home": ["Kansas", "Fresno St."], "Road": ["Fresno St.", "Kansas"],
        "line": [14.0, -3.0], "lineopen": [13.0, -2.5], "linemidweek": [13.5, -3.0],
        "lineespn": [10.0, -1.0], "linesag": [12.0, -2.0], "linemass": [11.0, -4.0],
        "linefoo": [9.0, 0.0], "lineavg": [99.0, 99.0],
        "hscore": [31, 20], "vscore": [7, 24], "week": [1, 9], "neutral": [0, 0]})


class TestFeatures(unittest.TestCase):
    def test_home_spread_convention(self):
        f = pt.to_features(raw_rows())
        self.assertEqual(f.pt_updated.iloc[0], -14.0)   # home favoured -> negative
        self.assertEqual(f.pt_open.iloc[0], -13.0)
        self.assertEqual(f.pt_fpi.iloc[0], -10.0)
        # aggregates exclude market and PT's own average columns
        self.assertEqual(f.pt_sys_n.iloc[0], 4)
        self.assertAlmostEqual(f.pt_sys_median.iloc[0], -10.5)

    def test_flipped_market_sign_is_blanked(self):
        r = raw_rows()
        r.loc[0, "line"] = -14.0            # sign error in PT's market column
        f = pt.to_features(r)
        self.assertTrue(np.isnan(f.pt_updated.iloc[0]))
        self.assertEqual(f.pt_open.iloc[0], -13.0)


class TestMatching(unittest.TestCase):
    def test_completed_match_handles_swapped_orientation(self):
        games = pd.DataFrame({"game_id": [1, 2], "season": [2025, 2025],
                              "home_team": ["Kansas", "Kansas"],
                              "away_team": ["Fresno State", "Fresno State"],
                              "home_points": [31, 24], "away_points": [7, 20]})
        m = pt.match_completed(pt.to_features(raw_rows()), games, 2025).set_index("game_id")
        self.assertEqual(m.loc[1, "pt_updated"], -14.0)
        # row 2 lists Fresno St. at home in PT; CFBD has Kansas home -> negate
        self.assertEqual(m.loc[2, "pt_updated"], -3.0)  # Kansas favoured by 3
        self.assertEqual(m.loc[2, "pt_fpi"], -1.0)

    def test_upcoming_match_by_name(self):
        f = pt.to_features(raw_rows().iloc[[0]].drop(columns=["hscore", "vscore"]))
        sched = pd.DataFrame({"game_id": [7], "home_team": ["Kansas"],
                              "away_team": ["Fresno State"]})
        m = pt.match_upcoming(f, sched, team_map={})
        self.assertEqual(int(m.game_id.iloc[0]), 7)


class TestAttach(unittest.TestCase):
    def test_rule_uses_opener_and_fpi(self):
        out = pd.DataFrame({"game_id": [1, 2], "home_team": ["A", "B"], "away_team": ["C", "D"],
                            "spread": [-3.0, -3.0], "spread_open": [-3.0, np.nan]})
        live = pd.DataFrame({"game_id": [1, 2]})
        for c in pt.PT_COLS:
            live[c] = np.nan
        live["pt_fpi"] = [-11.0, -5.0]
        live["pt_open"] = [np.nan, -4.0]
        res = pt.attach(out, pt_live=live).set_index("game_id")
        rule = pt.load_rule()
        self.assertIsNotNone(rule)
        # edge in home-margin points: opener - fpi spread
        self.assertAlmostEqual(res.loc[1, "pt_edge"], 8.0)
        self.assertAlmostEqual(res.loc[2, "pt_opener"], -4.0)  # PT opener fills
        self.assertEqual(bool(res.loc[1, "flag_pt"]), 8.0 >= rule["thr"])


class TestLedgerBetLines(unittest.TestCase):
    def test_first_flagged_line_is_kept(self):
        import prediction_ledger as pl
        base = {"game_id": 1, "season": 2026, "week": 14, "flag_pt": True, "pt_edge": 8.0}
        first = pd.DataFrame([{**base, "spread": -3.0, "recorded_at": "t1"}],
                             columns=pl.LEDGER_COLS)
        first = pl._carry_bet_lines(first, None)
        self.assertEqual(first.pt_bet_line.iloc[0], -3.0)
        later = pd.DataFrame([{**base, "spread": -6.0, "recorded_at": "t2", "flag_pt": False}],
                             columns=pl.LEDGER_COLS)
        later = pl._carry_bet_lines(later, first)
        self.assertEqual(later.pt_bet_line.iloc[0], -3.0)
        self.assertEqual(later.pt_bet_ts.iloc[0], "t1")


if __name__ == "__main__":
    unittest.main()
