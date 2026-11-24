import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import margin_pricing as mp  # noqa: E402


class TestKeyNumberPricing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dist, cls.params = mp.load()

    def test_pmf_normalised_and_symmetric(self):
        p = self.dist.pmf(0.0)
        self.assertAlmostEqual(p.sum(), 1.0, places=9)
        i = lambda k: k - mp.K.min()  # noqa: E731
        self.assertAlmostEqual(p[i(3)], p[i(-3)], places=9)

    def test_key_numbers_carry_extra_mass(self):
        p = self.dist.pmf(3.0)
        i = lambda k: k - mp.K.min()  # noqa: E731
        self.assertGreater(p[i(3)], 1.5 * p[i(4)])
        self.assertGreater(p[i(7)], 1.5 * p[i(6)])

    def test_push_only_on_integer_lines(self):
        _, push_half, _ = self.dist.probs(3.0, -3.5)
        _, push_int, _ = self.dist.probs(3.0, -3.0)
        self.assertEqual(push_half, 0.0)
        self.assertGreater(push_int, 0.04)

    def test_probs_sum_to_one_and_ev(self):
        ph, pp, pa = self.dist.probs(-6.0, 7.0)
        self.assertAlmostEqual(ph + pp + pa, 1.0, places=9)
        self.assertAlmostEqual(mp.ev(0.5238, 0.4762), 0.0, places=3)

    def test_half_point_through_three_is_worth_more_than_through_five(self):
        a = mp.price_bet(self.dist, 3.0, -3.0, bet_home=False)   # +3 -> +3.5
        b = mp.price_bet(self.dist, 5.0, -5.0, bet_home=False)   # +5 -> +5.5
        self.assertGreater(a["half_point_cover_gain"], b["half_point_cover_gain"])

    def test_attach_prices_flagged_rows_only(self):
        out = pd.DataFrame({"game_id": [1, 2], "spread": [-3.0, -7.0], "spread_open": [-3.0, -7.0],
                            "flag_pt": [True, False], "pt_fpi": [-11.0, -7.0], "pt_edge": [8.0, 0.0],
                            "flag_opener_gap": [False, False], "pred_indep_margin": [3.0, 7.0],
                            "opener_gap_edge": [0.0, 0.0]})
        res = mp.attach(out)
        self.assertTrue(0 < res.pt_p_cover.iloc[0] < 1)
        self.assertTrue(np.isnan(res.pt_p_cover.iloc[1]))


if __name__ == "__main__":
    unittest.main()
