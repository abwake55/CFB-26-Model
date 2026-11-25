import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calibration_report as cr  # noqa: E402


class CalibrationReportTests(unittest.TestCase):
    def test_fav_view_is_symmetric(self):
        fp, fy = cr.fav_view([0.7, 0.3], [1, 0])
        self.assertTrue(np.allclose(fp, [0.7, 0.7]))
        self.assertTrue(np.allclose(fy, [1, 1]))

    def test_perfect_probabilities_have_zero_ece(self):
        p = np.array([0.75] * 4 + [0.25] * 4)
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(cr.ece(p, y), 0.0)

    def test_no_vig_sums_to_one_and_removes_margin(self):
        p = cr.no_vig(np.array([-150.0, 120.0]), np.array([130.0, -140.0]))
        self.assertTrue(0.5 < p[0] < 0.65 and 0.35 < p[1] < 0.5)
        self.assertAlmostEqual(float(cr.no_vig(-110.0, -110.0)), 0.5)

    def test_bin_table_counts_cover_all_games(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.99, 500)
        y = (rng.random(500) < p).astype(int)
        t = cr.bin_table(p, y, "t")
        self.assertEqual(sum(r["n"] for r in t), 500)

    def test_brier_and_logloss_known_values(self):
        self.assertAlmostEqual(cr.brier(np.array([0.5]), np.array([1])), 0.25)
        self.assertAlmostEqual(cr.logloss(np.array([0.5]), np.array([1])), np.log(2))


if __name__ == "__main__":
    unittest.main()
