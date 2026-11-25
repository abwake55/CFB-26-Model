"""High-total under signal: first-number line, threshold, ledger freeze and CLV."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import high_total_under as htu


class TestAttach(unittest.TestCase):
    def test_uses_snapshot_first_then_cfbd_open_then_current(self):
        df = pd.DataFrame({
            "game_id": [1, 2, 3, 4],
            "season": [2026] * 4,
            "week": [13] * 4,
            "over_under": [59.5, 61.0, 58.0, np.nan],
            "over_under_open": [np.nan, 58.0, 62.0, np.nan],
            "snap_open_total": [61.0, np.nan, np.nan, np.nan],
        })
        out = htu.attach(df)
        self.assertEqual(list(out.high_total_under_line.iloc[:3]), [61.0, 58.0, 62.0])
        self.assertTrue(pd.isna(out.high_total_under_line.iloc[3]))
        self.assertEqual(list(out.flag_high_total_under), [True, False, True, False])

    def test_threshold_is_sixty_inclusive(self):
        df = pd.DataFrame({"over_under": [59.5, 60.0, 60.5],
                           "game_id": [1, 2, 3], "season": [2024] * 3, "week": [1] * 3})
        out = htu.attach(df)
        self.assertEqual(list(out.flag_high_total_under), [False, True, True])

    def test_attach_reads_on_disk_snapshot_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = Path(tmp)
            (snap_dir / "lines_2026_W13.json").write_text(json.dumps({
                "season": 2026, "week": 13,
                "games": {"10": {"first": {"over_under": 64.5},
                                 "current": {"over_under": 59.0}}},
            }))
            df = pd.DataFrame({"game_id": [10], "season": [2026], "week": [13],
                               "over_under": [59.0]})
            with patch.object(htu, "SNAP_DIR", snap_dir):
                out = htu.attach(df)
            self.assertEqual(out.snap_open_total.iloc[0], 64.5)
            self.assertEqual(out.high_total_under_line.iloc[0], 64.5)
            self.assertTrue(bool(out.flag_high_total_under.iloc[0]))


class TestLedgerTotalsBet(unittest.TestCase):
    def test_first_flagged_total_is_kept_and_clv_is_opener_minus_close(self):
        import prediction_ledger as pl
        base = {"game_id": 1, "season": 2026, "week": 14,
                "flag_high_total_under": True, "high_total_under_line": 62.0,
                "over_under": 62.0, "recorded_at": "t1"}
        first = pd.DataFrame([base], columns=pl.LEDGER_COLS)
        first = pl._carry_bet_lines(first, None)
        self.assertEqual(first.high_total_under_bet_line.iloc[0], 62.0)
        later = pd.DataFrame([{**base, "over_under": 59.0, "high_total_under_line": 59.0,
                               "flag_high_total_under": False, "recorded_at": "t2"}],
                             columns=pl.LEDGER_COLS)
        later = pl._carry_bet_lines(later, first)
        self.assertEqual(later.high_total_under_bet_line.iloc[0], 62.0)
        self.assertEqual(later.high_total_under_bet_ts.iloc[0], "t1")

        later["closing_over_under"] = 60.0
        later["closing_sharp_total"] = 59.5
        later = pl._fill_clv(later)
        # under at 62, close 60 -> +2 (higher number is better for the under)
        self.assertEqual(later.high_total_under_clv.iloc[0], 2.0)
        self.assertEqual(later.high_total_under_clv_sharp.iloc[0], 2.5)

    def test_bucket_grades_under_at_the_placed_line(self):
        import prediction_ledger as pl
        g = pd.DataFrame({
            "completed": [True, True, True],
            "high_total_under_bet_line": [62.0, 62.0, 62.0],
            "actual_total": [61.0, 62.0, 63.0],
            "closing_over_under": [60.0, 60.0, 63.0],
            "closing_sharp_total": [np.nan, np.nan, np.nan],
        })
        stats = pl._bucket_stats(g, "under", "high_total_under_bet_line")
        self.assertEqual((stats["wins"], stats["losses"], stats["pushes"]), (1, 1, 1))
        self.assertEqual(stats["avg_clv_pts"], 1.0)  # (2 + 2 + -1) / 3


if __name__ == "__main__":
    unittest.main()
