import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import odds_api  # noqa: E402


class TestOddsCache(unittest.TestCase):
    def test_second_call_uses_cache_and_usage_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            resp = MagicMock()
            resp.json.return_value = [{"id": "e1"}]
            resp.headers = {"x-requests-last": "3", "x-requests-used": "103",
                            "x-requests-remaining": "397"}
            env = {"ODDS_API_CACHE": os.path.join(tmp, "c.json"), "GITHUB_WORKFLOW": "Unit Test"}
            with patch.dict(os.environ, env), \
                    patch.object(odds_api, "USAGE_DIR", Path(tmp) / "usage"), \
                    patch.object(odds_api.requests, "get", return_value=resp) as get:
                self.assertEqual(odds_api.fetch_ncaaf_events("k"), [{"id": "e1"}])
                self.assertFalse(odds_api.LAST_FROM_CACHE)
                self.assertEqual(odds_api.fetch_ncaaf_events("k"), [{"id": "e1"}])
                self.assertTrue(odds_api.LAST_FROM_CACHE)
                self.assertEqual(get.call_count, 1)
                logs = list((Path(tmp) / "usage").glob("*_unit-test.csv"))
                self.assertEqual(len(logs), 1)
                u = pd.read_csv(logs[0])
                self.assertEqual(int(u.credits_last.iloc[0]), 3)
                self.assertEqual(int(u.remaining.iloc[0]), 397)


class TestSnapshot(unittest.TestCase):
    def test_record_carries_sharp_line(self):
        import snapshot_lines as sl
        rec = sl.snapshot_record(pd.Series({"spread": -3.0, "over_under": 50.5,
                                            "sharp_spread": -3.5, "sharp_total": 51.0,
                                            "sharp_book": "lowvig"}))
        self.assertEqual(rec["sharp_spread"], -3.5)
        self.assertEqual(rec["sharp_book"], "lowvig")
        self.assertIsNone(rec["home_ml"])


class TestLedgerCLV(unittest.TestCase):
    def test_sharp_close_prefers_latest_history(self):
        import prediction_ledger as pl
        rec = {"history": [{"ts": "a", "spread": -3, "sharp_spread": -3.5, "sharp_book": "lowvig"},
                           {"ts": "b", "spread": -4}],
               "current": {"spread": -4}, "updated": "b"}
        self.assertEqual(pl._sharp_close(rec), (-3.5, "lowvig", "a"))

    def test_clv_sign_follows_bet_side(self):
        import prediction_ledger as pl
        df = pd.DataFrame({"pt_bet_line": [-3.0, -3.0], "pt_bet_edge": [5.0, -5.0],
                           "closing_spread": [-5.0, -5.0], "closing_sharp_spread": [-4.5, np.nan]})
        for c in pl.LEDGER_COLS:
            if c not in df.columns:
                df[c] = np.nan
        out = pl._fill_clv(df)
        # home bet at -3 that closed -5: +2 pts; away bet at +3 (home -3) closed +5: -2
        self.assertEqual(out.pt_clv.tolist(), [2.0, -2.0])
        self.assertEqual(out.pt_clv_sharp.iloc[0], 1.5)
        self.assertTrue(np.isnan(out.pt_clv_sharp.iloc[1]))


if __name__ == "__main__":
    unittest.main()
