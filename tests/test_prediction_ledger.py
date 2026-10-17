"""Tests for scripts/prediction_ledger.py: weather + closing-ML columns."""
import sys
import types
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prediction_ledger as pl


def _pred_row(**kw):
    base = {
        "game_id": 1, "home_team": "A", "away_team": "B",
        "home_conference": "SEC", "away_conference": "SEC",
        "neutral_site": 0, "start_date": "2030-09-05",
        "pred_spread": -3.0, "pred_total": 55.0, "pred_win_p": 0.6,
        "spread": -3.0, "over_under": 55.0, "spread_open": -3.0,
        "home_moneyline": -150, "away_moneyline": 130,
        "spread_edge": 4.5, "totals_edge": -1.0,
        "wind_speed": 12.0, "is_dome": 0,
        "home_ml_ev": 0.05, "away_ml_ev": 0.01,
    }
    base.update(kw)
    return base


class LedgerColumnTests(unittest.TestCase):
    def test_new_columns_in_schema(self):
        for col in ("wind_speed", "is_dome",
                    "closing_home_moneyline", "closing_away_moneyline"):
            self.assertIn(col, pl.LEDGER_COLS)

    def test_record_writes_weather_columns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            orig_ledger, orig_summary = pl.LEDGER, pl.SUMMARY
            pl.LEDGER, pl.SUMMARY = td / "ledger.csv", td / "summary.json"
            try:
                preds = pd.DataFrame([_pred_row()])
                fake_wp = types.SimpleNamespace(
                    fetch_schedule=lambda s, w: pd.DataFrame(
                        [{"game_id": 1, "start_date": "2030-09-05"}]),
                    fetch_lines=lambda g, s, w: pd.DataFrame(),
                    load_models=lambda: (None, None, None, None),
                    build_predictions=lambda *a, **k: preds,
                )
                sys.modules["weekly_pipeline"] = fake_wp
                try:
                    pl.record(2030, 1)
                finally:
                    del sys.modules["weekly_pipeline"]
                df = pd.read_csv(td / "ledger.csv")
                self.assertEqual(df.loc[0, "wind_speed"], 12.0)
                self.assertEqual(df.loc[0, "is_dome"], 0)
                # |4.5| in the 4-7 flag range
                self.assertTrue(bool(df.loc[0, "flag_spread"]))
            finally:
                pl.LEDGER, pl.SUMMARY = orig_ledger, orig_summary

    def test_grade_backfills_and_fills_closing_ml(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            orig_ledger, orig_summary = pl.LEDGER, pl.SUMMARY
            pl.LEDGER, pl.SUMMARY = td / "ledger.csv", td / "summary.json"
            try:
                # old-schema CSV: none of the four new columns
                new_cols = {"wind_speed", "is_dome",
                            "closing_home_moneyline", "closing_away_moneyline"}
                old_cols = [c for c in pl.LEDGER_COLS if c not in new_cols]
                row = {c: 0.0 for c in old_cols}
                row.update({"game_id": 1, "season": 2030, "week": 1,
                            "completed": False, "spread": -3.0,
                            "spread_edge": 4.5, "over_under": 55.0})
                pd.DataFrame([row]).to_csv(td / "ledger.csv", index=False)
                pl._fetch_week_results = lambda s, w, key: {1: (24, 17, True)}
                pl._fetch_week_closing = lambda s, w, key: {
                    1: (-3.5, 54.5, -160, 140)}
                try:
                    pl.grade(2030)
                finally:
                    del pl._fetch_week_results
                    del pl._fetch_week_closing
                out = pd.read_csv(td / "ledger.csv")
                for col in new_cols:
                    self.assertIn(col, out.columns)
                self.assertEqual(out.loc[0, "closing_home_moneyline"], -160)
                self.assertEqual(out.loc[0, "closing_away_moneyline"], 140)
                self.assertEqual(out.loc[0, "closing_spread"], -3.5)
                self.assertTrue(bool(out.loc[0, "completed"]))
            finally:
                pl.LEDGER, pl.SUMMARY = orig_ledger, orig_summary


if __name__ == "__main__":
    unittest.main()
