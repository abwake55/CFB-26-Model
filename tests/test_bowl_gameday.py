"""Bowl module, game-day update and edge ranges (offline)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bowl_context  # noqa: E402
import edge_ranges  # noqa: E402
import history  # noqa: E402
import qb_watch  # noqa: E402


def _passing():
    rows = []
    # Team A: QB 1 starts weeks 1-3, QB 2 starts week 4 (QB 1 absent)
    for w, (pid, name) in enumerate([(1, "Ann One")] * 3 + [(2, "Bob Two")], start=1):
        rows.append({"game_id": 100 + w, "season": 2030, "week": w, "season_type": "regular",
                     "team": "A", "player_id": pid, "player": name, "attempts": 30, "yards": 200})
    rows.append({"game_id": 101, "season": 2030, "week": 1, "season_type": "regular",
                 "team": "A", "player_id": 0, "player": " Team", "attempts": 99, "yards": 0})
    for w in range(1, 5):
        rows.append({"game_id": 100 + w, "season": 2030, "week": w, "season_type": "regular",
                     "team": "B", "player_id": 9, "player": "Cal Nine", "attempts": 25, "yards": 180})
    return pd.DataFrame(rows)


class TestEdgeRanges(unittest.TestCase):
    def test_params_and_attach(self):
        p = edge_ranges.load()
        self.assertIsNotNone(p)
        for sig in ("spread", "opener_gap", "pt"):
            self.assertIn(sig, p["signals"])
        out = edge_ranges.attach(pd.DataFrame({"pt_edge": [8.0, -8.0, np.nan], "pred_spread": [3, -3, 0]}))
        self.assertAlmostEqual(out.pt_edge_exp[0], out.pt_edge_exp[1])   # symmetric in side
        self.assertTrue((out.pt_edge_lo80[:2] < out.pt_edge_exp[:2]).all())
        self.assertTrue((out.pt_edge_hi80[:2] > out.pt_edge_exp[:2]).all())
        self.assertTrue(np.isnan(out.pt_edge_exp[2]))
        self.assertTrue((out.pred_margin_lo80 < out.pred_spread).all())


class TestQbWatch(unittest.TestCase):
    def test_status_and_team_rows(self):
        st = qb_watch.team_status(_passing(), ["A", "B"], 2030).set_index("team")
        self.assertEqual(st.loc["A", "primary_qb"], "Ann One")          # 'Team' row ignored
        self.assertEqual(st.loc["A", "likely_starter"], "Bob Two")
        self.assertEqual(st.loc["A", "qb_change_likely"], 1.0)
        self.assertEqual(st.loc["A", "qb_out_last"], 1.0)
        self.assertEqual(st.loc["B", "qb_change_likely"], 0.0)

    def test_portal_marks_primary_out(self):
        po = pd.DataFrame({"season": [2030], "player": ["Cal Nine"], "position": ["QB"], "origin": ["B"],
                           "transfer_date": ["2030-12-01T00:00:00Z"]})
        st = qb_watch.team_status(_passing(), ["B"], 2030, portal=po,
                                  now=pd.Timestamp("2030-12-10", tz="UTC")).set_index("team")
        self.assertTrue(st.loc["B", "primary_in_portal"])


class TestBowlContext(unittest.TestCase):
    def test_benching_is_not_a_change(self):
        p = _passing()
        # bowl: Team B's primary throws 3 passes, backup throws 20 (benched) -> no change
        bowl = pd.DataFrame([
            {"game_id": 900, "season": 2030, "week": 1, "season_type": "postseason", "team": "B",
             "player_id": 9, "player": "Cal Nine", "attempts": 3, "yards": 10},
            {"game_id": 900, "season": 2030, "week": 1, "season_type": "postseason", "team": "B",
             "player_id": 8, "player": "Dee Eight", "attempts": 20, "yards": 150},
            {"game_id": 900, "season": 2030, "week": 1, "season_type": "postseason", "team": "A",
             "player_id": 2, "player": "Bob Two", "attempts": 30, "yards": 150}])
        tg = pd.DataFrame({"game_id": [900, 900], "season": [2030, 2030], "team": ["A", "B"]})
        q = bowl_context.qb_features(tg, pd.concat([p, bowl])).set_index("team")
        self.assertEqual(q.loc["B", "qb_change"], 0.0)   # benched, not out
        self.assertEqual(q.loc["A", "qb_change"], 1.0)   # primary absent

    def test_adjust_model_is_paper(self):
        adj = json.loads((ROOT / "models/bowl_adjust.json").read_text())
        self.assertEqual(adj["status"], "paper")
        self.assertFalse(any("draft" in k for k in adj["beta"]))  # post-hoc proxy never live


class TestHistory(unittest.TestCase):
    def test_append_keeps_rows_and_grows_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.csv"
            history.append(pd.DataFrame({"a": [1]}), p)
            history.append(pd.DataFrame({"a": [2]}), p)
            history.append(pd.DataFrame({"a": [3], "b": [4]}), p)
            h = pd.read_csv(p)
            self.assertEqual(h.a.tolist(), [1, 2, 3])
            self.assertEqual(list(h.columns), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
