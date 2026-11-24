import sys, unittest
from pathlib import Path
from unittest import mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import opener_gap as og


class OpenerGapTest(unittest.TestCase):
    def test_edge_uses_opener_then_current_line(self):
        out = pd.DataFrame({"game_id": [1, 2], "spread": [-7.0, 3.0], "spread_open": [-3.0, None]})
        ff = pd.DataFrame({"game_id": [1, 2], "is_margin_diff": [0.0, 0.0]})
        with mock.patch.object(og, "predict", return_value=pd.Series([10.0, 4.0], index=[1, 2])), \
             mock.patch.object(og.MODEL_PATH.__class__, "exists", return_value=True):
            r = og.attach(out, ff)
        self.assertEqual(list(r.opener_gap_line), [-3.0, 3.0])       # opener, else current
        self.assertEqual(list(r.opener_gap_edge), [7.0, 7.0])        # margin + home line
        self.assertEqual(list(r.flag_opener_gap), [True, True])

    def test_no_model_is_noop(self):
        out = pd.DataFrame({"game_id": [1], "spread": [-7.0]})
        with mock.patch.object(og.MODEL_PATH.__class__, "exists", return_value=False):
            self.assertNotIn("flag_opener_gap", og.attach(out, out).columns)


if __name__ == "__main__":
    unittest.main()
