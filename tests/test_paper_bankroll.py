"""Paper bankroll: unit size, freeze-at-kickoff, pricing fallback, grading."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dk_predictions as dkp
import paper_bankroll as pb


class TestUnitsAndStakes(unittest.TestCase):
    def test_unit_is_flat_twenty(self):
        self.assertEqual(pb.UNIT_USD, 20.0)
        self.assertEqual(pb.STARTING_BANKROLL, 1000.0)
        self.assertEqual(pb.money_stake(0.25), 5.0)
        self.assertEqual(pb.money_stake(0.5), 10.0)
        self.assertEqual(pb.money_stake(1), 20.0)

    def test_pt_half_unit_only_when_edge_at_least_ten(self):
        self.assertEqual(pb.pt_units(7.0), 0.25)
        self.assertEqual(pb.pt_units(9.77), 0.25)
        self.assertEqual(pb.pt_units(9.99), 0.25)
        self.assertEqual(pb.pt_units(10.0), 0.50)
        self.assertEqual(pb.pt_units(11.37), 0.50)

    def test_bowl_signals_are_paper(self):
        u, reason = pb.stake_for("pt_fpi", 11.0, is_bowl=True)
        self.assertEqual(u, 0.0)
        self.assertIn("bowls", reason)
        u, _ = pb.stake_for("high_total_under", None, is_bowl=True)
        self.assertEqual(u, 0.0)
        u, _ = pb.stake_for("opener_gap", 6.0, is_bowl=True)
        self.assertEqual(u, 0.0)
        # CORE unders stay 1u even in bowls (betting rule did not demote them).
        u, _ = pb.stake_for("core_under", None, is_bowl=True)
        self.assertEqual(u, 1.0)
        u, _ = pb.stake_for("spread_model", 4.0, is_bowl=False)
        self.assertEqual(u, 0.0)


class TestPricing(unittest.TestCase):
    def test_american_profit_at_minus_110(self):
        self.assertEqual(pb.american_profit(5, -110, "win"), 4.55)
        self.assertEqual(pb.american_profit(5, -110, "loss"), -5.0)
        self.assertEqual(pb.american_profit(5, -110, "push"), 0.0)
        self.assertEqual(pb.american_profit(10, 150, "win"), 15.0)

    def test_fee_inclusive_american_from_contract(self):
        # $0.50 + $0.02 fee -> cost 0.52, decimal 1/0.52 ≈ 1.923 -> about -108
        amer = dkp.american_from_contract(0.50)
        self.assertEqual(amer, -108)
        self.assertEqual(dkp.dkex_fee_per_contract(0.50), 0.02)

    def test_fallback_when_lookup_returns_none(self):
        price = pb.price_bet("Miami", "Boston College", "spread", -25.5, "home",
                             lookup=lambda *a, **k: None)
        self.assertTrue(price["fallback"])
        self.assertEqual(price["american"], -110)
        self.assertEqual(price["price_source"], "sportsbook_fallback")

    def test_uses_public_dk_price_when_provided(self):
        hit = {
            "source": "draftkings_predictions",
            "url": "https://predictions.draftkings.com/en/football",
            "fetched_at": "2026-11-24T18:00:00+00:00",
            "contract_price": 0.50,
            "fee_per_contract": 0.02,
            "american": -108,
            "fallback": False,
        }
        price = pb.price_bet("Miami", "Boston College", "spread", -25.5, "home",
                             lookup=lambda *a, **k: hit)
        self.assertFalse(price["fallback"])
        self.assertEqual(price["american"], -108)
        self.assertEqual(price["contract_price"], 0.50)


class TestWorstAcceptable(unittest.TestCase):
    def test_home_favorite_just_above_threshold(self):
        # Miami -25.5, edge 7.31, thr 7 -> cannot take -26
        w = pb.worst_acceptable_spread(-25.5, 7.31, 7.0)
        self.assertEqual(w, -25.5)

    def test_home_favorite_with_cushion(self):
        # North Texas -3.5, edge 11.37 -> can take down to -7.5
        w = pb.worst_acceptable_spread(-3.5, 11.37, 7.0)
        self.assertEqual(w, -7.5)

    def test_under_does_not_go_below_logged_line(self):
        self.assertEqual(pb.worst_acceptable_total(67.5), 67.5)


class TestGradeAndClv(unittest.TestCase):
    def test_spread_cover_and_push(self):
        self.assertEqual(pb.grade_spread(35, 7, "home", -25.5), "win")   # margin 28
        self.assertEqual(pb.grade_spread(31, 7, "home", -25.5), "loss")  # margin 24
        self.assertEqual(pb.grade_spread(21, 14, "home", -7), "push")
        self.assertEqual(pb.grade_spread(21, 14, "away", -7), "push")
        self.assertEqual(pb.grade_spread(28, 14, "away", -7), "loss")
        self.assertEqual(pb.grade_spread(20, 14, "away", -7), "win")

    def test_under(self):
        self.assertEqual(pb.grade_total(24, 20, "under", 61), "win")
        self.assertEqual(pb.grade_total(40, 30, "under", 61), "loss")
        self.assertEqual(pb.grade_total(30, 31, "under", 61), "push")

    def test_clv_signs(self):
        # Bet Miami -25.5, close -26.5 -> +1 CLV (got a better number)
        self.assertEqual(pb.spread_clv(-25.5, -26.5, "home"), 1.0)
        # Under 67.5, close 65.5 -> +2 CLV
        self.assertEqual(pb.total_clv(67.5, 65.5, "under"), 2.0)


class TestFreezeAndSeed(unittest.TestCase):
    def test_refuses_to_log_after_kickoff(self):
        df = pb.load_bankroll(Path("/tmp/does-not-exist.csv"))
        past = "2020-01-01T00:00:00+00:00"
        bet = pb.empty_row(
            bet_id="x", season=2026, week=1, game_id=1, kickoff=past,
            signal="pt_fpi", side="home", line=-3, market="spread",
            stake_usd=5, american=-110, home_team="A", away_team="B",
        )
        with self.assertRaises(ValueError):
            pb.add_bet(df, bet, at=datetime(2026, 11, 24, tzinfo=timezone.utc))

    def test_freeze_marks_started_games_and_grade_does_not_change_line(self):
        at_log = datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc)
        at_after = datetime(2026, 11, 28, 18, 0, tzinfo=timezone.utc)
        df = pd.DataFrame([pb.empty_row(
            bet_id="2026-W13-01", season=2026, week=13, game_id=401858311,
            date="2026-11-28", kickoff="2026-11-28T17:00:00+00:00",
            home_team="Miami", away_team="Boston College",
            signal="pt_fpi", side="home", line=-25.5, market="spread",
            american=-110, fallback=True, price_source="sportsbook_fallback",
            stake_units=0.25, stake_usd=5.0, stake_reason="pt_fpi edge 7.31 < 10: 0.25u",
            line_logged=-25.5, logged_at=at_log.isoformat(),
            frozen_at_kickoff=False, is_bowl=False, status="open",
        )], columns=pb.CSV_COLS)
        frozen = pb.freeze_kicked_off(df, at=at_after)
        self.assertTrue(bool(frozen.iloc[0].frozen_at_kickoff))
        ledger = pd.DataFrame([{
            "game_id": 401858311, "completed": True, "home_score": 42, "away_score": 10,
            "closing_spread": -26.5, "closing_sharp_spread": -26.0,
            "closing_over_under": 50, "closing_sharp_total": 50,
        }])
        graded = pb.grade_bankroll(frozen, ledger, at=at_after)
        row = graded.iloc[0]
        self.assertEqual(row.line, -25.5)
        self.assertEqual(row.stake_usd, 5.0)
        self.assertEqual(row.american, -110)
        self.assertEqual(row.result, "win")  # 32 point win covers -25.5
        self.assertEqual(row.profit_usd, 4.55)
        self.assertEqual(row.clv_pts, 1.0)
        self.assertEqual(row.running_balance, 1004.55)
        # Mutating the line after kickoff is not something grade does
        self.assertEqual(row.line_logged, -25.5)

    def test_seed_is_idempotent_and_uses_group_lines(self):
        at = datetime(2026, 11, 24, 18, 30, tzinfo=timezone.utc)
        df = pd.DataFrame(columns=pb.CSV_COLS)
        df = pb.seed_week13(df, at=at, lookup=lambda *a, **k: None)
        self.assertEqual(len(df), 9)
        # All fallback, all before kickoff, week 13 lines match the group text.
        self.assertTrue(df.fallback.astype(str).str.lower().isin(["true", "1"]).all())
        miami = df[df.home_team == "Miami"].iloc[0]
        self.assertEqual(miami.line, -25.5)
        self.assertEqual(miami.stake_usd, 5.0)
        self.assertEqual(miami.signal, "pt_fpi")
        ntx = df[df.home_team == "North Texas"].iloc[0]
        self.assertEqual(ntx.line, -3.5)
        self.assertEqual(ntx.stake_usd, 10.0)
        kans = df[(df.home_team == "Oklahoma State") & (df.signal == "high_total_under")].iloc[0]
        self.assertEqual(kans.line, 67.5)
        self.assertEqual(kans.stake_usd, 5.0)
        again = pb.seed_week13(df, at=at, lookup=lambda *a, **k: None)
        self.assertEqual(len(again), 9)
        # Total open risk: 4 FPI (5+5+10+5) + 5 unders * 5 = $50
        self.assertEqual(df.stake_usd.sum(), 50.0)

    def test_seed_aborts_if_a_game_already_started(self):
        at = datetime(2026, 11, 27, 21, 0, tzinfo=timezone.utc)  # after Temple/Memphis 20:30Z
        with self.assertRaises(ValueError):
            pb.seed_week13(pd.DataFrame(columns=pb.CSV_COLS), at=at,
                           lookup=lambda *a, **k: None)

    def test_sync_does_not_edit_frozen_rows(self):
        at_log = datetime(2026, 11, 24, tzinfo=timezone.utc)
        at_after = datetime(2026, 11, 28, 18, 0, tzinfo=timezone.utc)
        df = pb.seed_week13(pd.DataFrame(columns=pb.CSV_COLS), at=at_log,
                            lookup=lambda *a, **k: None)
        df = pb.freeze_kicked_off(df, at=at_after)
        before = df[df.game_id == 401858311].iloc[0].copy()
        ledger = pd.DataFrame([{
            "game_id": 401858311, "season": 2026, "week": 13,
            "home_team": "Miami", "away_team": "Boston College",
            "start_date": "2026-11-28T17:00:00+00:00",
            "flag_pt": True, "pt_bet_line": -27.0, "pt_edge": 9.0,
            "spread": -27.0, "completed": False,
        }])
        out = pb.sync_from_ledger(df, ledger, at=at_after, lookup=lambda *a, **k: None)
        after = out[out.game_id == 401858311].iloc[0]
        self.assertEqual(after.line, before.line)
        self.assertEqual(after.line, -25.5)

    def test_sync_skips_kicked_off_new_flags(self):
        at = datetime(2026, 11, 28, 18, 0, tzinfo=timezone.utc)
        ledger = pd.DataFrame([{
            "game_id": 1, "season": 2026, "week": 13,
            "home_team": "A", "away_team": "B",
            "start_date": "2026-11-28T17:00:00+00:00",
            "flag_core": True, "over_under": 55.0, "completed": False,
        }])
        df = pb.sync_from_ledger(pd.DataFrame(columns=pb.CSV_COLS), ledger, at=at,
                                 lookup=lambda *a, **k: None)
        self.assertEqual(len(df), 0)

    def test_no_stacked_stakes_on_same_market(self):
        at = datetime(2026, 11, 24, tzinfo=timezone.utc)
        ledger = pd.DataFrame([{
            "game_id": 99, "season": 2026, "week": 14,
            "home_team": "HomeU", "away_team": "AwayU",
            "start_date": "2026-12-05T17:00:00+00:00",
            "flag_pt": True, "pt_bet_line": -7.0, "pt_edge": 8.0,
            "flag_opener_gap": True, "opener_gap_bet_line": -7.0, "opener_gap_edge": 6.0,
            "spread": -7.0, "completed": False,
        }])
        df = pb.sync_from_ledger(pd.DataFrame(columns=pb.CSV_COLS), ledger, at=at,
                                 lookup=lambda *a, **k: None)
        self.assertEqual(len(df), 2)
        pt = df[df.signal == "pt_fpi"].iloc[0]
        og = df[df.signal == "opener_gap"].iloc[0]
        self.assertEqual(pt.stake_usd, 5.0)
        self.assertEqual(og.stake_usd, 0.0)


class TestSlipAndSummary(unittest.TestCase):
    def test_summary_paths_and_open_week(self):
        at = datetime(2026, 11, 24, 18, 30, tzinfo=timezone.utc)
        df = pb.seed_week13(pd.DataFrame(columns=pb.CSV_COLS), at=at,
                            lookup=lambda *a, **k: None)
        summary = pb.build_summary(df, at=at)
        self.assertEqual(summary["current_balance"], 1000.0)
        self.assertEqual(summary["record"], {"wins": 0, "losses": 0, "pushes": 0})
        self.assertEqual(summary["open_risk_usd"], 50.0)
        self.assertEqual(summary["paths"]["summary_json"],
                         "outputs/bankroll/paper_bankroll_2026_summary.json")
        self.assertEqual(summary["paths"]["bankroll_csv"],
                         "outputs/bankroll/paper_bankroll_2026.csv")
        self.assertEqual(summary["unit_usd"], 20.0)
        text = pb.render_slip(df, 13)
        self.assertNotIn("🏈", text)
        self.assertIn("Miami -25.5 vs Boston College", text)
        self.assertIn("Ohio State -10 vs Michigan", text)
        self.assertIn("North Texas -3.5 vs UAB", text)
        self.assertIn("Hawai'i -13.5 vs Sacramento State", text)
        self.assertIn("Boise State/Colorado State under 75.5", text)
        self.assertIn("West Virginia/Utah under 68.5", text)
        self.assertIn("Kansas/Oklahoma State under 67.5", text)
        self.assertIn("Temple/Memphis under 64.5", text)
        self.assertIn("SMU/Stanford under 61", text)
        self.assertIn("$5 at -110", text)
        self.assertIn("$10 at -110", text)
        self.assertIn("sportsbook snapshot at -110", text)

    def test_write_slip_to_temp(self):
        at = datetime(2026, 11, 24, tzinfo=timezone.utc)
        df = pb.seed_week13(pd.DataFrame(columns=pb.CSV_COLS), at=at,
                            lookup=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slip.txt"
            pb.write_slip(df, 13, path=path)
            self.assertTrue(path.exists())
            self.assertIn("ready-to-place slip", path.read_text())


class TestDkMatcher(unittest.TestCase):
    def test_finds_matching_contract_only(self):
        markets = [
            {"event": "Boston College at Miami", "market": "spread", "line": -25.5,
             "side": "home", "contract_price": 0.52, "american": -110},
        ]
        hit = dkp.find_cfb_contract(markets, "Miami", "Boston College", "spread", -25.5, "home")
        self.assertIsNotNone(hit)
        miss = dkp.find_cfb_contract(markets, "Miami", "Boston College", "spread", -10, "home")
        self.assertIsNone(miss)
        miss2 = dkp.find_cfb_contract(markets, "Ohio State", "Michigan", "spread", -10, "home")
        self.assertIsNone(miss2)


if __name__ == "__main__":
    unittest.main()
