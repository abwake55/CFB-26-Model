"""
CFB Betting Model — Backtester
================================
Simulates historical betting performance given a model's predictions
and actual Vegas lines.

Usage:
    from src.backtester import Backtester
    results = Backtester.run(predictions_df, edge_threshold=3.0)
    print(results.summary())
"""

import pandas as pd
import numpy as np


class Backtester:
    """
    Backtests a set of game predictions against historical Vegas lines.

    Input DataFrame columns required:
      - predicted_spread: your model's spread (negative = home favored)
      - vegas_spread: actual Vegas opening spread
      - actual_margin: home_score - away_score (positive = home won)
      - over_under: Vegas total
      - total_points: actual combined score

    Optional:
      - game_id, season, week, home_team, away_team
    """

    @staticmethod
    def calculate_ats_result(actual_margin: float, spread: float) -> str:
        """
        Determine ATS result from the home team's perspective.
        spread is from home team's view (negative = home favored).
        """
        cover_margin = actual_margin + spread  # home team covers if > 0
        if cover_margin > 0:
            return "home_cover"
        elif cover_margin < 0:
            return "away_cover"
        else:
            return "push"

    @staticmethod
    def bet_return(result: str, side: str, vig: float = -110) -> float:
        """
        Calculate P&L on a 1-unit bet.
        vig of -110 means you risk 110 to win 100.
        Returns +1.0 for win, -1.0 for loss, 0 for push.
        """
        win_amount = 100 / abs(vig)  # e.g., 100/110 ≈ 0.909 units

        if result == "push":
            return 0.0
        if (result == "home_cover" and side == "home") or (
            result == "away_cover" and side == "away"
        ):
            return win_amount
        return -1.0

    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        edge_threshold: float = 3.0,
        vig: float = -110,
        bet_side: str = "auto",  # "auto", "home", "away"
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Parameters:
          - df: DataFrame with predictions and actual results
          - edge_threshold: minimum point difference to trigger a bet
          - vig: standard juice (default -110)
          - bet_side: which side to bet ("auto" = bet the side your model favors)

        Returns a DataFrame with bet results and running P&L.
        """
        results = []

        for _, row in df.iterrows():
            pred_spread = row.get("predicted_spread")
            vegas_spread = row.get("vegas_spread")
            actual_margin = row.get("actual_margin")

            if pd.isna(pred_spread) or pd.isna(vegas_spread) or pd.isna(actual_margin):
                continue

            # Edge = model's predicted home margin minus Vegas expected home margin.
            # Both values use the same sign convention: positive = home team advantage.
            # Positive edge  → model likes home MORE than Vegas → bet home
            # Negative edge  → model likes away MORE than Vegas → bet away
            edge = pred_spread - vegas_spread

            if abs(edge) < edge_threshold:
                continue  # No bet

            # Determine which side to bet
            if bet_side == "auto":
                side = "home" if edge > 0 else "away"
            else:
                side = bet_side

            # ATS result — calculate_ats_result expects standard spread notation
            # (negative = home favored), so negate vegas_spread to convert.
            ats_result = cls.calculate_ats_result(actual_margin, -vegas_spread)
            pnl = cls.bet_return(ats_result, side, vig)

            record = {
                "game_id": row.get("game_id"),
                "season": row.get("season"),
                "week": row.get("week"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "predicted_spread": pred_spread,
                "vegas_spread": vegas_spread,
                "edge": edge,
                "bet_side": side,
                "actual_margin": actual_margin,
                "ats_result": ats_result,
                "pnl": pnl,
                "result": "win" if pnl > 0 else ("push" if pnl == 0 else "loss"),
            }

            # Totals (if available)
            if "over_under" in row and "total_points" in row:
                ou = row["over_under"]
                total = row["total_points"]
                if not pd.isna(ou) and not pd.isna(total):
                    record["over_under"] = ou
                    record["total_points"] = total
                    record["ou_result"] = "over" if total > ou else ("push" if total == ou else "under")

            results.append(record)

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("No bets found. Try lowering edge_threshold.")
            return results_df

        # Running P&L
        results_df = results_df.sort_values(["season", "week"]).reset_index(drop=True)
        results_df["cumulative_pnl"] = results_df["pnl"].cumsum()
        results_df["running_roi"] = results_df["cumulative_pnl"] / (results_df.index + 1)

        return results_df

    @classmethod
    def run_totals(
        cls,
        df: pd.DataFrame,
        edge_threshold: float = 3.0,
        vig: float = -110,
    ) -> pd.DataFrame:
        """
        Backtest the totals (over/under) model.

        Required columns:
          - pred_total:   model's predicted combined score
          - over_under:   Vegas total
          - total_points: actual combined score

        Positive edge → model thinks MORE scoring than Vegas → bet OVER
        Negative edge → model thinks LESS scoring than Vegas → bet UNDER
        """
        results = []

        for _, row in df.iterrows():
            pred_total  = row.get("pred_total")
            over_under  = row.get("over_under")
            total_points = row.get("total_points")

            if pd.isna(pred_total) or pd.isna(over_under) or pd.isna(total_points):
                continue

            over_under   = float(over_under)
            total_points = float(total_points)
            edge = pred_total - over_under   # positive = bet over, negative = bet under

            if abs(edge) < edge_threshold:
                continue

            bet_side = "over" if edge > 0 else "under"

            # Determine result
            if total_points > over_under:
                ou_result = "over"
            elif total_points < over_under:
                ou_result = "under"
            else:
                ou_result = "push"

            win_amount = 100 / abs(vig)
            if ou_result == "push":
                pnl = 0.0
            elif ou_result == bet_side:
                pnl = win_amount
            else:
                pnl = -1.0

            results.append({
                "game_id":      row.get("game_id"),
                "season":       row.get("season"),
                "week":         row.get("week"),
                "home_team":    row.get("home_team"),
                "away_team":    row.get("away_team"),
                "pred_total":   pred_total,
                "over_under":   over_under,
                "total_points": total_points,
                "edge":         edge,
                "bet_side":     bet_side,
                "ou_result":    ou_result,
                "pnl":          pnl,
                "result":       "win" if pnl > 0 else ("push" if pnl == 0 else "loss"),
                # Pass through weather columns if present
                "wind_speed":   row.get("wind_speed", np.nan),
                "temp_avg":     row.get("temp_avg", np.nan),
                "is_dome":      row.get("is_dome", np.nan),
            })

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("No totals bets found. Try lowering edge_threshold.")
            return results_df

        results_df = results_df.sort_values(["season", "week"]).reset_index(drop=True)
        results_df["cumulative_pnl"] = results_df["pnl"].cumsum()
        results_df["running_roi"]    = results_df["cumulative_pnl"] / (results_df.index + 1)

        return results_df

    @classmethod
    def run_moneyline(
        cls,
        df: pd.DataFrame,
        ev_threshold: float = 0.03,
        vig: float = -110,
    ) -> pd.DataFrame:
        """
        Backtest moneyline bets using the win probability model.

        Required columns in df:
          - pred_home_win_p:  model's home win probability
          - home_moneyline:   actual American odds for home team
          - away_moneyline:   actual American odds for away team
          - home_win:         1 if home team won, 0 if away team won
          - game_id, season, week, home_team, away_team

        Strategy:
          1. Convert book moneylines to fair probs (remove vig)
          2. Compare model win prob to fair prob → compute EV for each side
          3. Bet whichever side has EV ≥ ev_threshold (home or away)
          4. P&L is variable based on actual moneyline payout
        """
        results = []

        for _, row in df.iterrows():
            home_wp   = row.get("pred_home_win_p")
            home_ml   = row.get("home_moneyline")
            away_ml   = row.get("away_moneyline")
            home_win  = row.get("home_win")

            # Need all four values to evaluate a bet
            if any(pd.isna(v) for v in [home_wp, home_ml, away_ml, home_win]):
                continue

            away_wp = 1 - home_wp

            # ── Implied probs (raw, with vig) ──────────────────────────────
            def ml_to_prob(odds):
                if odds < 0:
                    return abs(odds) / (abs(odds) + 100)
                return 100 / (odds + 100)

            home_implied = ml_to_prob(home_ml)
            away_implied = ml_to_prob(away_ml)
            total_implied = home_implied + away_implied
            if total_implied <= 0:
                continue

            # Fair (no-vig) probs from the book
            fair_home = home_implied / total_implied
            fair_away = away_implied / total_implied

            # ── EV for each side ───────────────────────────────────────────
            def ml_ev(model_prob, american_odds):
                payout = 100 / abs(american_odds) if american_odds < 0 else american_odds / 100
                return model_prob * payout - (1 - model_prob)

            home_ev = ml_ev(home_wp, home_ml)
            away_ev = ml_ev(away_wp, away_ml)

            # ── Pick best side if it clears the threshold ──────────────────
            best_side = None
            best_ev   = max(home_ev, away_ev)
            if best_ev < ev_threshold:
                continue   # No +EV bet

            if home_ev >= away_ev and home_ev >= ev_threshold:
                best_side = "home"
                bet_odds  = home_ml
                model_wp  = home_wp
                fair_prob = fair_home
            elif away_ev > home_ev and away_ev >= ev_threshold:
                best_side = "away"
                bet_odds  = away_ml
                model_wp  = away_wp
                fair_prob = fair_away
            else:
                continue

            # ── P&L calculation ────────────────────────────────────────────
            # Variable payout based on actual moneyline (not flat -110)
            payout = 100 / abs(bet_odds) if bet_odds < 0 else bet_odds / 100
            won    = (best_side == "home" and home_win == 1) or \
                     (best_side == "away" and home_win == 0)
            pnl    = payout if won else -1.0

            results.append({
                "game_id":       row.get("game_id"),
                "season":        row.get("season"),
                "week":          row.get("week"),
                "home_team":     row.get("home_team"),
                "away_team":     row.get("away_team"),
                "bet_side":      best_side,
                "bet_team":      row.get("home_team") if best_side == "home" else row.get("away_team"),
                "book_odds":     bet_odds,
                "model_wp":      model_wp,
                "fair_book_prob":fair_prob,
                "edge_prob":     model_wp - fair_prob,   # model prob minus fair book prob
                "ev":            best_ev,
                "won":           int(won),
                "home_win":      int(home_win),
                "pnl":           pnl,
                "result":        "win" if won else "loss",
            })

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("No moneyline bets found. Try lowering ev_threshold.")
            return results_df

        results_df = results_df.sort_values(["season", "week"]).reset_index(drop=True)
        results_df["cumulative_pnl"] = results_df["pnl"].cumsum()
        results_df["running_roi"]    = results_df["cumulative_pnl"] / (results_df.index + 1)

        return results_df

    @staticmethod
    def summary_moneyline(results_df: pd.DataFrame) -> None:
        """Print a moneyline backtest summary with underdog/favorite breakdown."""
        if len(results_df) == 0:
            print("No moneyline results to summarize.")
            return

        bets   = len(results_df)
        wins   = results_df["won"].sum()
        losses = bets - wins
        total_pnl  = results_df["pnl"].sum()
        roi        = total_pnl / bets * 100
        win_rate   = wins / bets * 100 if bets > 0 else 0
        avg_odds   = results_df["book_odds"].mean()
        avg_ev     = results_df["ev"].mean()

        # Breakeven win rate depends on average odds (not fixed at 52.4%)
        def breakeven_wr(odds):
            if odds < 0:
                return abs(odds) / (abs(odds) + 100) * 100
            return 100 / (odds + 100) * 100
        avg_breakeven = results_df["book_odds"].apply(breakeven_wr).mean()

        print("\n" + "=" * 55)
        print("MONEYLINE BACKTEST SUMMARY")
        print("=" * 55)
        print(f"Total bets:        {bets}")
        print(f"Record:            {wins}W - {losses}L")
        print(f"Win rate:          {win_rate:.1f}%  (avg breakeven: {avg_breakeven:.1f}%)")
        print(f"Total P&L:         {total_pnl:+.2f} units")
        print(f"ROI per bet:       {roi:+.2f}%")
        print(f"Avg model EV:      {avg_ev:+.1%}")
        print(f"Avg book odds:     {avg_odds:+.0f}")

        if roi > 0:
            print(f"\n✅ PROFITABLE — {roi:.1f}% ROI")
        else:
            print(f"\n❌ NOT PROFITABLE — {abs(roi):.1f}% below breakeven")

        # ── Underdog vs favourite breakdown ──────────────────────────────
        dogs = results_df[results_df["book_odds"] > 0]
        favs = results_df[results_df["book_odds"] < 0]

        def side_stats(subset, label):
            if len(subset) == 0:
                return
            w   = subset["won"].sum()
            pnl = subset["pnl"].sum()
            r   = pnl / len(subset) * 100
            wr  = w / len(subset) * 100
            print(f"  {label:12s}  {len(subset):>5} bets  {wr:>5.1f}% win  "
                  f"{r:>+6.1f}% ROI  {pnl:>+7.2f}u")

        print("\nBy bet type:")
        side_stats(dogs,  "Underdogs")
        side_stats(favs,  "Favorites")

        # ── EV bucket breakdown ───────────────────────────────────────────
        ev_bins = [
            ("3–5% EV",  (results_df["ev"] >= 0.03) & (results_df["ev"] < 0.05)),
            ("5–8% EV",  (results_df["ev"] >= 0.05) & (results_df["ev"] < 0.08)),
            ("8%+ EV",    results_df["ev"] >= 0.08),
        ]
        print("\nBy EV bucket:")
        print(f"  {'EV Range':12s}  {'Bets':>5}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
        print("  " + "-" * 45)
        for label, mask in ev_bins:
            sub = results_df[mask]
            if len(sub) == 0:
                continue
            w   = sub["won"].sum()
            wr  = w / len(sub) * 100
            pnl = sub["pnl"].sum()
            r   = pnl / len(sub) * 100
            flag = " ✅" if pnl > 0 else ""
            print(f"  {label:12s}  {len(sub):>5}  {wr:>5.1f}%  {r:>+6.1f}%  {pnl:>+7.2f}u{flag}")

        print("\nBy season:")
        season_summary = (
            results_df.groupby("season")
            .agg(bets=("pnl","count"),
                 wins=("won","sum"),
                 pnl=("pnl","sum"),
                 roi=("pnl", lambda x: x.sum()/len(x)*100))
            .round(2)
        )
        print(season_summary.to_string())
        print("=" * 55)

    @staticmethod
    def summary_totals(results_df: pd.DataFrame) -> None:
        """Print a totals backtest summary, including weather breakdowns."""
        if len(results_df) == 0:
            print("No totals results to summarize.")
            return

        bets   = len(results_df)
        wins   = (results_df["pnl"] > 0).sum()
        losses = (results_df["pnl"] < 0).sum()
        pushes = (results_df["pnl"] == 0).sum()
        total_pnl = results_df["pnl"].sum()
        roi = total_pnl / bets * 100
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        breakeven = 52.38

        print("\n" + "=" * 55)
        print("TOTALS BACKTEST SUMMARY")
        print("=" * 55)
        print(f"Total bets:        {bets}")
        print(f"Record:            {wins}W - {losses}L - {pushes}P")
        print(f"Win rate:          {win_rate:.1f}%  (breakeven: {breakeven:.1f}%)")
        print(f"Total P&L:         {total_pnl:+.2f} units")
        print(f"ROI:               {roi:+.2f}%")

        over_bets  = results_df[results_df["bet_side"] == "over"]
        under_bets = results_df[results_df["bet_side"] == "under"]

        def side_stats(subset, label):
            if len(subset) == 0:
                return
            w = (subset["pnl"] > 0).sum()
            l = (subset["pnl"] < 0).sum()
            wr = w / (w + l) * 100 if (w + l) else 0
            print(f"  {label:8s}  {len(subset):>5} bets  {wr:>5.1f}%  {subset['pnl'].sum():>+7.2f}u")

        print("\nBy side:")
        side_stats(over_bets,  "OVER")
        side_stats(under_bets, "UNDER")

        # ── Weather breakdown ──────────────────────────────────────────────
        if "wind_speed" in results_df.columns and results_df["wind_speed"].notna().any():
            print("\nWeather breakdown (outdoor games only):")
            outdoor = results_df[(results_df["is_dome"] != 1) &
                                  results_df["wind_speed"].notna()]

            wind_bins = [
                ("Calm  (<8 mph)",  outdoor["wind_speed"] <  8),
                ("Mod   (8–15)",   (outdoor["wind_speed"] >= 8) & (outdoor["wind_speed"] < 15)),
                ("Windy (15–20)",  (outdoor["wind_speed"] >= 15) & (outdoor["wind_speed"] < 20)),
                ("High  (>20 mph)", outdoor["wind_speed"] >= 20),
            ]
            print(f"  {'Condition':20s}  {'Bets':>5}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
            print("  " + "-"*52)
            for label, mask in wind_bins:
                sub = outdoor[mask]
                if len(sub) < 10:
                    continue
                w  = (sub["pnl"] > 0).sum()
                l  = (sub["pnl"] < 0).sum()
                wr = w / (w + l) * 100 if (w + l) else 0
                p  = sub["pnl"].sum()
                r  = p / len(sub) * 100
                flag = " ✅" if wr > breakeven else ""
                print(f"  {label:20s}  {len(sub):>5}  {wr:>5.1f}%  {r:>+6.1f}%  {p:>+7.2f}u{flag}")

            # Under bets in high wind specifically
            high_wind_under = outdoor[
                (outdoor["wind_speed"] >= 15) & (outdoor["bet_side"] == "under")
            ]
            if len(high_wind_under) >= 10:
                w  = (high_wind_under["pnl"] > 0).sum()
                l  = (high_wind_under["pnl"] < 0).sum()
                wr = w / (w + l) * 100 if (w + l) else 0
                p  = high_wind_under["pnl"].sum()
                r  = p / len(high_wind_under) * 100
                flag = " ✅" if wr > breakeven else ""
                print(f"\n  HIGH WIND UNDERS specifically:")
                print(f"  {len(high_wind_under)} bets  {wr:.1f}% win  "
                      f"{r:+.1f}% ROI  {p:+.2f}u{flag}")

        print("\nBy season:")
        season_summary = (
            results_df.groupby("season")
            .agg(bets=("pnl","count"), wins=("result", lambda x: (x=="win").sum()),
                 pnl=("pnl","sum"), roi=("pnl", lambda x: x.sum()/len(x)*100))
            .round(2)
        )
        print(season_summary.to_string())
        print("=" * 55)

    @staticmethod
    def summary(results_df: pd.DataFrame) -> pd.DataFrame:
        """Print and return a summary of backtest performance."""
        if len(results_df) == 0:
            print("No results to summarize.")
            return pd.DataFrame()

        bets = len(results_df)
        wins = (results_df["pnl"] > 0).sum()
        losses = (results_df["pnl"] < 0).sum()
        pushes = (results_df["pnl"] == 0).sum()
        total_pnl = results_df["pnl"].sum()
        roi = total_pnl / bets * 100

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        breakeven = 52.38  # at -110 vig

        print("\n" + "=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Total bets:        {bets}")
        print(f"Record:            {wins}W - {losses}L - {pushes}P")
        print(f"Win rate:          {win_rate:.1f}% (breakeven: {breakeven:.1f}%)")
        print(f"Total P&L:         {total_pnl:+.2f} units")
        print(f"ROI:               {roi:+.2f}%")
        print(f"Edge threshold:    {results_df['edge'].abs().min():.1f}+ points")
        print(f"Avg edge (bets):   {results_df['edge'].abs().mean():.2f} points")

        if win_rate > breakeven:
            print(f"\n✅ PROFITABLE — {win_rate - breakeven:.1f}% above breakeven")
        else:
            print(f"\n❌ NOT PROFITABLE — {breakeven - win_rate:.1f}% below breakeven")

        print("\nBy season:")
        season_summary = (
            results_df.groupby("season")
            .agg(
                bets=("pnl", "count"),
                wins=("result", lambda x: (x == "win").sum()),
                pnl=("pnl", "sum"),
                roi=("pnl", lambda x: x.sum() / len(x) * 100),
            )
            .round(2)
        )
        print(season_summary.to_string())
        print("=" * 50)

        return season_summary


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    results_path = Path("outputs/predictions/model_results.csv")
    if not results_path.exists():
        print("No model results found. Run model.py first.")
        import sys; sys.exit(1)

    raw = pd.read_csv(results_path)

    # Rename columns to match what Backtester.run() expects
    df = raw.rename(columns={
        "pred_spread":       "predicted_spread",
        "vegas_home_margin": "vegas_spread",
        "point_diff":        "actual_margin",
    })

    print(f"Loaded {len(df):,} games from test set "
          f"({int(df['season'].min())}–{int(df['season'].max())})\n")

    # ── Run at several edge thresholds to find the sweet spot ─────────────
    print("="*60)
    print("EDGE THRESHOLD SWEEP  (how selective should we be?)")
    print("="*60)
    print(f"{'Threshold':>10}  {'Bets':>6}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
    print("-"*60)

    for threshold in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        r = Backtester.run(df, edge_threshold=threshold)
        if len(r) == 0:
            continue
        wins   = (r["pnl"] > 0).sum()
        losses = (r["pnl"] < 0).sum()
        total  = wins + losses
        win_pct = wins / total * 100 if total > 0 else 0
        pnl    = r["pnl"].sum()
        roi    = pnl / len(r) * 100
        flag   = " ✅" if win_pct > 52.38 else ""
        print(f"{threshold:>9.1f}+  {len(r):>6}  {win_pct:>5.1f}%  {roi:>+6.1f}%  {pnl:>+7.2f}u{flag}")

    # ── Windowed sweep: only bet when edge is between min and max ──────────
    print("\n" + "="*60)
    print("WINDOWED EDGE SWEEP  (avoid large disagreements with Vegas)")
    print("="*60)
    print(f"{'Window':>12}  {'Bets':>6}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
    print("-"*60)
    for lo, hi in [(2,5),(3,6),(3,7),(4,7),(4,8),(5,8),(5,10)]:
        r = Backtester.run(df, edge_threshold=lo)
        r = r[r["edge"].abs() <= hi].copy() if len(r) > 0 else r
        if len(r) < 20:
            continue
        wins = (r["pnl"] > 0).sum()
        total = wins + (r["pnl"] < 0).sum()
        win_pct = wins / total * 100 if total > 0 else 0
        pnl = r["pnl"].sum()
        roi = pnl / len(r) * 100
        flag = " ✅" if win_pct > 52.38 else ""
        print(f"  {lo:.0f}–{hi:.0f} pts    {len(r):>6}  {win_pct:>5.1f}%  {roi:>+6.1f}%  {pnl:>+7.2f}u{flag}")

    # ── Full summary at the 3–7 point window ───────────────────────────────
    print("\n")
    results_all = Backtester.run(df, edge_threshold=3.0)
    results = results_all[results_all["edge"].abs() <= 7.0].copy()
    Backtester.summary(results)

    # ── Save spread bet log ────────────────────────────────────────────────
    out = Path("outputs/bets/backtest_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\n✅ Full bet log saved → {out}")

    # ══════════════════════════════════════════════════════════════════════
    # TOTALS BACKTESTING
    # ══════════════════════════════════════════════════════════════════════

    # Merge weather into the test set (if available)
    weather_path = Path("data/processed/game_weather.csv")
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        want = [c for c in ["game_id", "wind_speed", "temp_avg",
                             "precipitation", "is_dome"] if c in weather.columns]
        df = df.merge(weather[want], on="game_id", how="left")

    print("\n\n" + "="*60)
    print("TOTALS — EDGE THRESHOLD SWEEP")
    print("="*60)
    print(f"{'Threshold':>10}  {'Bets':>6}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
    print("-"*60)

    for threshold in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        r = Backtester.run_totals(df, edge_threshold=threshold)
        if len(r) == 0:
            continue
        wins   = (r["pnl"] > 0).sum()
        losses = (r["pnl"] < 0).sum()
        total  = wins + losses
        win_pct = wins / total * 100 if total > 0 else 0
        pnl    = r["pnl"].sum()
        roi    = pnl / len(r) * 100
        flag   = " ✅" if win_pct > 52.38 else ""
        print(f"{threshold:>9.1f}+  {len(r):>6}  {win_pct:>5.1f}%  {roi:>+6.1f}%  {pnl:>+7.2f}u{flag}")

    print("\n" + "="*60)
    print("TOTALS — WINDOWED EDGE SWEEP")
    print("="*60)
    print(f"{'Window':>12}  {'Bets':>6}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
    print("-"*60)
    for lo, hi in [(2,5),(3,6),(3,7),(4,7),(4,8),(5,8),(5,10)]:
        r = Backtester.run_totals(df, edge_threshold=lo)
        r = r[r["edge"].abs() <= hi].copy() if len(r) > 0 else r
        if len(r) < 20:
            continue
        wins = (r["pnl"] > 0).sum()
        total = wins + (r["pnl"] < 0).sum()
        win_pct = wins / total * 100 if total > 0 else 0
        pnl = r["pnl"].sum()
        roi = pnl / len(r) * 100
        flag = " ✅" if win_pct > 52.38 else ""
        print(f"  {lo:.0f}–{hi:.0f} pts    {len(r):>6}  {win_pct:>5.1f}%  {roi:>+6.1f}%  {pnl:>+7.2f}u{flag}")

    # ── Full totals summary at 3–7 point window (with weather breakdown) ──
    totals_all = Backtester.run_totals(df, edge_threshold=3.0)
    totals = totals_all[totals_all["edge"].abs() <= 7.0].copy()
    Backtester.summary_totals(totals)

    # ── Save totals bet log ────────────────────────────────────────────────
    out_tot = Path("outputs/bets/totals_backtest_results.csv")
    totals.to_csv(out_tot, index=False)
    print(f"\n✅ Totals bet log saved → {out_tot}")

    # ══════════════════════════════════════════════════════════════════════
    # MONEYLINE BACKTESTING
    # ══════════════════════════════════════════════════════════════════════

    # Need pred_home_win_p + historical moneylines in the same DataFrame.
    # model_results.csv now includes home_moneyline/away_moneyline if
    # features.py included them (which it does via load_lines()).
    has_ml = ("home_moneyline" in raw.columns and "away_moneyline" in raw.columns and
              "pred_home_win_p" in raw.columns)

    if has_ml:
        print("\n\n" + "="*60)
        print("MONEYLINE — EV THRESHOLD SWEEP")
        print("="*60)
        print(f"{'EV Min':>10}  {'Bets':>6}  {'Win%':>6}  {'ROI':>7}  {'P&L':>8}")
        print("-"*60)

        for ev_min in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]:
            r = Backtester.run_moneyline(raw, ev_threshold=ev_min)
            if len(r) == 0:
                continue
            wins   = r["won"].sum()
            total  = len(r)
            win_pct = wins / total * 100
            pnl    = r["pnl"].sum()
            roi    = pnl / total * 100
            flag   = " ✅" if pnl > 0 else ""
            print(f"  EV>{ev_min:.0%}    {total:>6}  {win_pct:>5.1f}%  {roi:>+6.1f}%  {pnl:>+7.2f}u{flag}")

        # Full summary at 3% EV threshold
        # Best window: 4–8% EV (from threshold sweep analysis)
        ml_all = Backtester.run_moneyline(raw, ev_threshold=0.04)
        ml_results = ml_all[ml_all["ev"] < 0.08].copy() if len(ml_all) > 0 else ml_all
        Backtester.summary_moneyline(ml_results)

        out_ml = Path("outputs/bets/moneyline_backtest_results.csv")
        ml_results.to_csv(out_ml, index=False)
        print(f"\n✅ Moneyline bet log saved → {out_ml}")
    else:
        print("\n⚠️  Moneyline backtest skipped — no moneyline columns in model_results.csv.")
        print("   Re-run python3 src/model.py to regenerate model_results.csv with moneylines.")
