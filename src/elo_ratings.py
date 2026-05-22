"""
CFB Betting Model — Elo Rating System
=======================================
A simple Elo rating system for college football teams.

Elo ratings update after each game based on:
  - The margin of victory
  - The expected probability of winning (based on pre-game ratings)
  - A K-factor that controls how fast ratings move

Usage:
    from src.elo_ratings import EloSystem
    import pandas as pd

    games = pd.read_csv("data/processed/master_games.csv")
    elo = EloSystem()
    ratings_history = elo.run(games)
"""

import pandas as pd
import numpy as np
from typing import Optional


class EloSystem:
    """
    Elo rating system adapted for college football.

    Key parameters:
      - K: How fast ratings update (higher = faster, more volatile)
      - HOME_ADVANTAGE: Points added to home team's effective rating
      - MEAN_REVERSION: How much teams revert to average between seasons (0-1)
      - INITIAL_RATING: Starting rating for all teams
    """

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 55.0,   # in Elo points (~2.5 spread points)
        mean_reversion: float = 0.33,    # revert 1/3 toward mean each offseason
        initial_rating: float = 1500.0,
        scale: float = 400.0,            # standard Elo scale factor
        autocorrelation: float = 0.7,    # margin of victory multiplier factor
    ):
        self.K = k
        self.HOME_ADV = home_advantage
        self.REVERSION = mean_reversion
        self.INITIAL = initial_rating
        self.SCALE = scale
        self.AUTOCORR = autocorrelation

        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def get_rating(self, team: str) -> float:
        """Return a team's current Elo rating (default = INITIAL if new team)."""
        return self.ratings.get(team, self.INITIAL)

    def expected_win_prob(self, rating_a: float, rating_b: float) -> float:
        """Calculate team A's win probability given two Elo ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / self.SCALE))

    def margin_of_victory_multiplier(
        self, margin: float, elo_diff: float
    ) -> float:
        """
        Scale K-factor by margin of victory, diminishing returns.
        Prevents blowouts from overcounting.
        Based on FiveThirtyEight's NFL Elo methodology.
        """
        return (np.log(abs(margin) + 1) * 2.2) / (
            (elo_diff * self.AUTOCORR + 1) * 2.2
        )

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        neutral_site: bool = False,
    ) -> dict:
        """
        Update Elo ratings after a single game.
        Returns a dict with pre-game ratings, predictions, and post-game ratings.
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Apply home field advantage (skip for neutral sites)
        home_adj = home_elo + (0 if neutral_site else self.HOME_ADV)

        # Expected win probability
        home_exp = self.expected_win_prob(home_adj, away_elo)
        away_exp = 1.0 - home_exp

        # Actual result (1 = win, 0.5 = tie, 0 = loss)
        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif home_score < away_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        # Margin of victory adjustment
        margin = abs(home_score - away_score)
        elo_diff = abs(home_adj - away_elo)
        mov_mult = self.margin_of_victory_multiplier(margin, elo_diff)

        # Rating updates
        home_new = home_elo + self.K * mov_mult * (home_result - home_exp)
        away_new = away_elo + self.K * mov_mult * (away_result - away_exp)

        self.ratings[home_team] = home_new
        self.ratings[away_team] = away_new

        # Predicted spread: Elo diff / 25 ≈ points (rough conversion)
        predicted_spread = (home_adj - away_elo) / 25.0

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "home_win_prob": home_exp,
            "predicted_spread": predicted_spread,
            "home_score": home_score,
            "away_score": away_score,
            "actual_margin": home_score - away_score,
            "home_elo_post": home_new,
            "away_elo_post": away_new,
        }

    def apply_offseason_reversion(self):
        """
        Revert all ratings partially toward the mean between seasons.
        Accounts for roster turnover, coaching changes, etc.
        """
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.REVERSION)
                + self.INITIAL * self.REVERSION
            )

    def run(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a full games DataFrame and return a history of Elo updates.

        Required columns in games_df:
          - season, week, home_team, away_team, home_points, away_points
          - neutral_site (optional, defaults to False)

        Returns a DataFrame with one row per game including pre/post Elo ratings.
        """
        required = ["season", "week", "home_team", "away_team", "home_points", "away_points"]
        for col in required:
            if col not in games_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by season and week
        games = games_df.sort_values(["season", "week"]).copy()
        games = games.dropna(subset=["home_points", "away_points"])

        records = []
        current_season = None

        for _, game in games.iterrows():
            season = game["season"]

            # Apply offseason reversion when season changes
            if current_season is not None and season != current_season:
                self.apply_offseason_reversion()
            current_season = season

            neutral = bool(game.get("neutral_site", False))

            result = self.update(
                home_team=game["home_team"],
                away_team=game["away_team"],
                home_score=game["home_points"],
                away_score=game["away_points"],
                neutral_site=neutral,
            )
            result["season"] = season
            result["week"] = game["week"]

            if "id" in game:
                result["game_id"] = game["id"]

            records.append(result)

        return pd.DataFrame(records)

    def current_ratings_df(self) -> pd.DataFrame:
        """Return a sorted DataFrame of all current team ratings."""
        df = pd.DataFrame(
            [{"team": t, "elo": r} for t, r in self.ratings.items()]
        ).sort_values("elo", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["spread_vs_avg"] = (df["elo"] - self.INITIAL) / 25.0
        return df

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
    ) -> dict:
        """
        Predict a future game based on current ratings.
        Returns: win probability, predicted spread, moneyline.
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        home_adj = home_elo + (0 if neutral_site else self.HOME_ADV)
        home_win_prob = self.expected_win_prob(home_adj, away_elo)
        predicted_spread = -(home_adj - away_elo) / 25.0  # negative = home favored

        # Convert probability to American moneyline
        if home_win_prob >= 0.5:
            home_ml = -round((home_win_prob / (1 - home_win_prob)) * 100)
            away_ml = round(((1 - home_win_prob) / home_win_prob) * 100)
        else:
            home_ml = round((home_win_prob / (1 - home_win_prob)) * 100)
            away_ml = -round(((1 - home_win_prob) / home_win_prob) * 100)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_win_prob": round(home_win_prob, 3),
            "away_win_prob": round(1 - home_win_prob, 3),
            "predicted_spread": round(predicted_spread, 1),
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "neutral_site": neutral_site,
        }


# ─── MAIN (demo) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    games_path = Path("data/processed/master_games.csv")
    sp_path    = Path("data/processed/master_sp_ratings.csv")

    if not games_path.exists():
        print("No game data found. Run data_collection.py first.")
        sys.exit(1)

    print("Loading game data...")
    games = pd.read_csv(games_path)
    print(f"  Total games loaded: {len(games)}")

    # ── Filter to FBS-only games ──────────────────────────────────────────────
    # Use SP+ team list as the authoritative source of FBS programs.
    # We only run Elo on games where BOTH teams are FBS — this prevents D2/D3
    # schools from inflating their ratings by dominating lower-division play.
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        fbs_teams = set(sp["team"].unique())
        before = len(games)
        games = games[
            games["home_team"].isin(fbs_teams) &
            games["away_team"].isin(fbs_teams)
        ].copy()
        print(f"  After FBS filter:   {len(games)} games "
              f"({before - len(games)} non-FBS games removed)")
    else:
        print("  ⚠️  SP+ file not found — running on all games (may include D2/D3)")

    print("\nRunning Elo ratings...")
    elo = EloSystem()
    history = elo.run(games)

    print("\nTop 25 FBS teams by current Elo rating:")
    print(elo.current_ratings_df().head(25).to_string(index=False))

    print("\n\nExample predictions:")
    for home, away in [("Alabama", "Georgia"), ("Ohio State", "Oregon"), ("Texas", "Michigan")]:
        pred = elo.predict_game(home, away)
        sign = "+" if pred["predicted_spread"] > 0 else ""
        print(f"  {home:15s} vs {away:15s}  |  spread: {sign}{pred['predicted_spread']:.1f}  "
              f"win%: {pred['home_win_prob']:.1%}  "
              f"ML: {home} {pred['home_moneyline']:+d}")

    # Save ratings
    out_path = Path("outputs/predictions/current_elo_ratings.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df = elo.current_ratings_df()
    ratings_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(ratings_df)} team ratings → {out_path}")
