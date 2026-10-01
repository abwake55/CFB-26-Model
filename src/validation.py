"""Chronological validation and input integrity shared by training scripts."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def clean_games(df):
    """Remove exact join duplicates; reject conflicting rows for a game."""
    df = df.drop_duplicates().copy()
    if df["game_id"].isna().any() or df["game_id"].duplicated().any():
        raise ValueError("Expected one non-null row per game_id; fix conflicting source joins")
    required = ["season", "point_diff", "total_points", "spread", "over_under"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    good = np.isfinite(df[required]).all(axis=1)
    df = df.loc[good].copy()
    df["vegas_home_margin"] = -df["spread"]
    # Recompute targets from scores/lines, including pushes and missing data.
    df["home_win"] = (df["point_diff"] > 0).astype(int)
    df["covered_spread"] = np.sign(df["point_diff"] + df["spread"])
    df["covered_spread"] = df["covered_spread"].map({-1: 0., 0: .5, 1: 1.})
    df["went_over"] = np.sign(df["total_points"] - df["over_under"])
    df["went_over"] = df["went_over"].map({-1: 0., 0: .5, 1: 1.})
    return df


def split_seasons(df, test_season, first_season=2017):
    """Last available prior season is validation, earlier seasons train."""
    prior = sorted(s for s in df["season"].unique()
                   if first_season <= s < test_season and s != 2020)
    if len(prior) < 2:
        raise ValueError("Need at least two prior non-COVID seasons")
    train = df[df["season"].isin(prior[:-1])].copy()
    val = df[df["season"] == prior[-1]].copy()
    test = df[df["season"] == test_season].copy()
    assert train["season"].max() < val["season"].min() < test_season
    return train, val, test


def temporal_calibrator(estimator, seasons):
    """Calibrate on whole later seasons, never on future-trained classifiers.

    With one season, return the uncalibrated estimator rather than inventing
    a random split. ensemble=True supports expanding folds whose validation
    indices do not partition the earliest season.
    """
    s = np.asarray(seasons)
    years = sorted(np.unique(s))
    folds = [(np.flatnonzero(s < year), np.flatnonzero(s == year))
             for year in years[1:]][-3:]
    if not folds:
        return estimator
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=folds, ensemble=True)
