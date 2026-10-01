"""Identical probability blending and preseason adjustment on every surface."""
import numpy as np
import pandas as pd
from preseason import apply_preseason_shrinkage, DEFAULT_SPREAD_SIGMA, norm_cdf


def tune_probability_blend(margins, actual_margins, classifier_probs, home_wins):
    """Select probability blending on held-out validation predictions only."""
    sigma = max(float(np.std(np.asarray(margins) - np.asarray(actual_margins))), 1.)
    implied = np.array([norm_cdf(m / sigma) for m in margins])
    candidates = np.linspace(0, 1, 11)
    scores = [np.mean((a * implied + (1-a) * classifier_probs - home_wins)**2)
              for a in candidates]
    return {"spread_sigma": sigma, "blend_alpha": float(candidates[np.argmin(scores)])}


def adjust_predictions(df, calibration=None, *, probability_col="pred_win_p"):
    """Apply cross-calibration, then early-season shrinkage, exactly once."""
    out = df.copy()
    calibration = calibration or {}
    sigma = float(calibration.get("spread_sigma", DEFAULT_SPREAD_SIGMA))
    alpha = float(calibration.get("blend_alpha", 0.))
    if not np.isfinite(sigma) or sigma <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid probability calibration parameters")
    implied = out["pred_spread"].apply(
        lambda m: norm_cdf(m / sigma) if pd.notna(m) else np.nan)
    raw = out[probability_col]
    out[probability_col] = (alpha * implied + (1-alpha) * raw).fillna(raw).clip(.01, .99)
    market = -pd.to_numeric(out["spread"], errors="coerce")
    out = apply_preseason_shrinkage(
        out, week_col="week", pred_spread_col="pred_spread", market_margin=market,
        over_under_col="over_under", pred_total_col="pred_total",
        pred_win_col=probability_col, sigma=sigma)
    out["pred_away_win_p"] = 1 - out[probability_col]
    out["vegas_home_margin"] = market
    out["spread_edge"] = out["pred_spread"] - market
    out["totals_edge"] = out["pred_total"] - pd.to_numeric(out["over_under"], errors="coerce")
    return out
