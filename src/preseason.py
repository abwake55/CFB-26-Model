"""
Preseason Shrinkage
===================
Early in the season the model leans on tiny samples — one or two games of
EPA data, freshmen in new roles, portal classes that haven't played a snap
together. Small samples swing projections, so early-season outputs are
blended toward preseason priors:

  spread / total : the model's *residual vs the market line* is scaled by
                   f(week) — the market open is the best public preseason
                   prior for a game
  win probability: blended toward the market-implied probability
                   Φ(vegas_margin / σ), falling back to 0.5 when no line
                   is posted yet

Schedule — share of model signal kept (rest comes from the prior):
  Week 1 → 60%  ·  Week 2 → 75%  ·  Week 3 → 90%  ·  Week 4+ → 100%

Validated out-of-sample on the 2019–25 walk-forward (~5,200 games):
  CORE under portfolio  54.9% → 56.4% hit,  +4.8% → +7.7% ROI (n=530)
  win-prob Brier        0.186 → 0.185 overall,  0.181 → 0.174 weeks 1–3
  spread ATS wk1-3 edge≥3  53.7% → 55.7%
(A same-environment A/B against unshrunk outputs puts the CORE delta
within fold noise, ±0.7pp ROI; calibration improves consistently.)
Revalidate with scripts/walk_forward.py after any change here.
"""

from math import erf, sqrt

import numpy as np
import pandas as pd

# Share of the model signal kept, by week. Week 4+ is full model.
SHRINK_SCHEDULE = {1: 0.60, 2: 0.75, 3: 0.90}

# Fallback σ for the margin→win-prob normal CDF when the tuned value
# (models/win_prob_calibration.json, written by src/model.py) is absent —
# e.g. in walk_forward.py, which has no saved calibration file. Matches the
# currently trained model's σ ≈ 15.4.
DEFAULT_SPREAD_SIGMA = 15.4


def shrinkage_factor(week) -> float:
    """Share of model signal kept this week (1.0 = full model)."""
    try:
        w = int(week)
    except (TypeError, ValueError):
        return 1.0
    return SHRINK_SCHEDULE.get(w, 1.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(float(x) / sqrt(2)))


def apply_preseason_shrinkage(df: pd.DataFrame, *,
                              week_col: str,
                              pred_spread_col: str,
                              market_margin,          # Series or column name
                              over_under_col: str,
                              pred_total_col: str,
                              pred_win_col: str,
                              sigma: float = DEFAULT_SPREAD_SIGMA,
                              factor_col: str = "prior_shrink") -> pd.DataFrame:
    """
    Blend early-season predictions toward preseason (market) priors in place
    on a copy of `df`. Adds `factor_col` with the per-game model share so the
    UI can badge shrunk picks.

    market_margin: home margin implied by the market line (home − away,
    negative = home favored), as a Series aligned with df or a column name.
    Rows without a market line shrink win prob toward 0.5 and leave
    spread/total untouched (no residual to scale).
    """
    out = df.copy()
    margin = (out[market_margin] if isinstance(market_margin, str)
              else pd.Series(market_margin, index=out.index))
    margin = pd.to_numeric(margin, errors="coerce")
    f = out[week_col].map(shrinkage_factor).astype(float)

    if factor_col not in out.columns:
        out[factor_col] = f

    # ── Spread: scale the residual vs the market margin ──────────────────────
    if pred_spread_col in out.columns:
        ps = pd.to_numeric(out[pred_spread_col], errors="coerce")
        ok = ps.notna() & margin.notna()
        out.loc[ok, pred_spread_col] = margin[ok] + (ps[ok] - margin[ok]) * f[ok]

    # ── Total: scale the deviation vs the O/U line ───────────────────────────
    if pred_total_col in out.columns and over_under_col in out.columns:
        pt = pd.to_numeric(out[pred_total_col], errors="coerce")
        ou = pd.to_numeric(out[over_under_col], errors="coerce")
        ok = pt.notna() & ou.notna()
        out.loc[ok, pred_total_col] = ou[ok] + (pt[ok] - ou[ok]) * f[ok]

    # ── Win probability: blend toward the market-implied prior ───────────────
    if pred_win_col in out.columns:
        pw = pd.to_numeric(out[pred_win_col], errors="coerce")
        p_mkt = margin.apply(lambda m: norm_cdf(m / sigma) if pd.notna(m) else np.nan)
        prior = p_mkt.fillna(0.5)          # no line → uninformative prior
        ok = pw.notna()
        out.loc[ok, pred_win_col] = (f[ok] * pw[ok] + (1 - f[ok]) * prior[ok]).clip(0.01, 0.99)

    return out


def sample_badge(week) -> tuple[str, str] | None:
    """
    (label, css_color) confidence badge for pick cards in shrinkage weeks,
    else None once the model runs on full signal (Week 4+).
    """
    f = shrinkage_factor(week)
    if f >= 1.0:
        return None
    model_pct = round(f * 100)
    label = f"WK{int(week)} SAMPLE · {model_pct}% MODEL / {100 - model_pct}% PRIOR"
    color = ("var(--orange)" if f < 0.70 else
             "var(--gold)"   if f < 0.85 else
             "var(--blue)")
    return label, color
