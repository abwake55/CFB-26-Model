"""
Unified recommendation gate — the single source of truth for what is a bet.

Both the Streamlit app and the weekly pipeline import from here so the rule
can never drift between surfaces.

Walk-forward 2019-25 evidence (see scripts/build_core_history.py):
  - CORE unders: model leans UNDER, edge 2-7 pts, power-conference team
    involved, forecast wind < 15 mph outdoors, market total >= 48.
    The ONLY validated unit play. Flat 1u; hit rate does not rise with edge.
  - Everything else — spreads, moneylines, overs, non-CORE totals — has no
    validated edge and is paper/research only. Never sized.

A "row" is anything dict-like with the prediction fields (pd.Series, dict).
Weather fields (wind_speed, is_dome) may be absent in contexts without a
weather feed (e.g. the weekly pipeline); missing wind is treated as calm,
which matches how the gate behaved before live weather existed.
"""

POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
               "Big East", "FBS Independents"}

# Flat unit size for CORE plays. Hit rate does not rise with edge size, so
# edge-scaled sizing just adds variance; 1u (=1% of bankroll) is conservative.
CORE_UNITS = 1


def _get(row, key, default=None):
    try:
        v = row.get(key, default)
    except AttributeError:
        return default
    return default if v is None else v


def _isna(v) -> bool:
    return v is None or v != v  # None or NaN


def power_involved(row) -> bool:
    """A power-conference team is playing. Totals edge only exists there —
    G5-vs-G5 totals hit 48.2% in the walk-forward and are excluded."""
    return (_get(row, "home_conference") in POWER_CONFS
            or _get(row, "away_conference") in POWER_CONFS)


def wind15(row) -> bool:
    """Forecast wind >= 15 mph outdoors. CORE unders hit only 50.0% in high
    wind (n=88) — the market prices obvious weather itself."""
    ws = _get(row, "wind_speed")
    if _isna(ws) or bool(_get(row, "is_dome", 0)):
        return False
    return float(ws) >= 15


def low_total(row) -> bool:
    """Market total < 48. Low-total games have no over-bias to fade — CORE
    unders there hit ~49%. The inflation the edge exploits lives in higher
    totals (public backs overs in expected shootouts)."""
    ou = _get(row, "over_under")
    return not _isna(ou) and float(ou) < 48


def core_total(row) -> bool:
    """The CORE gate: under, edge 2-7 pts, power-conf involved, wind < 15,
    market total >= 48. See module docstring for the walk-forward record;
    current numbers live in outputs/predictions/core_metrics.json."""
    edge = _get(row, "totals_edge")
    if _isna(edge):
        return False
    edge = float(edge)
    return bool(edge <= -2 and edge >= -7 and power_involved(row)
                and not wind15(row) and not low_total(row))


def is_play(kind: str, row) -> bool:
    """True when the pick's segment has a validated walk-forward edge —
    i.e. it may carry real units. Non-plays are research/paper only.

    2026-09-10: spreads and moneylines are paper-only on every surface.
    The old `spread: week <= 9` sizing path is removed — early-season
    spread cards went 2-5 (-3.18u) in weeks 1-2 of 2026, matching the
    long-run ~50% ATS record.
    """
    if kind == "total":
        return core_total(row)
    return False  # spreads, moneylines: paper record only
