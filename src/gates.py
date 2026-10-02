"""Shared historical candidate rules and fail-closed live recommendation gate.

Candidate rules were selected retrospectively; they are not proof of live edge.
Use core_candidate for historical research and core_total for live decisions.
"""

import math

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


def core_candidate(row) -> bool:
    """The CORE gate: under, edge 2-7 pts, power-conf involved, wind < 15,
    market total >= 48. See module docstring for the walk-forward record;
    current numbers live in outputs/predictions/core_metrics.json."""
    edge = _get(row, "totals_edge")
    if _isna(edge):
        return False
    edge = float(edge)
    ou = _get(row, "over_under")
    if _isna(ou) or not math.isfinite(float(ou)) or not math.isfinite(edge):
        return False
    return bool(edge <= -2 and edge >= -7 and power_involved(row)
                and not wind15(row) and not low_total(row))


def core_total(row) -> bool:
    """A live play requires an approved price-specific decision, never missing context."""
    return core_candidate(row) and _get(row, "decision_status") == "APPROVED"


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
