"""Settlement at a stated price; score equality is a returned stake."""
import math


def settle(actual, line, direction, odds=-110):
    """Return (result, profit, risk) for a direction-adjusted score difference.

    direction=+1 means home margin/over, -1 means away margin/under.
    line is expected home margin for spreads, O/U for totals.
    Stakes target one unit profit at negative odds, risk one at positive odds.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    actual, line, odds = float(actual), float(line), float(odds)
    if not all(math.isfinite(v) for v in (actual, line, odds)) or odds == 0:
        raise ValueError("Settlement requires finite scores, line and nonzero odds")
    risk = abs(odds) / 100 if odds < 0 else 1.
    payout = 1. if odds < 0 else odds / 100
    delta = (actual - line) * direction
    if abs(delta) < 1e-9:
        return "push", 0., risk
    return ("win", payout, risk) if delta > 0 else ("loss", -risk, risk)
