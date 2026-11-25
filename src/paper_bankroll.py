"""$1,000 paper bankroll for the 2026 CFB betting rule.

A flat 1u = $20 (2% of the $1,000 starting bankroll), held constant through
the 2026 regular season, bowls and playoff. Stakes follow
docs/betting_rule_2026-11-24.md. Bets are frozen at kickoff: after that
only result, close, CLV, profit and the derived running balance may be
filled in.

Prices: try DraftKings Predictions public pages first (src/dk_predictions.py).
If no price is available, fall back to the sportsbook snapshot line at -110
and mark the row `price_source=sportsbook_fallback`.

Files
-----
outputs/bankroll/paper_bankroll_2026.csv
outputs/bankroll/paper_bankroll_2026_summary.json
outputs/picks/slip_2026_WNN.txt
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LEDGER_CSV = ROOT / "outputs" / "predictions" / "ledger_2026.csv"
BANKROLL_DIR = ROOT / "outputs" / "bankroll"
BANKROLL_CSV = BANKROLL_DIR / "paper_bankroll_2026.csv"
SUMMARY_JSON = BANKROLL_DIR / "paper_bankroll_2026_summary.json"
SLIP_DIR = ROOT / "outputs" / "picks"
SNAP_DIR = ROOT / "data" / "lines_snapshots"

STARTING_BANKROLL = 1000.0
UNIT_USD = 20.0  # flat; 2% of the starting bankroll
SEASON = 2026
FALLBACK_AMERICAN = -110
PT_THRESHOLD = 7.0
PT_LARGE_EDGE = 10.0  # 0.50u at or above this; otherwise 0.25u
OPENER_GAP_THRESHOLD = 5.0
HIGH_TOTAL_THRESHOLD = 60.0

# Regular-season stake in units, before the bowl paper override.
# main-model spreads are always paper (0u).
STAKE_UNITS = {
    "core_under": 1.00,
    "pt_fpi": None,  # 0.25 or 0.50, decided by edge
    "opener_gap": 0.25,
    "high_total_under": 0.25,
    "spread_model": 0.0,
    "bowl_adj": 0.0,
}

# In bowls / CFP these signals stay on the board but stake 0.
BOWL_PAPER_SIGNALS = {
    "pt_fpi", "opener_gap", "high_total_under", "bowl_adj", "spread_model",
}

CSV_COLS = [
    "bet_id", "season", "week", "game_id", "date", "kickoff",
    "home_team", "away_team", "signal", "side", "line", "market",
    "american", "contract_price", "fee_per_contract",
    "price_source", "price_url", "price_ts", "fallback",
    "stake_units", "stake_usd", "stake_reason",
    "line_logged", "logged_at", "frozen_at_kickoff",
    "closing_line", "clv_pts", "clv_sharp_pts",
    "result", "profit_usd", "running_balance",
    "is_bowl", "status",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return None
    return ts


def has_kicked_off(kickoff, at: datetime | None = None) -> bool:
    ts = parse_ts(kickoff)
    if ts is None:
        return False
    now = at or now_utc()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return ts <= pd.Timestamp(now)


def american_profit(stake: float, american: int, result: str) -> float:
    """Profit (not returned stake) for a graded bet. Push = 0."""
    r = (result or "").strip().lower()
    if r == "push":
        return 0.0
    if r == "loss":
        return -float(stake)
    if r == "win":
        a = int(american) if american not in (None, "", np.nan) else FALLBACK_AMERICAN
        if a < 0:
            return round(float(stake) * 100.0 / abs(a), 2)
        return round(float(stake) * a / 100.0, 2)
    raise ValueError(f"unknown result {result!r}")


def pt_units(edge: float) -> float:
    """0.50u when |FPI edge| >= 10, else 0.25u (threshold to flag is 7)."""
    if abs(float(edge)) >= PT_LARGE_EDGE:
        return 0.50
    return 0.25


def pt_units_reason(edge: float) -> str:
    e = abs(float(edge))
    if e >= PT_LARGE_EDGE:
        return f"pt_fpi edge {e:.2f} >= {PT_LARGE_EDGE:g}: 0.50u"
    return f"pt_fpi edge {e:.2f} < {PT_LARGE_EDGE:g}: 0.25u"


def stake_for(signal: str, edge: float | None, is_bowl: bool) -> tuple[float, str]:
    """Return (units, reason). Bowl override sets listed signals to 0u."""
    sig = signal
    if is_bowl and sig in BOWL_PAPER_SIGNALS:
        return 0.0, f"{sig} is paper in bowls (stake 0)"
    if sig == "pt_fpi":
        if edge is None or (isinstance(edge, float) and math.isnan(edge)):
            return 0.25, "pt_fpi edge missing: default 0.25u"
        return pt_units(edge), pt_units_reason(edge)
    if sig not in STAKE_UNITS:
        raise ValueError(f"unknown signal {sig}")
    units = float(STAKE_UNITS[sig])
    if units == 0:
        return 0.0, f"{sig} is paper only (stake 0)"
    return units, f"{sig}: {units:g}u per betting rule"


def money_stake(units: float) -> float:
    return round(float(units) * UNIT_USD, 2)


def half_point_ceil(x: float) -> float:
    return math.ceil(x * 2.0) / 2.0


def half_point_floor(x: float) -> float:
    return math.floor(x * 2.0) / 2.0


def worst_acceptable_spread(bet_line: float, edge: float, threshold: float) -> float:
    """Worst home-perspective spread still at or inside `threshold` points of edge.

    Ledger convention: positive edge means bet the home side at `bet_line`.
    Fair home line is bet_line - edge. Betting home, a more negative line is
    worse; betting away, a more positive home line is worse.
    """
    fair = float(bet_line) - float(edge)
    if float(edge) >= 0:
        # home: accept L >= fair + threshold (worst = that floor)
        raw = fair + float(threshold)
        return half_point_ceil(raw)
    raw = fair - float(threshold)
    return half_point_floor(raw)


def worst_acceptable_total(bet_line: float) -> float:
    """Under bets: do not take a lower number than the one that was logged."""
    return float(bet_line)


def pick_label(home: str, away: str, signal: str, side: str, line: float) -> str:
    """One human line like 'Miami -25.5 vs Boston College' or 'SMU/Stanford under 61'."""
    if signal in ("high_total_under", "core_under") or side.lower() in ("under", "over"):
        return f"{away}/{home} {side.lower()} {_fmt_total(line)}"
    team = home if side.lower() == "home" else away
    # Display the line from the picked team's perspective.
    shown = float(line) if side.lower() == "home" else -float(line)
    opp = away if side.lower() == "home" else home
    return f"{team} {_fmt_line(shown)} vs {opp}"


def _fmt_line(line: float) -> str:
    x = float(line)
    if abs(x - round(x)) < 1e-9:
        s = f"{int(round(x))}"
    else:
        s = f"{x:.1f}"
    if x > 0:
        return f"+{s}"
    return s


def _fmt_total(line: float) -> str:
    x = float(line)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.1f}"


def side_from_spread_edge(edge: float) -> str:
    return "home" if float(edge) >= 0 else "away"


def grade_spread(home_score, away_score, side: str, line: float) -> str:
    """ATS result from the home-perspective line."""
    margin = float(home_score) - float(away_score)
    L = float(line)
    # Home covers if margin + line > 0 when betting home? Home betting L:
    # home favorite -25.5 covers if margin > 25.5, i.e. margin + L > 0? margin=26, L=-25.5, 0.5>0 yes.
    # Home dog +3 covers if margin + 3 > 0.
    cover_margin = margin + L
    if abs(cover_margin) < 1e-9:
        return "push"
    home_covers = cover_margin > 0
    won = home_covers if side.lower() == "home" else (not home_covers)
    return "win" if won else "loss"


def grade_total(home_score, away_score, side: str, line: float) -> str:
    total = float(home_score) + float(away_score)
    if abs(total - float(line)) < 1e-9:
        return "push"
    went_under = total < float(line)
    if side.lower() == "under":
        return "win" if went_under else "loss"
    return "loss" if went_under else "win"


def spread_clv(bet_line: float, close: float, side: str) -> float:
    """Points of CLV, bet side. Same sign convention as the prediction ledger."""
    sgn = 1.0 if side.lower() == "home" else -1.0
    return round(sgn * (float(bet_line) - float(close)), 2)


def total_clv(bet_line: float, close: float, side: str) -> float:
    """Under: a higher number than the close is positive CLV."""
    diff = float(bet_line) - float(close)
    return round(diff if side.lower() == "under" else -diff, 2)


def load_bankroll(path: Path | None = None) -> pd.DataFrame:
    p = path or BANKROLL_CSV
    if not p.exists():
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.read_csv(p)
    for c in CSV_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CSV_COLS].copy()


def save_bankroll(df: pd.DataFrame, path: Path | None = None) -> None:
    p = path or BANKROLL_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df[CSV_COLS].copy()
    out.to_csv(p, index=False)


def empty_row(**kwargs) -> dict:
    row = {c: None for c in CSV_COLS}
    row.update(kwargs)
    return row


def next_bet_id(df: pd.DataFrame, season: int, week: int, n: int = 1) -> list[str]:
    prefix = f"{season}-W{int(week):02d}-"
    existing = []
    if len(df) and "bet_id" in df.columns:
        existing = [str(x) for x in df["bet_id"].astype(str) if str(x).startswith(prefix)]
    used = set()
    for b in existing:
        try:
            used.add(int(b.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    out = []
    i = 1
    while len(out) < n:
        if i not in used:
            out.append(f"{prefix}{i:02d}")
        i += 1
    return out


def snapshot_first(season: int, week: int, game_id) -> dict | None:
    path = SNAP_DIR / f"lines_{int(season)}_W{int(week):02d}.json"
    if not path.exists():
        return None
    try:
        games = json.loads(path.read_text()).get("games") or {}
    except (OSError, json.JSONDecodeError):
        return None
    return games.get(str(int(game_id)))


def price_bet(home: str, away: str, market: str, line: float, side: str,
              lookup=None) -> dict:
    """DK Predictions if a public price is found, else -110 fallback."""
    fn = lookup
    if fn is None:
        from dk_predictions import lookup_price as fn
    try:
        hit = fn(home, away, market, line, side)
    except Exception:
        hit = None
    if hit and hit.get("american") is not None:
        return {
            "american": int(hit["american"]),
            "contract_price": hit.get("contract_price"),
            "fee_per_contract": hit.get("fee_per_contract"),
            "price_source": hit.get("source") or "draftkings_predictions",
            "price_url": hit.get("url"),
            "price_ts": hit.get("fetched_at") or now_utc().isoformat(timespec="seconds"),
            "fallback": False,
        }
    return {
        "american": FALLBACK_AMERICAN,
        "contract_price": None,
        "fee_per_contract": None,
        "price_source": "sportsbook_fallback",
        "price_url": None,
        "price_ts": now_utc().isoformat(timespec="seconds"),
        "fallback": True,
    }


def _row_frozen(row, at=None) -> bool:
    if _is_true(row.get("frozen_at_kickoff")):
        return True
    return has_kicked_off(row.get("kickoff"), at=at)


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def freeze_kicked_off(df: pd.DataFrame, at=None) -> pd.DataFrame:
    """Mark rows whose kickoff has passed. Never un-freezes a frozen row."""
    out = df.copy()
    if "frozen_at_kickoff" not in out.columns:
        out["frozen_at_kickoff"] = False
    for i, r in out.iterrows():
        if _is_true(r.get("frozen_at_kickoff")):
            continue
        if has_kicked_off(r.get("kickoff"), at=at):
            out.at[i, "frozen_at_kickoff"] = True
    return out


def add_bet(df: pd.DataFrame, bet: dict, at=None) -> pd.DataFrame:
    """Append a bet if the game has not kicked off and it is not a duplicate."""
    if has_kicked_off(bet.get("kickoff"), at=at):
        raise ValueError(
            f"refusing to log {bet.get('signal')} on {bet.get('game_id')}: already kicked off"
        )
    key = (int(bet["game_id"]), str(bet["signal"]))
    if len(df):
        have = set(zip(pd.to_numeric(df["game_id"], errors="coerce").fillna(-1).astype(int),
                       df["signal"].astype(str)))
        if key in have:
            return df
    out = pd.concat([df, pd.DataFrame([bet])], ignore_index=True)
    return out[CSV_COLS]


# ── Week 13 seed (the slips the user sent his group) ─────────────────────────

# Lines are the numbers in the group text, which match the first captured
# total (unders) or the FPI bet line (spreads) in the 2026 ledger / snapshot.
WEEK13_PICKS = [
    {
        "game_id": 401858311, "home_team": "Miami", "away_team": "Boston College",
        "kickoff": "2026-11-28T17:00:00+00:00",
        "signal": "pt_fpi", "side": "home", "line": -25.5, "market": "spread",
        "edge": 7.306440510000002,
    },
    {
        "game_id": 401858540, "home_team": "Ohio State", "away_team": "Michigan",
        "kickoff": "2026-11-28T17:00:00+00:00",
        "signal": "pt_fpi", "side": "home", "line": -10.0, "market": "spread",
        "edge": 8.409546673000001,
    },
    {
        "game_id": 401862843, "home_team": "North Texas", "away_team": "UAB",
        "kickoff": "2026-11-28T21:00:00+00:00",
        "signal": "pt_fpi", "side": "home", "line": -3.5, "market": "spread",
        "edge": 11.367982719,
    },
    {
        "game_id": 401864547, "home_team": "Hawai'i", "away_team": "Sacramento State",
        "kickoff": "2026-11-29T04:00:00+00:00",
        "signal": "pt_fpi", "side": "home", "line": -13.5, "market": "spread",
        "edge": 9.770186431,
    },
    {
        "game_id": 401860926, "home_team": "Colorado State", "away_team": "Boise State",
        "kickoff": "2026-11-28T23:00:00+00:00",
        "signal": "high_total_under", "side": "under", "line": 75.5, "market": "total",
        "edge": None,
    },
    {
        "game_id": 401856880, "home_team": "Utah", "away_team": "West Virginia",
        "kickoff": "2026-11-28T02:00:00+00:00",
        "signal": "high_total_under", "side": "under", "line": 68.5, "market": "total",
        "edge": None,
    },
    {
        "game_id": 401856877, "home_team": "Oklahoma State", "away_team": "Kansas",
        "kickoff": "2026-11-28T23:15:00+00:00",
        "signal": "high_total_under", "side": "under", "line": 67.5, "market": "total",
        "edge": None,
    },
    {
        "game_id": 401862838, "home_team": "Memphis", "away_team": "Temple",
        "kickoff": "2026-11-27T20:30:00+00:00",
        "signal": "high_total_under", "side": "under", "line": 64.5, "market": "total",
        "edge": None,
    },
    {
        "game_id": 401858316, "home_team": "Stanford", "away_team": "SMU",
        "kickoff": "2026-11-28T20:30:00+00:00",
        "signal": "high_total_under", "side": "under", "line": 61.0, "market": "total",
        "edge": None,
    },
]


def seed_week13(df: pd.DataFrame, at=None, lookup=None) -> pd.DataFrame:
    """Log the week 13 group-text picks if they are not already present.

    Refuses to add a game that has already kicked off. Idempotent.
    """
    at = at or now_utc()
    out = df.copy()
    ids = next_bet_id(out, SEASON, 13, n=len(WEEK13_PICKS))
    id_i = 0
    logged_at = (at if isinstance(at, datetime) else now_utc()).astimezone(timezone.utc).isoformat(timespec="seconds")
    for p in WEEK13_PICKS:
        if has_kicked_off(p["kickoff"], at=at):
            raise ValueError(
                f"week 13 seed: {p['away_team']} @ {p['home_team']} already kicked off"
            )
        units, reason = stake_for(p["signal"], p.get("edge"), is_bowl=False)
        price = price_bet(p["home_team"], p["away_team"], p["market"], p["line"], p["side"],
                          lookup=lookup)
        bet_id = ids[id_i]
        id_i += 1
        row = empty_row(
            bet_id=bet_id, season=SEASON, week=13, game_id=int(p["game_id"]),
            date=_kickoff_date_pt(p["kickoff"]),
            kickoff=p["kickoff"],
            home_team=p["home_team"], away_team=p["away_team"],
            signal=p["signal"], side=p["side"], line=float(p["line"]),
            market=p["market"],
            american=price["american"],
            contract_price=price["contract_price"],
            fee_per_contract=price["fee_per_contract"],
            price_source=price["price_source"],
            price_url=price["price_url"],
            price_ts=price["price_ts"],
            fallback=bool(price["fallback"]),
            stake_units=units, stake_usd=money_stake(units),
            stake_reason=reason,
            line_logged=float(p["line"]),
            logged_at=logged_at,
            frozen_at_kickoff=False,
            closing_line=None, clv_pts=None, clv_sharp_pts=None,
            result=None, profit_usd=None, running_balance=None,
            is_bowl=False, status="open",
        )
        out = add_bet(out, row, at=at)
    return out


def _kickoff_date_pt(kickoff: str) -> str:
    ts = parse_ts(kickoff)
    if ts is None:
        return ""
    return ts.tz_convert("America/Los_Angeles").strftime("%Y-%m-%d")


def _is_bowl_row(r) -> bool:
    """True for postseason / bowl / CFP games.

    The regular-season ledger has no season_type column. A week number of
    16 or more, a 'bowl' flag, or a start date on/after 12 Dec (bowl season
    typically; conference title games are earlier) is treated as a bowl
    only when the caller already marked it, or the week is clearly
    postseason. Week 13–15 stay regular season.
    """
    if _is_true(r.get("is_bowl")):
        return True
    st = str(r.get("season_type") or "").lower()
    if st in ("postseason", "bowl", "cfp"):
        return True
    try:
        week = int(r.get("week"))
    except (TypeError, ValueError):
        week = 0
    return week >= 16


# Signals already covering a market on a game (don't stack two spread bets).
SPREAD_SIGNALS = ("pt_fpi", "opener_gap", "spread_model", "bowl_adj")
TOTAL_SIGNALS = ("core_under", "high_total_under")


def _occupied(df: pd.DataFrame, game_id, family) -> bool:
    if not len(df):
        return False
    sub = df[pd.to_numeric(df["game_id"], errors="coerce") == int(game_id)]
    return bool(sub["signal"].astype(str).isin(family).any())


def candidates_from_ledger(ledger: pd.DataFrame) -> list[dict]:
    """Build unsized candidate bets from live ledger flags."""
    out = []
    if ledger is None or not len(ledger):
        return out
    for _, r in ledger.iterrows():
        kick = r.get("start_date")
        base = {
            "game_id": int(r["game_id"]),
            "season": int(r.get("season") or SEASON),
            "week": int(r["week"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "kickoff": kick,
            "is_bowl": _is_bowl_row(r),
        }
        if _is_true(r.get("flag_core")):
            line = r.get("over_under")
            if pd.notna(line):
                out.append({**base, "signal": "core_under", "side": "under",
                            "line": float(line), "market": "total", "edge": None})
        if _is_true(r.get("flag_high_total_under")):
            line = r.get("high_total_under_bet_line")
            if pd.isna(line):
                line = r.get("high_total_under_line")
            if pd.notna(line):
                out.append({**base, "signal": "high_total_under", "side": "under",
                            "line": float(line), "market": "total", "edge": None})
        if _is_true(r.get("flag_pt")):
            line = r.get("pt_bet_line")
            if pd.isna(line):
                line = r.get("spread")
            edge = r.get("pt_bet_edge")
            if pd.isna(edge):
                edge = r.get("pt_edge")
            if pd.notna(line) and pd.notna(edge):
                out.append({**base, "signal": "pt_fpi", "side": side_from_spread_edge(edge),
                            "line": float(line), "market": "spread", "edge": float(edge)})
        if _is_true(r.get("flag_opener_gap")):
            line = r.get("opener_gap_bet_line")
            if pd.isna(line):
                line = r.get("opener_gap_line")
            if pd.isna(line):
                line = r.get("spread")
            edge = r.get("opener_gap_bet_edge")
            if pd.isna(edge):
                edge = r.get("opener_gap_edge")
            if pd.notna(line) and pd.notna(edge):
                out.append({**base, "signal": "opener_gap", "side": side_from_spread_edge(edge),
                            "line": float(line), "market": "spread", "edge": float(edge)})
        if _is_true(r.get("flag_spread")):
            line = r.get("spread_bet_line")
            if pd.isna(line):
                line = r.get("spread")
            edge = r.get("spread_bet_edge")
            if pd.isna(edge):
                edge = r.get("spread_edge")
            if pd.notna(line) and pd.notna(edge):
                out.append({**base, "signal": "spread_model", "side": side_from_spread_edge(edge),
                            "line": float(line), "market": "spread", "edge": float(edge)})
    return out


def sync_from_ledger(df: pd.DataFrame, ledger: pd.DataFrame, at=None, lookup=None) -> pd.DataFrame:
    """Add newly flagged, not-yet-kicked-off bets. Never edits frozen rows.

    Stacking: at most one staked spread bet and one staked totals bet per
    game. Preference is core_under > high_total_under, and
    pt_fpi > opener_gap > spread_model. The lower-priority flag can still be
    logged at 0u so it is tracked.
    """
    at = at or now_utc()
    out = freeze_kicked_off(df, at=at)
    cands = candidates_from_ledger(ledger)
    # Sort so higher-priority signals are considered first in a game.
    rank = {"core_under": 0, "pt_fpi": 1, "high_total_under": 2,
            "opener_gap": 3, "bowl_adj": 4, "spread_model": 5}
    cands.sort(key=lambda c: (c["week"], c["game_id"], rank.get(c["signal"], 9)))

    logged_at = (at if isinstance(at, datetime) else now_utc()).astimezone(timezone.utc).isoformat(timespec="seconds")
    for c in cands:
        if has_kicked_off(c["kickoff"], at=at):
            continue
        key = (int(c["game_id"]), str(c["signal"]))
        have = set(zip(pd.to_numeric(out["game_id"], errors="coerce").fillna(-1).astype(int),
                       out["signal"].astype(str))) if len(out) else set()
        if key in have:
            continue
        units, reason = stake_for(c["signal"], c.get("edge"), c["is_bowl"])
        family = SPREAD_SIGNALS if c["signal"] in SPREAD_SIGNALS else TOTAL_SIGNALS
        if units > 0 and _occupied_staked(out, c["game_id"], family):
            units, reason = 0.0, reason + "; stake 0 (same-market bet already staked)"
        price = price_bet(c["home_team"], c["away_team"], c["market"], c["line"], c["side"],
                          lookup=lookup)
        bet_id = next_bet_id(out, c["season"], c["week"], n=1)[0]
        row = empty_row(
            bet_id=bet_id, season=c["season"], week=c["week"], game_id=int(c["game_id"]),
            date=_kickoff_date_pt(c["kickoff"]) if c.get("kickoff") else "",
            kickoff=c["kickoff"],
            home_team=c["home_team"], away_team=c["away_team"],
            signal=c["signal"], side=c["side"], line=float(c["line"]),
            market=c["market"],
            american=price["american"],
            contract_price=price["contract_price"],
            fee_per_contract=price["fee_per_contract"],
            price_source=price["price_source"],
            price_url=price["price_url"],
            price_ts=price["price_ts"],
            fallback=bool(price["fallback"]),
            stake_units=units, stake_usd=money_stake(units),
            stake_reason=reason,
            line_logged=float(c["line"]),
            logged_at=logged_at,
            frozen_at_kickoff=False,
            closing_line=None, clv_pts=None, clv_sharp_pts=None,
            result=None, profit_usd=None, running_balance=None,
            is_bowl=bool(c["is_bowl"]), status="open",
        )
        out = add_bet(out, row, at=at)
    return out


def _occupied_staked(df: pd.DataFrame, game_id, family) -> bool:
    if not len(df):
        return False
    sub = df[(pd.to_numeric(df["game_id"], errors="coerce") == int(game_id))
             & df["signal"].astype(str).isin(family)]
    stakes = pd.to_numeric(sub.get("stake_usd"), errors="coerce").fillna(0)
    return bool((stakes > 0).any())


def grade_bankroll(df: pd.DataFrame, ledger: pd.DataFrame, at=None) -> pd.DataFrame:
    """Fill results, closes, CLV and profit for completed games.

    Frozen (or kicked-off) rows never have their bet fields changed. Open
    rows whose game is not yet complete are left alone. Running balance is
    recomputed from settled bets in kickoff order, starting at $1,000.
    """
    at = at or now_utc()
    out = freeze_kicked_off(df, at=at)
    if ledger is None or not len(ledger):
        return _recompute_balance(out)

    led = ledger.copy()
    led["game_id"] = pd.to_numeric(led["game_id"], errors="coerce")
    by_gid = led.drop_duplicates("game_id", keep="last").set_index("game_id")

    for i, r in out.iterrows():
        gid = pd.to_numeric(r.get("game_id"), errors="coerce")
        if pd.isna(gid) or int(gid) not in by_gid.index:
            continue
        g = by_gid.loc[int(gid)]
        completed = _is_true(g.get("completed"))
        hs, aus = g.get("home_score"), g.get("away_score")
        if not completed or pd.isna(hs) or pd.isna(aus):
            continue
        # Only fill outcome fields. Never touch line, price, stake.
        if r["market"] == "total" or r["signal"] in TOTAL_SIGNALS:
            result = grade_total(hs, aus, r["side"], r["line"])
            close = g.get("closing_over_under")
            sharp = g.get("closing_sharp_total")
            clv_fn = total_clv
        else:
            result = grade_spread(hs, aus, r["side"], r["line"])
            close = g.get("closing_spread")
            sharp = g.get("closing_sharp_spread")
            clv_fn = spread_clv
        profit = american_profit(float(r["stake_usd"] or 0), int(r["american"] or FALLBACK_AMERICAN), result)
        out.at[i, "result"] = result
        out.at[i, "profit_usd"] = profit
        out.at[i, "status"] = "settled"
        if pd.notna(close):
            out.at[i, "closing_line"] = float(close)
            out.at[i, "clv_pts"] = clv_fn(float(r["line"]), float(close), r["side"])
        if pd.notna(sharp):
            out.at[i, "clv_sharp_pts"] = clv_fn(float(r["line"]), float(sharp), r["side"])
        if not _is_true(r.get("frozen_at_kickoff")):
            out.at[i, "frozen_at_kickoff"] = True
    return _recompute_balance(out)


def _recompute_balance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not len(out):
        return out
    # Sort settled bets by kickoff, then bet_id, for a stable running total.
    kick = pd.to_datetime(out["kickoff"], errors="coerce", utc=True, format="mixed")
    order = kick.fillna(pd.Timestamp.max.tz_localize("UTC")).argsort(kind="mergesort")
    # argsort on a Series returns positions; keep original index mapping
    ranked = out.iloc[order]
    bal = STARTING_BANKROLL
    balances = {}
    for i, r in ranked.iterrows():
        if str(r.get("status")) == "settled" and pd.notna(r.get("profit_usd")):
            bal = round(bal + float(r["profit_usd"]), 2)
            balances[i] = bal
        else:
            balances[i] = np.nan
    out["running_balance"] = out.index.map(lambda i: balances.get(i, np.nan))
    return out


def current_balance(df: pd.DataFrame) -> float:
    settled = df[df["status"].astype(str) == "settled"] if len(df) else df
    if not len(settled) or settled["profit_usd"].isna().all():
        return STARTING_BANKROLL
    last = pd.to_numeric(settled["running_balance"], errors="coerce").dropna()
    if len(last):
        # The latest kickoff's balance
        kick = pd.to_datetime(settled["kickoff"], errors="coerce", utc=True, format="mixed")
        idx = kick.fillna(pd.Timestamp.min.tz_localize("UTC")).idxmax()
        val = settled.at[idx, "running_balance"]
        if pd.notna(val):
            return float(val)
    return round(STARTING_BANKROLL + float(settled["profit_usd"].sum()), 2)


def record_tuple(df: pd.DataFrame) -> tuple[int, int, int]:
    settled = df[df["status"].astype(str) == "settled"] if len(df) else df
    # Count only staked bets in the headline record; 0u paper stays in by_signal.
    staked = settled[pd.to_numeric(settled.get("stake_usd"), errors="coerce").fillna(0) > 0] if len(settled) else settled
    w = int((staked["result"] == "win").sum()) if len(staked) else 0
    l = int((staked["result"] == "loss").sum()) if len(staked) else 0
    p = int((staked["result"] == "push").sum()) if len(staked) else 0
    return w, l, p


def build_summary(df: pd.DataFrame, at=None) -> dict:
    at = at or now_utc()
    settled = df[df["status"].astype(str) == "settled"] if len(df) else df
    staked = settled[pd.to_numeric(settled.get("stake_usd"), errors="coerce").fillna(0) > 0] if len(settled) else settled
    w, l, p = record_tuple(df)
    profit = float(pd.to_numeric(staked.get("profit_usd"), errors="coerce").fillna(0).sum()) if len(staked) else 0.0
    staked_sum = float(pd.to_numeric(staked.get("stake_usd"), errors="coerce").fillna(0).sum()) if len(staked) else 0.0
    roi = (profit / staked_sum) if staked_sum else None
    clv = pd.to_numeric(staked.get("clv_pts"), errors="coerce") if len(staked) else pd.Series(dtype=float)
    clv_s = pd.to_numeric(staked.get("clv_sharp_pts"), errors="coerce") if len(staked) else pd.Series(dtype=float)
    open_rows = df[df["status"].astype(str) != "settled"] if len(df) else df
    open_risk = float(pd.to_numeric(open_rows.get("stake_usd"), errors="coerce").fillna(0).sum()) if len(open_rows) else 0.0
    bal = current_balance(df)

    def bucket(sub: pd.DataFrame) -> dict:
        staked = sub[pd.to_numeric(sub.get("stake_usd"), errors="coerce").fillna(0) > 0] if len(sub) else sub
        ss = staked[staked["status"].astype(str) == "settled"] if len(staked) else staked
        ww = int((ss["result"] == "win").sum()) if len(ss) else 0
        ll = int((ss["result"] == "loss").sum()) if len(ss) else 0
        pp = int((ss["result"] == "push").sum()) if len(ss) else 0
        pr = float(pd.to_numeric(ss.get("profit_usd"), errors="coerce").fillna(0).sum()) if len(ss) else 0.0
        sta = float(pd.to_numeric(ss.get("stake_usd"), errors="coerce").fillna(0).sum()) if len(ss) else 0.0
        c = pd.to_numeric(ss.get("clv_pts"), errors="coerce") if len(ss) else pd.Series(dtype=float)
        return {
            "n_settled": int(len(ss)),
            "n_open": int((sub["status"].astype(str) != "settled").sum()) if len(sub) else 0,
            "wins": ww, "losses": ll, "pushes": pp,
            "profit_usd": round(pr, 2),
            "staked_usd": round(sta, 2),
            "roi": None if not sta else round(pr / sta, 4),
            "avg_clv_pts": None if c.dropna().empty else round(float(c.mean()), 3),
            "clv_n": int(c.notna().sum()) if len(c) else 0,
        }

    by_signal = {}
    if len(df):
        for sig, sub in df.groupby(df["signal"].astype(str)):
            by_signal[sig] = bucket(sub)

    n_decided = w + l  # pushes excluded from hit rate
    return {
        "season": SEASON,
        "starting_bankroll": STARTING_BANKROLL,
        "unit_usd": UNIT_USD,
        "unit_rule": "flat $20 (2% of the $1,000 starting bankroll), held constant through 2026 bowls and the playoff",
        "current_balance": round(bal, 2),
        "open_risk_usd": round(open_risk, 2),
        "profit_usd": round(profit, 2),
        "staked_usd": round(staked_sum, 2),
        "record": {"wins": w, "losses": l, "pushes": p},
        "record_str": f"{w}-{l}" + (f"-{p}" if p else ""),
        "hit_rate": None if not n_decided else round(w / n_decided, 4),
        "roi": None if roi is None else round(roi, 4),
        "avg_clv_pts": None if clv.dropna().empty else round(float(clv.mean()), 3),
        "clv_n": int(clv.notna().sum()) if len(clv) else 0,
        "avg_clv_sharp_pts": None if clv_s.dropna().empty else round(float(clv_s.mean()), 3),
        "clv_sharp_n": int(clv_s.notna().sum()) if len(clv_s) else 0,
        "n_bets": int(len(df)),
        "n_settled": int((df["status"].astype(str) == "settled").sum()) if len(df) else 0,
        "n_open": int((df["status"].astype(str) != "settled").sum()) if len(df) else 0,
        "by_signal": by_signal,
        "paths": {
            "bankroll_csv": "outputs/bankroll/paper_bankroll_2026.csv",
            "summary_json": "outputs/bankroll/paper_bankroll_2026_summary.json",
            "slip_pattern": "outputs/picks/slip_2026_WNN.txt",
        },
        "pricing_note": (
            "Prefer DraftKings Predictions public contract prices (fee-inclusive). "
            "When those are not reachable, use the sportsbook snapshot line at -110 "
            "and mark the row price_source=sportsbook_fallback."
        ),
        "last_updated": (at if isinstance(at, datetime) else now_utc()).astimezone(timezone.utc).isoformat(timespec="seconds"),
    }


def write_summary(df: pd.DataFrame, path: Path | None = None, at=None) -> dict:
    summary = build_summary(df, at=at)
    p = path or SUMMARY_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def worst_for_row(r) -> tuple[str, str]:
    """(worst line label, worst price label) for the slip."""
    american = int(r["american"] if pd.notna(r.get("american")) else FALLBACK_AMERICAN)
    price_s = f"{american}"
    if r["signal"] in TOTAL_SIGNALS or r["market"] == "total":
        worst = worst_acceptable_total(float(r["line"]))
        return f"{r['side'].lower()} {_fmt_total(worst)}", price_s
    edge = None
    # Recover edge from stake_reason if needed; for worst line we need edge.
    # Store nothing extra: recompute from line vs threshold using stake_reason? Better
    # to accept an optional edge. For seeded PT bets we know the line and can
    # use a conservative worst = the logged line when edge is unknown and the
    # edge was only slightly above threshold.
    return _worst_spread_label(r), price_s


def _edge_from_reason_or_none(r) -> float | None:
    # Not stored as a column; PT reason includes the edge figure.
    reason = str(r.get("stake_reason") or "")
    if r["signal"] == "pt_fpi" and "edge" in reason:
        try:
            # "pt_fpi edge 7.31 < 10: 0.25u"
            part = reason.split("edge", 1)[1].strip().split()[0]
            e = float(part)
            return e if str(r["side"]).lower() == "home" else -e
        except (IndexError, ValueError):
            return None
    return None


def set_row_edge(r, edge: float | None) -> float | None:
    return edge


def _worst_spread_label(r, edge: float | None = None) -> str:
    e = edge if edge is not None else _edge_from_reason_or_none(r)
    thr = PT_THRESHOLD if r["signal"] == "pt_fpi" else OPENER_GAP_THRESHOLD
    if e is None:
        # Conservative: do not take a worse number than the one logged.
        shown = float(r["line"]) if str(r["side"]).lower() == "home" else -float(r["line"])
        team = r["home_team"] if str(r["side"]).lower() == "home" else r["away_team"]
        return f"{team} {_fmt_line(shown)}"
    worst_home = worst_acceptable_spread(float(r["line"]), float(e), thr)
    if str(r["side"]).lower() == "home":
        return f"{r['home_team']} {_fmt_line(worst_home)}"
    return f"{r['away_team']} {_fmt_line(-worst_home)}"


def render_slip(df: pd.DataFrame, week: int, season: int = SEASON, edges: dict | None = None) -> str:
    """Plain-text ready-to-place slip. No emojis. Staked bets first, 0u paper last."""
    sub = df[(pd.to_numeric(df.get("season"), errors="coerce") == season)
             & (pd.to_numeric(df.get("week"), errors="coerce") == int(week))].copy() if len(df) else df
    lines = [
        f"Week {int(week)} CFB — ready-to-place slip",
        f"Paper bankroll ${STARTING_BANKROLL:,.0f}. 1u = ${UNIT_USD:.0f} (flat).",
        "",
    ]
    if not len(sub):
        lines.append("No paper bets logged this week.")
        return "\n".join(lines) + "\n"

    def sort_key(r):
        staked = 0 if float(r.get("stake_usd") or 0) > 0 else 1
        kick = str(r.get("kickoff") or "")
        return (staked, kick, str(r.get("bet_id") or ""))

    rows = sorted((r for _, r in sub.iterrows()), key=sort_key)
    staked_rows = [r for r in rows if float(r.get("stake_usd") or 0) > 0]
    paper_n = len(rows) - len(staked_rows)
    any_fallback = False
    any_dk = False
    if not staked_rows:
        lines.append("No staked paper bets this week.")
    for r in staked_rows:
        stake = float(r.get("stake_usd") or 0)
        american = int(r["american"] if pd.notna(r.get("american")) else FALLBACK_AMERICAN)
        label = pick_label(r["home_team"], r["away_team"], r["signal"], r["side"], float(r["line"]))
        e = None
        if edges and r.get("bet_id") in edges:
            e = edges[r["bet_id"]]
        if r["signal"] in TOTAL_SIGNALS or r["market"] == "total":
            worst_line = f"{r['side'].lower()} {_fmt_total(worst_acceptable_total(float(r['line'])))}"
        else:
            worst_line = _worst_spread_label(r, edge=e)
        price_note = f"{american}"
        if _is_true(r.get("fallback")) or r.get("price_source") == "sportsbook_fallback":
            any_fallback = True
            src = "sportsbook fallback"
        else:
            any_dk = True
            src = "DK Predictions"
        paper = "" if stake > 0 else "  [paper, $0]"
        lines.append(
            f"{label}    ${stake:.0f} at {american} ({src})    "
            f"worst acceptable: {worst_line} at {price_note}{paper}"
        )
    if paper_n:
        lines.append("")
        lines.append(f"{paper_n} additional flag(s) are paper only ($0) and are not on this ticket.")
    lines.append("")
    if any_fallback and not any_dk:
        lines.append(
            "Prices: sportsbook snapshot at -110. DraftKings Predictions contract "
            "prices were not publicly reachable from this environment (see docs/paper_bankroll.md)."
        )
    elif any_fallback and any_dk:
        lines.append(
            "Prices: DraftKings Predictions where a public contract was found; "
            "otherwise sportsbook snapshot at -110."
        )
    else:
        lines.append("Prices: DraftKings Predictions, fee-inclusive implied odds.")
    return "\n".join(lines) + "\n"


def write_slip(df: pd.DataFrame, week: int, season: int = SEASON, path: Path | None = None,
               edges: dict | None = None) -> Path:
    text = render_slip(df, week, season=season, edges=edges)
    p = path or (SLIP_DIR / f"slip_{season}_W{int(week):02d}.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def current_week_from_df(df: pd.DataFrame, at=None) -> int | None:
    if not len(df):
        return None
    at = at or now_utc()
    # Prefer a week that still has an open (not settled) bet; else the latest week.
    open_ = df[df["status"].astype(str) != "settled"]
    src = open_ if len(open_) else df
    return int(pd.to_numeric(src["week"], errors="coerce").max())
