"""Best-effort public fetch of DraftKings Predictions prices.

DraftKings Predictions has a public website (https://predictions.draftkings.com)
and no official public developer API. This module only issues unauthenticated
GETs to documented public pages, with a short timeout and no cookies or
login. If the request is blocked, the page has no matching college-football
market, or the response cannot be parsed, it returns None so the caller can
fall back to a sportsbook snapshot at -110.

This environment and typical CI runners are blocked by Akamai (HTTP 403).
Do not add auth, paid third-party feeds, or scrapers that require an account.
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

PUBLIC_ORIGIN = "https://predictions.draftkings.com"
# Paths that search engines index as public football pages. None of these
# require a login in a regular browser; they may still be blocked from
# datacenter IPs.
PUBLIC_PATHS = (
    "/en/football",
    "/en/cfb",
    "/en/ncaaf",
    "/en/college-football",
)
USER_AGENT = "CFB-26-Model paper-bankroll (unauthenticated public GET)"
TIMEOUT_SEC = 8

# Published DKeX / Crypto.com fee schedule, as reported from DraftKings's
# fee disclosure (read by third-party summaries on 11 Nov 2026). Used only
# when a contract price is actually captured. CME markets use a slightly
# different all-in table; we assume DKeX (sports) unless the source says
# otherwise.
# price band start (inclusive) -> fee per contract in dollars.
# The mid-range is a flat $0.02 from $0.30 through $0.94.


def dkex_fee_per_contract(price: float) -> float:
    """Exchange + broker fee per contract on a DKeX sports market."""
    p = float(price)
    if p < 0 or p > 1:
        raise ValueError(f"contract price must be in [0, 1], got {p}")
    if p <= 0.10:
        return 0.0069
    if p < 0.30:
        # Published examples: $0.20 -> $0.0143, $0.29 -> $0.0184.
        # Linear interpolation between $0.10/$0.0069 and $0.30/$0.02.
        return 0.0069 + (p - 0.10) * (0.02 - 0.0069) / 0.20
    if p <= 0.94:
        return 0.02
    if p <= 0.96:
        return 0.0105
    return 0.0045


def american_from_contract(price: float, fee: float | None = None) -> int:
    """Implied American odds for a $1-settlement Yes, including the fee.

    Cost to buy one contract is price + fee. Win pays $1, so decimal odds
    are 1 / (price + fee). That is a more conservative (worse) price than
    the raw contract.
    """
    p = float(price)
    f = dkex_fee_per_contract(p) if fee is None else float(fee)
    cost = p + f
    if cost <= 0 or cost >= 1:
        raise ValueError(f"effective cost {cost} is not a valid probability")
    decimal = 1.0 / cost
    if decimal >= 2:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))


def profit_on_win(stake: float, american: int) -> float:
    """Profit (not including returned stake) on a winning bet of `stake`."""
    a = int(american)
    if a < 0:
        return float(stake) * 100.0 / abs(a)
    return float(stake) * a / 100.0


def try_fetch_public_page(path: str) -> tuple[Optional[str], dict]:
    """GET a public Predictions path. Returns (body or None, meta)."""
    url = PUBLIC_ORIGIN + path
    meta = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": None,
        "error": None,
    }
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=ssl.create_default_context()) as resp:
            meta["http_status"] = getattr(resp, "status", None) or resp.getcode()
            body = resp.read().decode("utf-8", errors="replace")
            return body, meta
    except urllib.error.HTTPError as e:
        meta["http_status"] = e.code
        meta["error"] = f"HTTP {e.code}"
        return None, meta
    except Exception as e:  # timeout, DNS, TLS, etc.
        meta["error"] = f"{type(e).__name__}: {e}"
        return None, meta


def find_cfb_contract(markets: list, home: str, away: str, market: str, line: float, side: str):
    """Match a parsed market list to one game/line/side. No match -> None.

    `markets` items are dicts with keys like event, market ('spread'/'total'),
    line, side, contract_price, american. This is a pure matcher; it does
    not invent prices.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    h, a = norm(home), norm(away)
    want_mkt = "total" if str(market).lower() in ("total", "totals", "under", "over") else "spread"
    want_side = str(side).lower()
    want_line = float(line)
    for m in markets or []:
        ev = norm(str(m.get("event") or ""))
        teams_ok = (h and a and h in ev and a in ev) or (
            norm(str(m.get("home") or "")) == h and norm(str(m.get("away") or "")) == a
        )
        if not teams_ok:
            continue
        if str(m.get("market") or "").lower() not in (want_mkt, str(market).lower()):
            continue
        try:
            if abs(float(m.get("line")) - want_line) > 1e-9:
                continue
        except (TypeError, ValueError):
            continue
        if str(m.get("side") or "").lower() != want_side:
            continue
        if m.get("contract_price") is None and m.get("american") is None:
            continue
        return m
    return None


def lookup_price(home: str, away: str, market: str, line: float, side: str) -> Optional[dict]:
    """Try public pages for one price. Returns a result dict or None.

    On success the dict has: source, url, fetched_at, contract_price (maybe),
    fee_per_contract, american (fee-inclusive), raw_american (if shown).
    """
    notes = []
    for path in PUBLIC_PATHS:
        body, meta = try_fetch_public_page(path)
        notes.append(meta)
        if not body:
            continue
        markets = _extract_markets(body)
        hit = find_cfb_contract(markets, home, away, market, line, side)
        if not hit:
            continue
        price = hit.get("contract_price")
        raw_amer = hit.get("american")
        fee = dkex_fee_per_contract(float(price)) if price is not None else None
        if price is not None:
            american = american_from_contract(float(price), fee)
        elif raw_amer is not None:
            american = int(raw_amer)
            fee = None
        else:
            continue
        return {
            "source": "draftkings_predictions",
            "url": meta["url"],
            "fetched_at": meta["fetched_at"],
            "contract_price": None if price is None else float(price),
            "fee_per_contract": fee,
            "american": int(american),
            "raw_american": None if raw_amer is None else int(raw_amer),
            "fallback": False,
        }
    return None


def _extract_markets(body: str) -> list:
    """Pull any JSON-looking market blobs out of a public HTML page.

    DraftKings Predictions is a JS app; a blocked or empty shell yields [].
    We never invent a market that is not in the page.
    """
    markets = []
    # __NEXT_DATA__ or a similar SSR blob, if the public page ever includes one.
    for m in re.finditer(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', body, re.S | re.I):
        try:
            blob = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        markets.extend(_walk_for_markets(blob))
    return markets


def _walk_for_markets(obj, acc=None) -> list:
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if {"line", "side"} <= keys or {"point", "price"} <= keys:
            acc.append(obj)
        for v in obj.values():
            _walk_for_markets(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_markets(v, acc)
    return acc
