"""Price-specific research decisions. New probability estimates stay paper-only."""
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICY = {"version": "decision_v3", "max_quote_minutes": 60,
          "max_context_hours": 24, "minimum_ev": .02,
          "minimum_samples": 300, "live_probability_approved": False}


def utc(value):
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else result


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def payout(odds):
    if not finite(odds) or abs(float(odds)) < 100:
        raise ValueError("American odds must be finite and at least 100 in magnitude")
    return float(odds) / 100 if float(odds) > 0 else 100 / abs(float(odds))


def ev(probabilities, odds):
    return probabilities["win"] * payout(odds) - probabilities["loss"]


def probabilities(actual, predicted, projection, line, side):
    """Empirical OOS errors, rounded to football's integer score lattice.

    Only call with forecasts made without access to their game's outcome.
    Integer lines have explicit push mass; half-point lines cannot push.
    """
    if side not in ("under", "over", "home", "away"):
        raise ValueError("Unknown side")
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(a) & np.isfinite(p)
    if not valid.any() or not finite(projection) or not finite(line):
        raise ValueError("Finite projection, line and calibration games required")
    scores = np.rint(float(projection) + a[valid] - p[valid])
    if side in ("under", "over"):
        scores = np.maximum(scores, 0)
    delta = scores - float(line)
    if side in ("under", "away"):
        delta = -delta
    return {"win": float(np.mean(delta > 0)), "loss": float(np.mean(delta < 0)),
            "push": float(np.mean(delta == 0)), "samples": int(valid.sum())}


def _tokens(name):
    return re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).split()


def match_quote_side(side_name, home_team, away_team):
    """Map a sportsbook spread quote's side (a team name like
    "Ohio State Buckeyes") to "home" or "away". CFBD names are normally a
    token prefix of the book's name; spelling variants fall back to a fuzzy
    ratio. Returns None when neither team matches."""
    quote_tokens = _tokens(side_name)
    if not quote_tokens:
        return None
    best_label, best_score = None, 0.0
    for label, team in (("home", home_team), ("away", away_team)):
        team_tokens = _tokens(team)
        if not team_tokens:
            continue
        if len(team_tokens) <= len(quote_tokens) and quote_tokens[:len(team_tokens)] == team_tokens:
            score = 1.0 + 0.1 * len(team_tokens)
        else:
            ratio = SequenceMatcher(None, " ".join(team_tokens), " ".join(quote_tokens)).ratio()
            score = ratio if ratio >= 0.72 else 0.0
        if score > best_score:
            best_label, best_score = label, score
    return best_label


def spread_decision(row, history, now=None):
    """Price-specific spread review. Spreads are PAPER-only on every surface
    (gates.is_play("spread") is False), so this never approves a live play and
    always assigns zero units.

    Evaluates fresh sportsbook spread quotes: each quote is matched to the
    home or away team, cover/push/loss probabilities come from the empirical
    out-of-sample point_diff vs pred_spread residuals on football's integer
    score lattice, and EV is computed at the quote's actual American odds.
    """
    now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    reasons = []
    result = {"decision_status": "PASS", "units": 0, "market": "spreads",
              "policy_version": POLICY["version"],
              "decision_at": now.isoformat(), "decision_reasons": reasons}
    home_team, away_team = row.get("home_team"), row.get("away_team")
    kickoff = utc(row.get("start_date"))
    if kickoff is None or kickoff <= now:
        reasons.append("missing_or_started_kickoff")
    if history.empty:
        reasons.append("no_prior_oos_calibration")
        return result
    season = row.get("season", now.year)
    hist = history[pd.to_numeric(history.season, errors="coerce") < int(season)]
    if "evaluation_version" not in hist or not hist.evaluation_version.eq("season_holdout_v2").all():
        reasons.append("unverified_calibration_provenance")
        return result
    if not finite(row.get("pred_spread")):
        reasons.append("missing_model_spread")
        return result
    quotes = row.get("book_quotes", [])
    if not isinstance(quotes, list):
        quotes = []
    evaluated = []
    for q in quotes:
        if q.get("market") != "spreads":
            continue
        side = match_quote_side(q.get("side"), home_team, away_team)
        if side is None:
            continue
        if not q.get("book") or not fresh(q.get("updated_at"), now, POLICY["max_quote_minutes"] / 60):
            continue
        if not fresh(q.get("observed_at"), now, POLICY["max_quote_minutes"] / 60):
            continue
        if not finite(q.get("line")):
            continue
        try:
            probs = probabilities(hist.point_diff, hist.pred_spread,
                                  row["pred_spread"], q["line"], side)
        except (ValueError, TypeError, KeyError):
            continue
        if probs["samples"] < POLICY["minimum_samples"]:
            continue
        try:
            expected = ev(probs, q["odds"])
        except (ValueError, TypeError):
            continue
        evaluated.append({"quote": q, "side": side, "probabilities": probs,
                          "ev": expected})
    if not evaluated:
        reasons.append("no_fresh_priced_spread_quote")
        return result
    best = max(evaluated, key=lambda x: x["ev"])
    result.update(best)
    result["quote_candidates"] = evaluated
    if best["ev"] < POLICY["minimum_ev"]:
        reasons.append("insufficient_price_specific_ev")
    if not POLICY["live_probability_approved"]:
        reasons.append("probability_model_pending_prospective_validation")
    # Paper-only: a spread can never be APPROVED, and units stay 0.
    result["decision_status"] = ("PAPER"
                                 if reasons == ["probability_model_pending_prospective_validation"]
                                 else "REVIEW")
    return result


def append_record(directory, record):
    """Content-addressed, write-once JSON; repeated snapshots never overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, sort_keys=True, allow_nan=False, default=str)
    digest = hashlib.sha256(data.encode()).hexdigest()
    path = directory / (digest + ".json")
    try:
        with path.open("x") as f:
            f.write(data + "\n")
    except FileExistsError:
        if path.read_text().strip() != data:
            raise ValueError("Snapshot collision")
    return digest


def model_fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*")):
        if path.is_file() and path.suffix in (".pkl", ".json"):
            digest.update(path.name.encode())
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def fresh(stamp, now, hours):
    t = utc(stamp)
    return t is not None and 0 <= (now - t).total_seconds() <= hours * 3600


def total_decision(row, history, now=None):
    from gates import core_candidate
    now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    reasons = []
    result = {"decision_status": "PASS", "units": 0, "policy_version": POLICY["version"],
              "decision_at": now.isoformat(), "decision_reasons": reasons}
    if not core_candidate(row):
        reasons.append("outside_core_candidate_rule")
        return result
    result["decision_status"] = "REVIEW"
    kickoff = utc(row.get("start_date"))
    if kickoff is None or kickoff <= now:
        reasons.append("missing_or_started_kickoff")
    dome = row.get("is_dome") in (True, 1, "1")
    weather_ok = (row.get("weather_source") and fresh(row.get("weather_observed_at"), now, 24)
                  and (dome or (finite(row.get("wind_speed")) and 0 <= float(row["wind_speed"]) < 15)))
    if not weather_ok:
        reasons.append("weather_missing_stale_or_high_wind")
    if row.get("availability_status") != "reviewed" or not fresh(row.get("availability_observed_at"), now, 24):
        reasons.append("availability_review_required")
    if history.empty:
        reasons.append("no_prior_oos_calibration")
        return result
    season = row.get("season", now.year)
    hist = history[pd.to_numeric(history.season, errors="coerce") < int(season)]
    if "evaluation_version" not in hist or not hist.evaluation_version.eq("season_holdout_v2").all():
        reasons.append("unverified_calibration_provenance")
        return result
    quotes = row.get("book_quotes", [])
    if not isinstance(quotes, list):
        quotes = []
    evaluated = []
    for q in quotes:
        if q.get("market") != "totals" or q.get("side") != "Under":
            continue
        if not q.get("book") or not fresh(q.get("updated_at"), now, POLICY["max_quote_minutes"] / 60):
            continue
        if not fresh(q.get("observed_at"), now, POLICY["max_quote_minutes"] / 60):
            continue
        try:
            probs = probabilities(hist.total_points, hist.pred_total, row["pred_total"], q["line"], "under")
            expected = ev(probs, q["odds"])
        except (ValueError, TypeError, KeyError):
            continue
        if probs["samples"] < POLICY["minimum_samples"]:
            continue
        # Each price gets its own acceptable minimum total. Never compare bare lines.
        acceptable = [line / 2 for line in range(0, 301)
                      if ev(probabilities(hist.total_points, hist.pred_total, row["pred_total"], line / 2, "under"), q["odds"]) >= POLICY["minimum_ev"]]
        required_payout = (probs["loss"] + POLICY["minimum_ev"]) / probs["win"] if probs["win"] else None
        worst_odds = (100 * required_payout if required_payout >= 1 else -100 / required_payout) if required_payout else None
        evaluated.append({"quote": q, "probabilities": probs, "ev": expected,
                          "minimum_total_at_quoted_odds": min(acceptable) if acceptable else None,
                          "worst_american_odds_at_quoted_line": worst_odds})
    if not evaluated:
        reasons.append("no_fresh_priced_book_quote")
        return result
    best = max(evaluated, key=lambda x: x["ev"])
    result.update(best)
    result["quote_candidates"] = evaluated
    if best["ev"] < POLICY["minimum_ev"]:
        reasons.append("insufficient_price_specific_ev")
    if not POLICY["live_probability_approved"]:
        reasons.append("probability_model_pending_prospective_validation")
    result["decision_status"] = "PAPER" if reasons == ["probability_model_pending_prospective_validation"] else "REVIEW"
    return result


def attach_decisions(frame, model_dir=None, history=None, now=None, archive=False):
    if frame.empty:
        return frame
    if history is None:
        path = ROOT / "outputs/predictions/walk_forward_results.csv"
        history = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fingerprint = model_fingerprint(model_dir or ROOT / "models")
    records = []
    for _, row in frame.iterrows():
        result = total_decision(row, history, now)
        result["model_fingerprint"] = fingerprint
        # Price-specific spread review rides along with every attached
        # decision: paper-only, zero units, never APPROVED.
        result["spread_decision"] = spread_decision(row, history, now)
        records.append(result)
        if archive:
            clean_row = json.loads(row.to_json())
            append_record(ROOT / "outputs/predictions/decisions", {"prediction": clean_row, "decision": result})
    out = frame.copy()
    out["decision"] = records
    out["decision_status"] = [r["decision_status"] for r in records]
    out["recommended_units"] = 0
    return out
