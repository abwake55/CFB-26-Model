"""Timestamped reports: retain publication AND collection times for replay."""
from pathlib import Path
import json
import pandas as pd
from decision_quality import ROOT, utc, append_record


def ingest(record, directory=None, now=None):
    now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    required = ("game_id", "kind", "source_url", "published_at", "payload")
    if any(key not in record for key in required):
        raise ValueError("Required: " + ", ".join(required))
    if record["kind"] not in ("availability", "weather", "analyst"):
        raise ValueError("Unsupported context kind")
    if not str(record["source_url"]).startswith("https://"):
        raise ValueError("An HTTPS source URL is required")
    published = utc(record["published_at"])
    if published is None or published > now:
        raise ValueError("Publication time must be known and no later than collection")
    p = record["payload"]
    if record["kind"] == "availability":
        # Explicit review includes both teams; missing is never interpreted as healthy.
        for side in ("home", "away"):
            for field in ("qb_status", "ol_status", "defense_status"):
                if p.get(f"{side}_{field}") not in ("available", "out", "limited", "questionable", "unknown"):
                    raise ValueError(f"Missing/invalid {side}_{field}")
    if record["kind"] == "analyst":
        from decision_quality import payout, finite
        for field in ("analyst", "market", "side", "odds", "book", "rationale"):
            if field not in p:
                raise ValueError("Analyst record missing " + field)
        if p["market"] not in ("totals", "spreads", "h2h"):
            raise ValueError("Unknown market")
        if p["market"] != "h2h" and not finite(p.get("line")):
            raise ValueError("Spread/total needs exact line")
        payout(p["odds"])
    record = {**record, "observed_at": now.isoformat()}
    return append_record(directory or ROOT / "data/context", record)


def asof(game_id, cutoff, directory=None):
    cutoff = utc(cutoff)
    if cutoff is None:
        raise ValueError("A decision cutoff is required")
    records = []
    for path in Path(directory or ROOT / "data/context").glob("*.json"):
        r = json.loads(path.read_text())
        published, observed = utc(r.get("published_at")), utc(r.get("observed_at"))
        if (str(r.get("game_id")) == str(game_id) and published is not None and observed is not None
                and published <= cutoff and observed <= cutoff):
            records.append(r)
    return sorted(records, key=lambda r: (r["published_at"], r["observed_at"]))


def attach_context(frame, now=None):
    out = frame.copy()
    now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    for idx, row in out.iterrows():
        kickoff = utc(row.get("start_date"))
        cutoff = min(now, kickoff) if kickoff is not None else now
        for r in asof(row["game_id"], cutoff):
            if r["kind"] == "weather":
                for key in ("wind_speed", "temp_avg", "precipitation", "is_dome"):
                    out.loc[idx, key] = r["payload"].get(key)
                out.loc[idx, "weather_source"] = r["source_url"]
                out.loc[idx, "weather_observed_at"] = r["observed_at"]
            elif r["kind"] == "availability":
                p = r["payload"]
                # Any unresolved injury gets review, not an invented points adjustment.
                out.loc[idx, "availability_status"] = "reviewed" if all(
                    p.get(f"{side}_{field}") == "available" for side in ("home", "away")
                    for field in ("qb_status", "ol_status", "defense_status")) else "needs_review"
                out.loc[idx, "availability_observed_at"] = r["observed_at"]
    return out
