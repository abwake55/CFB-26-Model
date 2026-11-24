"""One-off: fold legacy one-file-per-record archives into daily JSONL shards.

    python scripts/migrate_archives.py          # migrate + delete legacy files
    python scripts/migrate_archives.py --check  # report counts only

Each legacy file is named by the SHA-256 of its canonical JSON; that digest
is kept as ``_id`` so write-once de-duplication keeps working. Records are
sharded by their own observation/decision date (UTC).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORES = ["data/lines_snapshots/quotes", "outputs/predictions/decisions",
          "outputs/predictions/market_decisions", "data/context"]


def record_day(r: dict) -> str:
    for v in (r.get("observed_at"), (r.get("decision") or {}).get("decision_at")):
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return "legacy"


def migrate(store: Path, check: bool) -> int:
    files = sorted(store.glob("*.json"))
    shards = defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        shards[record_day(r)].append({"_id": f.stem, **r})
    if check:
        print(f"{store.relative_to(ROOT)}: {len(files)} files -> {len(shards)} shards")
        return len(files)
    for day, rows in shards.items():
        path = store / f"{day}.jsonl"
        existing = set()
        if path.exists():
            existing = {json.loads(l)["_id"] for l in path.read_text().splitlines() if l.strip()}
        with path.open("a") as out:
            for row in sorted(rows, key=lambda x: x["_id"]):
                if row["_id"] not in existing:
                    out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    for f in files:
        f.unlink()
    print(f"{store.relative_to(ROOT)}: migrated {len(files)} files into {len(shards)} shards")
    return len(files)


if __name__ == "__main__":
    check = "--check" in sys.argv
    for s in STORES:
        if (ROOT / s).exists():
            migrate(ROOT / s, check)
