"""
Probe CFBD /stats/season to understand turnover field names.
Run: /opt/homebrew/bin/python3 scripts/probe_turnovers.py
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load API key
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import toml as tomllib
        tomllib.load = lambda f: toml.load(f)

secrets_path = ROOT / ".streamlit" / "secrets.toml"
if secrets_path.exists():
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    os.environ["CFB_API_KEY"] = secrets.get("CFB_API_KEY", "")

from data_collection import cfb_get

print("=== /stats/season field names for 2024 ===")
data = cfb_get("stats/season", params={"year": 2024})
print(f"Total rows: {len(data)}")
names = sorted(set(row.get("statName", "") for row in data))
print("All stat names:", names)

print("\n=== Turnover-related stats for Alabama ===")
for row in data:
    stat = row.get("statName", "")
    if row.get("team") == "Alabama" and any(k in stat.lower() for k in
            ["turnover", "fumble", "intercept", "takeaway", "giveaway"]):
        print(f"  {stat}: {row.get('statValue')}")

print("\n=== /stats/season/advanced — checking for turnover fields ===")
adv = cfb_get("stats/season/advanced", params={"year": 2024, "excludeGarbageTime": "true"})
import pandas as pd
df = pd.DataFrame(adv)
print("Columns:", list(df.columns))
if "offense" in df.columns:
    sample = df[df.get("school", df.get("team", pd.Series())) == "Alabama"]["offense"]
    if not sample.empty:
        val = sample.iloc[0]
        if isinstance(val, dict):
            print("offense keys:", sorted(val.keys()))
if "defense" in df.columns:
    sample = df[df.get("school", df.get("team", pd.Series())) == "Alabama"]["defense"]
    if not sample.empty:
        val = sample.iloc[0]
        if isinstance(val, dict):
            print("defense keys:", sorted(val.keys()))
