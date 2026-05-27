"""
Probe the CFBD /stats/season/advanced endpoint to see what
opponent-adjusted fields are available and their exact structure.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

if not os.getenv("CFB_API_KEY"):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        os.environ["CFB_API_KEY"] = secrets.get("CFB_API_KEY", "")

from data_collection import cfb_get
import pandas as pd

# Pull one season to inspect structure
data = cfb_get("stats/season/advanced", params={"year": 2024, "excludeGarbageTime": "true"})
df = pd.DataFrame(data)

print("Columns returned:", list(df.columns))
print()

# Show one row fully expanded
sample = data[0]
print("Sample team:", sample.get("team") or sample.get("school"))
for key, val in sample.items():
    if isinstance(val, dict):
        print(f"\n  {key}:")
        for k2, v2 in val.items():
            if isinstance(v2, dict):
                print(f"    {k2}: {json.dumps(v2)}")
            else:
                print(f"    {k2}: {v2}")
    else:
        print(f"  {key}: {val}")
