"""
Re-pull havoc + explosiveness for all seasons and save master_havoc.csv.
Run: python3 scripts/refresh_havoc.py
"""
import sys, os, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load API key from secrets.toml if not already in environment
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

from data_collection import collect_havoc
import pandas as pd

PROC_DIR = ROOT / "data" / "processed"
seasons  = list(range(2015, 2026))

all_havoc = []
for season in seasons:
    df = collect_havoc(season)
    if not df.empty:
        all_havoc.append(df)
    time.sleep(0.3)

if all_havoc:
    master = pd.concat(all_havoc, ignore_index=True)
    master.to_csv(PROC_DIR / "master_havoc.csv", index=False)
    print(f"\n✅ Saved master_havoc.csv — {len(master):,} rows")
    print(f"   Columns: {list(master.columns)}")
    if "explosiveness_off" in master.columns:
        cov = master["explosiveness_off"].notna().mean()
        print(f"   Explosiveness coverage: {cov:.1%}")
        sample = (master[master["explosiveness_off"].notna()]
                  [["season", "team", "explosiveness_off", "explosiveness_def"]]
                  .head(8))
        print(f"\nSample rows:\n{sample.to_string(index=False)}")
else:
    print("❌ No havoc data returned — check API key.")
