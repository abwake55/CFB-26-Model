"""Quick check: are 2026 Week 1 games and lines available in the CFBD API?"""
import sys, os, re, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Read API key from secrets.toml
secrets_path = __import__("pathlib").Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
api_key = ""
if secrets_path.exists():
    txt = secrets_path.read_text()
    m = re.search(r'CFB_API_KEY\s*=\s*["\']([^"\']+)', txt)
    if m:
        api_key = m.group(1)
if not api_key:
    api_key = os.getenv("CFB_API_KEY", "")

if not api_key:
    print("ERROR: No CFB_API_KEY found")
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}"}
BASE = "https://api.collegefootballdata.com"

# 1. Check games
r = requests.get(f"{BASE}/games", params={"year": 2026, "week": 1, "seasonType": "regular"}, headers=headers)
games = r.json() if r.ok else []
print(f"\n=== Week 1 2026 GAMES: {len(games)} found ===")
for g in games[:5]:
    print(f"  {g.get('away_team')} @ {g.get('home_team')} | {g.get('start_date','?')[:10]}")
if len(games) > 5:
    print(f"  ... and {len(games)-5} more")

# 2. Check lines
r2 = requests.get(f"{BASE}/lines", params={"year": 2026, "week": 1, "seasonType": "regular"}, headers=headers)
lines_data = r2.json() if r2.ok else []
print(f"\n=== Week 1 2026 LINES: {len(lines_data)} games with line data ===")
games_with_lines = 0
for g in lines_data:
    ls = g.get("lines", [])
    if ls:
        games_with_lines += 1
        spread = ls[0].get("spread", "n/a")
        ou = ls[0].get("overUnder", "n/a")
        print(f"  {g.get('awayTeam')} @ {g.get('homeTeam')}: spread={spread}, O/U={ou} ({len(ls)} books)")
print(f"\nGames with at least one line: {games_with_lines} / {len(lines_data)}")

# 3. Check if schedule (future games) endpoint has week 1
r3 = requests.get(f"{BASE}/calendar", params={"year": 2026}, headers=headers)
if r3.ok:
    cal = r3.json()
    wk1 = [w for w in cal if w.get("week") == 1]
    if wk1:
        print(f"\n=== Calendar: Week 1 window ===")
        print(f"  First game: {wk1[0].get('firstGameStart', '?')[:10]}")
        print(f"  Last game:  {wk1[0].get('lastGameStart', '?')[:10]}")
