"""
Quick diagnostic — prints the actual field names returned by the API
so we can fix the data collection script.
"""
import requests, json

API_KEY = "uxvnvwwBh6dQBE/hxA+GK+srmnfZ1mkRSr8E7gOg/BuIL/TeNHw5aHbbZDbi4TMt"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Fetch 3 games from 2024 to inspect the response shape
r = requests.get(
    "https://api.collegefootballdata.com/games",
    headers=headers,
    params={"year": 2024, "seasonType": "regular", "week": 1}
)
r.raise_for_status()
games = r.json()

print(f"Got {len(games)} games\n")
if games:
    print("Fields in a game record:")
    for k, v in games[0].items():
        print(f"  {k!r}: {v!r}")
