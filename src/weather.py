"""
CFB Weather Enrichment
=======================
Fetches historical weather for each outdoor game using the free OpenMeteo
archive API. No API key required.

Strategy: group games by home team, fetch a full season's daily weather
for each venue in one API call, then join by game date. This keeps total
requests to ~130 teams × 11 seasons ≈ 1,400 calls instead of 8,000+.

Run:
    python3 src/weather.py

Output:
    data/processed/game_weather.csv  — wind_speed, temp_avg, precipitation,
                                        is_dome per game_id
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROC_DIR = DATA_DIR / "processed"

# ─── STADIUM LOOKUP ──────────────────────────────────────────────────────────
# (lat, lon, is_dome)  — is_dome=True means weather has no effect on scoring

TEAM_VENUES = {
    "Air Force":          (38.9969, -104.8406, False),
    "Akron":              (41.0748,  -81.5130, False),
    "Alabama":            (33.2098,  -87.5503, False),
    "Appalachian State":  (36.2157,  -81.6746, False),
    "Arizona":            (32.2286, -110.9488, False),
    "Arizona State":      (33.4255, -111.9329, False),
    "Arkansas":           (36.0684,  -94.1756, False),
    "Arkansas State":     (35.8318,  -90.6709, False),
    "Army":               (41.3912,  -73.9547, False),
    "Auburn":             (32.6013,  -85.4913, False),
    "Ball State":         (40.2731,  -85.4097, False),
    "Baylor":             (31.5594,  -97.1258, False),
    "Boise State":        (43.6018, -116.2004, False),
    "Boston College":     (42.3358,  -71.1687, False),
    "Bowling Green":      (41.3745,  -83.6524, False),
    "Buffalo":            (42.9956,  -78.7811, False),
    "BYU":                (40.2574, -111.6546, False),
    "California":         (37.8720, -122.2508, False),
    "Central Michigan":   (43.5789,  -84.7742, False),
    "Charlotte":          (35.3097,  -80.7453, False),
    "Cincinnati":         (39.1451,  -84.5139, False),
    "Clemson":            (34.6785,  -82.8437, False),
    "Coastal Carolina":   (33.7793,  -79.0193, False),
    "Colorado":           (40.0097, -105.2669, False),
    "Colorado State":     (40.5955, -105.0831, False),
    "Connecticut":        (41.7681,  -72.6936, False),
    "Duke":               (36.0014,  -78.9378, False),
    "East Carolina":      (35.6052,  -77.3662, False),
    "Eastern Michigan":   (42.2536,  -83.6219, False),
    "FIU":                (25.7574,  -80.3728, False),
    "Florida":            (29.6499,  -82.3486, False),
    "Florida Atlantic":   (26.3764,  -80.1026, False),
    "Florida State":      (30.4396,  -84.3041, False),
    "Fresno State":       (36.8143, -119.7474, False),
    "Georgia":            (33.9497,  -83.3733, False),
    "Georgia Southern":   (32.0835,  -81.8915, False),
    "Georgia State":      (33.7552,  -84.4012, False),
    "Georgia Tech":       (33.7729,  -84.3921, False),
    "Hawaii":             (21.2969, -157.8587, False),
    "Hawai'i":            (21.2969, -157.8587, False),
    "Houston":            (29.7207,  -95.4102, False),
    "Illinois":           (40.0999,  -88.2353, False),
    "Indiana":            (39.1840,  -86.5257, False),
    "Iowa":               (41.6584,  -91.5508, False),
    "Iowa State":         (42.0140,  -93.6357, False),
    "Jacksonville State": (33.8175,  -85.7648, False),
    "James Madison":      (38.4359,  -78.8690, False),
    "Kansas":             (38.9634,  -95.2524, False),
    "Kansas State":       (39.1989,  -96.5981, False),
    "Kennesaw State":     (34.0233,  -84.5816, False),
    "Kent State":         (41.1545,  -81.3417, False),
    "Kentucky":           (38.0220,  -84.5054, False),
    "Liberty":            (37.3531,  -79.1718, False),
    "Louisiana":          (30.2127,  -92.0193, False),
    "Louisiana Monroe":   (32.5320,  -92.0779, False),
    "Louisiana Tech":     (32.5293,  -93.7404, False),
    "Louisville":         (38.2531,  -85.7601, False),
    "LSU":                (30.4121,  -91.1837, False),
    "Marshall":           (38.4198,  -82.4453, False),
    "Maryland":           (38.9896,  -76.9479, False),
    "Memphis":            (35.1167,  -89.9370, False),
    "Miami":              (25.9580,  -80.2388, False),
    "Miami (OH)":         (39.5124,  -84.7344, False),
    "Michigan":           (42.2659,  -83.7485, False),
    "Michigan State":     (42.7278,  -84.4800, False),
    "Middle Tennessee":   (35.8456,  -86.3697, False),
    "Minnesota":          (44.9537,  -93.2231, False),
    "Mississippi State":  (33.4552,  -88.7887, False),
    "Missouri":           (38.9353,  -92.3334, False),
    "Navy":               (38.9887,  -76.4762, False),
    "Nebraska":           (40.8209,  -96.7052, False),
    "Nevada":             (39.5499, -119.8185, False),
    "New Mexico":         (35.0844, -106.6504, False),
    "New Mexico State":   (32.3085, -106.7711, False),
    "North Carolina":     (35.9046,  -79.0469, False),
    "NC State":           (35.7698,  -78.6757, False),
    "North Texas":        (33.2149,  -97.1478, False),
    "Northern Illinois":  (41.9340,  -88.7731, False),
    "Northwestern":       (42.0591,  -87.6726, False),
    "Notre Dame":         (41.6985,  -86.2338, False),
    "Ohio":               (39.3238,  -82.1046, False),
    "Ohio State":         (40.0016,  -83.0196, False),
    "Oklahoma":           (35.2059,  -97.4455, False),
    "Oklahoma State":     (36.1253,  -97.0681, False),
    "Old Dominion":       (36.8862,  -76.3059, False),
    "Ole Miss":           (34.3604,  -89.5391, False),
    "Oregon":             (44.0566, -123.0688, False),
    "Oregon State":       (44.5634, -123.2784, False),
    "Penn State":         (40.7990,  -77.8591, False),
    "Pittsburgh":         (40.4440,  -79.9589, False),
    "Purdue":             (40.4594,  -86.9981, False),
    "Rice":               (29.7165,  -95.4065, False),
    "Rutgers":            (40.5202,  -74.4374, False),
    "Sam Houston":        (30.7110,  -95.5477, False),
    "Sam Houston State":  (30.7110,  -95.5477, False),
    "San Diego State":    (32.7831, -117.1197, False),
    "San Jose State":     (37.3334, -121.9000, False),
    "SMU":                (32.8416,  -96.7843, False),
    "South Alabama":      (30.6954,  -88.1702, False),
    "South Carolina":     (33.9905,  -81.0213, False),
    "South Florida":      (28.0655,  -82.4152, False),
    "Southern Miss":      (31.3279,  -89.3273, False),
    "Stanford":           (37.4346, -122.1609, False),
    "Syracuse":           (43.0366,  -76.1364, True),   # JMA Wireless Dome
    "TCU":                (32.7103,  -97.2882, False),
    "Temple":             (39.9007,  -75.1675, False),
    "Tennessee":          (35.9548,  -83.9252, False),
    "Texas":              (30.2840,  -97.7326, False),
    "Texas A&M":          (30.6096,  -96.3407, False),
    "Texas State":        (29.8900,  -97.9403, False),
    "Texas Tech":         (33.5906, -101.8734, False),
    "Toledo":             (41.6586,  -83.6028, False),
    "Troy":               (31.8085,  -85.9706, False),
    "Tulane":             (29.9511,  -90.1210, False),
    "Tulsa":              (36.1549,  -95.9929, False),
    "UAB":                (33.5013,  -86.8104, False),
    "UCF":                (28.6025,  -81.1888, False),
    "UCLA":               (34.1614, -118.1685, False),
    "UMass":              (42.3868,  -72.5301, False),
    "UNLV":               (36.0903, -115.1831, True),   # Allegiant Stadium
    "USC":                (34.0141, -118.2879, False),
    "Utah":               (40.7600, -111.8480, False),
    "Utah State":         (41.7555, -111.8128, False),
    "UTEP":               (31.7719, -106.5042, False),
    "UTSA":               (29.5851,  -98.6196, False),
    "Vanderbilt":         (36.1447,  -86.8072, False),
    "Virginia":           (38.0297,  -78.5095, False),
    "Virginia Tech":      (37.2241,  -80.4179, False),
    "Wake Forest":        (36.1337,  -80.2599, False),
    "Washington":         (47.6499, -122.3018, False),
    "Washington State":   (46.7278, -117.1552, False),
    "West Virginia":      (39.6498,  -79.9536, False),
    "Western Kentucky":   (36.9728,  -86.4749, False),
    "Western Michigan":   (42.2942,  -85.5908, False),
    "Wisconsin":          (43.0701,  -89.4125, False),
    "Wyoming":            (41.3143, -105.5660, False),
}


# ─── OPENMETEO FETCH ─────────────────────────────────────────────────────────

def fetch_season_weather(lat: float, lon: float, season: int) -> pd.DataFrame:
    """
    Pull daily max wind speed, avg temperature, and precipitation for an
    entire CFB season (Aug 24 – Jan 15) at a given latitude/longitude.

    Returns a DataFrame indexed by date with columns:
      wind_speed, temp_avg, precipitation
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":           lat,
        "longitude":          lon,
        "start_date":         f"{season}-08-24",
        "end_date":           f"{season + 1}-01-15",
        "daily":              [
            "wind_speed_10m_max",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "timezone":           "auto",
        "wind_speed_unit":    "mph",
        "temperature_unit":   "fahrenheit",
        "precipitation_unit": "inch",
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})

    dates = daily.get("time", [])
    wind  = daily.get("wind_speed_10m_max", [None] * len(dates))
    tmax  = daily.get("temperature_2m_max",  [None] * len(dates))
    tmin  = daily.get("temperature_2m_min",  [None] * len(dates))
    precip = daily.get("precipitation_sum",  [None] * len(dates))

    df = pd.DataFrame({
        "date":          dates,
        "wind_speed":    wind,
        "temp_max":      tmax,
        "temp_min":      tmin,
        "precipitation": precip,
    })
    df["date"]     = pd.to_datetime(df["date"])
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    return df.set_index("date")[["wind_speed", "temp_avg", "precipitation"]]


# ─── MAIN ────────────────────────────────────────────────────────────────────

def build_game_weather():
    """
    For every game in the feature matrix, attach weather data.
    Dome games get wind_speed=0, is_dome=1.
    Outdoor games get weather from OpenMeteo for the home venue.
    Neutral site games are treated as outdoor with the home team's typical climate.
    """
    fm_path = PROC_DIR / "feature_matrix.csv"
    if not fm_path.exists():
        print("feature_matrix.csv not found — run features.py first.")
        return

    print("Loading feature matrix...")
    fm = pd.read_csv(fm_path, usecols=["game_id", "season", "home_team",
                                        "neutral_site", "start_date"])
    fm["game_date"] = pd.to_datetime(fm["start_date"], utc=True).dt.date
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # Check for existing results (to allow resuming a partial run)
    out_path = PROC_DIR / "game_weather.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_ids = set(existing["game_id"])
        print(f"  Resuming — {len(done_ids):,} games already have weather.")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    remaining = fm[~fm["game_id"].isin(done_ids)].copy()
    print(f"  Games needing weather: {len(remaining):,}")

    if len(remaining) == 0:
        print("All games already have weather data.")
        return

    # Build (team, season) pairs to fetch in bulk
    team_seasons = (
        remaining[["home_team", "season"]]
        .drop_duplicates()
        .values.tolist()
    )

    # Cache: (team, season) → daily weather DataFrame
    weather_cache: dict = {}
    records = []
    errors  = 0

    print(f"\nFetching weather for {len(team_seasons)} team-seasons "
          f"(~{len(team_seasons)} API calls)...")

    for i, (team, season) in enumerate(team_seasons):
        venue = TEAM_VENUES.get(team)
        if venue is None:
            continue  # skip teams not in our lookup

        lat, lon, is_dome = venue
        key = (lat, lon, season)

        if key not in weather_cache:
            try:
                weather_cache[key] = fetch_season_weather(lat, lon, season)
                if i % 50 == 0:
                    print(f"  [{i+1}/{len(team_seasons)}] {team} {season} ✓")
            except Exception as e:
                print(f"  ⚠️  {team} {season}: {e}")
                errors += 1
                weather_cache[key] = None
            time.sleep(0.12)  # ~8 requests/sec — well within free tier limits

    print(f"\nBuilding per-game weather records...")
    for _, row in remaining.iterrows():
        team   = row["home_team"]
        season = row["season"]
        gdate  = row["game_date"]
        venue  = TEAM_VENUES.get(team)

        rec = {"game_id": row["game_id"]}

        if venue is None:
            # Unknown venue — leave as NaN (imputed in model)
            rec.update({"wind_speed": np.nan, "temp_avg": np.nan,
                         "precipitation": np.nan, "is_dome": 0})
        elif venue[2]:
            # Dome stadium — weather irrelevant
            rec.update({"wind_speed": 0.0, "temp_avg": 68.0,
                         "precipitation": 0.0, "is_dome": 1})
        else:
            lat, lon, _ = venue
            key = (lat, lon, season)
            daily = weather_cache.get(key)

            if daily is not None and gdate in daily.index:
                w = daily.loc[gdate]
                rec.update({
                    "wind_speed":    w["wind_speed"],
                    "temp_avg":      w["temp_avg"],
                    "precipitation": w["precipitation"],
                    "is_dome":       0,
                })
            else:
                rec.update({"wind_speed": np.nan, "temp_avg": np.nan,
                             "precipitation": np.nan, "is_dome": 0})

        records.append(rec)

    new_df = pd.DataFrame(records)

    # Combine with any previously fetched data
    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates("game_id").reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    print(f"\n✅ Saved weather for {len(combined):,} games → {out_path}")
    print(f"   Coverage: {combined['wind_speed'].notna().mean():.1%} of games")
    if errors:
        print(f"   ⚠️  {errors} team-seasons had API errors (those games get NaN)")

    # Quick summary
    outdoor = combined[combined["is_dome"] == 0]
    print(f"\nOutdoor games summary:")
    print(f"  Avg wind speed:      {outdoor['wind_speed'].mean():.1f} mph")
    print(f"  High wind (>15 mph): {(outdoor['wind_speed'] > 15).mean():.1%} of games")
    print(f"  Avg temperature:     {outdoor['temp_avg'].mean():.1f}°F")
    print(f"  Cold games (<40°F):  {(outdoor['temp_avg'] < 40).mean():.1%} of games")
    print(f"\nNext step: re-run python3 src/features.py to merge weather into feature matrix.")


if __name__ == "__main__":
    build_game_weather()
