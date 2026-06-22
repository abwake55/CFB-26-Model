"""
Stadium coordinates for FBS programs.
Used to compute travel distance (away team home → game venue).

Coordinates are approximate lat/lon of each team's home stadium.
Unknown teams get a fallback of (None, None) and travel features are NaN.

Priority:
  1. data/processed/team_locations_cfbd.json  (CFBD API — run fetch_team_locations.py)
  2. STADIUM_COORDS dict below                (hardcoded fallback ~130 teams)

Run `python3 scripts/fetch_team_locations.py` to refresh the API cache.
"""

import json
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# ── Load CFBD API cache (if it exists) ───────────────────────────────────────

def _load_cfbd_cache() -> dict[str, tuple[float, float]]:
    """Load team → (lat, lon) from the CFBD-fetched JSON cache."""
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "team_locations_cfbd.json"
    if not cache_path.exists():
        return {}
    try:
        raw: dict = json.loads(cache_path.read_text())
        return {team: tuple(coords) for team, coords in raw.items()
                if isinstance(coords, list) and len(coords) == 2}
    except Exception:
        return {}

_CFBD_COORDS: dict[str, tuple[float, float]] = _load_cfbd_cache()

# team name → (latitude, longitude)
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    # ── SEC ───────────────────────────────────────────────────────────────────
    "Alabama":          (33.2082, -87.5503),
    "Auburn":           (32.6030, -85.4895),
    "Georgia":          (33.9497, -83.3733),
    "Florida":          (29.6499, -82.3485),
    "Tennessee":        (35.9550, -83.9250),
    "LSU":              (30.4121, -91.1837),
    "Ole Miss":         (34.3628, -89.5414),
    "Mississippi State":(33.4527, -88.7892),
    "Arkansas":         (36.0685, -94.1761),
    "South Carolina":   (33.9719, -81.0720),
    "Kentucky":         (38.0219, -84.5057),
    "Missouri":         (38.9359, -92.3330),
    "Texas A&M":        (30.6100, -96.3408),
    "Texas":            (30.2838, -97.7324),
    "Oklahoma":         (35.2059, -97.4426),
    "Vanderbilt":       (36.1447, -86.8140),

    # ── Big Ten ───────────────────────────────────────────────────────────────
    "Michigan":         (42.2659, -83.7490),
    "Ohio State":       (40.0014, -83.0196),
    "Penn State":       (40.8120, -77.8561),
    "Iowa":             (41.6584, -91.5507),
    "Nebraska":         (40.8202, -96.7055),
    "Wisconsin":        (43.0704, -89.4117),
    "Michigan State":   (42.7271, -84.4826),
    "Minnesota":        (44.9775, -93.2277),
    "Indiana":          (39.1773, -86.5244),
    "Illinois":         (40.0979, -88.2355),
    "Purdue":           (40.4533, -86.9201),
    "Maryland":         (38.9872, -76.9456),
    "Rutgers":          (40.5265, -74.4593),
    "Northwestern":     (42.0585, -87.6672),
    "Oregon":           (44.0568, -123.0677),
    "Washington":       (47.6506, -122.3018),
    "USC":              (34.0141, -118.2879),
    "UCLA":             (34.1613, -118.1676),

    # ── Big 12 ────────────────────────────────────────────────────────────────
    "TCU":              (32.7357, -97.2827),
    "Baylor":           (31.5614, -97.1141),
    "Kansas":           (38.9577, -95.2426),
    "Kansas State":     (39.1956, -96.5831),
    "Iowa State":       (42.0140, -93.6358),
    "West Virginia":    (39.6500, -79.9545),
    "Oklahoma State":   (36.1226, -97.0673),
    "Texas Tech":       (33.5908, -101.8713),
    "Cincinnati":       (39.1301, -84.5155),
    "Houston":          (29.7218, -95.4089),
    "Arizona":          (32.2289, -110.9488),
    "Arizona State":    (33.4264, -111.9327),
    "Colorado":         (40.0095, -105.2680),
    "Utah":             (40.7597, -111.8472),
    "BYU":              (40.2571, -111.6536),
    "UCF":              (28.6017, -81.1924),

    # ── ACC ───────────────────────────────────────────────────────────────────
    "Clemson":          (34.6782, -82.8438),
    "Florida State":    (30.4382, -84.3066),
    "North Carolina":   (35.9053, -79.0470),
    "NC State":         (35.8011, -78.7220),
    "Virginia":         (38.0310, -78.5097),
    "Virginia Tech":    (37.2209, -80.4176),
    "Miami":            (25.9580, -80.2389),
    "Louisville":       (38.2156, -85.7671),
    "Pittsburgh":       (40.4468, -80.0157),
    "Syracuse":         (43.0360, -76.1368),
    "Duke":             (36.0014, -78.9399),
    "Wake Forest":      (36.1328, -80.2640),
    "Georgia Tech":     (33.7731, -84.3918),
    "Boston College":   (42.3356, -71.1675),
    "California":       (37.8689, -122.2506),
    "Stanford":         (37.4346, -122.1609),
    "SMU":              (32.8378, -96.7841),

    # ── Mountain West ─────────────────────────────────────────────────────────
    "Boise State":      (43.6037, -116.2014),
    "Colorado State":   (40.5769, -105.0825),
    "Nevada":           (39.5413, -119.8113),
    "UNLV":             (36.0909, -115.1833),
    "Wyoming":          (41.3141, -105.5906),
    "Air Force":        (38.9962, -104.8544),
    "San Diego State":  (32.7741, -117.1210),
    "Fresno State":     (36.8105, -119.7471),
    "Utah State":       (41.7428, -111.8176),
    "New Mexico":       (35.0845, -106.6213),
    "Hawai'i":          (21.2967, -157.8142),
    "Hawaii":           (21.2967, -157.8142),
    "San José State":   (37.3429, -121.9139),
    "San Jose State":   (37.3429, -121.9139),

    # ── American Athletic ─────────────────────────────────────────────────────
    "Memphis":          (35.1138, -90.0029),
    "Tulsa":            (36.1383, -95.9407),
    "South Florida":    (27.9759, -82.5033),
    "East Carolina":    (35.6059, -77.3662),
    "Navy":             (38.9768, -76.4846),
    "Army":             (41.3903, -73.9561),
    "Tulane":           (29.9382, -90.1280),
    "Temple":           (39.9008, -75.1675),
    "UTSA":             (29.4241, -98.4648),
    "North Texas":      (33.2134, -97.1573),

    # ── Sun Belt ──────────────────────────────────────────────────────────────
    "Louisiana":        (30.2119, -92.0194),
    "Appalachian State":(36.2089, -81.6847),
    "Troy":             (31.7826, -85.9731),
    "Georgia Southern": (32.4173, -81.7738),
    "Georgia State":    (33.7353, -84.3882),
    "Arkansas State":   (35.8350, -90.7021),
    "UL Monroe":        (32.5254, -92.0774),
    "South Alabama":    (30.6953, -88.0981),
    "Texas State":      (29.8930, -97.9415),
    "Marshall":         (38.4200, -82.4400),
    "Old Dominion":     (36.8944, -76.3074),
    "James Madison":    (38.4412, -78.8785),
    "Southern Miss":    (31.3302, -89.3232),
    "Coastal Carolina": (33.8167, -79.0108),

    # ── MAC ───────────────────────────────────────────────────────────────────
    "Western Michigan": (42.2968, -85.5890),
    "Eastern Michigan": (42.2390, -83.6298),
    "Central Michigan": (43.5993, -84.7892),
    "Northern Illinois":(41.9278, -88.7680),
    "Ball State":       (40.1990, -85.3841),
    "Bowling Green":    (41.3773, -83.6398),
    "Buffalo":          (42.9634, -78.7900),
    "Akron":            (41.0854, -81.5270),
    "Ohio":             (39.3277, -82.1032),
    "Kent State":       (41.1535, -81.3490),
    "Miami (OH)":       (39.5134, -84.7367),
    "Toledo":           (41.6609, -83.6111),

    # ── C-USA / Independents ──────────────────────────────────────────────────
    "Western Kentucky": (36.9679, -86.4726),
    "Florida Atlantic": (26.3757, -80.1043),
    "Middle Tennessee": (35.8468, -86.3606),
    "Charlotte":        (35.3071, -80.7459),
    "FIU":              (25.7599, -80.3737),
    "UAB":              (33.5227, -86.8155),
    "UTEP":             (31.7727, -106.5029),
    "Rice":             (29.7165, -95.4096),
    "Jacksonville State":(33.8166, -85.7666),
    "Liberty":          (37.3530, -79.1697),
    "Sam Houston":      (30.7180, -95.5543),
    "New Mexico State": (32.2962, -106.7597),
    "Connecticut":      (41.8084, -72.2547),
    "UMass":            (42.3898, -72.5319),
    "Notre Dame":       (41.6973, -86.2336),
    "Kennesaw State":   (34.0404, -84.5758),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_coords(team: str) -> tuple[float, float] | tuple[None, None]:
    """
    Return (lat, lon) for a team, or (None, None) if unknown.
    Checks CFBD API cache first, then falls back to hardcoded STADIUM_COORDS.
    """
    if team in _CFBD_COORDS:
        return _CFBD_COORDS[team]
    return STADIUM_COORDS.get(team, (None, None))


def travel_miles(origin_team: str, dest_team: str) -> float | None:
    """
    Distance in miles from origin_team's stadium to dest_team's stadium.
    Returns None if either team is unknown.
    """
    lat1, lon1 = get_coords(origin_team)
    lat2, lon2 = get_coords(dest_team)
    if lat1 is None or lat2 is None:
        return None
    return haversine_miles(lat1, lon1, lat2, lon2)
