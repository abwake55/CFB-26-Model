"""Prediction Tracker (thepredictiontracker.com) per-system CFB lines.

Todd Beck's Prediction Tracker publishes, free, every tracked computer
system's predicted line for every FBS game plus the opening, midweek and
updated market lines:

  * ``ncaaYYYY.csv``       – a season's completed games (history, 2015+)
  * ``ncaapredictions.csv`` – the current week's upcoming games (live;
                              replaced weekly, usually by Tuesday)

PT's lines are the predicted HOME margin (home favoured = positive). This
module converts everything to the repo's home-spread convention
(home favoured = negative, i.e. ``spread = -margin``) so it lines up with
``spread``/``spread_open``/``close_line`` everywhere else.

Outputs (``build_history``):
  data/external/prediction_tracker/ncaaYYYY.csv  raw season files
  data/external/prediction_tracker/team_map.csv  PT name -> CFBD name
  data/processed/pt_lines.csv                    one row per CFBD game_id

Point-in-time: PT records each system's line before kickoff (collected
midweek), so every pt_* column is pre-game information. The market columns
(pt_open, pt_midweek, pt_updated) are PT's own record of the market.
"""
from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "external" / "prediction_tracker"
PROCESSED = ROOT / "data" / "processed" / "pt_lines.csv"
TEAM_MAP = RAW_DIR / "team_map.csv"
BASE_URL = "https://www.thepredictiontracker.com"
FIRST_SEASON = 2015
UA = {"User-Agent": "CFB-26-Model research (github.com/abwake55/CFB-26-Model)"}

# market / aggregate columns that are NOT independent computer systems
NON_SYSTEM = {"line", "lineopen", "linemidweek", "lineavg", "linemedian",
              "linestd", "lineca", "linel2hf"}
# canonical names for the systems we use directly (columns were renamed over
# the years: lineespn -> linefpi, linemass -> linemassey)
ALIASES = {"lineespn": "fpi", "linefpi": "fpi",
           "linemass": "massey", "linemassey": "massey",
           "linesag": "sagarin", "linefei": "fei",
           "lineopen": "open", "linemidweek": "midweek", "line": "updated",
           "lineca": "comp_adj"}
PT_COLS = ["pt_open", "pt_midweek", "pt_updated", "pt_fpi", "pt_sagarin",
           "pt_massey", "pt_fei", "pt_comp_adj", "pt_sys_median", "pt_sys_mean",
           "pt_sys_std", "pt_sys_n"]


# ─── fetch ───────────────────────────────────────────────────────────────────

def _get_csv(name: str, timeout: int = 30) -> pd.DataFrame:
    r = requests.get(f"{BASE_URL}/{name}", headers=UA, timeout=timeout)
    r.raise_for_status()
    if not r.text.lstrip().lower().startswith(("home", "lineopen", "line", "road")):
        raise ValueError(f"{name}: unexpected body (not a CSV)")
    return pd.read_csv(io.StringIO(r.text))


def fetch_season(season: int, save: bool = True) -> pd.DataFrame:
    df = _get_csv(f"ncaa{season}.csv")
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / f"ncaa{season}.csv", index=False)
    return df


def fetch_live(save: bool = True) -> pd.DataFrame:
    df = _get_csv("ncaapredictions.csv")
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / "ncaapredictions_latest.csv", index=False)
    return df


# ─── normalise ───────────────────────────────────────────────────────────────

def norm_name(s) -> str:
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"\bst\.?$", "state", s)
    s = re.sub(r"\bst\.", "state", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw PT frame -> home/away names plus pt_* columns in home-spread
    convention (negative = home favoured)."""
    cols = {c.lower(): c for c in raw.columns}
    home = raw[cols.get("home")]
    away = raw[cols.get("road")]
    line_cols = [c for c in raw.columns if c.lower().startswith("line")]
    num = raw[line_cols].apply(pd.to_numeric, errors="coerce")
    num.columns = [c.lower() for c in line_cols]
    out = pd.DataFrame({"pt_home": home.astype(str), "pt_away": away.astype(str)})
    for src, dst in ALIASES.items():
        if src in num.columns:
            out[f"pt_{dst}"] = -num[src]
    systems = [c for c in num.columns if c not in NON_SYSTEM]
    sys_lines = -num[systems]
    out["pt_sys_median"] = sys_lines.median(axis=1)
    out["pt_sys_mean"] = sys_lines.mean(axis=1)
    out["pt_sys_std"] = sys_lines.std(axis=1)
    out["pt_sys_n"] = sys_lines.notna().sum(axis=1)
    for c in PT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = qc_market_signs(out)
    for c in ("hscore", "vscore", "week", "neutral"):
        if c in cols:
            out[f"pt_{c}"] = pd.to_numeric(raw[cols[c]], errors="coerce")
    return out


def qc_market_signs(out: pd.DataFrame, far: float = 14.0, near: float = 7.0) -> pd.DataFrame:
    """PT's market columns occasionally carry a flipped sign (e.g. 2019 wk13
    pt_updated). Blank a market value that is far from the system median but
    close to its negation; system lines are left alone."""
    out = out.copy()
    med = out["pt_sys_median"]
    for c in ("pt_open", "pt_midweek", "pt_updated", "pt_comp_adj"):
        if c in out.columns:
            bad = ((out[c] - med).abs() > far) & ((-out[c] - med).abs() < near)
            out.loc[bad, c] = np.nan
    return out


# ─── match to CFBD games ─────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# names PT uses that fuzzy matching cannot resolve
MANUAL_MAP = {"Central Florida": "UCF", "Miami-Florida": "Miami", "Miami-Ohio": "Miami (OH)",
              "Southern California": "USC", "Louisiana State": "LSU",
              "Texas Christian": "TCU", "Brigham Young": "BYU", "Mississippi": "Ole Miss",
              "Texas-San Antonio": "UTSA", "Texas-El Paso": "UTEP",
              "Alabama-Birmingham": "UAB", "Nevada-Las Vegas": "UNLV",
              "Southern Methodist": "SMU", "Louisiana-Monroe": "UL Monroe"}


def load_team_map() -> dict:
    m = dict(MANUAL_MAP)
    if TEAM_MAP.exists():
        t = pd.read_csv(TEAM_MAP)
        m.update(dict(zip(t.pt_name, t.cfbd_name)))
    return m


def _swap(row: pd.Series) -> pd.Series:
    row = row.copy()
    for c in PT_COLS:
        if c != "pt_sys_std" and c != "pt_sys_n" and pd.notna(row.get(c)):
            row[c] = -row[c]
    row["pt_home"], row["pt_away"] = row["pt_away"], row["pt_home"]
    return row


def match_completed(pt: pd.DataFrame, games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Match a season's PT rows to CFBD games by final score + team names.
    games: CFBD rows with game_id, season, home_team, away_team, home_points,
    away_points. Returns pt_* keyed by game_id."""
    g = games[games["season"] == season].dropna(subset=["home_points", "away_points"])
    by_score: dict = {}
    for gi, r in g.iterrows():
        k = (int(r.home_points), int(r.away_points))
        by_score.setdefault(k, []).append(gi)
        by_score.setdefault(k[::-1], []).append(gi)
    rows = []
    for _, p in pt.iterrows():
        if pd.isna(p.get("pt_hscore")) or pd.isna(p.get("pt_vscore")):
            continue
        ph, pa = norm_name(p.pt_home), norm_name(p.pt_away)
        best = None
        for gi in by_score.get((int(p.pt_hscore), int(p.pt_vscore)), []):
            gh, ga = norm_name(g.at[gi, "home_team"]), norm_name(g.at[gi, "away_team"])
            same = _sim(ph, gh) + _sim(pa, ga)
            swap = _sim(ph, ga) + _sim(pa, gh)
            if int(p.pt_hscore) != int(g.at[gi, "home_points"]):
                same = -1  # scores only fit the swapped orientation
            if int(p.pt_hscore) != int(g.at[gi, "away_points"]):
                swap = -1
            sc, sw = max((same, False), (swap, True))
            if best is None or sc > best[0]:
                best = (sc, gi, sw)
        if best and best[0] >= 1.3:
            row = _swap(p) if best[2] else p.copy()
            row["game_id"] = g.at[best[1], "game_id"]
            row["cfbd_home"] = g.at[best[1], "home_team"]
            row["cfbd_away"] = g.at[best[1], "away_team"]
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["season"] = season
    return out.drop_duplicates("game_id")


def match_upcoming(pt: pd.DataFrame, games: pd.DataFrame, team_map: dict | None = None,
                   min_score: float = 1.5) -> pd.DataFrame:
    """Match the live predictions file to this week's CFBD schedule by names
    (team_map first, fuzzy fallback; either orientation)."""
    team_map = team_map if team_map is not None else load_team_map()
    gm = games.copy()
    gm["_h"] = gm.home_team.map(norm_name)
    gm["_a"] = gm.away_team.map(norm_name)
    rows = []
    for _, p in pt.iterrows():
        ph = norm_name(team_map.get(p.pt_home, p.pt_home))
        pa = norm_name(team_map.get(p.pt_away, p.pt_away))
        best = None
        for gi, r in gm.iterrows():
            for sw, (h, a) in ((False, (r._h, r._a)), (True, (r._a, r._h))):
                sc = (2.0 if ph == h else _sim(ph, h)) + (2.0 if pa == a else _sim(pa, a))
                if best is None or sc > best[0]:
                    best = (sc, gi, sw)
        if best and best[0] >= min_score:
            row = _swap(p) if best[2] else p.copy()
            row["game_id"] = gm.at[best[1], "game_id"]
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["game_id"] + PT_COLS)
    return out.drop_duplicates("game_id", keep="first")


def build_history(seasons=None, fetch: bool = True) -> pd.DataFrame:
    """(Re)build data/processed/pt_lines.csv and the team map."""
    games = pd.read_csv(ROOT / "data" / "processed" / "master_games.csv", low_memory=False,
                        usecols=["game_id", "season", "home_team", "away_team",
                                 "home_points", "away_points"]).drop_duplicates("game_id")
    seasons = seasons or sorted(int(s) for s in games.season.unique() if s >= FIRST_SEASON)
    parts = []
    for s in seasons:
        path = RAW_DIR / f"ncaa{s}.csv"
        if fetch or not path.exists():
            try:
                raw = fetch_season(s)
            except Exception as exc:  # keep any cached copy
                print(f"[pt] {s}: fetch failed ({type(exc).__name__}: {exc})")
                if not path.exists():
                    continue
                raw = pd.read_csv(path)
        else:
            raw = pd.read_csv(path)
        m = match_completed(to_features(raw), games, s)
        print(f"[pt] {s}: {len(raw)} PT rows, {len(m)} matched to CFBD games")
        parts.append(m)
    hist = pd.concat(parts, ignore_index=True)
    if PROCESSED.exists():  # keep seasons we did not rebuild
        old = pd.read_csv(PROCESSED)
        hist = pd.concat([old[~old.season.isin(seasons)], hist], ignore_index=True)
    keep = ["game_id", "season", "pt_week", "pt_home", "pt_away"] + PT_COLS
    hist = hist[[c for c in keep if c in hist.columns]].sort_values(["season", "game_id"])
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(PROCESSED, index=False)
    # team map from matched rows (orientation-corrected)
    if parts:
        mp = pd.concat(parts, ignore_index=True)
        pairs = pd.concat([mp[["pt_home", "cfbd_home"]].set_axis(["pt_name", "cfbd_name"], axis=1),
                           mp[["pt_away", "cfbd_away"]].set_axis(["pt_name", "cfbd_name"], axis=1)])
        tm = (pairs.groupby(["pt_name", "cfbd_name"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates("pt_name"))
        if TEAM_MAP.exists():
            old = pd.read_csv(TEAM_MAP)
            tm = pd.concat([tm, old[~old.pt_name.isin(tm.pt_name)]], ignore_index=True)
        tm.sort_values("pt_name").to_csv(TEAM_MAP, index=False)
    return hist


def live_features(games: pd.DataFrame, fetch: bool = True) -> pd.DataFrame:
    """pt_* for this week's schedule from the live predictions file (empty
    frame when PT has not posted the week or the fetch fails)."""
    try:
        raw = fetch_live(save=True) if fetch else pd.read_csv(RAW_DIR / "ncaapredictions_latest.csv")
    except Exception as exc:
        print(f"[pt] live predictions unavailable ({type(exc).__name__})")
        return pd.DataFrame(columns=["game_id"] + PT_COLS)
    m = match_upcoming(to_features(raw), games)
    return m[["game_id"] + PT_COLS]


# ─── live signal (frozen on dev 2019-23 by scripts/research/pt_signals.py) ───

RULE_PATH = ROOT / "models" / "pt_rule.json"
EDGE_DEFS = {
    # home-margin points vs the opener; > 0 -> bet home
    "fpi_gap": lambda o: o["pt_opener"] - o["pt_fpi"],
    "consensus_gap": lambda o: o["pt_opener"] - o["pt_sys_median"],
}


def load_rule() -> dict | None:
    import json
    return json.loads(RULE_PATH.read_text()) if RULE_PATH.exists() else None


def attach(out: pd.DataFrame, pt_live: pd.DataFrame | None = None, fetch: bool = True) -> pd.DataFrame:
    """Add pt_* columns, pt_edge and flag_pt to a predictions frame.

    The rule is the frozen one in models/pt_rule.json. The decision line is
    the opener (``spread_open``, else PT's own opener); the line a bet would
    actually be placed at is the CURRENT ``spread`` when the flag is seen,
    which the ledger stores as pt_bet_line / pt_bet_ts."""
    rule = load_rule()
    out = out.copy()
    if pt_live is None:
        sched = out[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
        pt_live = live_features(sched, fetch=fetch)
    out = out.drop(columns=[c for c in PT_COLS if c in out.columns])
    out = out.merge(pt_live[["game_id"] + PT_COLS], on="game_id", how="left")
    so = pd.to_numeric(out.get("spread_open"), errors="coerce")
    out["pt_opener"] = so.where(so.notna(), out["pt_open"])
    if rule is None or rule.get("rule") not in EDGE_DEFS:
        out["pt_edge"], out["flag_pt"] = np.nan, False
        return out
    out["pt_edge"] = EDGE_DEFS[rule["rule"]](out)
    out["flag_pt"] = out["pt_edge"].abs() >= float(rule["thr"])
    out["pt_rule"] = f"{rule['rule']}>={rule['thr']:g}"
    return out
