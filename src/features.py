"""
CFB Betting Model — Feature Engineering
=========================================
Transforms raw data into a game-level feature matrix ready for modeling.

Each row in the output = one game, with features for both the home and
away team side by side, plus the Vegas line and actual result.

Run:
    python3 src/features.py

Output:
    data/processed/feature_matrix.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
PROC_DIR  = DATA_DIR / "processed"
RAW_DIR   = DATA_DIR / "raw"

# ─── 1. LOADERS ──────────────────────────────────────────────────────────────

def load_games() -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / "master_games.csv")
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    return df


def load_lines() -> pd.DataFrame:
    """
    Return one line per game using a priority order of providers.
    Prefer 'consensus' → 'Bovada' → first available.
    """
    df = pd.read_csv(PROC_DIR / "master_lines.csv")

    priority = ["consensus", "Bovada", "DraftKings", "ESPN Bet",
                "William Hill (New Jersey)", "teamrankings"]

    # Assign a numeric rank to each provider (lower = preferred)
    rank_map = {p: i for i, p in enumerate(priority)}
    df["_rank"] = df["provider"].map(rank_map).fillna(len(priority))

    # Sort by rank and keep the best line per game
    best = (
        df.sort_values("_rank")
          .drop_duplicates("game_id", keep="first")
          .drop(columns=["_rank"])
          .reset_index(drop=True)
    )

    # Only keep columns that actually exist in the CSV
    want = ["game_id", "provider", "spread", "over_under",
            "spread_open", "over_under_open",
            "home_moneyline", "away_moneyline"]
    keep = [c for c in want if c in best.columns]

    return best[keep].copy()


def load_sp_ratings() -> pd.DataFrame:
    """
    Flatten nested offense/defense dicts and return clean SP+ table.

    IMPORTANT — leakage fix: SP+ ratings are season-FINAL values that
    incorporate all games played that year. To use only genuinely pre-game
    available data, we shift ratings forward by one year: a team's 2023
    final SP+ becomes their pre-season strength estimate for 2024.
    This is standard practice — the prior season's final rating is the
    best publicly available pre-season predictor.
    """
    df = pd.read_csv(PROC_DIR / "master_sp_ratings.csv")

    import ast

    def safe_parse(val):
        if pd.isna(val):
            return {}
        if isinstance(val, dict):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}

    df["off_dict"] = df["offense"].apply(safe_parse)
    df["def_dict"] = df["defense"].apply(safe_parse)

    df["sp_offense"] = df["off_dict"].apply(lambda d: d.get("rating"))
    df["sp_defense"] = df["def_dict"].apply(lambda d: d.get("rating"))

    rename = {"year": "season", "rating": "sp_rating", "sos": "sp_sos"}
    df = df.rename(columns=rename)

    df = df[["season", "team", "sp_rating", "sp_offense",
             "sp_defense", "sp_sos"]].copy()

    # ── Shift ratings forward one year (leakage fix) ──────────────────────
    # Season N final rating → used as pre-season estimate for season N+1.
    # Games in season N itself will get NaN (no prior-season data available),
    # which the model handles via median imputation — acceptable for 2019.
    df["season"] = df["season"] + 1

    return df


def load_fpi_ratings() -> pd.DataFrame:
    """
    Load ESPN FPI ratings with the same +1 year leakage fix as SP+.
    FPI is structured differently by season — we normalise to (season, team, fpi).
    """
    path = PROC_DIR / "master_fpi_ratings.csv"
    if not path.exists():
        print("  ⚠️  FPI ratings not found — run data_collection.py to pull them.")
        return pd.DataFrame(columns=["season", "team", "fpi"])

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Normalise team name column (API sometimes returns 'school' or 'team')
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    # Season column
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})

    # FPI value — API returns 'fpi' directly
    keep_cols = [c for c in ["season", "team", "fpi"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["team"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["fpi"]    = pd.to_numeric(df["fpi"],    errors="coerce")

    # Leakage fix: same shift as SP+
    df["season"] = df["season"] + 1
    return df


def load_srs_ratings() -> pd.DataFrame:
    """
    Load SRS (Simple Rating System) ratings with the same +1 year leakage fix.
    SRS = adjusted point differential per game vs. schedule.
    """
    path = PROC_DIR / "master_srs_ratings.csv"
    if not path.exists():
        print("  ⚠️  SRS ratings not found — run data_collection.py to pull them.")
        return pd.DataFrame(columns=["season", "team", "rating"])

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})

    # SRS value is called 'rating' in the API response
    if "rating" in df.columns:
        df = df.rename(columns={"rating": "srs"})

    keep_cols = [c for c in ["season", "team", "srs"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["team"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "srs" in df.columns:
        df["srs"] = pd.to_numeric(df["srs"], errors="coerce")

    # Leakage fix
    df["season"] = df["season"] + 1
    return df


def load_recruiting() -> pd.DataFrame:
    """
    Build a 4-year rolling average recruiting composite per team per season.
    Recruiting data has a 3–4 year lag before players appear on the field.
    """
    df = pd.read_csv(PROC_DIR / "master_recruiting.csv")

    # Standardise column names (API returns camelCase or snake_case)
    df.columns = [c.lower() for c in df.columns]
    if "points" not in df.columns and "total" in df.columns:
        df = df.rename(columns={"total": "points"})

    # Sort and compute 4-year rolling mean per team
    df = df.sort_values(["team", "year"])
    df["recruiting_4yr"] = (
        df.groupby("team")["points"]
          .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    return df[["year", "team", "recruiting_4yr"]].rename(
        columns={"year": "season"}
    )


def load_ppa_games() -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / "master_ppa_games.csv")


def load_portal_features() -> pd.DataFrame:
    """
    Load pre-computed transfer portal team features.

    If the raw portal CSV exists but the features file doesn't, build it on the fly.
    Returns a DataFrame keyed by (season, team) with columns:
      portal_net_rating, portal_qb_in, portal_qb_out,
      portal_net_count, portal_stars_in_avg, portal_talent_in, portal_talent_out

    NO year-shift needed: portal[year=N] reflects roster changes heading into season N.
    All transfers happen in the Dec-Aug offseason before the fall season starts.
    """
    feat_path = PROC_DIR / "master_portal_features.csv"
    raw_path  = PROC_DIR / "master_portal.csv"

    if feat_path.exists():
        df = pd.read_csv(feat_path)
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        num_cols = ["portal_talent_in", "portal_talent_out", "portal_net_rating",
                    "portal_count_in", "portal_count_out", "portal_net_count",
                    "portal_stars_in_avg", "portal_qb_in", "portal_qb_out"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df

    if raw_path.exists():
        print("  Building portal features from raw data...")
        import sys
        sys.path.insert(0, str(PROC_DIR.parent.parent / "src"))
        from data_collection import build_portal_team_features
        raw = pd.read_csv(raw_path)
        feat = build_portal_team_features(raw)
        if not feat.empty:
            feat.to_csv(feat_path, index=False)
            print(f"  Saved portal features → {feat_path.name}")
        return feat

    print("  ⚠️  No portal data found — run data_collection.py to pull it.")
    print("      Run:  python3 src/data_collection.py  (or refresh_portal_only())")
    return pd.DataFrame()


def load_wepa() -> pd.DataFrame:
    """
    Load WEPA (opponent-adjusted EPA) ratings with +1 year leakage fix.

    WEPA is computed from full-season results, so we shift forward one year:
    a team's 2023 WEPA becomes their pre-season strength estimate for 2024.
    Same logic as SP+ — use prior season's final value as next season's prior.

    wepa_offense: positive = efficient offense vs. good defenses
    wepa_defense: negative = disruptive defense (lower = better)
    """
    path = PROC_DIR / "master_wepa.csv"
    if not path.exists():
        print("  ⚠️  WEPA data not found — run refresh_advanced_stats() in data_collection.py")
        return pd.DataFrame(columns=["season", "team", "wepa_offense", "wepa_defense"])

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})

    keep = [c for c in ["season", "team", "wepa_offense", "wepa_defense",
                        "wepa_success_off", "wepa_success_def",
                        "wepa_explosiveness", "wepa_explosiveness_def"] if c in df.columns]
    df = df[keep].dropna(subset=["team"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    for col in ["wepa_offense", "wepa_defense"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Leakage fix: same shift as SP+
    df["season"] = df["season"] + 1
    print(f"  WEPA rows: {len(df):,} (seasons {df['season'].min():.0f}–{df['season'].max():.0f})")
    return df


def load_talent() -> pd.DataFrame:
    """
    Load team talent composite ratings.

    Unlike SP+ and WEPA, talent IS available pre-season (247Sports publishes roster
    ratings before fall games start). No year-shift needed.

    talent: composite score (higher = more talent on roster; ~600-800 for P5 teams)
    """
    path = PROC_DIR / "master_talent.csv"
    if not path.exists():
        print("  ⚠️  Talent data not found — run refresh_advanced_stats() in data_collection.py")
        return pd.DataFrame(columns=["season", "team", "talent"])

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})

    keep = [c for c in ["season", "team", "talent"] if c in df.columns]
    df = df[keep].dropna(subset=["team"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "talent" in df.columns:
        df["talent"] = pd.to_numeric(df["talent"], errors="coerce")

    # No leakage shift — talent reflects current roster, available pre-season
    print(f"  Talent rows: {len(df):,} (seasons {df['season'].min():.0f}–{df['season'].max():.0f})")
    return df


def load_havoc() -> pd.DataFrame:
    """
    Load advanced team stats (havoc rate, success rates) with +1 year leakage fix.

    havoc_total: % of opponent plays resulting in TFL/sack/FF/PBU — lower is better D
    rush_success_rate / pass_success_rate: offensive efficiency metrics
    All are end-of-season values, so shifted +1 year like SP+.
    """
    path = PROC_DIR / "master_havoc.csv"
    if not path.exists():
        print("  ⚠️  Havoc data not found — run refresh_advanced_stats() in data_collection.py")
        return pd.DataFrame(columns=["season", "team", "havoc_total"])

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})

    havoc_cols = ["havoc_total", "havoc_front_seven", "havoc_db",
                  "rush_success_rate", "pass_success_rate",
                  "explosiveness_off", "explosiveness_off_rush",
                  "explosiveness_off_pass", "explosiveness_def"]
    keep = [c for c in ["season", "team"] + havoc_cols if c in df.columns]
    df = df[keep].dropna(subset=["team"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    for col in havoc_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Leakage fix: same shift as SP+
    df["season"] = df["season"] + 1
    print(f"  Havoc rows: {len(df):,} (seasons {df['season'].min():.0f}–{df['season'].max():.0f})")
    return df


def build_home_field_advantage(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's home field advantage (HFA) estimate from historical data.

    Method: for each team-season, calculate their average home point margin
    minus average away point margin over the PRIOR two seasons.
    This captures venue effects, fan atmosphere, and travel burden.

    Returns a DataFrame keyed by (season, team) with column hfa_estimate.
    A value of +3.5 means that team historically scores ~3.5 more points
    at home than away (vs. equivalent opponents on neutral ground).
    """
    # Build per-game home/away margins.
    # Include neutral_site so we can drop those games — neutral venues don't
    # reflect a true home or away environment and would pollute the HFA estimate.
    ns_col = games["neutral_site"] if "neutral_site" in games.columns else 0

    home = games[["season","home_team","point_diff"]].rename(
        columns={"home_team":"team","point_diff":"margin"}).copy()
    home["is_home"]     = 1
    home["neutral_site"] = ns_col.values

    away = games[["season","away_team","point_diff"]].copy()
    away["margin"]       = -away["point_diff"]   # flip: away margin = -(home margin)
    away = away.rename(columns={"away_team":"team"})
    away["is_home"]      = 0
    away["neutral_site"] = ns_col.values

    all_margins = pd.concat([home, away], ignore_index=True)
    # Filter AFTER concat — neutral_site is now guaranteed to exist
    all_margins = all_margins[all_margins["neutral_site"] == 0]

    # Season-level averages per team
    season_avg = all_margins.groupby(["season","team","is_home"])["margin"].mean().reset_index()
    home_avg = season_avg[season_avg["is_home"]==1][["season","team","margin"]].rename(columns={"margin":"home_margin_avg"})
    away_avg = season_avg[season_avg["is_home"]==0][["season","team","margin"]].rename(columns={"margin":"away_margin_avg"})

    hfa = home_avg.merge(away_avg, on=["season","team"], how="outer")
    hfa["hfa_raw"] = hfa["home_margin_avg"].fillna(0) - hfa["away_margin_avg"].fillna(0)

    # Rolling 2-season average, then shift forward 1 year (no leakage)
    hfa = hfa.sort_values(["team","season"])
    hfa["hfa_estimate"] = (
        hfa.groupby("team")["hfa_raw"]
           .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    )

    return hfa[["season","team","hfa_estimate"]]


def build_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days of rest for each team heading into each game.

    Uses start_date to find each team's previous game within the same season.
    Season openers get a default of 14 days (standard week + bye equivalent).
    Capped at 21 days so late-season extended byes don't dominate.

    Returns a flat DataFrame keyed by (game_id, team).
    """
    df = games[["game_id", "season", "home_team", "away_team", "start_date"]].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)

    # Flatten: one row per team per game
    home = df[["game_id", "season", "home_team", "start_date"]].rename(
        columns={"home_team": "team"})
    away = df[["game_id", "season", "away_team", "start_date"]].rename(
        columns={"away_team": "team"})
    flat = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "start_date"])

    # Previous game date — within the same season only (no cross-season spillover)
    flat["prev_date"] = flat.groupby(["team", "season"])["start_date"].shift(1)
    flat["rest_days"] = (
        (flat["start_date"] - flat["prev_date"]).dt.days
        .fillna(14)       # season opener → treat as standard 2-week rest
        .clip(upper=21)   # cap so extended byes don't outscore everything else
    )

    return flat[["game_id", "team", "rest_days"]]


# ─── 2. ROLLING EPA FEATURES ─────────────────────────────────────────────────

def build_rolling_epa(ppa: pd.DataFrame, windows: list = [3, 5]) -> pd.DataFrame:
    """
    For each team-season, compute rolling EPA averages over the last N games
    (calculated *before* the current game so there's no data leakage).

    Returns a DataFrame keyed by (game_id, team) with rolling EPA columns.
    """
    ppa = ppa.sort_values(["season", "team", "week"]).copy()

    for w in windows:
        for col in ["off_epa", "def_epa", "off_epa_pass", "off_epa_rush"]:
            if col not in ppa.columns:
                continue
            # shift(1) ensures we only use games *before* the current one
            ppa[f"{col}_roll{w}"] = (
                ppa.groupby(["season", "team"])[col]
                   .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    # Also compute season-to-date average (all prior games this season)
    for col in ["off_epa", "def_epa"]:
        if col not in ppa.columns:
            continue
        ppa[f"{col}_ytd"] = (
            ppa.groupby(["season", "team"])[col]
               .transform(lambda x: x.shift(1).expanding().mean())
        )

    return ppa


# ─── 3. ATTACH TEAM FEATURES TO GAMES ────────────────────────────────────────

def attach_team_features(
    games: pd.DataFrame,
    ppa_rolled: pd.DataFrame,
    sp: pd.DataFrame,
    recruiting: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each game, merge in pre-game features for both the home and away team.
    Returns games df with home_* and away_* feature columns appended.
    """

    # ── SP+ (pre-season rating, same value all season) ──
    sp_lookup = sp.set_index(["season", "team"])

    # ── Recruiting (4-year rolling, keyed by season) ──
    rec_lookup = recruiting.set_index(["season", "team"])

    # ── Per-game EPA (keyed by game_id + team) ──
    epa_cols = [c for c in ppa_rolled.columns
                if c not in ("game_id", "season", "week", "team",
                             "conference", "opponent",
                             "off_epa", "def_epa",        # raw values — use rolled versions
                             "off_epa_pass", "def_epa_pass",
                             "off_epa_rush", "def_epa_rush")]
    ppa_lookup = ppa_rolled.set_index(["game_id", "team"])[epa_cols]

    def get_team_features(game_id, season, team, prefix):
        row = {}

        # SP+ features
        try:
            s = sp_lookup.loc[(season, team)]
            row[f"{prefix}sp_rating"]  = s.get("sp_rating")
            row[f"{prefix}sp_offense"] = s.get("sp_offense")
            row[f"{prefix}sp_defense"] = s.get("sp_defense")
            row[f"{prefix}sp_sos"]     = s.get("sp_sos")
        except KeyError:
            row[f"{prefix}sp_rating"]  = np.nan
            row[f"{prefix}sp_offense"] = np.nan
            row[f"{prefix}sp_defense"] = np.nan
            row[f"{prefix}sp_sos"]     = np.nan

        # Recruiting
        try:
            row[f"{prefix}recruiting_4yr"] = rec_lookup.loc[(season, team), "recruiting_4yr"]
        except KeyError:
            row[f"{prefix}recruiting_4yr"] = np.nan

        # Rolling EPA
        try:
            epa = ppa_lookup.loc[(game_id, team)]
            for col in epa_cols:
                row[f"{prefix}{col}"] = epa[col] if not isinstance(epa, pd.DataFrame) else epa[col].iloc[0]
        except KeyError:
            for col in epa_cols:
                row[f"{prefix}{col}"] = np.nan

        return row

    feature_rows = []
    for _, g in games.iterrows():
        home_feats = get_team_features(g["game_id"], g["season"], g["home_team"], "home_")
        away_feats = get_team_features(g["game_id"], g["season"], g["away_team"], "away_")
        feature_rows.append({**home_feats, **away_feats})

    feat_df = pd.DataFrame(feature_rows, index=games.index)
    return pd.concat([games, feat_df], axis=1)


# ─── 4. MERGE BETTING LINES ──────────────────────────────────────────────────

def attach_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """
    Join betting lines onto games. Only keep games that have a line
    (i.e. FBS games that sportsbooks priced).
    """
    merged = games.merge(
        lines[["game_id", "spread", "over_under",
               "spread_open", "over_under_open",
               "home_moneyline", "away_moneyline"]],
        on="game_id",
        how="inner",   # drop games with no line
    )
    return merged


# ─── 5. DERIVED TARGETS & CONTEXT FEATURES ───────────────────────────────────

def add_targets_and_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model target variables and a few context features.
    """
    # Targets
    df["home_win"]         = (df["point_diff"] > 0).astype(int)
    df["covered_spread"]   = (df["point_diff"] + df["spread"].astype(float) > 0).astype(int)
    df["went_over"]        = (df["total_points"] > df["over_under"].astype(float)).astype(int)

    # Context
    df["neutral_site"]     = df["neutral_site"].fillna(False).astype(int)
    df["conference_game"]  = df["conference_game"].fillna(False).astype(int)

    # SP+ differential (home advantage in ratings)
    df["sp_diff"]          = df["home_sp_rating"] - df["away_sp_rating"]
    df["sp_off_diff"]      = df["home_sp_offense"] - df["away_sp_offense"]
    df["sp_def_diff"]      = df["home_sp_defense"] - df["away_sp_defense"]

    # EPA differential (rolling 3-game)
    if "home_off_epa_roll3" in df.columns:
        df["epa_off_diff_roll3"] = df["home_off_epa_roll3"] - df["away_off_epa_roll3"]
        df["epa_def_diff_roll3"] = df["home_def_epa_roll3"] - df["away_def_epa_roll3"]

    # Recruiting differential
    df["recruiting_diff"]  = df["home_recruiting_4yr"] - df["away_recruiting_4yr"]

    # Vegas implied home margin (spread is from home team's perspective: negative = home favored)
    df["vegas_home_margin"] = -df["spread"].astype(float)

    # ── Line movement features ────────────────────────────────────────────────
    # Sign convention: negative movement = line moved toward home team
    #   (home got more expensive, sharp money went to home side)
    # Coverage: opening lines only available for 2023+ (~60% of recent games).
    # For earlier seasons: has_line_data=0 and movement columns stay NaN
    # so LightGBM handles them natively; Ridge uses median imputation.
    if "spread_open" in df.columns:
        spread      = pd.to_numeric(df["spread"],       errors="coerce")
        spread_open = pd.to_numeric(df["spread_open"],  errors="coerce")
        ou          = pd.to_numeric(df.get("over_under"),      errors="coerce")
        ou_open     = pd.to_numeric(df.get("over_under_open"), errors="coerce")

        # Flag: do we actually have opening line data for this game?
        df["has_line_data"] = spread_open.notna().astype(int)

        # Raw movement (NaN when no opener available)
        df["line_movement"]      = spread - spread_open
        df["line_movement_abs"]  = df["line_movement"].abs()

        # Sharp-money flags: 2+ point moves are typically sharp, not public
        # Negative move = home team got more expensive = sharp money on HOME side
        # Positive move = home team got cheaper      = sharp money on AWAY side
        df["sharp_move_home"] = (df["line_movement"] <= -2.0).astype(int)
        df["sharp_move_away"] = (df["line_movement"] >=  2.0).astype(int)

        # Finer-grained direction flags (1+ pt threshold, kept for backwards compat)
        df["line_moved_home"] = (df["line_movement"] < -1.0).astype(int)
        df["line_moved_away"] = (df["line_movement"] >  1.0).astype(int)

        # Opening spread itself — the "sharp-only" estimate before public money
        # Only populated where we have it; NaN elsewhere (handled by imputer/LightGBM)
        df["spread_open_val"] = spread_open   # renamed to avoid collision

        # Totals movement (same logic as spread)
        if ou_open.notna().any():
            df["total_movement"]     = ou - ou_open
            df["total_movement_abs"] = df["total_movement"].abs()
            # Sharp-money flags for totals: 2+ pt move suggests sharp interest
            df["sharp_total_under"] = (df["total_movement"] <= -2.0).astype(int)
            df["sharp_total_over"]  = (df["total_movement"] >=  2.0).astype(int)
    else:
        df["has_line_data"]    = 0
        df["line_movement_abs"] = np.nan
        df["sharp_move_home"]  = 0
        df["sharp_move_away"]  = 0

    return df


# ─── 6. MAIN PIPELINE ────────────────────────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    print("Loading raw data...")
    games      = load_games()
    lines      = load_lines()
    sp         = load_sp_ratings()
    recruiting = load_recruiting()
    ppa        = load_ppa_games()
    fpi        = load_fpi_ratings()
    srs        = load_srs_ratings()
    portal     = load_portal_features()
    wepa       = load_wepa()
    talent     = load_talent()
    havoc      = load_havoc()

    print(f"  Games:       {len(games):,}")
    print(f"  Lines:       {len(lines):,} (one per game)")
    print(f"  SP+ rows:    {len(sp):,}")
    print(f"  Recruiting:  {len(recruiting):,}")
    print(f"  PPA rows:    {len(ppa):,}")
    print(f"  FPI rows:    {len(fpi):,}")
    print(f"  SRS rows:    {len(srs):,}")
    print(f"  Portal rows: {len(portal):,}")
    print(f"  WEPA rows:   {len(wepa):,}")
    print(f"  Talent rows: {len(talent):,}")
    print(f"  Havoc rows:  {len(havoc):,}")

    print("\nBuilding rolling EPA features...")
    ppa_rolled = build_rolling_epa(ppa, windows=[3, 5])

    print("Building home field advantage estimates...")
    hfa = build_home_field_advantage(games)

    print("Attaching team features to games...")
    games_feat = attach_team_features(games, ppa_rolled, sp, recruiting)

    # Merge HFA for home team
    games_feat = games_feat.merge(
        hfa.rename(columns={"team": "home_team", "hfa_estimate": "home_hfa"}),
        on=["season", "home_team"], how="left"
    )
    # Merge HFA for away team (their home-field effect doesn't apply here,
    # but their travel/away comfort does — use negative of their HFA)
    games_feat = games_feat.merge(
        hfa.rename(columns={"team": "away_team", "hfa_estimate": "away_hfa"}),
        on=["season", "away_team"], how="left"
    )
    # Net HFA: home team's advantage minus away team's away-game penalty
    games_feat["hfa_diff"] = games_feat["home_hfa"].fillna(0) - games_feat["away_hfa"].fillna(0)

    # ── Merge FPI (home and away) ──────────────────────────────────────────
    if len(fpi) > 0 and "fpi" in fpi.columns:
        games_feat = games_feat.merge(
            fpi.rename(columns={"team": "home_team", "fpi": "home_fpi"}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            fpi.rename(columns={"team": "away_team", "fpi": "away_fpi"}),
            on=["season", "away_team"], how="left"
        )
        games_feat["fpi_diff"] = games_feat["home_fpi"] - games_feat["away_fpi"]

    # ── Merge SRS (home and away) ──────────────────────────────────────────
    if len(srs) > 0 and "srs" in srs.columns:
        games_feat = games_feat.merge(
            srs.rename(columns={"team": "home_team", "srs": "home_srs"}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            srs.rename(columns={"team": "away_team", "srs": "away_srs"}),
            on=["season", "away_team"], how="left"
        )
        games_feat["srs_diff"] = games_feat["home_srs"] - games_feat["away_srs"]

    # ── Merge WEPA (home and away) ────────────────────────────────────────
    if len(wepa) > 0 and "wepa_offense" in wepa.columns:
        wepa_cols = [c for c in ["wepa_offense", "wepa_defense",
                                  "wepa_success_off", "wepa_success_def",
                                  "wepa_explosiveness", "wepa_explosiveness_def"]
                     if c in wepa.columns]
        games_feat = games_feat.merge(
            wepa[["season", "team"] + wepa_cols].rename(
                columns={"team": "home_team",
                         **{c: f"home_{c}" for c in wepa_cols}}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            wepa[["season", "team"] + wepa_cols].rename(
                columns={"team": "away_team",
                         **{c: f"away_{c}" for c in wepa_cols}}),
            on=["season", "away_team"], how="left"
        )
        if "home_wepa_offense" in games_feat.columns:
            games_feat["wepa_off_diff"] = (
                games_feat["home_wepa_offense"] - games_feat["away_wepa_offense"])
        if "home_wepa_defense" in games_feat.columns:
            games_feat["wepa_def_diff"] = (
                games_feat["home_wepa_defense"] - games_feat["away_wepa_defense"])
        if "home_wepa_success_off" in games_feat.columns:
            games_feat["wepa_success_off_diff"] = (
                games_feat["home_wepa_success_off"] - games_feat["away_wepa_success_off"])
        if "home_wepa_success_def" in games_feat.columns:
            games_feat["wepa_success_def_diff"] = (
                games_feat["home_wepa_success_def"] - games_feat["away_wepa_success_def"])
        if "home_wepa_explosiveness" in games_feat.columns:
            games_feat["wepa_explosiveness_diff"] = (
                games_feat["home_wepa_explosiveness"] - games_feat["away_wepa_explosiveness"])
        cov = games_feat["home_wepa_offense"].notna().mean()
        print(f"  WEPA coverage: {cov:.1%} of games")

    # ── Merge Talent composite (home and away) ────────────────────────────
    if len(talent) > 0 and "talent" in talent.columns:
        games_feat = games_feat.merge(
            talent[["season", "team", "talent"]].rename(
                columns={"team": "home_team", "talent": "home_talent"}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            talent[["season", "team", "talent"]].rename(
                columns={"team": "away_team", "talent": "away_talent"}),
            on=["season", "away_team"], how="left"
        )
        if "home_talent" in games_feat.columns and "away_talent" in games_feat.columns:
            games_feat["talent_diff"] = (
                games_feat["home_talent"] - games_feat["away_talent"])
        cov = games_feat["home_talent"].notna().mean()
        print(f"  Talent coverage: {cov:.1%} of games")

    # ── Merge Havoc & advanced stats (home and away) ──────────────────────
    havoc_stat_cols = ["havoc_total", "havoc_front_seven", "havoc_db",
                       "rush_success_rate", "pass_success_rate",
                       "explosiveness_off", "explosiveness_off_rush",
                       "explosiveness_off_pass", "explosiveness_def"]
    if len(havoc) > 0 and "havoc_total" in havoc.columns:
        avail_havoc = [c for c in havoc_stat_cols if c in havoc.columns]
        games_feat = games_feat.merge(
            havoc[["season", "team"] + avail_havoc].rename(
                columns={"team": "home_team",
                         **{c: f"home_{c}" for c in avail_havoc}}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            havoc[["season", "team"] + avail_havoc].rename(
                columns={"team": "away_team",
                         **{c: f"away_{c}" for c in avail_havoc}}),
            on=["season", "away_team"], how="left"
        )
        if "home_havoc_total" in games_feat.columns:
            games_feat["havoc_diff"] = (
                games_feat["home_havoc_total"] - games_feat["away_havoc_total"])
        if "home_rush_success_rate" in games_feat.columns:
            games_feat["rush_sr_diff"] = (
                games_feat["home_rush_success_rate"] - games_feat["away_rush_success_rate"])
        # Explosiveness differentials
        # Off: home offense big-play rate vs. away offense big-play rate
        # Def: how much big-play EPA each team's defense allows (lower = better)
        # Net: (home_off - away_off) - (home_def_allowed - away_def_allowed)
        if "home_explosiveness_off" in games_feat.columns:
            games_feat["explosiveness_off_diff"] = (
                games_feat["home_explosiveness_off"] - games_feat["away_explosiveness_off"])
        if "home_explosiveness_def" in games_feat.columns:
            games_feat["explosiveness_def_diff"] = (
                games_feat["home_explosiveness_def"] - games_feat["away_explosiveness_def"])
        if ("home_explosiveness_off" in games_feat.columns and
                "home_explosiveness_def" in games_feat.columns):
            # Net explosiveness advantage: home's big-play offense minus their big-play allowed
            # vs. same for away team — positive = home team creates more explosives than it gives up
            home_net = (games_feat["home_explosiveness_off"].fillna(0) -
                        games_feat["home_explosiveness_def"].fillna(0))
            away_net = (games_feat["away_explosiveness_off"].fillna(0) -
                        games_feat["away_explosiveness_def"].fillna(0))
            games_feat["explosiveness_net_diff"] = home_net - away_net
        cov = games_feat["home_havoc_total"].notna().mean()
        exp_cov = games_feat["home_explosiveness_off"].notna().mean() if "home_explosiveness_off" in games_feat.columns else 0
        print(f"  Havoc coverage: {cov:.1%} of games")
        print(f"  Explosiveness coverage: {exp_cov:.1%} of games")

    # ── Merge Transfer Portal features (home and away) ────────────────────
    PORTAL_COLS = ["portal_net_rating", "portal_qb_in", "portal_qb_out",
                   "portal_net_count", "portal_stars_in_avg",
                   "portal_talent_in", "portal_talent_out"]
    if len(portal) > 0 and "portal_net_rating" in portal.columns:
        avail_portal = [c for c in PORTAL_COLS if c in portal.columns]
        games_feat = games_feat.merge(
            portal[["season", "team"] + avail_portal].rename(
                columns={"team": "home_team",
                         **{c: f"home_{c}" for c in avail_portal}}),
            on=["season", "home_team"], how="left"
        )
        games_feat = games_feat.merge(
            portal[["season", "team"] + avail_portal].rename(
                columns={"team": "away_team",
                         **{c: f"away_{c}" for c in avail_portal}}),
            on=["season", "away_team"], how="left"
        )
        # Fill NaN portal values with 0 — no portal activity = no change
        for col in avail_portal:
            for side in ["home_", "away_"]:
                col_name = f"{side}{col}"
                if col_name in games_feat.columns:
                    games_feat[col_name] = games_feat[col_name].fillna(0)
        # Net differential (positive = home team had better portal offseason)
        if "home_portal_net_rating" in games_feat.columns:
            games_feat["portal_net_rating_diff"] = (
                games_feat["home_portal_net_rating"] - games_feat["away_portal_net_rating"]
            )
        print(f"  Portal coverage: {(games_feat['home_portal_net_rating'] != 0).mean():.1%} of games")

    print("Building rest-day features...")
    rest = build_rest_features(games)
    games_feat = games_feat.merge(
        rest.rename(columns={"team": "home_team", "rest_days": "home_rest_days"}),
        on=["game_id", "home_team"], how="left"
    )
    games_feat = games_feat.merge(
        rest.rename(columns={"team": "away_team", "rest_days": "away_rest_days"}),
        on=["game_id", "away_team"], how="left"
    )
    games_feat["rest_diff"] = (
        games_feat["home_rest_days"].fillna(14) - games_feat["away_rest_days"].fillna(14)
    )

    print("Merging betting lines (filtering to FBS games with lines)...")
    games_with_lines = attach_lines(games_feat, lines)
    print(f"  Games with lines: {len(games_with_lines):,}")

    print("Adding targets and context features...")
    final = add_targets_and_context(games_with_lines)

    # Optionally merge weather (run src/weather.py first to generate this file)
    weather_path = PROC_DIR / "game_weather.csv"
    if weather_path.exists():
        print("Merging weather data...")
        weather = pd.read_csv(weather_path)
        want_weather = [c for c in ["game_id", "wind_speed", "temp_avg",
                                     "precipitation", "is_dome"] if c in weather.columns]
        final = final.merge(weather[want_weather], on="game_id", how="left")
        print(f"  Weather coverage: {final['wind_speed'].notna().mean():.1%} of games")
    else:
        print("  (no weather data — run src/weather.py to add wind/temp features)")

    # Drop rows missing spread or total (can't train/backtest without them)
    before = len(final)
    final = final.dropna(subset=["spread", "over_under", "point_diff"])
    print(f"  Dropped {before - len(final)} rows with missing spread/total/score")
    print(f"  Final feature matrix: {len(final):,} rows")

    # Save
    out_path = PROC_DIR / "feature_matrix.csv"
    final.to_csv(out_path, index=False)
    print(f"\n✅ Saved feature matrix → {out_path}")
    print(f"   Columns: {len(final.columns)}")
    print(f"   Rows:    {len(final):,}")
    print(f"\nNext step: run python3 src/elo_ratings.py to build team power ratings.")

    return final


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # First check that per-game EPA data exists — run data_collection.py if not
    ppa_path = PROC_DIR / "master_ppa_games.csv"
    if not ppa_path.exists():
        print("⚠️  Per-game EPA data not found.")
        print("   Run data_collection.py first (it now pulls ppa/games too).")
        import sys; sys.exit(1)

    df = build_feature_matrix()

    # Quick preview
    print("\nSample rows (2024 season):")
    sample_cols = [
        "season", "week", "home_team", "away_team",
        "home_points", "away_points", "point_diff",
        "spread", "over_under",
        "home_sp_rating", "away_sp_rating", "sp_diff",
    ]
    sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[df["season"] == 2024][sample_cols].head(10).to_string(index=False))

    print("\nNull rates for key features:")
    key_feats = [c for c in df.columns if any(x in c for x in
                 ["sp_rating", "epa_roll", "recruiting", "spread", "over_under"])]
    null_rates = df[key_feats].isna().mean().sort_values(ascending=False)
    print(null_rates[null_rates > 0].to_string())
