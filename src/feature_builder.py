"""
CFB Betting Model — Shared Feature Builder
============================================
Single source of truth for loading team ratings and building game-level
feature vectors.  Used by:

  • src/predict.py   — CLI predictor
  • app.py           — Streamlit app

Having one module here means training (features.py) and live prediction
always use the same loading logic, so feature drift is impossible.

Public API
----------
    load_rating_sources(pred_season, data_dir)  → dict
    load_recent_epa(pred_season, data_dir)      → pd.DataFrame
    attach_team_features(df, ratings, epa, elo) → pd.DataFrame
    feature_coverage_report(df)                  → dict

All functions are pure Python / pandas — no Streamlit imports.
Callers can wrap load_rating_sources with @st.cache_data as needed.
"""

import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── FEATURE GROUP DEFINITIONS ────────────────────────────────────────────────
# Used by feature_coverage_report() to bucket columns into human-readable groups.

FEATURE_GROUPS = {
    "SP+":        ["home_sp_rating",     "away_sp_rating"],
    "FPI":        ["home_fpi",           "away_fpi"],
    "SRS":        ["home_srs",           "away_srs"],
    "Elo":        ["home_pregame_elo",   "away_pregame_elo"],
    "Recruiting": ["home_recruiting_4yr","away_recruiting_4yr"],
    "Portal":     ["home_portal_net_rating","away_portal_net_rating"],
    "WEPA":       ["home_wepa_offense",  "away_wepa_offense"],
    "Talent":     ["home_talent",        "away_talent"],
    "Havoc":      ["home_havoc_total",   "away_havoc_total"],
    "EPA (roll3)":["home_off_epa_roll3", "away_off_epa_roll3"],
    "Lines":      ["spread",             "over_under"],
    "Line Move":  ["spread_open",        "line_movement"],
}

PORTAL_FEAT_COLS = [
    "portal_net_rating", "portal_qb_in", "portal_qb_out",
    "portal_net_count", "portal_stars_in_avg",
    "portal_talent_in", "portal_talent_out",
]

WEPA_COLS = [
    "wepa_offense", "wepa_defense",
    "wepa_success_off", "wepa_success_def",
    "wepa_explosiveness", "wepa_explosiveness_def",
]

HAVOC_COLS = [
    "havoc_total", "havoc_front_seven", "havoc_db",
    "rush_success_rate", "pass_success_rate",
    # Extra advanced-stat columns needed by the trained models' derived
    # features (explosiveness_net_diff, tempo_combined, turnover_margin_diff,
    # points_per_opp_combined, …) — must match what features.py merges in
    # training, otherwise these are silently NaN at prediction time.
    "explosiveness_off", "explosiveness_def",
    "plays_per_drive", "rush_rate",
    "points_per_opp", "def_points_per_opp",
    "turnovers_off_pg", "turnovers_def_pg", "turnover_margin",
]


# ─── 1. RATING SOURCE LOADERS ─────────────────────────────────────────────────

def _safe_parse_sp_dict(val):
    """Parse SP+ offense/defense dict stored as a string in the CSV."""
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}


def load_rating_sources(pred_season: int, data_dir: Path) -> dict:
    """
    Load all per-team rating sources for *pred_season*.

    Returns a dict keyed by source name.  Each value is a DataFrame
    indexed by team name, with one or more rating columns.

    Sources loaded (if files exist):
        "sp"         → sp_rating, sp_offense, sp_defense
        "fpi"        → fpi
        "srs"        → srs
        "recruiting" → recruiting_4yr
        "portal"     → portal_net_rating, portal_qb_in, …
        "wepa"       → wepa_offense, wepa_defense, …
        "talent"     → talent
        "havoc"      → havoc_total, rush_success_rate, …
        "hfa"        → hfa_estimate  (computed fresh from master_games.csv)

    Year-shift notes
    ----------------
    SP+, FPI, SRS, WEPA, Havoc: season-final values → shifted +1 year in
        features.py (and again here for the raw CSVs), so we query by
        pred_season directly after applying the +1 shift.
    Recruiting: rolling average keyed by year; we use pred_season − 1 as the
        latest complete cycle.
    Talent: available pre-season, no shift needed — query pred_season directly.
    Portal: transfers happen before the season; query pred_season directly.
    HFA: computed from the 2 most recently completed seasons (< pred_season);
        the rolling average is equivalent to the +1 shift in features.py.
    """
    data_dir = Path(data_dir)
    ratings: dict = {}

    # ── SP+ ───────────────────────────────────────────────────────────────────
    sp_path = data_dir / "master_sp_ratings.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        sp["off_dict"]   = sp["offense"].apply(_safe_parse_sp_dict)
        sp["def_dict"]   = sp["defense"].apply(_safe_parse_sp_dict)
        sp["sp_offense"] = sp["off_dict"].apply(lambda d: d.get("rating"))
        sp["sp_defense"] = sp["def_dict"].apply(lambda d: d.get("rating"))
        year_col = "year" if "year" in sp.columns else "season"
        sp = sp.rename(columns={year_col: "season", "rating": "sp_rating"})
        sp["season"] = pd.to_numeric(sp["season"], errors="coerce") + 1
        cur = sp[sp["season"] == pred_season][
            ["team", "sp_rating", "sp_offense", "sp_defense"]
        ].set_index("team")
        if not cur.empty:
            ratings["sp"] = cur

    # ── FPI ───────────────────────────────────────────────────────────────────
    fpi_path = data_dir / "master_fpi_ratings.csv"
    if fpi_path.exists():
        fpi = pd.read_csv(fpi_path)
        fpi.columns = [c.lower() for c in fpi.columns]
        if "school" in fpi.columns and "team" not in fpi.columns:
            fpi = fpi.rename(columns={"school": "team"})
        if "year" in fpi.columns and "season" not in fpi.columns:
            fpi = fpi.rename(columns={"year": "season"})
        fpi["season"] = pd.to_numeric(fpi["season"], errors="coerce") + 1
        if "fpi" in fpi.columns:
            cur = fpi[fpi["season"] == pred_season][["team", "fpi"]].set_index("team")
            if not cur.empty:
                ratings["fpi"] = cur

    # ── SRS ───────────────────────────────────────────────────────────────────
    srs_path = data_dir / "master_srs_ratings.csv"
    if srs_path.exists():
        srs = pd.read_csv(srs_path)
        srs.columns = [c.lower() for c in srs.columns]
        if "school" in srs.columns and "team" not in srs.columns:
            srs = srs.rename(columns={"school": "team"})
        if "year" in srs.columns and "season" not in srs.columns:
            srs = srs.rename(columns={"year": "season"})
        if "rating" in srs.columns:
            srs = srs.rename(columns={"rating": "srs"})
        srs["season"] = pd.to_numeric(srs["season"], errors="coerce") + 1
        if "srs" in srs.columns:
            cur = srs[srs["season"] == pred_season][["team", "srs"]].set_index("team")
            if not cur.empty:
                ratings["srs"] = cur

    # ── Recruiting (4-year rolling, no shift) ─────────────────────────────────
    rec_path = data_dir / "master_recruiting.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rec.columns = [c.lower() for c in rec.columns]
        if "points" not in rec.columns and "total" in rec.columns:
            rec = rec.rename(columns={"total": "points"})
        rec = rec.sort_values(["team", "year"])
        rec["recruiting_4yr"] = (
            rec.groupby("team")["points"]
               .transform(lambda x: x.rolling(4, min_periods=1).mean())
        )
        # Use the last complete recruiting class (season − 1)
        cur = rec[rec["year"] == pred_season - 1][
            ["team", "recruiting_4yr"]
        ].set_index("team")
        if not cur.empty:
            ratings["recruiting"] = cur

    # ── Transfer Portal (no shift — reflects the offseason heading into season) ─
    portal_path = data_dir / "master_portal_features.csv"
    if portal_path.exists():
        portal = pd.read_csv(portal_path)
        portal.columns = [c.lower() for c in portal.columns]
        portal["season"] = pd.to_numeric(portal["season"], errors="coerce")
        avail = [c for c in PORTAL_FEAT_COLS if c in portal.columns]
        cur = portal[portal["season"] == pred_season]
        if not cur.empty and avail:
            ratings["portal"] = cur[["team"] + avail].set_index("team")

    # ── WEPA (+1 year shift) ──────────────────────────────────────────────────
    wepa_path = data_dir / "master_wepa.csv"
    if wepa_path.exists():
        try:
            wepa = pd.read_csv(wepa_path)
            wepa.columns = [c.lower() for c in wepa.columns]
            if "school" in wepa.columns and "team" not in wepa.columns:
                wepa = wepa.rename(columns={"school": "team"})
            if "year" in wepa.columns and "season" not in wepa.columns:
                wepa = wepa.rename(columns={"year": "season"})
            wepa["season"] = pd.to_numeric(wepa["season"], errors="coerce") + 1
            avail = [c for c in WEPA_COLS if c in wepa.columns]
            cur = wepa[wepa["season"] == pred_season]
            if not cur.empty and avail:
                ratings["wepa"] = cur[["team"] + avail].set_index("team")
        except Exception as exc:
            print(f"  ⚠️  WEPA loader failed: {exc}")

    # ── Talent (no shift — available pre-season from 247Sports) ──────────────
    # Fall back to the most recently available season when the exact pred_season
    # row is missing (e.g. 2026 talent hasn't been pulled yet).
    talent_path = data_dir / "master_talent.csv"
    if talent_path.exists():
        try:
            talent = pd.read_csv(talent_path)
            talent.columns = [c.lower() for c in talent.columns]
            if "school" in talent.columns and "team" not in talent.columns:
                talent = talent.rename(columns={"school": "team"})
            if "year" in talent.columns and "season" not in talent.columns:
                talent = talent.rename(columns={"year": "season"})
            talent["season"] = pd.to_numeric(talent["season"], errors="coerce")
            cur = talent[talent["season"] == pred_season]
            if cur.empty and "talent" in talent.columns:
                # Fallback: use the most recent available season
                available_seasons = sorted(talent["season"].dropna().unique())
                if available_seasons:
                    fallback_season = available_seasons[-1]
                    cur = talent[talent["season"] == fallback_season]
                    print(f"  ⚠️  Talent: no {pred_season} data — using {fallback_season:.0f} (stale prior-year talent)")
            if not cur.empty and "talent" in cur.columns:
                ratings["talent"] = cur[["team", "talent"]].set_index("team")
        except Exception as exc:
            print(f"  ⚠️  Talent loader failed: {exc}")

    # ── Havoc (+1 year shift) ─────────────────────────────────────────────────
    havoc_path = data_dir / "master_havoc.csv"
    if havoc_path.exists():
        try:
            havoc = pd.read_csv(havoc_path)
            havoc.columns = [c.lower() for c in havoc.columns]
            if "school" in havoc.columns and "team" not in havoc.columns:
                havoc = havoc.rename(columns={"school": "team"})
            if "year" in havoc.columns and "season" not in havoc.columns:
                havoc = havoc.rename(columns={"year": "season"})
            havoc["season"] = pd.to_numeric(havoc["season"], errors="coerce") + 1
            avail = [c for c in HAVOC_COLS if c in havoc.columns]
            cur = havoc[havoc["season"] == pred_season]
            if not cur.empty and avail:
                ratings["havoc"] = cur[["team"] + avail].set_index("team")
        except Exception as exc:
            print(f"  ⚠️  Havoc loader failed: {exc}")

    # ── Returning production (no shift — describes the upcoming roster) ──────
    # NO stale-season fallback on purpose: 2025's returning numbers describe
    # 2025 rosters and would be wrong for 2026. Missing season → NaN → imputed.
    ret_path = data_dir / "master_returning.csv"
    if ret_path.exists():
        try:
            ret = pd.read_csv(ret_path)
            ret["season"] = pd.to_numeric(ret["season"], errors="coerce")
            for col in ["ret_ppa_pct", "ret_pass_ppa_pct"]:
                if col in ret.columns:
                    ret[col] = pd.to_numeric(ret[col], errors="coerce").clip(0, 1.2)
            avail = [c for c in ["ret_ppa_pct", "ret_pass_ppa_pct"] if c in ret.columns]
            cur = ret[ret["season"] == pred_season]
            if not cur.empty and avail:
                ratings["returning"] = cur[["team"] + avail].set_index("team")
            elif avail:
                print(f"  ⚠️  Returning production: no {pred_season} data yet "
                      f"(CFBD publishes in summer) — feature will be imputed")
        except Exception as exc:
            print(f"  ⚠️  Returning-production loader failed: {exc}")

    # ── HFA — computed fresh from master_games.csv ───────────────────────────
    # We derive each team's home-field advantage from the 2 most recently
    # completed seasons (< pred_season) rather than reading the stale
    # feature_matrix.csv.  This is the same rolling-2-season logic as
    # build_home_field_advantage() in features.py, just applied at query time.
    games_path = data_dir / "master_games.csv"
    if games_path.exists():
        try:
            g = pd.read_csv(games_path)
            g["season"]       = pd.to_numeric(g["season"],      errors="coerce")
            g["home_points"]  = pd.to_numeric(g.get("home_points",  pd.Series(dtype=float)), errors="coerce")
            g["away_points"]  = pd.to_numeric(g.get("away_points",  pd.Series(dtype=float)), errors="coerce")
            g["neutral_site"] = g.get("neutral_site", pd.Series(0, index=g.index)).fillna(0).astype(int)
            g = g.dropna(subset=["home_points", "away_points"])
            g["point_diff"] = g["home_points"] - g["away_points"]

            completed = sorted(
                g[g["season"] < pred_season]["season"].dropna().unique()
            )
            recent2 = completed[-2:] if len(completed) >= 2 else completed

            g2 = g[(g["season"].isin(recent2)) & (g["neutral_site"] == 0)]
            if not g2.empty:
                h_m = g2.groupby("home_team")["point_diff"].mean().rename("home_margin")
                # Negate AFTER aggregation to avoid SeriesGroupBy TypeError
                a_m = (-g2.groupby("away_team")["point_diff"].mean()).rename("away_margin")
                hfa_df = pd.concat([h_m, a_m], axis=1).fillna(0)
                hfa_df["hfa_estimate"] = hfa_df["home_margin"] - hfa_df["away_margin"]
                ratings["hfa"] = hfa_df[["hfa_estimate"]]
        except Exception as exc:
            print(f"  ⚠️  HFA loader failed: {exc}")

    return ratings


# ─── 2. EPA LOADER ────────────────────────────────────────────────────────────

def load_recent_epa(pred_season: int, data_dir: Path) -> pd.DataFrame:
    """
    Return each team's current-form EPA features for predicting pred_season games.

    In-season aware: once pred_season games exist in master_ppa_games.csv
    (refresh_in_season.py appends them weekly), the rolling windows use them;
    pre-season they fall back to the tail of pred_season − 1. This mirrors the
    training features, where roll3/ytd are computed from prior games of the
    same season.

    Also computes the opponent-adjusted EPA features used by the models
    (adj_off_epa_roll3, adj_def_epa_roll3), de-meaned by the opponent's
    prior-season average — same formula as features.py build_rolling_epa().

    Returns a DataFrame indexed by team with columns:
        off_epa_roll3/5, def_epa_roll3/5, off_epa_pass_roll3/5, off_epa_rush_roll3/5,
        adj_off_epa_roll3, adj_def_epa_roll3, off_epa_ytd, def_epa_ytd
    """
    data_dir = Path(data_dir)
    ppa_path = data_dir / "master_ppa_games.csv"
    if not ppa_path.exists():
        return pd.DataFrame()

    ppa = pd.read_csv(ppa_path)
    cols = ["off_epa", "def_epa", "off_epa_pass", "off_epa_rush"]
    available = [c for c in cols if c in ppa.columns]
    if not available:
        return pd.DataFrame()

    # Pool current + prior season so early-season windows can reach back
    pool = ppa[ppa["season"].isin([pred_season - 1, pred_season])].copy()
    if pool.empty:
        return pd.DataFrame()
    pool = pool.sort_values(["team", "season", "week"])

    # ── Opponent-adjusted EPA (training formula: raw − opponent prior-season avg) ──
    if "opponent" in pool.columns and "def_epa" in pool.columns:
        prior_means = (
            ppa.groupby(["season", "team"])[["off_epa", "def_epa"]]
               .mean().reset_index()
               .rename(columns={"team": "opponent",
                                "off_epa": "opp_prior_off_epa",
                                "def_epa": "opp_prior_def_epa"})
        )
        prior_means["season"] = prior_means["season"] + 1   # season N stats adjust season N+1 games
        pool = pool.merge(
            prior_means[["season", "opponent", "opp_prior_off_epa", "opp_prior_def_epa"]],
            on=["season", "opponent"], how="left",
        )
        pool["adj_off_epa"] = pool["off_epa"] - pool["opp_prior_def_epa"].fillna(0)
        pool["adj_def_epa"] = pool["def_epa"] - pool["opp_prior_off_epa"].fillna(0)
        available = available + ["adj_off_epa", "adj_def_epa"]

    last3 = pool.groupby("team").tail(3).groupby("team")[available].mean()
    last3.columns = [f"{c}_roll3" for c in available]

    last5 = pool.groupby("team").tail(5).groupby("team")[available].mean()
    last5.columns = [f"{c}_roll5" for c in available]

    # ── Season-to-date baseline ────────────────────────────────────────────────
    # In-season: mean over the team's pred_season games played so far.
    # Pre-season fallback: full prior-season mean (best available proxy).
    ytd_cols = [c for c in ["off_epa", "def_epa"] if c in pool.columns]
    cur   = pool[pool["season"] == pred_season]
    prior = pool[pool["season"] == pred_season - 1]
    ytd = (
        prior.groupby("team")[ytd_cols].mean()
        if cur.empty else
        cur.groupby("team")[ytd_cols].mean()
            .combine_first(prior.groupby("team")[ytd_cols].mean())
    )
    ytd.columns = [f"{c}_ytd" for c in ytd_cols]

    return last3.join(last5, how="outer").join(ytd, how="outer")


# ─── 3. ATTACH TEAM FEATURES ─────────────────────────────────────────────────

def attach_team_features(
    df: pd.DataFrame,
    ratings: dict,
    epa: pd.DataFrame,
    elo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Given a games-plus-lines DataFrame, attach all pre-game team features
    for both the home and away team and compute all differentials.

    Parameters
    ----------
    df       : DataFrame with at least columns home_team, away_team, spread,
               over_under, spread_open.  Other line columns (home_moneyline,
               away_moneyline) are used if present.
    ratings  : dict returned by load_rating_sources()
    epa      : DataFrame returned by load_recent_epa() (indexed by team)
    elo      : DataFrame indexed by team with column 'elo' (optional)

    Returns
    -------
    df with all feature columns attached in-place (copy).
    """
    df = df.copy()

    def get_r(team: str, src_key: str, col: str) -> float:
        """Look up a single rating value; return NaN if missing."""
        src = ratings.get(src_key)
        if src is None or team not in src.index:
            return np.nan
        if col not in src.columns:
            return np.nan
        val = src.loc[team, col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        try:
            return np.nan if pd.isna(val) else float(val)
        except Exception:
            return np.nan

    # ── Per-team rating lookups ────────────────────────────────────────────────
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        ts = df[team_col]

        # SP+
        df[f"{side}_sp_rating"]      = ts.map(lambda t: get_r(t, "sp", "sp_rating"))
        df[f"{side}_sp_offense"]     = ts.map(lambda t: get_r(t, "sp", "sp_offense"))
        df[f"{side}_sp_defense"]     = ts.map(lambda t: get_r(t, "sp", "sp_defense"))

        # FPI / SRS
        df[f"{side}_fpi"] = ts.map(lambda t: get_r(t, "fpi", "fpi"))
        df[f"{side}_srs"] = ts.map(lambda t: get_r(t, "srs", "srs"))

        # Recruiting
        df[f"{side}_recruiting_4yr"] = ts.map(
            lambda t: get_r(t, "recruiting", "recruiting_4yr"))

        # HFA
        df[f"{side}_hfa"] = ts.map(lambda t: get_r(t, "hfa", "hfa_estimate"))

        # Elo
        if elo is not None and not elo.empty:
            df[f"{side}_pregame_elo"] = ts.map(
                lambda t: float(elo.loc[t, "elo"])
                if t in elo.index else np.nan)
        else:
            df[f"{side}_pregame_elo"] = np.nan

        # EPA rolling windows
        if epa is not None and not epa.empty:
            for col in epa.columns:
                df[f"{side}_{col}"] = ts.map(
                    lambda t, c=col: float(epa.loc[t, c])
                    if t in epa.index else np.nan)

        # Transfer portal (fill 0 — no portal activity = no change)
        portal_src = ratings.get("portal")
        for pcol in PORTAL_FEAT_COLS:
            if portal_src is not None and pcol in portal_src.columns:
                df[f"{side}_{pcol}"] = ts.map(
                    lambda t, c=pcol: float(portal_src.loc[t, c])
                    if t in portal_src.index else 0.0)
            else:
                df[f"{side}_{pcol}"] = 0.0

        # WEPA
        wepa_src = ratings.get("wepa")
        for col in WEPA_COLS:
            df[f"{side}_{col}"] = ts.map(
                lambda t, c=col: float(wepa_src.loc[t, c])
                if (wepa_src is not None
                    and t in wepa_src.index
                    and c in wepa_src.columns)
                else np.nan)

        # Talent
        talent_src = ratings.get("talent")
        df[f"{side}_talent"] = ts.map(
            lambda t: float(talent_src.loc[t, "talent"])
            if (talent_src is not None and t in talent_src.index)
            else np.nan)

        # Returning production
        ret_src = ratings.get("returning")
        for col in ["ret_ppa_pct", "ret_pass_ppa_pct"]:
            df[f"{side}_{col}"] = ts.map(
                lambda t, c=col: float(ret_src.loc[t, c])
                if (ret_src is not None
                    and t in ret_src.index
                    and c in ret_src.columns
                    and pd.notna(ret_src.loc[t, c]))
                else np.nan)

        # Havoc / advanced stats
        havoc_src = ratings.get("havoc")
        for col in HAVOC_COLS:
            df[f"{side}_{col}"] = ts.map(
                lambda t, c=col: float(havoc_src.loc[t, c])
                if (havoc_src is not None
                    and t in havoc_src.index
                    and c in havoc_src.columns)
                else np.nan)

    # ── Differentials (home − away) ────────────────────────────────────────────
    df["sp_diff"]          = df["home_sp_rating"]      - df["away_sp_rating"]
    df["sp_off_diff"]      = df["home_sp_offense"]     - df["away_sp_offense"]
    df["sp_def_diff"]      = df["home_sp_defense"]     - df["away_sp_defense"]
    df["fpi_diff"]         = df["home_fpi"]            - df["away_fpi"]
    df["srs_diff"]         = df["home_srs"]            - df["away_srs"]
    df["elo_diff"]         = df["home_pregame_elo"]    - df["away_pregame_elo"]
    df["recruiting_diff"]  = df["home_recruiting_4yr"] - df["away_recruiting_4yr"]
    df["hfa_diff"]         = df["home_hfa"].fillna(0)  - df["away_hfa"].fillna(0)
    # Neutral-site games get no home-field edge for either team — zero the
    # differential so the designated "home" team doesn't collect phantom
    # points. Mirrors model.load_data() (training) and features.py (matrix).
    if "neutral_site" in df.columns:
        _ns = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0)
        df.loc[_ns == 1, "hfa_diff"] = 0.0
    df["talent_diff"]      = df["home_talent"]         - df["away_talent"]

    df["portal_net_rating_diff"] = (
        df["home_portal_net_rating"] - df["away_portal_net_rating"])

    df["ret_ppa_diff"] = df["home_ret_ppa_pct"] - df["away_ret_ppa_pct"]
    if "home_ret_pass_ppa_pct" in df.columns:
        df["ret_pass_ppa_diff"] = (df["home_ret_pass_ppa_pct"]
                                   - df["away_ret_pass_ppa_pct"])

    if ratings.get("wepa") is not None:
        df["wepa_off_diff"]           = df["home_wepa_offense"]       - df["away_wepa_offense"]
        df["wepa_def_diff"]           = df["home_wepa_defense"]       - df["away_wepa_defense"]
        df["wepa_success_off_diff"]   = df["home_wepa_success_off"]   - df["away_wepa_success_off"]
        df["wepa_success_def_diff"]   = df["home_wepa_success_def"]   - df["away_wepa_success_def"]
        df["wepa_explosiveness_diff"] = (df["home_wepa_explosiveness"]
                                         - df["away_wepa_explosiveness"])

    if ratings.get("havoc") is not None:
        df["havoc_diff"]   = df["home_havoc_total"]       - df["away_havoc_total"]
        df["rush_sr_diff"] = df["home_rush_success_rate"] - df["away_rush_success_rate"]

        # Derived combos — identical formulas to features.py (training) so the
        # models see the same distributions at prediction time.
        if "home_explosiveness_off" in df.columns and "home_explosiveness_def" in df.columns:
            home_net = (df["home_explosiveness_off"].fillna(0) -
                        df["home_explosiveness_def"].fillna(0))
            away_net = (df["away_explosiveness_off"].fillna(0) -
                        df["away_explosiveness_def"].fillna(0))
            df["explosiveness_net_diff"] = home_net - away_net
        if "home_plays_per_drive" in df.columns:
            df["tempo_combined"] = (df["home_plays_per_drive"].fillna(0) +
                                    df["away_plays_per_drive"].fillna(0))
        if "home_rush_rate" in df.columns:
            df["rush_rate_combined"] = (df["home_rush_rate"].fillna(0) +
                                        df["away_rush_rate"].fillna(0))
        if "home_points_per_opp" in df.columns:
            df["points_per_opp_combined"] = (df["home_points_per_opp"].fillna(0) +
                                             df["away_points_per_opp"].fillna(0))
        if "home_def_points_per_opp" in df.columns:
            df["def_points_per_opp_combined"] = (df["home_def_points_per_opp"].fillna(0) +
                                                 df["away_def_points_per_opp"].fillna(0))
        if "home_turnover_margin" in df.columns:
            df["turnover_margin_diff"] = (df["home_turnover_margin"] -
                                          df["away_turnover_margin"])
            df["turnovers_combined"] = (df["home_turnovers_off_pg"].fillna(0) +
                                        df["away_turnovers_off_pg"].fillna(0))

    if "home_off_epa_roll3" in df.columns and "away_off_epa_roll3" in df.columns:
        df["epa_off_diff_roll3"] = df["home_off_epa_roll3"] - df["away_off_epa_roll3"]
        df["epa_def_diff_roll3"] = df["home_def_epa_roll3"] - df["away_def_epa_roll3"]

    # ── Rest days (default 14 — callers can override) ─────────────────────────
    if "home_rest_days" not in df.columns:
        df["home_rest_days"] = 14
    if "away_rest_days" not in df.columns:
        df["away_rest_days"] = 14
    if "rest_diff" not in df.columns:
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # ── Line-derived features ──────────────────────────────────────────────────
    df["spread"]      = pd.to_numeric(df.get("spread",      pd.Series(dtype=float)), errors="coerce")
    df["over_under"]  = pd.to_numeric(df.get("over_under",  pd.Series(dtype=float)), errors="coerce")
    df["spread_open"] = pd.to_numeric(df.get("spread_open", pd.Series(dtype=float)), errors="coerce")

    df["vegas_home_margin"] = -df["spread"]   # NaN when no line — don't fabricate a pick'em
    df["line_movement"]     = df["spread"] - df["spread_open"]
    df["line_moved_home"]   = (df["line_movement"] < -1.0).astype(int)
    df["line_moved_away"]   = (df["line_movement"] >  1.0).astype(int)

    # spread_magnitude: |expected margin| — top GBM feature in training, must
    # exist at prediction time. NaN when no line (imputer/LightGBM handle it).
    df["spread_magnitude"] = df["spread"].abs()

    # Clipped opening line used by MarketAnchoredEnsemble for shrinkage
    # (same ±45 clip as features.py applies in training).
    df["spread_open_val"] = df["spread_open"].clip(-45, 45)

    # ── Season-phase context (same definitions as features.py) ────────────────
    if "week_num" not in df.columns:
        df["week_num"] = (pd.to_numeric(df["week"], errors="coerce").fillna(0)
                          if "week" in df.columns else 0)
    if "late_season" not in df.columns:
        df["late_season"] = (df["week_num"] >= 11).astype(int)
    if "is_postseason" not in df.columns:
        df["is_postseason"] = (
            df["season_type"].astype(str).str.lower().str.contains("post", na=False).astype(int)
            if "season_type" in df.columns else 0
        )

    # ── Unrated-opponent flag ──────────────────────────────────────────────────
    # When a team has no SP+, FPI, or SRS, the model imputes medians and
    # effectively sees an "average FBS team" — badly underpredicting blowouts.
    key_rating_cols = ["sp_rating", "fpi", "srs"]
    for side in ("home", "away"):
        rating_cols_present = [
            f"{side}_{c}" for c in key_rating_cols if f"{side}_{c}" in df.columns
        ]
        df[f"{side}_unrated"] = (
            df[rating_cols_present].isna().all(axis=1)
            if rating_cols_present
            else True
        )
    df["has_unrated_opponent"] = df["home_unrated"] | df["away_unrated"]

    return df


# ─── 4. FEATURE COVERAGE REPORT ───────────────────────────────────────────────

def feature_coverage_report(df: pd.DataFrame) -> dict:
    """
    For each feature group in FEATURE_GROUPS, return the fraction of rows
    (games) where *all* columns in that group are non-null.

    Returns
    -------
    dict mapping group_name → float (0.0 – 1.0)
    Example: {"SP+": 0.98, "FPI": 0.76, "WEPA": 0.42, …}
    """
    report = {}
    n = len(df)
    if n == 0:
        return {g: 0.0 for g in FEATURE_GROUPS}

    for group, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            report[group] = 0.0
        else:
            # A row is "covered" when every column for this group is non-null
            coverage = df[present].notna().all(axis=1).mean()
            report[group] = float(coverage)

    return report


# ─── 5. ELO LOADER (convenience wrapper) ─────────────────────────────────────

def load_current_elo(pred_season: int, data_dir: Path) -> pd.DataFrame:
    """
    Recompute Elo through the end of pred_season − 1 using EloSystem.run().

    Returns a DataFrame indexed by team with column 'elo'.
    Returns an empty DataFrame if games data or EloSystem is unavailable.
    """
    data_dir = Path(data_dir)
    games_path = data_dir / "master_games.csv"
    sp_path    = data_dir / "master_sp_ratings.csv"

    if not games_path.exists():
        return pd.DataFrame(columns=["elo"])

    # Make sure src/ is importable regardless of caller location
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from elo_ratings import EloSystem
    except ImportError:
        return pd.DataFrame(columns=["elo"])

    games = pd.read_csv(games_path)

    # Restrict to FBS teams only
    if sp_path.exists():
        sp  = pd.read_csv(sp_path)
        fbs = set(sp["team"].unique())
        games = games[
            games["home_team"].isin(fbs) & games["away_team"].isin(fbs)
        ]

    games = (
        games[games["season"] <= pred_season - 1]
        .dropna(subset=["home_points", "away_points"])
    )

    elo = EloSystem()
    elo.run(games)
    return elo.current_ratings_df().set_index("team")[["elo"]]
