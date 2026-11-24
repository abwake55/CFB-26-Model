"""Point-in-time, opponent-adjusted in-season team ratings.

For every game, ratings are fitted ONLY on games played on earlier dates
(US/Eastern calendar day) in the same season, shrunk toward a preseason
prior. Nothing from the game itself, or from later games, is used.

Two ratings per team:
* PPA (EPA/play) offense & defense: off_epa(team vs opp) ~ mu + home*h
  + O[team] + D[opp], ridge-shrunk toward the prior.
* Margin (SRS-like): point_diff ~ h*home + R[home] - R[away], ridge-shrunk.

Prior: the previous season's END-OF-SEASON fit of the same model, regressed
toward zero by PRIOR_REGRESS (teams churn year to year). Ridge's penalty is a
fixed number of "pseudo-games", so the prior's weight decays naturally as
games accumulate.

Output columns (per game): is_home_off, is_home_def, is_away_off, is_away_def,
is_ppa_net_diff, is_ppa_total, is_margin_diff, is_games_home, is_games_away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

PRIOR_REGRESS = 0.6
LAMBDA_PPA = 4.0      # pseudo-games of prior weight per team parameter
LAMBDA_MARGIN = 3.0
VERSION = "inseason_v1"


def _ridge_offset(X, y, lam, w=None):
    """Solve min sum w (y - Xb)^2 + lam ||b||^2 (dense normal equations)."""
    if w is not None:
        W = sparse.diags(w)
        XtW = X.T @ W
        XtX = XtW @ X
        XtX = XtX.toarray() if sparse.issparse(XtX) else XtX
        XtX[np.diag_indices_from(XtX)] += lam
        return np.linalg.solve(XtX, XtW @ y)
    XtX = (X.T @ X).toarray() if sparse.issparse(X) else X.T @ X
    XtX[np.diag_indices_from(XtX)] += lam
    return np.linalg.solve(XtX, X.T @ y)


def _recency(days_ago, half_life):
    """Weights 0.5**(age/half_life); None -> unweighted."""
    if not half_life:
        return None
    return np.power(0.5, np.asarray(days_ago, dtype=float) / float(half_life))


def _eastern_day(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")
    return t.dt.tz_convert("America/New_York").dt.normalize()


def _fit_ppa(rows: pd.DataFrame, teams: list, prior_o: dict, prior_d: dict, lam: float,
             w=None):
    """rows: team, opponent, off_epa, is_home(+1/0/-1). Returns (O, D, mu, h)."""
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(rows), len(teams)
    if n == 0:
        return ({t: prior_o.get(t, 0.) for t in teams},
                {t: prior_d.get(t, 0.) for t in teams}, 0., 0.)
    ti = rows["team"].map(idx).to_numpy()
    oi = rows["opponent"].map(idx).to_numpy()
    po = np.array([prior_o.get(t, 0.) for t in teams])
    pd_ = np.array([prior_d.get(t, 0.) for t in teams])
    mu = float(rows["off_epa"].mean())
    y = rows["off_epa"].to_numpy() - mu - po[ti] - pd_[oi]
    data = np.ones(2 * n)
    r = np.r_[np.arange(n), np.arange(n)]
    c = np.r_[ti, k + oi]
    X = sparse.csr_matrix((data, (r, c)), shape=(n, 2 * k))
    hcol = sparse.csr_matrix(rows["is_home"].to_numpy(dtype=float).reshape(-1, 1))
    X = sparse.hstack([X, hcol]).tocsr()
    b = _ridge_offset(X, y, lam, w)
    O = {t: po[i] + b[i] for t, i in idx.items()}
    D = {t: pd_[i] + b[k + i] for t, i in idx.items()}
    return O, D, mu, float(b[-1])


def _fit_margin(g: pd.DataFrame, teams: list, prior: dict, lam: float, w=None):
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    if n == 0:
        return {t: prior.get(t, 0.) for t in teams}
    hi = g["home_team"].map(idx).to_numpy(); ai = g["away_team"].map(idx).to_numpy()
    pr = np.array([prior.get(t, 0.) for t in teams])
    home = 1 - g["neutral_site"].fillna(0).astype(float).to_numpy()
    y = g["point_diff"].to_numpy(dtype=float) - (pr[hi] - pr[ai])
    y = np.clip(y, -45, 45)  # cap blowouts
    data = np.r_[np.ones(n), -np.ones(n)]
    X = sparse.csr_matrix((data, (np.r_[np.arange(n), np.arange(n)], np.r_[hi, ai])),
                          shape=(n, k))
    X = sparse.hstack([X, sparse.csr_matrix(home.reshape(-1, 1))]).tocsr()
    b = _ridge_offset(X, y, lam, w)
    return {t: pr[i] + b[i] for t, i in idx.items()}


def build_inseason_ratings(games: pd.DataFrame, ppa: pd.DataFrame,
                           lam_ppa: float = LAMBDA_PPA,
                           lam_margin: float = LAMBDA_MARGIN,
                           prior_regress: float = PRIOR_REGRESS,
                           half_life_days: float | None = None) -> pd.DataFrame:
    """Return one row per game_id in `games` with point-in-time rating features.

    games: game_id, season, start_date, home_team, away_team, neutral_site,
           point_diff (NaN allowed for unplayed games).
    ppa:   game_id, team, opponent, off_epa (one row per team per game).
    half_life_days: optional recency weighting of earlier games (research).
    """
    g = games.copy()
    g["day"] = _eastern_day(g["start_date"])
    g = g.dropna(subset=["day"]).sort_values("day")
    p = ppa.merge(g[["game_id", "day", "season", "home_team", "neutral_site"]],
                  on="game_id", how="inner", suffixes=("", "_g"))
    if "season_g" in p:
        p["season"] = p["season_g"]
    p["is_home"] = np.where(p["neutral_site"].fillna(0).astype(float) == 1, 0,
                            np.where(p["team"] == p["home_team"], 1, -1))
    p = p.dropna(subset=["off_epa"])
    played = g.dropna(subset=["point_diff"])

    out = []
    prior_o, prior_d, prior_r = {}, {}, {}
    for season in sorted(g["season"].unique()):
        gs = g[g["season"] == season]
        ps, pl = p[p["season"] == season], played[played["season"] == season]
        teams = sorted(set(gs["home_team"]) | set(gs["away_team"])
                       | set(ps["team"]) | set(ps["opponent"]))
        po = {t: prior_regress * v for t, v in prior_o.items()}
        pdd = {t: prior_regress * v for t, v in prior_d.items()}
        prr = {t: prior_regress * v for t, v in prior_r.items()}
        for day, todays in gs.groupby("day"):
            prev_p, prev_g = ps[ps["day"] < day], pl[pl["day"] < day]
            wp = _recency((day - prev_p["day"]).dt.days, half_life_days)
            wg = _recency((day - prev_g["day"]).dt.days, half_life_days)
            O, D, _, _ = _fit_ppa(prev_p, teams, po, pdd, lam_ppa, wp)
            R = _fit_margin(prev_g, teams, prr, lam_margin, wg)
            cnt = pd.concat([prev_g["home_team"], prev_g["away_team"]]).value_counts()
            for r in todays.itertuples():
                h, a = r.home_team, r.away_team
                out.append({
                    "game_id": r.game_id,
                    "is_home_off": O[h], "is_home_def": D[h],
                    "is_away_off": O[a], "is_away_def": D[a],
                    "is_ppa_net_diff": (O[h] - D[h]) - (O[a] - D[a]),
                    "is_ppa_total": O[h] + D[a] + O[a] + D[h],
                    "is_margin_diff": R[h] - R[a],
                    "is_games_home": int(cnt.get(h, 0)),
                    "is_games_away": int(cnt.get(a, 0)),
                })
        # end-of-season fit (all season games) becomes next season's prior
        prior_o, prior_d, _, _ = _fit_ppa(ps, teams, po, pdd, lam_ppa)
        prior_r = _fit_margin(pl, teams, prr, lam_margin)
    res = pd.DataFrame(out)
    res["inseason_version"] = VERSION
    return res


INSEASON_SPREAD_FEATURES = ["is_ppa_net_diff", "is_margin_diff",
                            "is_home_off", "is_home_def", "is_away_off", "is_away_def"]
INSEASON_TOTALS_FEATURES = ["is_ppa_total", "is_home_off", "is_home_def",
                            "is_away_off", "is_away_def"]


def load_inseason_ratings(proc_dir) -> pd.DataFrame:
    from pathlib import Path
    proc_dir = Path(proc_dir)
    games = pd.read_csv(proc_dir / "master_games.csv")
    ppa = pd.read_csv(proc_dir / "master_ppa_games.csv")
    games = games.drop_duplicates("game_id")
    return build_inseason_ratings(games, ppa)
