"""Fit the edge ranges (80% intervals) shown next to every pick.

Two ranges per signal, both from backtest residuals on dev seasons 2019-2023
and checked once on 2024-2025:

1. Expected-edge range. For the side a signal bets, regress the realized
   cover margin c (points by which the bet side beat the line) on the model
   edge e = |model margin - market margin|:  c = a + b*e.  A game-level
   bootstrap gives the (a, b) covariance; the 80% interval for a new pick is
   a + b*e +/- 1.2816 * sqrt([1 e] Cov [1 e]').  This is the honest version of
   "edge 4, likely 1 to 7": it says how much of a stated edge survived in the
   backtest, and how sure we are of that.
   Checked by fitting on dev only and scoring 2024-2025; the live
   parameters are then refit on every season 2019-2025.
2. Margin range. 10th-90th percentile of (actual margin - model margin),
   added to the model's projected margin. This is the game-outcome range.

Inputs (research files, not all tracked in git):
  outputs/predictions/walk_forward_results.csv     main model (bet at close)
  outputs/research/lab_is_pr0.8_lam2_hl0.csv        opener-gap model (bet at opener)
  /workspace/pt/matched.csv (or PT_MATCHED env)     Prediction Tracker FPI (bet at opener)

Writes models/edge_ranges.json and outputs/research/edge_ranges_check.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV = [2019, 2020, 2021, 2022, 2023]
HOLD = [2024, 2025]
Z80 = 1.2816
RNG = np.random.default_rng(26)


def _src(path_env: str, default: Path) -> Path:
    p = Path(os.environ.get(path_env, default))
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return p


def frames() -> dict[str, tuple[pd.DataFrame, float]]:
    """signal -> (frame with season, edge (home pts, >0 bet home), line (home
    spread convention), model_margin, actual_margin), flag threshold."""
    w = pd.read_csv(ROOT / "outputs/predictions/walk_forward_results.csv")
    main = pd.DataFrame({"season": w.season, "game_id": w.game_id,
                         "model_margin": w.pred_spread, "line": w.spread,
                         "actual_margin": w.point_diff})
    lab = pd.read_csv(_src("OPENER_GAP_LAB", ROOT / "outputs/research/lab_is_pr0.8_lam2_hl0.csv"))
    og = pd.DataFrame({"season": lab.season, "game_id": lab.game_id,
                       "model_margin": lab.pred_indep, "line": lab.open_line,
                       "actual_margin": lab.point_diff})
    m = pd.read_csv(_src("PT_MATCHED", Path("/workspace/pt/matched.csv")))
    opener = m.open_line.where(m.open_line.notna(), -m.lineopen)
    pt = pd.DataFrame({"season": m.season, "game_id": m.game_id,
                       "model_margin": m.lineespn, "line": opener,
                       "actual_margin": m.point_diff})
    rule = json.loads((ROOT / "models/pt_rule.json").read_text())
    out = {}
    for name, f, thr in (("spread", main, 3.0), ("opener_gap", og, 5.0),
                         ("pt", pt, float(rule["thr"]))):
        f = f.dropna(subset=["model_margin", "line", "actual_margin"]).copy()
        f["edge"] = f.model_margin + f.line              # >0 -> bet home
        side = np.sign(f.edge)
        f["e"] = f.edge.abs()
        f["c"] = side * (f.actual_margin + f.line)        # cover margin of the bet side
        f["resid"] = f.actual_margin - f.model_margin
        out[name] = (f[f.e >= 0.5], thr)
    return out


def fit(f: pd.DataFrame, n_boot: int = 2000) -> dict:
    X = np.column_stack([np.ones(len(f)), f.e.values])
    y = f.c.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    seasons = f.season.values
    groups = {s: np.flatnonzero(seasons == s) for s in np.unique(seasons)}
    keys = list(groups)
    boots = np.empty((n_boot, 2))
    for i in range(n_boot):
        pick = RNG.choice(keys, len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        boots[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    cov = np.cov(boots.T)
    return {"a": float(beta[0]), "b": float(beta[1]), "cov": cov.round(8).tolist(),
            "resid_q10": float(np.quantile(f.resid, 0.10)),
            "resid_q90": float(np.quantile(f.resid, 0.90)),
            "n": int(len(f)), "seasons": sorted(int(x) for x in keys)}


def interval(p: dict, e) -> tuple:
    e = np.asarray(e, dtype=float)
    mu = p["a"] + p["b"] * e
    c = np.array(p["cov"])
    se = np.sqrt(c[0, 0] + 2 * e * c[0, 1] + e * e * c[1, 1])
    return mu, mu - Z80 * se, mu + Z80 * se


def main() -> None:
    params, rows = {}, []
    for name, (f, thr) in frames().items():
        dev, hold = f[f.season.isin(DEV)], f[f.season.isin(HOLD)]
        p = fit(dev)                       # dev-only fit, scored on holdout below
        live = fit(f)                      # live parameters: every season
        live["flag_threshold"] = thr
        live["dev_only_fit"] = {k: p[k] for k in ("a", "b", "cov", "n", "seasons")}
        params[name] = live
        for part, d in (("dev", dev), ("holdout", hold)):
            for lo, hi in ((0.5, thr), (thr, 99)):
                s = d[(d.e >= lo) & (d.e < hi)]
                if s.empty:
                    continue
                mu, l80, h80 = interval(p, s.e)
                m = s.model_margin
                cov = ((s.actual_margin >= m + p["resid_q10"]) & (s.actual_margin <= m + p["resid_q90"])).mean()
                rows.append({"signal": name, "part": part, "edge_bucket": f"{lo:g}-{hi:g}",
                             "n": len(s), "mean_stated_edge": round(s.e.mean(), 2),
                             "pred_cover_margin": round(mu.mean(), 2),
                             "pred_lo80": round(l80.mean(), 2), "pred_hi80": round(h80.mean(), 2),
                             "realized_cover_margin": round(s.c.mean(), 2),
                             "realized_se": round(s.c.std() / np.sqrt(len(s)), 2),
                             "margin_range_coverage": round(float(cov), 3)})
    meta = {"method": "see scripts/research/fit_edge_ranges.py", "dev_seasons": DEV,
            "holdout_seasons": HOLD, "interval": 0.8, "signals": params}
    (ROOT / "models/edge_ranges.json").write_text(json.dumps(meta, indent=2))
    chk = pd.DataFrame(rows)
    (ROOT / "outputs/research").mkdir(parents=True, exist_ok=True)
    chk.to_csv(ROOT / "outputs/research/edge_ranges_check.csv", index=False)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk != "cov"} for k, v in params.items()}, default=str, indent=1))
    print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
