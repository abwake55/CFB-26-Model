"""Run spread-lab feature variants and summarise dev/holdout results.

  python scripts/research/run_variants.py qb        # idea 3
  python scripts/research/run_variants.py inseason  # idea 2 (grid in outputs/research/is_cache)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spread_lab as L  # noqa: E402
from lab_eval import evaluate  # noqa: E402

M = L.M
DEV_OPEN = [2021, 2022, 2023]


def mae(d, col, seasons):
    x = d[d.season.isin(seasons) & d[col].notna()]
    return round(float(np.mean(np.abs(x.point_diff + x.close_line - x[col]))), 4)


def summarize(name, d):
    out = {"variant": name,
           "dev_mae_resid": mae(d, "pred_resid_close", L.DEV),
           "hold_mae_resid": mae(d, "pred_resid_close", L.HOLD)}
    rows = []
    for rule, fn, thr in [("main@open", lambda x: x.pred_resid_close, (2, 3, 4, 5, 6)),
                          ("indep-gap@open", lambda x: x.pred_indep + x.open_line, (3, 5, 7, 9, 11, 13))]:
        g, t = evaluate(d, fn, "open", thr, label=rule, dev=DEV_OPEN)
        for _, r in g[g.thr == t].iterrows():
            rows.append({**out, "rule": rule, **r.to_dict()})
    return rows


def run_variant(base, extra, feats, name):
    df = base.merge(extra, on="game_id", how="left") if extra is not None else base
    d = L.run(df, list(M.SPREAD_FEATURES) + feats, ("close", "indep"), name)
    return summarize(name, d)


if __name__ == "__main__":
    which = sys.argv[1]
    base = L.load_base()
    rows = run_variant(base, None, [], "base")
    if which == "qb":
        import qb_features as Q
        qf = Q.game_features(base)
        for c in [c for c in qf.columns if "ypa_gap" in c]:
            qf[c] = qf[c].clip(-5, 5)
        pre = [f"{s}_{c}" for c in Q.PRE for s in ("home", "away")] + [f"{c}_diff" for c in Q.PRE]
        lin = [f"{s}_{c}" for c in Q.LINEUP for s in ("home", "away")] + [f"{c}_diff" for c in Q.LINEUP]
        rows += run_variant(base, qf, pre, "qb_pre")
        rows += run_variant(base, qf, lin, "qb_lineup_DIAGNOSTIC")
    elif which == "inseason":
        import inseason_ratings as IR
        cache = sorted((L.ROOT / "outputs/research/is_cache").glob("is_*.csv"))
        for f in cache:
            r = pd.read_csv(f).drop(columns=["inseason_version"], errors="ignore")
            rows += run_variant(base, r, IR.INSEASON_SPREAD_FEATURES, f.stem)
    res = pd.DataFrame(rows)
    res.to_csv(L.OUT / f"variants_{which}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["variant", "rule", "period", "thr", "n", "w", "l", "p", "win%", "roi%", "clv",
               "close_roi%", "dev_mae_resid", "hold_mae_resid"]].to_string(index=False))
