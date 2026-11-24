"""Idea 4: where does edge persist? Fixed rules, pre-declared segments,
dev (choose) vs holdout (confirm). A segment 'persists' only if dev ROI > 0
with n >= 100 AND holdout ROI > 0. Holdout one-sided binomial p vs 52.38%."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_lab import grade, OUT  # noqa: E402

P4 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}


def segs(d):
    hp = d.home_conference.isin(P4) | (d.home_team == "Notre Dame")
    ap = d.away_conference.isin(P4) | (d.away_team == "Notre Dame")
    L = d.close_line.abs()
    return {
        "all": pd.Series(True, index=d.index),
        "P4 vs P4": hp & ap, "G5 vs G5": ~hp & ~ap, "P4 vs G5": hp ^ ap,
        "weeks 1-4": d.week <= 4, "weeks 5-9": d.week.between(5, 9), "weeks 10+": d.week >= 10,
        "line <7": L < 7, "line 7-14": L.between(7, 14, inclusive="left"), "line 14+": L >= 14,
        "conference game": d.conference_game.fillna(0).astype(bool),
        "non-conference": ~d.conference_game.fillna(0).astype(bool),
    }


def side_segs(d, edge):
    home = edge > 0
    fav_home = d.close_line < 0
    return {"bet home": home, "bet away": ~home,
            "bet favorite": home == fav_home, "bet underdog": home != fav_home}


def run(d, edge_fn, at, thr, dev, hold, label):
    rows = []
    for period, ss in (("dev", dev), ("holdout", hold)):
        x = d[d.season.isin(ss)]
        e = edge_fn(x)
        line = x.open_line if at == "open" else x.close_line
        allsegs = {**segs(x), **side_segs(x, e)}
        for name, m in allsegs.items():
            r = grade(x, e, line, thr, m)
            p = binomtest(r["w"], r["w"] + r["l"], 0.5238, alternative="greater").pvalue if r["w"] + r["l"] else np.nan
            rows.append({"rule": label, "segment": name, "period": period, **r, "p_vs_breakeven": round(p, 3)})
    g = pd.DataFrame(rows)
    dv = g[g.period == "dev"].set_index("segment"); ho = g[g.period == "holdout"].set_index("segment")
    g["persists"] = g.segment.map(lambda s: bool(dv.loc[s, "n"] >= 100 and dv.loc[s, "roi%"] > 0 and ho.loc[s, "roi%"] > 0))
    return g


if __name__ == "__main__":
    d = pd.read_csv(OUT / "lab_base.csv")
    a = run(d, lambda x: x.pred_resid_close, "close", 3, [2019, 2020, 2021, 2022, 2023], [2024, 2025], "main@close edge>=3")
    b = run(d, lambda x: x.pred_resid_close, "open", 3, [2021, 2022, 2023], [2024, 2025], "main@open edge>=3")
    g = pd.concat([a, b]); g.to_csv(OUT / "segments.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(g[["rule", "segment", "period", "n", "w", "l", "win%", "roi%", "clv", "p_vs_breakeven", "persists"]].to_string(index=False))
