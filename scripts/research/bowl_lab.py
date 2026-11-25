"""Bowl module backtest. Honest split: tune on bowls of seasons 2017-2022,
score once on the 2023-24 and 2024-25 bowls (seasons 2023, 2024); the
2025-26 bowls (season 2025) are reported as an extra unseen season.

    python scripts/research/bowl_lab.py

Writes outputs/research/bowl_lab_*.csv and models/bowl_adjust.json.
Signs: margins are HOME margins; lines are home spreads (negative = home
favored); edge > 0 -> bet home. Cover margin c = side * (actual + line).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import bowl_context as B  # noqa: E402

OUT = ROOT / "outputs" / "research"
DEV = list(range(2017, 2023))
HOLD = [2023, 2024]
EXTRA = [2025]
PARTS = {"dev 2017-22": DEV, "holdout 2023-24": HOLD, "extra 2025": EXTRA}
WIN = 100 / 110


def lines() -> pd.DataFrame:
    l = pd.read_csv(ROOT / "data/research/cfbd_lines_by_provider.csv")
    l = l[l.season_type == "postseason"]
    cons = l[l.provider == "consensus"].groupby("game_id").spread.first()
    med = l.groupby("game_id").spread.median()
    close = cons.reindex(med.index).fillna(med)
    opener = l.groupby("game_id").spread_open.median()
    return pd.DataFrame({"close_line": close, "open_line": opener}).reset_index()


def frame() -> pd.DataFrame:
    g = B.bowl_games()
    g = g.dropna(subset=["home_points", "away_points"])
    g["actual"] = g.home_points - g.away_points
    f = g.merge(lines(), on="game_id", how="left").merge(B.game_features(g), on="game_id", how="left")
    pt_path = Path(os.environ.get("PT_MATCHED", "/workspace/pt/matched.csv"))
    if pt_path.exists():
        pt = pd.read_csv(pt_path)[["game_id", "lineespn", "lineopen"]]
        f = f.merge(pt, on="game_id", how="left")
        f["fpi_margin"] = f.lineespn
        f["open_line"] = f.open_line.fillna(-f.lineopen)     # PT's opener where CFBD has none
    wf = pd.read_csv(ROOT / "outputs/predictions/walk_forward_results.csv")[["game_id", "pred_spread"]]
    f = f.merge(wf, on="game_id", how="left")
    lab = Path(os.environ.get("OPENER_GAP_LAB", ROOT / "outputs/research/lab_is_pr0.8_lam2_hl0.csv"))
    if lab.exists():
        f = f.merge(pd.read_csv(lab)[["game_id", "pred_indep"]], on="game_id", how="left")
    return f.dropna(subset=["close_line"])


def grade(edge: pd.Series, line: pd.Series, actual: pd.Series, close: pd.Series | None = None) -> dict:
    m = edge.notna() & line.notna() & (edge != 0)
    side = np.sign(edge[m])
    c = side * (actual[m] + line[m])
    w, l_, p = int((c > 0).sum()), int((c < 0).sum()), int((c == 0).sum())
    n = w + l_ + p
    out = {"n": n, "w": w, "l": l_, "p": p,
           "win%": round(100 * w / (w + l_), 1) if w + l_ else np.nan,
           "roi%": round(100 * (w * WIN - l_) / n, 1) if n else np.nan}
    if close is not None:
        cm = close[m].notna()
        clv = side[cm] * (line[m][cm] - close[m][cm])  # points gained vs the close
        out["clv"] = round(float(clv.mean()), 2) if len(clv) else np.nan
        out["n_clv"] = int(cm.sum())
    return out


def by_part(f, edge, line, close=None, label=""):
    rows = []
    for part, ss in PARTS.items():
        s = f.season.isin(ss)
        r = grade(edge[s], f.loc[s, line], f.actual[s], f.loc[s, close] if close else None)
        rows.append({"rule": label, "period": part, **r})
    return rows


def fit_adjust(d: pd.DataFrame, feats: list[str], target: pd.Series, lam: float = 20.0) -> dict:
    X = d[feats].fillna(0).values
    y = target.values
    mu = X.mean(0)
    Xc = X - mu
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(len(feats)), Xc.T @ (y - y.mean()))
    return {"features": feats, "beta": dict(zip(feats, beta.round(3).tolist())),
            "lambda": lam, "n": int(len(d))}


def apply_adjust(d: pd.DataFrame, adj: dict) -> pd.Series:
    return sum(d[k].fillna(0) * v for k, v in adj["beta"].items())


def main() -> None:
    f = frame()
    OUT.mkdir(parents=True, exist_ok=True)
    print("bowls with closing line by season:", f.groupby("season").size().to_dict())
    print("with opener:", f[f.open_line.notna()].groupby("season").size().to_dict())
    f["y_close"] = f.actual + f.close_line        # home cover margin at the close
    f["y_open"] = f.actual + f.open_line
    f["move"] = f.open_line - f.close_line          # >0: market moved toward home

    # 1. one feature at a time on dev: does the close (or the opener) under-react?
    diag = []
    live = [f"{x}_diff" for x in B.LIVE_FEATS]
    allf = live + [f"{x}_diff" for x in B.RESEARCH_ONLY]
    for x in allf:
        for part, ss in PARTS.items():
            d = f[f.season.isin(ss) & f[x].notna()]
            if len(d) < 10 or d[x].std() == 0:
                continue
            row = {"feature": x, "period": part, "n": len(d), "nonzero": int((d[x] != 0).sum())}
            for y in ("y_close", "y_open", "move"):
                dd = d[d[y].notna()]
                if len(dd) > 10 and dd[x].std() > 0:
                    b = np.polyfit(dd[x], dd[y], 1)[0]
                    row[f"slope_{y}"] = round(float(b), 2)
                    row[f"n_{y}"] = len(dd)
            diag.append(row)
    diag = pd.DataFrame(diag)
    diag.to_csv(OUT / "bowl_lab_features.csv", index=False)
    print("\n== per-feature slopes (points per unit of home-minus-away) ==")
    print(diag.to_string(index=False))

    rows = []
    # 2. existing signals in bowls (frozen rules, no bowl tuning)
    rows += by_part(f, f.pred_spread + f.close_line, "close_line", label="main model edge (all, at close)")
    e = (f.pred_spread + f.close_line).where(lambda s: s.abs() >= 3)
    rows += by_part(f, e, "close_line", label="main model edge>=3 (at close)")
    pt_edge = (-f.fpi_margin + 0) if "fpi_margin" in f else None
    if pt_edge is not None:
        e = (f.fpi_margin + f.open_line).where(lambda s: s.abs() >= 7)
        rows += by_part(f, e, "open_line", "close_line", label="PT FPI-vs-opener>=7 (at opener)")
    if "pred_indep" in f:
        e = (f.pred_indep + f.open_line).where(lambda s: s.abs() >= 5)
        rows += by_part(f, e, "open_line", "close_line", label="opener-gap>=5 (at opener)")
    # bowl folklore baselines
    dog = np.sign(f.close_line).replace(0, np.nan)  # home dog -> +1 (bet home)
    rows += by_part(f, dog, "close_line", label="every bowl underdog (at close)")
    big = dog.where(f.close_line.abs() >= 7)
    rows += by_part(f, big, "close_line", label="bowl underdogs of 7+ (at close)")

    # 3. bowl adjustment fitted on dev bowls (target: cover margin at the close)
    dev = f[f.season.isin(DEV)]
    # draft picks are a post-hoc opt-out proxy: reported, never selectable
    candidates = {"live": live, "qb": ["qb_change_diff", "qb_starter_share_diff"],
                  "qb+coach": ["qb_change_diff", "qb_starter_share_diff",
                               "interim_coach_diff", "coach_change_diff"],
                  "RESEARCH live+draft": allf}
    adj_out = {}
    for name, feats in candidates.items():
        feats = [x for x in feats if x in f]
        for lam in (5.0, 20.0, 80.0):
            adj = fit_adjust(dev, feats, dev.y_close, lam)
            a = apply_adjust(f, adj)
            for thr in (1.0, 2.0, 3.0):
                e = a.where(a.abs() >= thr)
                r = by_part(f, e, "close_line", label=f"bowl adj [{name}] lam{lam:g} |adj|>={thr:g} (at close)")
                for x in r:
                    x.update({"set": name, "lam": lam, "thr": thr})
                rows += r
            adj_out[(name, lam)] = adj
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "bowl_lab_results.csv", index=False)

    # 4. select on dev only: max dev ROI with dev n >= 30 among the adjustments
    adjrows = res[res.rule.str.startswith("bowl adj") & ~res.rule.str.contains("RESEARCH")]
    print(f"\n{adjrows.rule.nunique()} selectable adjustment variants tried (dev picks one)")
    devr = adjrows[(adjrows.period == "dev 2017-22") & (adjrows.n >= 30)]
    best = devr.sort_values("roi%", ascending=False).iloc[0]
    sel = adjrows[adjrows.rule == best.rule]
    print("\n== frozen signals in bowls ==")
    print(res[~res.rule.str.startswith("bowl adj")].to_string(index=False))
    print("\n== dev-selected bowl adjustment ==")
    print(sel.drop(columns=["set", "lam", "thr"]).to_string(index=False))
    adj = adj_out[(best.set, best.lam)]
    a = apply_adjust(f, adj)
    e = a.where(a.abs() >= best.thr)
    op = pd.DataFrame(by_part(f, e, "open_line", "close_line",
                              label="same adjustment, bet at OPENER (upper bound: lineup not known then)"))
    knowable = [x for x in adj["features"] if "qb_" not in x]
    if knowable:
        a2 = sum(f[k].fillna(0) * adj["beta"][k] for k in knowable)
        op = pd.concat([op, pd.DataFrame(by_part(f, a2.where(a2.abs() >= best.thr), "open_line", "close_line",
                        label="non-QB part only (coach/portal), bet at opener"))])
    print(op.to_string(index=False))
    op.to_csv(OUT / "bowl_lab_opener.csv", index=False)
    hold = sel[sel.period == "holdout 2023-24"].iloc[0]
    meta = {"selected": best.rule, "threshold": float(best.thr), **adj,
            "fit_target": "home cover margin at the closing line, bowls 2017-2022",
            "dev": sel[sel.period == "dev 2017-22"].iloc[0].drop(["rule", "period", "set", "lam", "thr"]).to_dict(),
            "holdout_2023_24": hold.drop(["rule", "period", "set", "lam", "thr"]).to_dict(),
            "extra_2025": sel[sel.period == "extra 2025"].iloc[0].drop(["rule", "period", "set", "lam", "thr"]).to_dict(),
            "status": "paper"}
    (ROOT / "models" / "bowl_adjust.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, indent=1, default=float))
    f.to_csv(OUT / "bowl_lab_frame.csv", index=False)


if __name__ == "__main__":
    main()
