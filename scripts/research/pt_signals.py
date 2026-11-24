"""Prediction Tracker signals: dev chooses ONE frozen rule, holdout scored once.

Inputs
  outputs/research/lab_is_pr0.8_lam2_hl0.csv  walk-forward lab rows (point_diff,
      close_line, open_line, pred_indep = adopted opener-gap model, OOS)
  data/processed/pt_lines.csv                 src/prediction_tracker.build_history()

Decision line ("opener"): CFBD opener when present, else Prediction Tracker's
recorded opener (fills 2019-2020). Edges are in home-margin points:
edge > 0 -> bet home. Every rule is graded at the opener (the decision line),
and, as a realism check, at PT's midweek line (when PT's own numbers are
public) and at the close. CLV = points beaten vs the consensus close.

Pre-registered selection (written before looking at holdout):
  candidates = rule x threshold below; choose the single candidate with the
  highest dev ROI among those with dev n >= 150 AND dev CLV > 0. The chosen
  rule is written to models/pt_rule.json and scored on 2024-25 exactly once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_lab import grade, DEV, HOLD, OUT, ROOT  # noqa: E402

THRESHOLDS = (2, 3, 4, 5, 6, 7)
MIN_DEV_N = 150
RULE_PATH = ROOT / "models" / "pt_rule.json"


def load() -> pd.DataFrame:
    lab = pd.read_csv(OUT / "lab_is_pr0.8_lam2_hl0.csv")
    pt = pd.read_csv(ROOT / "data/processed/pt_lines.csv").drop(columns=["season"])
    d = lab.merge(pt, on="game_id", how="inner")
    d["opener"] = d["open_line"].where(d["open_line"].notna(), d["pt_open"])
    d["opener_src"] = np.where(d["open_line"].notna(), "cfbd", np.where(d["pt_open"].notna(), "pt", None))
    return d


def edges(d: pd.DataFrame, std_med: float) -> dict:
    """name -> (edge Series in home-margin pts vs opener, extra mask or None)."""
    fpi = d.opener - d.pt_fpi
    cons = d.opener - d.pt_sys_median
    ours = d.pred_indep + d.opener                    # adopted opener-gap edge
    return {
        "fpi_gap": (fpi, None),
        "consensus_gap": (cons, None),
        "consensus_gap_low_disagreement": (cons, d.pt_sys_std <= std_med),
        "fpi_gap_consensus_agrees": (fpi, np.sign(cons) == np.sign(fpi)),
        "opener_gap_consensus_agrees": (ours, np.sign(cons) == np.sign(ours)),
        "opener_gap_consensus_disagrees": (ours, np.sign(cons) != np.sign(ours)),
        "opener_gap_baseline": (ours, None),
        "sagarin_gap": (d.opener - d.pt_sagarin, None),
        "massey_gap": (d.opener - d.pt_massey, None),
    }


def grid(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    std_med = float(d[d.season.isin(DEV)].pt_sys_std.median())  # dev-only cut
    for period, seasons in (("dev", DEV), ("holdout", HOLD)):
        x = d[d.season.isin(seasons)]
        for name, (e, mask) in edges(x, std_med).items():
            for t in THRESHOLDS:
                r = grade(x, e, x.opener, t, mask)
                # same picks priced at PT's midweek line and at the close
                sel = e.abs().ge(t) & x.opener.notna()
                if mask is not None:
                    sel &= mask.fillna(False)
                rm = grade(x, e, x.pt_midweek, t, sel & x.pt_midweek.notna())
                rc = grade(x, e, x.close_line, t, sel & x.close_line.notna())
                rows.append({"rule": name, "period": period, "thr": t, "bet_at": "opener", **r,
                             "midweek_win%": rm["win%"], "midweek_roi%": rm["roi%"],
                             "close_win%": rc["win%"], "close_roi%": rc["roi%"]})
    return pd.DataFrame(rows)


def mae_table(d: pd.DataFrame) -> pd.DataFrame:
    cols = {"opener": "opener", "pt_midweek": "PT midweek", "close_line": "close",
            "pt_sys_median": "system median", "pt_fpi": "FPI", "pt_sagarin": "Sagarin",
            "pt_massey": "Massey", "pt_fei": "FEI", "pt_comp_adj": "PT computer-adjusted"}
    rows = []
    for period, seasons in (("dev", DEV), ("holdout", HOLD)):
        x = d[d.season.isin(seasons)]
        for c, lab in cols.items():
            y = x.dropna(subset=[c, "point_diff"])
            rows.append({"period": period, "predictor": lab, "n": len(y),
                         "mae": round(float((y.point_diff + y[c]).abs().mean()), 3)})
        y = x.dropna(subset=["pred_indep", "point_diff"])
        rows.append({"period": period, "predictor": "CFB-26 market-free (pred_indep)", "n": len(y),
                     "mae": round(float((y.point_diff - y.pred_indep).abs().mean()), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    print(f"{len(d)} lab games joined to PT; opener source: "
          f"{d.opener_src.value_counts(dropna=False).to_dict()}")
    g = grid(d)
    g.to_csv(OUT / "pt_signals_grid.csv", index=False)
    m = mae_table(d)
    m.to_csv(OUT / "pt_mae.csv", index=False)

    dev = g[(g.period == "dev") & (g.n >= MIN_DEV_N) & (g.clv > 0)]
    pick = dev.sort_values("roi%", ascending=False).iloc[0]
    rule = {"rule": pick.rule, "thr": float(pick.thr), "bet_at": "opener",
            "selection": f"max dev ROI with dev n>={MIN_DEV_N} and dev CLV>0",
            "dev_seasons": DEV, "holdout_seasons": HOLD,
            "dev": {k: (None if pd.isna(pick[k]) else float(pick[k])) for k in
                    ["n", "w", "l", "p", "win%", "roi%", "clv"]}}
    hold = g[(g.period == "holdout") & (g.rule == pick.rule) & (g.thr == pick.thr)].iloc[0]
    rule["holdout"] = {k: (None if pd.isna(hold[k]) else float(hold[k])) for k in
                       ["n", "w", "l", "p", "win%", "roi%", "clv", "midweek_roi%", "close_roi%"]}
    # 2026 to date: frozen rule, out of sample (never used for selection).
    # Not live: the ledger's live record starts when the flag went live.
    try:
        from spread_lab import load_base
        b = load_base()
        b = b[b.season == 2026][["game_id", "season", "week", "point_diff", "open_line", "close_line"]]
        pt26 = pd.read_csv(ROOT / "data/processed/pt_lines.csv").drop(columns=["season"])
        x = b.merge(pt26, on="game_id")
        x["opener"] = x.open_line.where(x.open_line.notna(), x.pt_open)
        e = edges(x.assign(pred_indep=np.nan), 0.0)[pick.rule][0]
        r26 = grade(x, e, x.opener, float(pick.thr))
        rule["oos_2026_to_date"] = {"weeks": f"{int(x.week.min())}-{int(x.week.max())}",
                                    **{k: (None if pd.isna(v) else float(v)) for k, v in r26.items()}}
        print("\n2026 to date (frozen rule, OOS, not live):", rule["oos_2026_to_date"])
    except Exception as exc:
        print(f"2026 OOS check skipped: {exc}")
    RULE_PATH.write_text(json.dumps(rule, indent=2))
    cols = ["rule", "period", "thr", "n", "w", "l", "p", "win%", "roi%", "clv", "clv+%",
            "midweek_win%", "midweek_roi%", "close_roi%"]
    print("\nFROZEN (dev):", pick.rule, pick.thr)
    print(g[(g.rule == pick.rule) & (g.thr == pick.thr)][cols].to_string(index=False))
    print("\nPer-rule at each rule's own best dev threshold (n>=150):")
    best = (g[(g.period == "dev") & (g.n >= MIN_DEV_N)].sort_values("roi%", ascending=False)
            .drop_duplicates("rule")[["rule", "thr"]])
    print(g.merge(best, on=["rule", "thr"])[cols].sort_values(["rule", "period"]).to_string(index=False))
    print("\nMAE:\n", m.pivot(index="predictor", columns="period", values="mae").to_string())


if __name__ == "__main__":
    main()
