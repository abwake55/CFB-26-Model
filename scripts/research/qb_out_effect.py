"""Regular-season QB-out effect, for the game-day re-projection.

primary  = the team's most-started QB this season BEFORE the game (>= 2 starts)
qb_out   = the primary threw no pass in this game (out; NOT a mid-game benching)
qb_out_last = the primary threw no pass in the team's previous game
              (knowable on Saturday morning without lineup news)

Slopes (points of home margin per unit home-minus-away) of:
  y_close = actual - closing market margin   (does the close under-react?)
  y_open  = actual - opening market margin
  y_indep = actual - market-free model margin (src/opener_gap.py lab)
Dev 2019-2023, holdout 2024-2025. Writes models/qb_adjust.json and
outputs/research/qb_out_effect.csv.

    python scripts/research/qb_out_effect.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV, HOLD = list(range(2019, 2024)), [2024, 2025]


def team_games() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/research/cfbd_passing_by_game.csv")
    p = p[p.season_type == "regular"]
    g = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False,
                    usecols=["game_id", "start_date"]).drop_duplicates("game_id")
    p = p.merge(g, on="game_id", how="left")
    p["t"] = pd.to_datetime(p.start_date, utc=True, errors="coerce", format="mixed")
    st = p.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    st = st[["game_id", "season", "team", "t", "player_id"]].sort_values("t")
    played = p.groupby(["game_id", "team"]).player_id.apply(lambda x: set(x.astype(str)))
    rows = []
    for (season, team), grp in st.groupby(["season", "team"]):
        hist, prev_out = [], np.nan
        for r in grp.itertuples():
            s = pd.Series(hist)
            if len(hist) >= 2:
                prim = str(s.value_counts().idxmax())
                out = float(prim not in played.get((r.game_id, team), set()))
            else:
                prim, out = None, np.nan
            rows.append({"game_id": r.game_id, "team": team, "qb_out": out,
                         "qb_out_last": prev_out})
            prev_out = out
            hist.append(str(r.player_id))
    return pd.DataFrame(rows)


def main() -> None:
    tg = team_games()
    w = pd.read_csv(ROOT / "outputs/predictions/walk_forward_results.csv")
    f = w[["game_id", "season", "home_team", "away_team", "point_diff", "spread"]]
    lab = Path(os.environ.get("OPENER_GAP_LAB", ROOT / "outputs/research/lab_is_pr0.8_lam2_hl0.csv"))
    if lab.exists():
        f = f.merge(pd.read_csv(lab)[["game_id", "pred_indep", "open_line"]], on="game_id", how="left")
    for side in ("home", "away"):
        f = f.merge(tg.rename(columns={"team": f"{side}_team", "qb_out": f"{side}_qb_out",
                                       "qb_out_last": f"{side}_qb_out_last"}),
                    on=["game_id", f"{side}_team"], how="left")
    for k in ("qb_out", "qb_out_last"):
        f[f"{k}_diff"] = f[f"home_{k}"] - f[f"away_{k}"]
    f["y_close"] = f.point_diff + f.spread
    f["y_open"] = f.point_diff + f.get("open_line")
    f["y_indep"] = f.point_diff - f.get("pred_indep")
    rows, fit = [], {}
    for x in ("qb_out_diff", "qb_out_last_diff"):
        for part, ss in (("dev 2019-23", DEV), ("holdout 2024-25", HOLD)):
            d = f[f.season.isin(ss) & f[x].notna()]
            row = {"feature": x, "period": part, "n": len(d), "nonzero": int((d[x] != 0).sum())}
            for y in ("y_close", "y_open", "y_indep"):
                dd = d[d[y].notna()]
                if len(dd) > 30 and dd[x].std() > 0:
                    b, a = np.polyfit(dd[x], dd[y], 1)
                    resid = dd[y] - (a + b * dd[x])
                    se = resid.std() / (dd[x].std() * np.sqrt(len(dd)))
                    row[f"slope_{y}"], row[f"se_{y}"] = round(float(b), 2), round(float(se), 2)
                    if part.startswith("dev"):
                        fit[(x, y)] = float(b)
            rows.append(row)
    res = pd.DataFrame(rows)
    out = ROOT / "outputs/research"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "qb_out_effect.csv", index=False)
    print(res.to_string(index=False))
    meta = {"source": "scripts/research/qb_out_effect.py", "dev_seasons": DEV,
            "per_qb_out_vs_model_pts": round(fit.get(("qb_out_diff", "y_indep"), np.nan), 2),
            "per_qb_out_vs_close_pts": round(fit.get(("qb_out_diff", "y_close"), np.nan), 2),
            "per_qb_out_last_vs_close_pts": round(fit.get(("qb_out_last_diff", "y_close"), np.nan), 2),
            "per_qb_out_last_vs_model_pts": round(fit.get(("qb_out_last_diff", "y_indep"), np.nan), 2),
            "note": "slopes are home-margin points per unit of (home - away) QB-out; "
                    "dev-fitted, holdout in outputs/research/qb_out_effect.csv"}
    (ROOT / "models/qb_adjust.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
