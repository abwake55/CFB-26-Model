"""Idea 5b (2026 live archive only): do sharp-leaning books (LowVig,
BetOnline) disagreeing with soft US books predict covers? Uses the earliest
and the last pre-kickoff snapshot per game. Bets the soft-book median at that
snapshot in the sharp direction when |sharp - soft| >= 0.5. Also intersects
with the live ledger's model side."""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
SHARP = {"lowvig", "betonlineag"}
SOFT = {"draftkings", "fanduel", "betmgm", "betrivers"}

sys.path.insert(0, str(ROOT / "src"))
from odds_api import _team_match  # noqa: E402
_mg = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False).drop_duplicates("game_id")
HOME = dict(zip(_mg.game_id, _mg.home_team))
rows = []
for f in sorted(glob.glob(str(ROOT / "data/lines_snapshots/quotes/*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r.get("market") != "spreads" or r.get("line") is None:
            continue
        sel = r.get("selection")
        if sel not in ("home", "away"):   # older records: infer from team name
            h = HOME.get(r["game_id"])
            sel = "home" if h and _team_match(h, r.get("side", "")) > 0 else "away"
        if sel == "home":
            rows.append((r["game_id"], r["book"], float(r["line"]), r["observed_at"], r.get("kickoff")))
q = pd.DataFrame(rows, columns=["game_id", "book", "line", "obs", "kickoff"])
q = q.drop_duplicates(["game_id", "book", "obs"])
q["obs"] = pd.to_datetime(q.obs, utc=True, format="mixed"); q["kickoff"] = pd.to_datetime(q.kickoff, utc=True, format="mixed")
q = q[q.obs < q.kickoff]
q["snap"] = q.obs.dt.floor("h")
g = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False).drop_duplicates("game_id")
g = g[g.season == 2026][["game_id", "point_diff"]].dropna() if "point_diff" in g else None
if g is None:
    gg = pd.read_csv(ROOT / "data/processed/master_games.csv", low_memory=False).drop_duplicates("game_id")
    gg = gg[gg.season == 2026]
    hp = [c for c in gg.columns if c in ("home_points", "home_score")][0]
    ap = [c for c in gg.columns if c in ("away_points", "away_score")][0]
    g = gg.assign(point_diff=gg[hp] - gg[ap])[["game_id", "point_diff"]].dropna()
led = pd.read_csv(ROOT / "outputs/predictions/ledger_2026.csv")
led = led[led.provenance == "live"].drop_duplicates("game_id", keep="first")[["game_id", "pred_spread", "spread"]]

out = []
for which in ("first", "last"):
    per = []
    for gid, x in q.groupby("game_id"):
        groups = list(x.groupby("snap"))
        for s, y in (groups if which == "first" else groups[::-1]):
            sh, so = y[y.book.isin(SHARP)].line, y[y.book.isin(SOFT)].line
            if len(sh) and len(so):
                per.append({"game_id": gid, "sharp": sh.mean(), "soft": so.median()}); break
    p = pd.DataFrame(per).merge(g, on="game_id").merge(led, on="game_id", how="left")
    p["dev"] = p.sharp - p.soft                       # <0: sharp books more on home
    for name, m, side in [
        ("sharp direction |dev|>=0.5", p.dev.abs() >= 0.5, -np.sign(p.dev)),
        ("sharp direction & model agrees", (p.dev.abs() >= 0.5) & (np.sign(-p.dev) == np.sign(p.pred_spread + p.soft)), -np.sign(p.dev)),
        ("model side (edge>=3 vs soft) & sharp agrees", ((p.pred_spread + p.soft).abs() >= 3) & (np.sign(-p.dev) == np.sign(p.pred_spread + p.soft)) & (p.dev != 0), np.sign(p.pred_spread + p.soft)),
        ("model side (edge>=3 vs soft), all", (p.pred_spread + p.soft).abs() >= 3, np.sign(p.pred_spread + p.soft)),
    ]:
        res = side[m] * (p.point_diff[m] + p.soft[m])
        w, l, pu = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
        out.append({"snapshot": which, "rule": name, "n": int(m.sum()), "w": w, "l": l, "p": pu,
                    "win%": round(100 * w / (w + l), 2) if w + l else np.nan,
                    "roi%": round(100 * (w - 1.1 * l) / (1.1 * (w + l)), 2) if w + l else np.nan,
                    "p_vs_breakeven": round(binomtest(w, w + l, 0.5238, alternative="greater").pvalue, 3) if w + l else np.nan,
                    "games": len(p)})
r = pd.DataFrame(out)
r.to_csv(ROOT / "outputs/research/sharp_vs_soft_2026.csv", index=False)
print(r.to_string(index=False))
