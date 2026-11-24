"""Idea 5: model edge at the close, filtered by opener->close movement.
Movement is known at the close, so these rules bet the CLOSING line.
agree: market moved toward the model's side (side*(open-close) > 0)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_eval import evaluate  # noqa: E402
from spread_lab import OUT  # noqa: E402

DEV = [2021, 2022, 2023]; HOLD = [2024, 2025]


def mv(x, e):
    return np.sign(e) * (x.open_line - x.close_line)


if __name__ == "__main__":
    rows = []
    for v in ["base", "is_pr0.8_lam2_hl0"]:
        d = pd.read_csv(OUT / f"lab_{v}.csv"); d = d[d.open_line.notna()]
        for col, nm in [("pred_resid_close", "main"), ("indep", "indep-gap")]:
            fn = (lambda x: x.pred_resid_close) if nm == "main" else (lambda x: x.pred_indep + x.close_line)
            thr = (2, 3, 4, 5, 6) if nm == "main" else (3, 5, 7, 9, 11)
            for flt, mfn in [("none", None),
                             ("agree", lambda x, fn=fn: mv(x, fn(x)) > 0),
                             ("agree>=1pt", lambda x, fn=fn: mv(x, fn(x)) >= 1),
                             ("no move against", lambda x, fn=fn: mv(x, fn(x)) >= 0),
                             ("against", lambda x, fn=fn: mv(x, fn(x)) < 0)]:
                g, t = evaluate(d, fn, "close", thr, min_dev_n=60, mask_fn=mfn,
                                label=f"{v}:{nm}@close+{flt}", dev=DEV, hold=HOLD)
                if t is not None:
                    rows.append(g[g.thr == t])
    r = pd.concat(rows); r.to_csv(OUT / "movement_filter.csv", index=False)
    pd.set_option("display.width", 250)
    print(r[["rule", "period", "thr", "n", "w", "l", "p", "win%", "roi%"]].to_string(index=False))
