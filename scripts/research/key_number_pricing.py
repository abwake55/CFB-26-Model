"""Key-number pricing research: calibration + ROI by EV bucket.

Fits src/margin_pricing.MarginDistribution on closing-line pairs from dev
seasons only (2016-2023), fits each signal's shrink weight w on dev
(2019-2023) by cover log loss, then scores 2024-25 holdout once:

  * push calibration on integer lines (key numbers vs others), vs a normal
  * cover Brier / log loss vs a normal-distribution pricer and vs 0.5
  * ROI by EV bucket (bet the +EV side at -110), with CLV
  * half-point buying (+0.5 at -120) when the pricer says it is worth it

Writes outputs/research/pricing_*.csv and, for production, models/margin_pairs.csv
(all completed seasons) + models/margin_pricing.json (dev-fitted params).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_lab import load_base, DEV, HOLD, OUT, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
import margin_pricing as mp  # noqa: E402

W_GRID = [round(x, 2) for x in np.arange(0, 1.01, 0.05)]
EV_BUCKETS = [(-1, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 9)]


def normal_probs(mu, line, sigma):
    """Continuous normal with a 1-point push window when the line is integer."""
    t = -line
    if float(line).is_integer():
        ph = 1 - norm.cdf(t + 0.5, mu, sigma)
        pa = norm.cdf(t - 0.5, mu, sigma)
        return ph, 1 - ph - pa, pa
    ph = 1 - norm.cdf(t, mu, sigma)
    return ph, 0.0, 1 - ph


def table(dist, sigma, mus, lines):
    emp = np.array([dist.probs(m, l) for m, l in zip(mus, lines)])
    nrm = np.array([normal_probs(m, l, sigma) for m, l in zip(mus, lines)])
    return emp, nrm


def cover_metrics(p_home, p_push, y_home, y_push):
    keep = ~y_push
    p = np.clip(p_home[keep] / np.clip(1 - p_push[keep], 1e-9, None), 1e-6, 1 - 1e-6)
    y = y_home[keep].astype(float)
    return {"n": int(keep.sum()), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5)}


def signals(base: pd.DataFrame) -> dict:
    lab = pd.read_csv(OUT / "lab_is_pr0.8_lam2_hl0.csv")
    labb = pd.read_csv(OUT / "lab_base.csv")[["game_id", "pred_resid_close"]] \
        .rename(columns={"pred_resid_close": "main_resid"})
    pt = pd.read_csv(ROOT / "data/processed/pt_lines.csv").drop(columns=["season"])
    d = lab.merge(pt, on="game_id", how="left").merge(labb, on="game_id", how="left")
    d["opener"] = d.open_line.where(d.open_line.notna(), d.pt_open)
    return {
        "opener_gap": d.assign(model=d.pred_indep, line=d.opener),
        "fpi": d.assign(model=-d.pt_fpi, line=d.opener),
        "main": d.assign(model=-d.close_line + d.main_resid, line=d.close_line),
    }


def main() -> None:
    base = load_base()
    done = base.dropna(subset=["close_line", "point_diff"])
    dev_pairs = done[done.season <= 2023]
    # tune bandwidth / tail weight inside dev: fit 2016-2020, score the exact
    # final-margin log-likelihood on 2021-2023
    fit_p, val_p = dev_pairs[dev_pairs.season <= 2020], dev_pairs[dev_pairs.season >= 2021]
    tune = []
    for bw in (0.75, 1.0, 1.5, 2.5, 4.0):
        for a in (0.02, 0.05, 0.10, 0.20):
            dd = mp.MarginDistribution(-fit_p.close_line.values, fit_p.point_diff.values, bw, a)
            ll = np.mean([np.log(dd.pmf(m)[int(k) - mp.K.min()] + 1e-12)
                          for m, k in zip(-val_p.close_line.values, val_p.point_diff.round().values)])
            tune.append({"bandwidth": bw, "tail_weight": a, "val_loglik": round(float(ll), 5)})
    sig_fit = float((fit_p.point_diff + fit_p.close_line).std())
    for sn in (10.0, 50.0, 200.0):
        dd = mp.KeyNumberNormal(-fit_p.close_line.values, fit_p.point_diff.values, sig_fit, sn)
        ll = np.mean([np.log(dd.pmf(m)[int(k) - mp.K.min()] + 1e-12)
                      for m, k in zip(-val_p.close_line.values, val_p.point_diff.round().values)])
        tune.append({"family": "key_number_normal", "shrink_n": sn, "val_loglik": round(float(ll), 5)})
    tune = pd.DataFrame(tune)
    tune["family"] = tune["family"].fillna("kernel")
    tune = tune.sort_values("val_loglik", ascending=False)
    tune.to_csv(OUT / "pricing_tuning.csv", index=False)
    best = tune.iloc[0]
    print("tuning (top 5):\n", tune.head(5).to_string(index=False))
    sigma_dev = float((dev_pairs.point_diff + dev_pairs.close_line).std())
    if best.family == "key_number_normal":
        FAM, BW, TW, SN = "key_number_normal", None, None, float(best.shrink_n)
        dist = mp.KeyNumberNormal(-dev_pairs.close_line.values, dev_pairs.point_diff.values, sigma_dev, SN)
    else:
        FAM, BW, TW, SN = "kernel", float(best.bandwidth), float(best.tail_weight), None
        dist = mp.MarginDistribution(-dev_pairs.close_line.values, dev_pairs.point_diff.values, BW, TW)
    resid = dev_pairs.point_diff + dev_pairs.close_line
    sigma = float(resid.std())
    print(f"pairs dev {len(dev_pairs)} (2016-2023); sigma {sigma:.2f}")

    # 1. key numbers: frequency of |margin| (dev) vs normal
    rows = []
    for k in [1, 2, 3, 4, 6, 7, 8, 10, 14, 17, 21]:
        emp = float((dev_pairs.point_diff.abs() == k).mean())
        mu = -dev_pairs.close_line.values
        nrm = float(np.mean(norm.cdf(k + 0.5, mu, sigma) - norm.cdf(k - 0.5, mu, sigma)
                            + norm.cdf(-k + 0.5, mu, sigma) - norm.cdf(-k - 0.5, mu, sigma)))
        rows.append({"abs_margin": k, "empirical_%": round(100 * emp, 2),
                     "normal_%": round(100 * nrm, 2)})
    kn = pd.DataFrame(rows)
    kn.to_csv(OUT / "pricing_key_numbers.csv", index=False)
    print(kn.to_string(index=False))

    # 2. push calibration on holdout integer closing lines (market mean)
    h = done[done.season.isin(HOLD)]
    hi = h[h.close_line.apply(lambda x: float(x).is_integer())]
    emp, nrm = table(dist, sigma, -hi.close_line.values, hi.close_line.values)
    hi = hi.assign(p_push_emp=emp[:, 1], p_push_nrm=nrm[:, 1],
                   push=(hi.point_diff + hi.close_line == 0))
    hi["group"] = np.where(hi.close_line.abs().isin([3, 7, 10, 14]),
                           "key (3/7/10/14)", "other integer")
    pc = hi.groupby("group").agg(n=("push", "size"), actual_push_pct=("push", "mean"),
                                 empirical_pred_pct=("p_push_emp", "mean"),
                                 normal_pred_pct=("p_push_nrm", "mean"))
    for c in ["actual_push_pct", "empirical_pred_pct", "normal_pred_pct"]:
        pc[c] = (100 * pc[c]).round(2)
    pc.to_csv(OUT / "pricing_push_calibration.csv")
    print("\nHoldout push calibration (closing lines):\n", pc.to_string())

    # 3. per signal: fit w on dev, score holdout
    sig = signals(base)
    params = {"family": FAM, "bandwidth": BW, "tail_weight": TW, "shrink_n": SN, "sigma": round(sigma, 3),
              "fit": "pairs 2016-2023 (dev); shrink w by dev cover log loss", "shrink": {}}
    cal_rows, roi_rows, buy_rows = [], [], []
    for name, d in sig.items():
        d = d.dropna(subset=["model", "line", "point_diff"])
        y_home = (d.point_diff + d.line > 0).values
        y_push = (d.point_diff + d.line == 0).values
        dv = d.season.isin(DEV).values
        best = None
        for w in W_GRID:
            mu = (-d.line + w * (d.model + d.line)).values
            e = np.array([dist.probs(m, l) for m, l in zip(mu[dv], d.line.values[dv])])
            ll = cover_metrics(e[:, 0], e[:, 1], y_home[dv], y_push[dv])["logloss"]
            if best is None or ll < best[0]:
                best = (ll, w)
        w = best[1]
        params["shrink"][name] = float(w)
        mu = (-d.line + w * (d.model + d.line)).values
        emp, nrm = table(dist, sigma, mu, d.line.values)
        for per, m in (("dev", dv), ("holdout", ~dv & d.season.isin(HOLD).values)):
            ce = cover_metrics(emp[m, 0], emp[m, 1], y_home[m], y_push[m])
            cn = cover_metrics(nrm[m, 0], nrm[m, 1], y_home[m], y_push[m])
            c5 = cover_metrics(np.full(m.sum(), .5), np.zeros(m.sum()), y_home[m], y_push[m])
            cal_rows.append({"signal": name, "w": w, "period": per, "n": ce["n"],
                             "brier_empirical": ce["brier"], "brier_normal": cn["brier"],
                             "brier_0.5": c5["brier"], "logloss_empirical": ce["logloss"],
                             "logloss_normal": cn["logloss"], "logloss_0.5": c5["logloss"]})
            # ROI by EV bucket (bet the +EV side)
            ev_h = np.array([mp.ev(p[0], p[2]) for p in emp[m]])
            ev_a = np.array([mp.ev(p[2], p[0]) for p in emp[m]])
            bet_home = ev_h >= ev_a
            evb = np.where(bet_home, ev_h, ev_a)
            x = d[m]
            res = np.where(bet_home, 1, -1) * (x.point_diff.values + x.line.values)
            clv = np.where(bet_home, 1, -1) * (x.line.values - x.close_line.values)
            for lo, hi_ in EV_BUCKETS:
                s = (evb >= lo) & (evb < hi_)
                wn, ls, ps = int((res[s] > 0).sum()), int((res[s] < 0).sum()), int((res[s] == 0).sum())
                roi_rows.append({"signal": name, "period": per, "ev_bucket": f"[{lo:g},{hi_:g})",
                                 "n": int(s.sum()), "w": wn, "l": ls, "p": ps,
                                 "win%": round(100 * wn / (wn + ls), 2) if wn + ls else np.nan,
                                 "roi%": round(100 * (wn - 1.1 * ls) / (1.1 * (wn + ls)), 2) if wn + ls else np.nan,
                                 "pred_win%": round(100 * float(np.mean(np.where(bet_home, emp[m, 0], emp[m, 2])[s]
                                                                        / (1 - emp[m, 1][s]))), 2) if s.any() else np.nan,
                                 "clv": round(float(np.mean(clv[s])), 3) if s.any() else np.nan})
            # half-point buying on +EV bets where pricer says buy
            pos = evb > 0
            bh = []
            for j in np.where(pos)[0]:
                pr = mp.price_bet(dist, mu[m][j], float(x.line.values[j]), bool(bet_home[j]))
                if pr["buy_half"]:
                    base_res = res[j]
                    bought = base_res + 0.5
                    bh.append((base_res, bought))
            if bh:
                a = np.array(bh)
                def units(r, price):
                    pay = 100 / -price
                    return float(np.sum(np.where(r > 0, pay, np.where(r < 0, -1, 0))))
                buy_rows.append({"signal": name, "period": per, "n_buy_flags": len(a),
                                 "units_no_buy_-110": round(units(a[:, 0], -110), 2),
                                 "units_buy_half_-120": round(units(a[:, 1], -120), 2),
                                 "units_buy_half_-125": round(units(a[:, 1], -125), 2),
                                 "units_buy_half_-130": round(units(a[:, 1], -130), 2),
                                 "pushes_avoided": int((a[:, 0] == 0).sum()),
                                 "losses_turned_push": int((a[:, 0] == -0.5).sum())})
    cal = pd.DataFrame(cal_rows); roi = pd.DataFrame(roi_rows); buy = pd.DataFrame(buy_rows)
    cal.to_csv(OUT / "pricing_calibration.csv", index=False)
    roi.to_csv(OUT / "pricing_roi_by_ev.csv", index=False)
    buy.to_csv(OUT / "pricing_half_point.csv", index=False)
    print("\nCover calibration:\n", cal.to_string(index=False))
    print("\nROI by EV bucket:\n", roi.to_string(index=False))
    print("\nHalf-point buys:\n", buy.to_string(index=False))

    # production artifacts: all completed seasons for the distribution;
    # params (w) stay the dev-fitted ones
    pairs = done[["season", "close_line", "point_diff"]].assign(mu=lambda x: -x.close_line,
                                                                 margin=lambda x: x.point_diff.round().astype(int))
    pairs[["season", "mu", "margin"]].to_csv(ROOT / "models/margin_pairs.csv", index=False)
    (ROOT / "models/margin_pricing.json").write_text(json.dumps(params, indent=2))
    print("\nparams:", params)


if __name__ == "__main__":
    main()
