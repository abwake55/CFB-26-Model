#!/usr/bin/env python3
"""
Win-probability calibration report
==================================
Question: when the model says a team wins 60% of the time, does it win ~60%?

Inputs (all already in the repo, nothing is fetched):
  outputs/predictions/walk_forward_results.csv  leak-free walk-forward predictions
        dev     = 2019-2023 (2020 is predicted but never trained on)
        holdout = 2024-2025 (scored once, never used to fit anything here)
  outputs/predictions/ledger_2026.csv           2026 live predictions (settled games)
  data/processed/master_lines.csv               closing moneylines -> no-vig market probability

Outputs (outputs/research/):
  calibration_bins.csv          favourite-view reliability table, per split
  calibration_summary.csv       Brier / log loss / ECE per split and per correction
  calibration_market.csv        model vs no-vig market on games that have a moneyline
  calibration_reliability.png   reliability chart
  calibration_summary.json      machine-readable result incl. the ship decision

Corrections (Platt on logit(p), isotonic) are fit on DEV only and scored once on
HOLDOUT. The ship rule is fixed in advance (SHIP_* below): both Brier and log loss
must improve on the holdout and the paired-bootstrap 95% interval for the Brier
difference must exclude zero. Nothing here changes production by itself.

    python3 scripts/calibration_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions"
OUT = ROOT / "outputs" / "research"
BINS = np.arange(0.50, 1.0001, 0.05)
DEV_SEASONS = (2019, 2023)
HOLDOUT_SEASONS = (2024, 2025)
SHIP_MIN_BRIER_GAIN = 0.0005   # holdout Brier must drop by at least this ...
SHIP_MIN_LL_GAIN = 0.001       # ... and log loss by at least this ...
EPS = 1e-6                     # ... and the bootstrap CI on the Brier gain must exclude 0
N_BOOT = 2000
SEED = 7
STATS = {}


# ── metrics ────────────────────────────────────────────────────────────────────
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fav_view(p, y):
    """Symmetric view: probability of the side the model favours, and whether it won."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    fav = p >= 0.5
    return np.where(fav, p, 1 - p), np.where(fav, y, 1 - y)


def bin_table(p, y, label):
    fp, fy = fav_view(p, y)
    idx = np.clip(np.digitize(fp, BINS[1:-1], right=False), 0, len(BINS) - 2)
    rows = []
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        pred = float(fp[m].mean()) if n else np.nan
        act = float(fy[m].mean()) if n else np.nan
        se = float(np.sqrt(act * (1 - act) / n)) if n else np.nan
        rows.append(dict(split=label, bin=f"{BINS[b]*100:.0f}-{BINS[b+1]*100:.0f}", n=n,
                         pred_win_rate=pred, actual_win_rate=act,
                         gap=(act - pred) if n else np.nan, std_err=se,
                         verdict=("n<30" if n < 30 else
                                  "over-confident" if (act - pred) < -1.96 * se else
                                  "under-confident" if (act - pred) > 1.96 * se else "ok")))
    return rows


def ece(p, y):
    fp, fy = fav_view(p, y)
    idx = np.clip(np.digitize(fp, BINS[1:-1]), 0, len(BINS) - 2)
    tot, n = 0.0, len(fp)
    for b in range(len(BINS) - 1):
        m = idx == b
        if m.any():
            tot += m.sum() / n * abs(fy[m].mean() - fp[m].mean())
    return float(tot)


def ece_noise_floor(p, n_sim=300, seed=11):
    """Mean ECE you would see from sampling noise alone if the probabilities were exactly right."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    return float(np.mean([ece(p, (rng.random(len(p)) < p).astype(float)) for _ in range(n_sim)]))


def summarise(p, y, label):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return dict(ece_noise_floor=ece_noise_floor(p), label=label, n=len(y), mean_pred_home=float(p.mean()), home_win_rate=float(y.mean()),
                brier=brier(p, y), log_loss=logloss(p, y), ece=ece(p, y))


def no_vig(h_ml, a_ml):
    def imp(ml):
        ml = np.asarray(ml, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    ph, pa = imp(h_ml), imp(a_ml)
    return ph / (ph + pa)


# ── data ───────────────────────────────────────────────────────────────────────
def closing_ml():
    """Median home/away moneyline across books per game (CFBD lines freeze at kickoff)."""
    ml = pd.read_csv(ROOT / "data" / "processed" / "master_lines.csv",
                     usecols=["game_id", "home_moneyline", "away_moneyline"])
    ml = ml.dropna(subset=["home_moneyline", "away_moneyline"])
    ml = ml[(ml.home_moneyline.abs() >= 100) & (ml.away_moneyline.abs() >= 100)]
    g = ml.groupby("game_id").agg(home_moneyline=("home_moneyline", "median"),
                                  away_moneyline=("away_moneyline", "median")).reset_index()
    g["market_p"] = no_vig(g.home_moneyline, g.away_moneyline)
    return g[["game_id", "market_p"]]


def load():
    wf = pd.read_csv(PRED / "walk_forward_results.csv")
    wf = wf.dropna(subset=["pred_home_win_p", "home_win"]).copy()
    wf["p"] = wf["pred_home_win_p"].astype(float)
    wf["y"] = wf["home_win"].astype(int)
    wf["split"] = np.where(wf.season.between(*DEV_SEASONS), "dev 2019-23",
                           np.where(wf.season.between(*HOLDOUT_SEASONS), "holdout 2024-25", "other"))
    lg = pd.read_csv(PRED / "ledger_2026.csv")
    lg = lg[lg["completed"].astype(str).str.lower().eq("true")].dropna(subset=["pred_win_p", "home_win"])
    # one row per game: the earliest record (closest to what was actually published first)
    lg = lg.sort_values("recorded_at").drop_duplicates("game_id", keep="first").copy()
    # Games with no posted line (FCS / non-FBS) get a flat base-rate fallback (~0.65), not a
    # real model prediction -> excluded. Weeks 1-3 replay and the Oct 29-Nov 17 backfill were
    # predicted after the fact -> excluded from the headline "live" set, reported separately.
    STATS["n_unpriced"] = int(lg["pred_spread"].isna().sum())
    lg = lg[lg["pred_spread"].notna()].copy()
    lg["p"] = lg["pred_win_p"].astype(float)
    lg["y"] = lg["home_win"].astype(int)
    lg["split"] = "live 2026"
    lg["is_live"] = lg["provenance"].eq("live")
    mk = closing_ml()
    wf = wf.merge(mk, on="game_id", how="left")
    lg = lg.merge(mk, on="game_id", how="left")
    lg_all = lg.copy()
    lg_all["split"] = "2026 all provenance"
    lg = lg[lg.is_live].copy()
    STATS["all"] = lg_all
    return wf, lg


# ── corrections ────────────────────────────────────────────────────────────────
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_corrections(p, y):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(p).reshape(-1, 1), y)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
    return dict(platt=lambda q: 1 / (1 + np.exp(-(a * logit(np.asarray(q)) + b))), iso=iso.predict,
                platt_params=dict(a=a, b=b),
                iso_x=[float(v) for v in iso.X_thresholds_], iso_y=[float(v) for v in iso.y_thresholds_])


def boot_gain(p0, p1, y, rng):
    """Paired bootstrap of (Brier_raw - Brier_corrected); positive = correction better."""
    d = (p0 - y) ** 2 - (p1 - y) ** 2
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── chart ──────────────────────────────────────────────────────────────────────
def chart(tables, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = [s for s in ("dev 2019-23", "holdout 2024-25", "live 2026") if s in tables]
    fig, axes = plt.subplots(2, len(splits), figsize=(5.2 * len(splits), 8.6),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, s in enumerate(splits):
        t = tables[s]
        t = t[t.n > 0]
        ax, hx = axes[0, j], axes[1, j]
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect")
        ax.errorbar(t.pred_win_rate, t.actual_win_rate, yerr=1.96 * t.std_err, fmt="o-", color="#1f77b4",
                    capsize=3, label="model (95% CI)")
        for _, r in t.iterrows():
            ax.annotate(f"n={int(r.n)}", (r.pred_win_rate, r.actual_win_rate), textcoords="offset points",
                        xytext=(4, -12), fontsize=7)
        ax.set_xlim(0.48, 1.02); ax.set_ylim(0.38, 1.08)
        sm = summ[s]
        ax.set_title(f"{s}\nn={sm['n']}  Brier {sm['brier']:.4f}  LL {sm['log_loss']:.3f}  ECE {sm['ece']:.3f}", fontsize=10)
        ax.set_xlabel("predicted win probability of the favourite")
        ax.set_ylabel("actual win rate of the favourite")
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
        hx.bar(t.pred_win_rate, t.n, width=0.04, color="#999")
        hx.set_xlim(0.48, 1.02); hx.set_ylabel("games"); hx.set_xlabel("predicted (favourite)")
    fig.suptitle("CFB model win-probability reliability (favourite view)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wf, lg = load()
    dev, hold = wf[wf.split == "dev 2019-23"], wf[wf.split == "holdout 2024-25"]
    sets = {"dev 2019-23": dev, "holdout 2024-25": hold, "live 2026": lg}
    lg_all = STATS["all"]
    sets["2026 all (incl. replay/backfill)"] = lg_all

    tables, summ, rows = {}, {}, []
    for s, d in sets.items():
        if len(d) == 0:
            continue
        t = pd.DataFrame(bin_table(d.p.values, d.y.values, s))
        tables[s] = t
        summ[s] = summarise(d.p.values, d.y.values, s)
    pd.concat(tables.values()).to_csv(OUT / "calibration_bins.csv", index=False, float_format="%.4f")

    # per-season detail (raw model)
    for s in sorted(wf.season.unique()):
        d = wf[wf.season == s]
        rows.append({**summarise(d.p.values, d.y.values, f"season {s}"), "correction": "raw"})
    for s, d in sets.items():
        if s in summ:
            rows.append({**summ[s], "correction": "raw"})

    # corrections: fit on dev only, score on holdout once
    corr = fit_corrections(dev.p.values, dev.y.values)
    rng = np.random.default_rng(SEED)
    result = {}
    raw_h = summ["holdout 2024-25"]
    for name in ("platt", "iso"):
        ph = np.asarray(corr[name](hold.p.values), float)
        sm = summarise(ph, hold.y.values, "holdout 2024-25")
        sm["correction"] = name
        gain, lo, hi = boot_gain(hold.p.values, ph, hold.y.values.astype(float), rng)
        sm.update(brier_gain=raw_h["brier"] - sm["brier"], ll_gain=raw_h["log_loss"] - sm["log_loss"],
                  brier_gain_ci95=[lo, hi])
        result[name] = sm
        rows.append(sm)
        # per-holdout-season check
        for s in HOLDOUT_SEASONS:
            m = (hold.season == s).values
            rows.append({**summarise(ph[m], hold.y.values[m], f"holdout season {s}"), "correction": name})
        if "live 2026" in sets:
            pl = np.asarray(corr[name](lg.p.values), float)
            rows.append({**summarise(pl, lg.y.values, "live 2026"), "correction": name})
        pdv = np.asarray(corr[name](dev.p.values), float)
        rows.append({**summarise(pdv, dev.y.values, "dev 2019-23 (in-sample fit)"), "correction": name})
    pd.DataFrame(rows).to_csv(OUT / "calibration_summary.csv", index=False, float_format="%.5f")

    def ships(r):
        return bool(r["brier_gain"] >= SHIP_MIN_BRIER_GAIN and r["ll_gain"] >= SHIP_MIN_LL_GAIN
                    and r["brier_gain_ci95"][0] > 0)
    for k, r in result.items():
        r["meets_ship_rule"] = ships(r)
    ship = [k for k, r in result.items() if r["meets_ship_rule"]]
    best = max(ship, key=lambda k: result[k]["brier_gain"]) if ship else None

    # market comparison on games with a moneyline
    mrows = []
    for s, d in sets.items():
        m = d.dropna(subset=["market_p"])
        if len(m) < 30:
            continue
        pm, y = m.market_p.values, m.y.values
        mrows.append(dict(split=s, n=len(m), coverage=len(m) / len(d),
                          model_brier=brier(m.p.values, y), market_brier=brier(pm, y),
                          model_logloss=logloss(m.p.values, y), market_logloss=logloss(pm, y),
                          model_ece=ece(m.p.values, y), market_ece=ece(pm, y),
                          mean_abs_model_minus_market=float(np.mean(np.abs(m.p.values - pm)))))
    mk = pd.DataFrame(mrows)
    mk.to_csv(OUT / "calibration_market.csv", index=False, float_format="%.5f")
    mk_bins = []
    for s, d in sets.items():
        m = d.dropna(subset=["market_p"])
        if len(m) >= 30:
            mk_bins += bin_table(m.market_p.values, m.y.values, f"market {s}")
    pd.DataFrame(mk_bins).to_csv(OUT / "calibration_market_bins.csv", index=False, float_format="%.4f")

    chart(tables, summ, OUT / "calibration_reliability.png")

    out = dict(generated_from=dict(walk_forward_games=int(len(wf)), live_2026_priced_settled=int(len(lg)),
                                   all_2026_priced_settled=int(len(lg_all)),
                                   excluded_2026_unpriced_fallback=STATS["n_unpriced"]),
               raw=summ, corrections=result,
               platt_params=corr["platt_params"],
               ship_rule=dict(min_brier_gain=SHIP_MIN_BRIER_GAIN, min_logloss_gain=SHIP_MIN_LL_GAIN,
                              bootstrap_ci_excludes_zero=True),
               ship=best, market=mrows)
    (OUT / "calibration_summary.json").write_text(json.dumps(out, indent=2))

    # console summary
    print(pd.concat(tables.values()).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for s, v in summ.items():
        print(f"{s:18s} n={v['n']:5d} brier={v['brier']:.4f} ll={v['log_loss']:.4f} ece={v['ece']:.4f}")
    for k, r in result.items():
        print(f"holdout {k:6s} brier={r['brier']:.4f} ll={r['log_loss']:.4f} ece={r['ece']:.4f} "
              f"gainB={r['brier_gain']:+.5f} CI={r['brier_gain_ci95']} gainLL={r['ll_gain']:+.5f} ship={r['meets_ship_rule']}")
    print(mk.to_string(index=False))
    print("SHIP:", best)


if __name__ == "__main__":
    sys.exit(main())
