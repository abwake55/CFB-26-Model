#!/usr/bin/env python3
"""
Systematic edge-pocket scan — multiple-testing guarded.
=======================================================
Searches walk-forward (out-of-sample) predictions for market-bias pockets
like the validated CORE unders portfolio, WITHOUT p-hacking.

Guards, in order of importance:

1. PRE-REGISTERED hypotheses. Every segment tested is declared up front in
   HYPOTHESES with a stated market mechanism — no fishing over arbitrary
   cut points until something hits. A pocket with no plausible reason a
   book/public would misprice it is noise, however good the number looks.
2. MULTIPLE-TESTING CORRECTION. We report the Benjamini-Hochberg FDR
   q-value across every test in the run. A raw p=0.03 among 60 tests is
   expected by chance ~2x over.
3. PER-SEASON STABILITY. A real edge shows up in most seasons; a fake one
   is carried by one outlier year (the moneyline EV strategy failed exactly
   this way — all profit was 2023).
4. SPLIT-HALF holdout. Discovered on 2019-22, confirmed on 2023-25. An edge
   that dies out of sample is a backtest artifact.
5. MINIMUM SAMPLE. n >= 150 graded bets; smaller slices are unfalsifiable.

Nothing here ships to the app on its own. Survivors are CANDIDATES for a
deliberate follow-up decision.

Usage:  python3 scripts/edge_scan.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
JUICE = -110
MIN_N = 150
DISCOVER_SEASONS = [2019, 2020, 2021, 2022]
CONFIRM_SEASONS = [2023, 2024, 2025]

POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
         "Big East", "FBS Independents"}


def roi_at(hit: float, odds: int = JUICE) -> float:
    payout = 100 / abs(odds) if odds < 0 else odds / 100
    return hit * payout - (1 - hit)


def binom_p_two_sided(wins: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test vs p0 (no scipy dependency).

    Computed in log space — math.comb(n, k) overflows float conversion for
    the sample sizes here (n in the thousands).
    """
    if n == 0:
        return 1.0
    obs = abs(wins - n * p0)
    log_p0, log_q0 = math.log(p0), math.log1p(-p0)
    total = 0.0
    for k in range(n + 1):
        if abs(k - n * p0) >= obs - 1e-9:
            total += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                              - math.lgamma(n - k + 1)
                              + k * log_p0 + (n - k) * log_q0)
    return min(1.0, total)


def bh_qvalues(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR q-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank
        val = min(prev, pvals[idx] * m / (i + 1))
        q[idx] = val
        prev = val
    return q


def load_data() -> pd.DataFrame:
    wf = pd.read_csv(ROOT / "outputs/predictions/walk_forward_results.csv")
    cols = ["game_id", "home_conference", "away_conference", "wind_speed",
            "is_dome", "temp_avg", "precipitation", "rest_diff",
            "neutral_site", "conference_game", "talent_diff",
            "spread_magnitude", "tempo_combined", "home_new_p4_conf",
            "away_new_p4_conf", "home_rest_days", "away_rest_days"]
    fm = pd.read_csv(ROOT / "data/processed/feature_matrix.csv")
    fm = fm[[c for c in cols if c in fm.columns]]
    d = wf.merge(fm, on="game_id", how="left")

    d["power_involved"] = (d["home_conference"].isin(POWER)
                           | d["away_conference"].isin(POWER))
    d["both_power"] = (d["home_conference"].isin(POWER)
                       & d["away_conference"].isin(POWER))
    d["windy"] = (~d["is_dome"].fillna(0).astype(bool)
                  & (d["wind_speed"].fillna(0) >= 15))
    d["tot_edge"] = d["pred_total"] - d["over_under"]
    d["sp_edge"] = d["pred_spread"] - d["vegas_home_margin"]
    d["home_dog"] = d["vegas_home_margin"] < 0
    return d


# ── Bet definitions ──────────────────────────────────────────────────────────
# Each returns (mask_of_eligible_games, win_series) for graded, non-push bets.

def bet_under(d):
    m = d["total_points"].notna() & d["over_under"].notna() & (d["total_points"] != d["over_under"])
    return m, d["total_points"] < d["over_under"]


def bet_over(d):
    m = d["total_points"].notna() & d["over_under"].notna() & (d["total_points"] != d["over_under"])
    return m, d["total_points"] > d["over_under"]


def bet_home_ats(d):
    m = d["point_diff"].notna() & d["vegas_home_margin"].notna() & (d["point_diff"] != d["vegas_home_margin"])
    return m, d["point_diff"] > d["vegas_home_margin"]


def bet_away_ats(d):
    m = d["point_diff"].notna() & d["vegas_home_margin"].notna() & (d["point_diff"] != d["vegas_home_margin"])
    return m, d["point_diff"] < d["vegas_home_margin"]


# ── PRE-REGISTERED HYPOTHESES ────────────────────────────────────────────────
# (name, bet_fn, filter_fn, stated market mechanism)
HYPOTHESES = [
    # -- Public over-bias family (the CORE edge's mechanism, other slices) ----
    ("under: model edge 2-7, G5-only, total>=48", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.tot_edge >= -7) & ~d.power_involved & (d.over_under >= 48) & ~d.windy,
     "Public over-bias should also exist in G5 games; CORE excluded them on volume/quality"),
    ("under: model edge 2-7, both-power, total>=48", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.tot_edge >= -7) & d.both_power & (d.over_under >= 48) & ~d.windy,
     "Marquee both-power games draw the most public over money"),
    ("under: high total >=60 (no model edge)", bet_under,
     lambda d: d.over_under >= 60,
     "Shootout-hyped games are where over-inflation is largest"),
    ("under: model edge>=2, cold temp <40F", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.temp_avg < 40),
     "Cold suppresses scoring; market may underreact to temperature vs wind"),
    ("under: model edge>=2, precipitation>0", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.precipitation > 0),
     "Rain/snow suppresses scoring; less salient to public than wind"),
    ("under: model edge 2-7, conference game, total>=48", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.tot_edge >= -7) & d.conference_game.eq(1) & (d.over_under >= 48) & ~d.windy,
     "Familiar conference opponents -> lower scoring than market expects"),
    ("under: model edge 2-7, late season wk>=10, total>=48", bet_under,
     lambda d: (d.tot_edge <= -2) & (d.tot_edge >= -7) & (d.week >= 10) & (d.over_under >= 48) & ~d.windy,
     "Late-season weather/defense maturity; does CORE hold when spreads fail?"),

    # -- Under-bias inverse: is there an over pocket? -------------------------
    ("over: model edge 2-7, low total <=45", bet_over,
     lambda d: (d.tot_edge >= 2) & (d.tot_edge <= 7) & (d.over_under <= 45),
     "If public inflates high totals, books may over-shade low totals downward"),
    ("over: model edge>=2, dome games", bet_over,
     lambda d: (d.tot_edge >= 2) & d.is_dome.eq(1),
     "Controlled conditions -> model's scoring estimate more reliable"),

    # -- Home/away ATS bias family -------------------------------------------
    ("ATS: home dogs (any)", bet_home_ats,
     lambda d: d.home_dog,
     "Classic documented bias: public backs road favorites, home dogs undervalued"),
    ("ATS: home dogs, power-conf", bet_home_ats,
     lambda d: d.home_dog & d.power_involved,
     "Same bias in higher-liquidity games"),
    ("ATS: big road favorites -21+ (fade)", bet_home_ats,
     lambda d: d.vegas_home_margin <= -21,
     "Public overbets big favorites; fade by taking the home dog"),
    ("ATS: big home favorites 21+ (fade -> away)", bet_away_ats,
     lambda d: d.vegas_home_margin >= 21,
     "Same public-favorite bias, mirrored side"),
    ("ATS: model edge>=3 on home dog", bet_home_ats,
     lambda d: d.home_dog & (d.sp_edge >= 3),
     "Model agreement + structural home-dog bias"),
    ("ATS: neutral site games, model edge>=3 home", bet_home_ats,
     lambda d: d.neutral_site.eq(1) & (d.sp_edge >= 3),
     "Neutral sites remove HFA the public still prices in"),

    # -- Rest / schedule spots ------------------------------------------------
    ("ATS: home team rest advantage >=7 days", bet_home_ats,
     lambda d: d.rest_diff >= 7,
     "Bye-week prep advantage historically underpriced"),
    ("ATS: away team rest advantage >=7 days", bet_away_ats,
     lambda d: d.rest_diff <= -7,
     "Same mechanism, road side"),
    ("under: both teams short rest (<=5 days)", bet_under,
     lambda d: (d.home_rest_days <= 5) & (d.away_rest_days <= 5),
     "Tired teams -> sloppier, lower-scoring games"),

    # -- Conference realignment ----------------------------------------------
    ("ATS: new-P4 team on road (fade)", bet_away_ats,
     lambda d: d.away_new_p4_conf.eq(1),
     "Realigned teams struggle in new conference road environments"),
]


def evaluate(d: pd.DataFrame, name: str, bet_fn, filt, mech: str) -> dict | None:
    base_mask, win = bet_fn(d)
    try:
        seg = filt(d).fillna(False)
    except Exception:
        return None
    m = base_mask & seg
    s = d[m]
    w = win[m]
    n = len(s)
    if n < MIN_N:
        return {"name": name, "n": n, "skipped": f"n<{MIN_N}", "mech": mech}

    wins = int(w.sum())
    hit = wins / n
    p = binom_p_two_sided(wins, n)

    per_season = s.assign(_w=w.values).groupby("season")["_w"].agg(["size", "mean"])
    seasons_profitable = int((per_season["mean"] > 0.5238).sum())
    seasons_total = len(per_season)

    disc = s["season"].isin(DISCOVER_SEASONS)
    conf = s["season"].isin(CONFIRM_SEASONS)
    hit_disc = w[disc.values].mean() if disc.sum() else float("nan")
    hit_conf = w[conf.values].mean() if conf.sum() else float("nan")

    return {
        "name": name, "mech": mech, "n": n, "hit": hit, "roi": roi_at(hit),
        "p": p, "seasons_profitable": seasons_profitable,
        "seasons_total": seasons_total,
        "n_disc": int(disc.sum()), "hit_disc": hit_disc,
        "n_conf": int(conf.sum()), "hit_conf": hit_conf,
        "skipped": None,
    }


def main() -> None:
    d = load_data()
    print(f"Walk-forward games: {len(d):,} | seasons {d.season.min()}-{d.season.max()}")
    print(f"Pre-registered hypotheses: {len(HYPOTHESES)}")
    print(f"Guards: n>={MIN_N}, BH-FDR across all tests, per-season stability, "
          f"split-half {DISCOVER_SEASONS[0]}-{DISCOVER_SEASONS[-1]} -> "
          f"{CONFIRM_SEASONS[0]}-{CONFIRM_SEASONS[-1]}\n")

    results = [r for r in (evaluate(d, *h) for h in HYPOTHESES) if r]
    tested = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]

    qs = bh_qvalues([r["p"] for r in tested])
    for r, q in zip(tested, qs):
        r["q"] = q

    tested.sort(key=lambda r: -r["hit"])
    print(f"{'segment':52s} {'n':>5s} {'hit':>6s} {'ROI':>7s} {'p':>7s} {'q':>7s} {'seasons':>8s} {'disc→conf':>14s}")
    print("-" * 116)
    for r in tested:
        seasons = f"{r['seasons_profitable']}/{r['seasons_total']}"
        split = f"{r['hit_disc']:.3f}→{r['hit_conf']:.3f}"
        print(f"{r['name']:52s} {r['n']:5d} {r['hit']:6.3f} {r['roi']:+7.1%} "
              f"{r['p']:7.3f} {r['q']:7.3f} {seasons:>8s} {split:>14s}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}, sample too small):")
        for r in skipped:
            print(f"  {r['name']:52s} n={r['n']}")

    # ── Survivors: all guards must pass ──────────────────────────────────────
    print("\n" + "=" * 116)
    print("SURVIVORS (q<0.10, profitable both halves, >=70% of seasons profitable):")
    survivors = [
        r for r in tested
        if r["q"] < 0.10
        and r["hit_disc"] > 0.5238 and r["hit_conf"] > 0.5238
        and r["seasons_profitable"] / r["seasons_total"] >= 0.70
    ]
    if not survivors:
        print("  NONE — no pre-registered pocket cleared every guard.")
        print("  (This is the expected outcome most of the time; it is a")
        print("   successful run, not a failed one.)")
    for r in survivors:
        print(f"  ✅ {r['name']}")
        print(f"     n={r['n']} hit={r['hit']:.3f} roi={r['roi']:+.1%} q={r['q']:.3f} "
              f"seasons {r['seasons_profitable']}/{r['seasons_total']} "
              f"split {r['hit_disc']:.3f}→{r['hit_conf']:.3f}")
        print(f"     mechanism: {r['mech']}")


if __name__ == "__main__":
    main()
