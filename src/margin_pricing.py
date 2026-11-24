"""Key-number pricing: predicted margin -> cover / push / lose probabilities.

College football margins are lumpy: 3, 7, 10, 14 (and 17, 21, 24, 28) land far
more often than a normal curve says. This module prices a spread bet from an
EMPIRICAL distribution of final margins conditional on the expected margin:

    P(M = k | mu) = (1-a) * sum_i K_h(mu_i - mu) 1[M_i = k] / sum_i K_h(mu_i - mu)
                    + a * Normal(k; mu, sigma)          (light tail smoothing)

fitted on (closing-line implied margin mu_i, final margin M_i) pairs, mirrored
((-mu_i, -M_i) added) so the distribution is symmetric. The model's own view
enters through a shrunk mean mu_hat = mu_mkt + w * (mu_model - mu_mkt), with w
fitted on dev seasons by cover log loss (scripts/research/key_number_pricing.py).

Convention: `line` is the HOME spread (negative = home favoured). Home covers
when M + line > 0, pushes when M + line == 0.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "models" / "margin_pairs.csv"
PARAMS_PATH = ROOT / "models" / "margin_pricing.json"
K = np.arange(-90, 91)            # supported integer margins
BANDWIDTH = 1.5                   # points of expected margin
TAIL_WEIGHT = 0.10
SIGMA = 15.5
WIN_PAYOUT = 100 / 110            # -110


class MarginDistribution:
    def __init__(self, mu: np.ndarray, margin: np.ndarray, bandwidth: float = BANDWIDTH,
                 tail_weight: float = TAIL_WEIGHT, sigma: float = SIGMA):
        mu = np.asarray(mu, float)
        margin = np.asarray(margin).round().astype(int)
        ok = np.isfinite(mu) & (np.abs(margin) <= K.max())
        mu, margin = mu[ok], margin[ok]
        self.mu = np.concatenate([mu, -mu])
        self.margin = np.concatenate([margin, -margin])
        self.h, self.a, self.sigma = bandwidth, tail_weight, sigma
        self._onehot_idx = self.margin - K.min()

    def pmf(self, mu: float) -> np.ndarray:
        w = np.exp(-0.5 * ((self.mu - mu) / self.h) ** 2)
        emp = np.bincount(self._onehot_idx, weights=w, minlength=len(K))
        emp = emp / emp.sum() if emp.sum() > 0 else np.zeros(len(K))
        nrm = np.exp(-0.5 * ((K - mu) / self.sigma) ** 2)
        nrm /= nrm.sum()
        return (1 - self.a) * emp + self.a * nrm

    def probs(self, mu: float, line: float) -> tuple[float, float, float]:
        """(P home covers, P push, P away covers) at home spread `line`."""
        p = self.pmf(mu)
        x = K + line
        return float(p[x > 0].sum()), float(p[x == 0].sum()), float(p[x < 0].sum())


class KeyNumberNormal(MarginDistribution):
    """Discretised normal N(mu, sigma) re-weighted by a per-|margin| factor
    r_|k| = observed count / normal-expected count (fitted on history):

        P(M = k | mu)  proportional to  N(k; mu, sigma) * r_|k|

    Key numbers (3, 7, 10, 14, ...) get their historical extra mass at every
    expected margin, without the data-sparsity of a kernel over mu."""

    def __init__(self, mu, margin, sigma: float = SIGMA, shrink_n: float = 50.0):
        mu = np.asarray(mu, float)
        margin = np.asarray(margin).round().astype(int)
        ok = np.isfinite(mu) & (np.abs(margin) <= K.max())
        mu, margin = mu[ok], margin[ok]
        self.sigma = float(sigma)
        absk = np.arange(0, K.max() + 1)
        obs = np.bincount(np.abs(margin), minlength=len(absk)).astype(float)
        exp = np.zeros(len(absk))
        for m in mu:
            d = np.exp(-0.5 * ((K - m) / self.sigma) ** 2)
            d /= d.sum()
            np.add.at(exp, np.abs(K), d)
        # shrink each ratio toward 1 with `shrink_n` pseudo-observations
        self.r = (obs + shrink_n) / (exp + shrink_n)
        self.r[0] = (obs[0] + shrink_n) / (exp[0] + shrink_n)

    def pmf(self, mu: float) -> np.ndarray:
        d = np.exp(-0.5 * ((K - mu) / self.sigma) ** 2) * self.r[np.abs(K)]
        return d / d.sum()


def ev(p_win: float, p_lose: float, price: float = -110) -> float:
    """Expected profit per unit staked at an american price."""
    payout = 100 / -price if price < 0 else price / 100
    return p_win * payout - p_lose


def price_bet(dist: MarginDistribution, mu: float, line: float, bet_home: bool,
              buy_price: float = -120) -> dict:
    """Cover/push probabilities, EV at -110, and the value of buying half a
    point (line moved 0.5 in the bettor's favour at `buy_price`)."""
    ph, pp, pa = dist.probs(mu, line)
    win, lose = (ph, pa) if bet_home else (pa, ph)
    better = line + 0.5 if bet_home else line - 0.5
    bh, bp, ba = dist.probs(mu, better)
    bwin, blose = (bh, ba) if bet_home else (ba, bh)
    ev_now = ev(win, lose)
    ev_buy = ev(bwin, blose, buy_price)
    return {"p_cover": round(win, 4), "p_push": round(pp, 4), "p_lose": round(lose, 4),
            "ev": round(ev_now, 4), "half_point_cover_gain": round(bwin - win, 4),
            "ev_buy_half": round(ev_buy, 4), "buy_half": bool(ev_buy > ev_now)}


@lru_cache(maxsize=1)
def load() -> tuple[MarginDistribution, dict] | None:
    if not PAIRS_PATH.exists():
        return None
    pairs = pd.read_csv(PAIRS_PATH)
    params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    if params.get("family") == "key_number_normal":
        return KeyNumberNormal(pairs.mu.values, pairs.margin.values,
                               params.get("sigma", SIGMA), params.get("shrink_n", 50.0)), params
    return MarginDistribution(pairs.mu.values, pairs.margin.values,
                              params.get("bandwidth", BANDWIDTH),
                              params.get("tail_weight", TAIL_WEIGHT)), params


def attach(out: pd.DataFrame) -> pd.DataFrame:
    """Price each flagged spread signal at the CURRENT line (where it would be
    bet): <sig>_p_cover, <sig>_p_push, <sig>_ev, <sig>_buy_half."""
    loaded = load()
    if loaded is None:
        return out
    dist, params = loaded
    w = params.get("shrink", {})
    out = out.copy()
    specs = {"pt": ("flag_pt", "pt_fpi", "fpi"),
             "opener_gap": ("flag_opener_gap", "pred_indep_margin", "opener_gap")}
    for sig, (flag, model_col, wkey) in specs.items():
        if flag not in out.columns or model_col not in out.columns:
            continue
        cols = {c: np.nan for c in (f"{sig}_p_cover", f"{sig}_p_push", f"{sig}_ev",
                                    f"{sig}_ev_buy_half")}
        for c, v in cols.items():
            out[c] = v
        out[f"{sig}_buy_half"] = False
        for i, r in out.iterrows():
            if not (r.get(flag) is True or r.get(flag) == 1):
                continue
            line = pd.to_numeric(r.get("spread"), errors="coerce")
            if pd.isna(line):
                line = pd.to_numeric(r.get("spread_open"), errors="coerce")
            m = pd.to_numeric(r.get(model_col), errors="coerce")
            if pd.isna(line) or pd.isna(m):
                continue
            model_margin = -m if model_col == "pt_fpi" else m   # pt_fpi is a spread
            mu = -line + float(w.get(wkey, 0.5)) * (model_margin + line)
            edge_col = "pt_edge" if sig == "pt" else "opener_gap_edge"
            bet_home = float(r.get(edge_col)) > 0
            pr = price_bet(dist, mu, float(line), bet_home)
            out.at[i, f"{sig}_p_cover"] = pr["p_cover"]
            out.at[i, f"{sig}_p_push"] = pr["p_push"]
            out.at[i, f"{sig}_ev"] = pr["ev"]
            out.at[i, f"{sig}_ev_buy_half"] = pr["ev_buy_half"]
            out.at[i, f"{sig}_buy_half"] = pr["buy_half"]
    return out
