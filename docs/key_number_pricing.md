# Key-number pricing — November 24, 2026

`src/margin_pricing.py` turns an expected home margin into cover / push /
lose probabilities at any spread, using an empirical CFB margin distribution,
and from those an EV at −110 and the value of buying half a point.
Research and fitting: `scripts/research/key_number_pricing.py` (outputs in
`outputs/research/pricing_*.csv`).

## Method

- Pairs: closing-line expected margin `mu = -close` and final margin, FBS vs
  FBS, 2016–2023 (5,125 games) for research; all completed seasons for the
  production table (`models/margin_pairs.csv`).
- Two families compared on a dev-internal split (fit 2016–2020, score exact
  final-margin log-likelihood on 2021–2023):
  1. kernel over `mu` (bandwidth 0.75–4, normal tail weight 0.02–0.2),
  2. **key-number normal**: N(mu, σ=15.61) re-weighted per |margin| by
     observed / normal-expected counts, shrunk with 10 pseudo-games.
  Winner: key-number normal (val log-lik −4.004 vs best kernel −4.029).
- Model view: `mu_hat = mu_market + w (mu_model − mu_market)`, `w` fitted on
  dev 2019–2023 by cover log loss: opener-gap 0.15, FPI 0.25, main 0.10.
  (`models/margin_pricing.json`).

## Key numbers (dev 2016–2023, share of games by |final margin|)

| |Margin| | Observed | Normal |
|---|---|---|
| 3 | 10.22% | 3.76% |
| 7 | 8.66% | 3.59% |
| 10 | 4.59% | 3.38% |
| 14 | 4.41% | 3.02% |
| 17 | 3.63% | 2.71% |
| 21 | 3.79% | 2.28% |
| 2 | 2.81% | 3.78% |
| 8 | 2.71% | 3.53% |

## Holdout 2024–25

Push calibration on integer closing lines:

| Lines | n | Actual push | Pricer | Normal |
|---|---|---|---|---|
| 3 / 7 / 10 / 14 | 308 | 8.44% | 5.58% | 2.56% |
| other integers | 497 | 2.21% | 2.30% | 2.56% |

The pricer is much closer than a normal on key numbers, but still
under-predicts pushes on them by about 3 points.

Cover calibration (pushes excluded; lower is better):

| Signal | Period | n | Brier pricer | Brier normal | Brier 0.5 | Log loss pricer | Log loss normal | Log loss 0.5 |
|---|---|---|---|---|---|---|---|---|
| opener-gap | dev | 3509 | 0.25003 | 0.24963 | 0.25 | 0.69321 | 0.69241 | 0.69315 |
| opener-gap | holdout | 1539 | 0.25061 | 0.25010 | 0.25 | 0.69436 | 0.69335 | 0.69315 |
| FPI | dev | 3440 | 0.24976 | 0.24940 | 0.25 | 0.69265 | 0.69192 | 0.69315 |
| FPI | holdout | 1468 | 0.25043 | 0.24982 | 0.25 | 0.69401 | 0.69279 | 0.69315 |
| main | dev | 3554 | 0.25049 | 0.24991 | 0.25 | 0.69413 | 0.69297 | 0.69315 |
| main | holdout | 1534 | 0.25092 | 0.24997 | 0.25 | 0.69499 | 0.69309 | 0.69315 |

No signal's cover probability beats a coin flip on holdout under either
pricer. The pricer's value is pushes and half points, not cover skill.

ROI by EV bucket (bet the +EV side at −110; holdout):

| Signal | EV bucket | n | W-L-P | Win % | ROI | Pred win % | CLV |
|---|---|---|---|---|---|---|---|
| opener-gap | 0–2% | 311 | 139-163-9 | 46.03 | −12.13% | 52.87 | −0.02 |
| opener-gap | 2–5% | 249 | 124-121-4 | 50.61 | −3.38% | 54.13 | +0.37 |
| opener-gap | 5–10% | 87 | 54-31-2 | 63.53 | +21.28% | 55.88 | +0.50 |
| FPI | 0–2% | 269 | 138-130-1 | 51.49 | −1.70% | 52.89 | +0.04 |
| FPI | 2–5% | 298 | 150-142-6 | 51.37 | −1.93% | 54.22 | +0.17 |
| FPI | 5–10% | 167 | 84-79-4 | 51.53 | −1.62% | 56.08 | +0.52 |
| FPI | ≥10% | 27 | 13-13-1 | 50.00 | −4.55% | 58.64 | +0.63 |
| main (close) | 0–2% | 329 | 145-174-10 | 45.45 | −13.22% | 52.86 | — |
| main (close) | 2–5% | 169 | 84-77-8 | 52.17 | −0.40% | 54.09 | — |
| main (close) | 5–10% | 26 | 13-13-0 | 50.00 | −4.55% | 55.30 | — |

Predicted win rates run 2–8 points above realised: EV is not a reliable bet
filter. CLV rises with EV for both opener signals. Full dev+holdout table:
`outputs/research/pricing_roi_by_ev.csv`.

Half-point buying, when the pricer says +0.5 is worth it (units over the
flagged +EV bets):

| Signal | Period | Bets | No buy (−110) | Buy (−120) | Buy (−125) | Buy (−130) |
|---|---|---|---|---|---|---|
| opener-gap | dev | 230 | +12.09 | +16.83 | +12.40 | +8.31 |
| opener-gap | holdout | 92 | +10.55 | +15.33 | +13.40 | +11.62 |
| FPI | dev | 240 | −7.73 | −7.83 | −12.00 | −15.85 |
| FPI | holdout | 107 | +10.64 | +12.33 | +10.20 | +8.23 |
| main | dev | 159 | +14.09 | +13.33 | +10.20 | +7.31 |
| main | holdout | 93 | −14.18 | −6.83 | −8.40 | −9.85 |

At −120, buying helped in 4 of 6 cells (all three holdout cells); at −125 or
worse it mostly does not pay. Rule: buy +0.5 only when the pricer flags it and
the price is −120 or better.

## Live

`scripts/weekly_pipeline.py` prices every flagged PT and opener-gap signal at
the current line (`<sig>_p_cover`, `_p_push`, `_ev`, `_buy_half`); the weekly
text shows cover/push and "buy +0.5 if ≤−120"; the ledger records them.
