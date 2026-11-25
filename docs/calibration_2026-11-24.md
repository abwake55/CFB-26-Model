# Win-probability calibration check (2026-11-24)

Question (the Nate Silver point): when the model says a team wins 60% of the time, does it win about 60% of the time?

Everything below is produced by `scripts/calibration_report.py` from data already in the repo. The weekly workflow reruns it (step 9d) and refreshes the files in `outputs/research/`, so the numbers in this doc are a snapshot as of 2026-11-24.

## Method

- **Probabilities tested:** `pred_home_win_p` from `outputs/predictions/walk_forward_results.csv` (the production probability: classifier blended with the spread-implied probability, then early-season shrinkage; every game is predicted by a model that never saw it). 5,184 games.
  - Dev: 2019-23 (3,613 games, includes 2020, which is predicted but never trained on).
  - Holdout: 2024-25 (1,571 games). Nothing is fit on it.
  - Live 2026: settled games in `ledger_2026.csv` that were recorded before kickoff by the scheduled job (301 games).
- **Excluded from live 2026:** 1,270 settled games with no posted line (FCS / non-FBS). They get a flat ~0.65 fallback probability, not a real prediction, and would swamp the 60-70% bins. Weeks 1-3 replay and the Oct 29 - Nov 17 backfill (410 more games) were predicted after the fact, so they are shown separately as "2026 all", not as live.
- **Favourite view:** every game is turned into "the side the model favours": predicted probability vs whether that side won. Bins are 5 points wide, 50-55 through 95-100. ECE is the game-weighted mean of |predicted - actual| per bin.
- **Noise floor:** with a finite sample, ECE is never 0 even for perfect probabilities. `ece_noise_floor` is the average ECE you would get from sampling noise alone if the model's probabilities were exactly right. An ECE near its floor is not evidence of miscalibration.
- **Market:** no-vig probability from the median closing moneyline across books (`master_lines.csv`, CFBD lines freeze at kickoff). Moneyline coverage is 60% of dev games (it is mostly missing before 2023), 98% of holdout, 96% of live.
- **Correction test:** Platt (logistic on logit p) and isotonic regression are fit on dev only and scored once on the holdout. Ship rule, fixed before looking: holdout Brier must improve by at least 0.0005, log loss by at least 0.001, and the paired-bootstrap 95% interval on the Brier gain must exclude zero.

## Result: calibrated, no correction shipped

| Split | n | Brier | Log loss | ECE | ECE noise floor |
|---|---|---|---|---|---|
| Dev 2019-23 | 3,613 | 0.1763 | 0.5248 | 0.0102 | 0.017 |
| Holdout 2024-25 | 1,571 | 0.1792 | 0.5279 | 0.0295 | 0.025 |
| Live 2026 (pre-kickoff, lined) | 301 | 0.1810 | 0.5362 | 0.0698 | 0.061 |

Reliability by bin (favourite view; predicted vs actual favourite win rate):

| Bin | Dev n | Dev pred / actual | Holdout n | Holdout pred / actual | Live n | Live pred / actual |
|---|---|---|---|---|---|---|
| 50-55 | 407 | 52.6 / 54.0 | 191 | 52.9 / 54.4 | 48 | 52.7 / 54.2 |
| 55-60 | 426 | 57.7 / 58.9 | 230 | 57.5 / 60.9 | 51 | 57.8 / 64.7 |
| 60-65 | 418 | 62.4 / 61.2 | 203 | 62.5 / 57.1 | 50 | 62.3 / 56.0 |
| 65-70 | 418 | 67.5 / 69.4 | 175 | 67.3 / 72.6 | 39 | 67.5 / 76.9 |
| 70-75 | 344 | 72.4 / 73.8 | 171 | 72.4 / 71.9 | 29 | 72.8 / 86.2 |
| 75-80 | 329 | 77.4 / 77.5 | 123 | 77.3 / 75.6 | 32 | 77.6 / 84.4 |
| 80-85 | 374 | 82.5 / 82.1 | 122 | 82.2 / 80.3 | 18 | 82.0 / 94.4 |
| 85-90 | 298 | 87.6 / 86.9 | 116 | 87.4 / 93.1 | 10 | 87.1 / 90.0 |
| 90-95 | 316 | 92.4 / 92.4 | 128 | 92.3 / 94.5 | 13 | 93.3 / 100 |
| 95-100 | 283 | 97.3 / 96.1 | 112 | 97.4 / 98.2 | 11 | 95.8 / 100 |

How to read it:

- **Dev:** every bin is within one standard error of the diagonal. No over- or under-confidence anywhere.
- **Holdout:** 9 of 10 bins are within 1.96 standard errors. The only flag is 85-90% (predicted 87.4%, actual 93.1%, n=116): the model was under-confident there, which is the safe direction. The 60-65 and 65-70 bins miss in opposite directions (-5.3 and +5.3 points), which looks like noise rather than a pattern. Overall ECE 0.030 is about 0.005 above its noise floor.
- **Live 2026:** the 301 pre-kickoff lined games lean under-confident in the 55-85% range (favourites win more often than predicted; most bins have 10-50 games, so none is individually reliable). ECE 0.070 against a noise floor of 0.061 is consistent with noise at this sample size. This is worth re-checking weekly, not acting on yet.
- **Home-side bias:** the mean home win probability is 57.5% on the holdout against an actual 58.8% (-1.3 points), and 59.5% against 61.1% live. Small, and in the same direction in both.
- **Where it is over-confident:** nowhere that survives the noise test. The model is, if anything, slightly cautious.

## Against the market

No-vig closing-moneyline probability on the same games (lower is better):

| Split | n | Model Brier | Market Brier | Model log loss | Market log loss | Model ECE | Market ECE |
|---|---|---|---|---|---|---|---|
| Dev 2019-23 (games with a moneyline) | 2,169 | 0.1860 | 0.1890 | 0.5513 | 0.5599 | 0.017 | 0.024 |
| Holdout 2024-25 | 1,536 | 0.1833 | 0.1836 | 0.5397 | 0.5429 | 0.030 | 0.016 |
| Live 2026 | 289 | 0.1885 | 0.1835 | 0.5565 | 0.5484 | 0.071 | 0.094 |

- Holdout: the model is level with the closing moneyline (Brier 0.1833 vs 0.1836, a difference far inside the noise). That matches the spread result: roughly equal to the market, not better. Dev scores slightly better than market, but dev is where the model's settings were tuned, so it does not count as evidence.
- Live 2026: the market is better by 0.005 Brier on 289 games. That is the same size as the sampling noise at this n, but it is not the direction anyone wants, and it argues against betting moneylines off this probability.
- The market is the better-calibrated of the two on the holdout (ECE 0.016 vs 0.030), so market probabilities are the safer input when pricing a bet.

## Corrections: fit on dev, scored once on holdout

| Holdout 2024-25 | Brier | Log loss | ECE | Brier gain vs raw (95% CI) | Log loss gain | Ships? |
|---|---|---|---|---|---|---|
| Raw model | 0.1792 | 0.5279 | 0.0295 | - | - | - |
| Platt (a=0.990, b=+0.033) | 0.1791 | 0.5277 | 0.0246 | +0.0001 (-0.0002 to +0.0004) | +0.0002 | No |
| Isotonic | 0.1793 | 0.5285 | 0.0283 | -0.0001 (-0.0011 to +0.0009) | -0.0006 | No |

- The fitted Platt curve is almost the identity line (slope 0.99, intercept +0.03): the dev data say the probabilities need essentially no stretch or shift.
- Platt's gain is 0.0001 Brier, and the interval includes zero. Isotonic is slightly worse than raw on the holdout (it overfits the dev bins).
- Neither passes the ship rule, so **no correction was applied**. Production probabilities are unchanged.
- Sample-size caveat: the holdout has 1,571 games, so Brier differences smaller than about 0.0003 cannot be detected. A correction that small would not move a bet either.

## What this means for betting

- The win probability can be taken at face value at the level of precision the data support. A "60%" call has won about 60% in dev and holdout.
- Calibration is not the same as edge. A calibrated model that only ties the closing moneyline has no moneyline edge (holdout Brier 0.1833 vs 0.1836). Moneyline bets still need a price better than the model's no-vig number by a margin, as in `docs/betting_rule_2026-11-24.md`.
- The live 2026 gap to the market (0.1885 vs 0.1835) is the number to watch in the weekly report. If it persists past ~700 lined games, it is a real weakness; at 289 it is not conclusive.

## Files

- `scripts/calibration_report.py` - the report (needs pandas, numpy, scikit-learn, matplotlib).
- `outputs/research/calibration_reliability.png` - reliability charts (dev, holdout, live) with game-count histograms.
- `outputs/research/calibration_bins.csv` - bin table per split, with standard error and over/under-confident flag.
- `outputs/research/calibration_summary.csv` / `.json` - Brier, log loss, ECE, noise floor per season and per correction, plus the ship decision.
- `outputs/research/calibration_market.csv`, `calibration_market_bins.csv` - model vs no-vig closing moneyline.
- `tests/test_calibration_report.py` - unit tests for the metric helpers.
- `.github/workflows/weekly_refresh.yml`, step 9d - refreshes the report with newly settled games. It is report-only and `continue-on-error`; the existing commit step (`commit_and_push.sh`, rebase-and-retry) pushes the outputs.

## Not covered

- This tests the home win probability. The spread, totals and signal models are calibrated separately (`outputs/research/pricing_calibration.csv`).
- Moneylines come from CFBD's book list (median across books), not a single sharp close.
- The `models/win_prob_platt.json` file left over from earlier work is not used by production and is not touched here.
