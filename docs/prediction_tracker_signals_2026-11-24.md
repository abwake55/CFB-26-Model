# Prediction Tracker signals — November 24, 2026

Source: Prediction Tracker (thepredictiontracker.com) free per-game CSVs,
2015–2026, about 50 computer systems' predicted lines plus PT's record of the
opening, midweek and updated market lines. `src/prediction_tracker.py` fetches
them, converts to the repo's home-spread convention (negative = home favoured),
blanks PT market values with an obvious flipped sign, and matches games to CFBD
`game_id`s by final score and team names (97–98% of PT rows per season;
`data/processed/pt_lines.csv`, 8,702 games). CI refreshes the current season
weekly (Weekly Refresh) and the live week daily (Line Snapshots).

## Setup

- Lab rows: `outputs/research/lab_is_pr0.8_lam2_hl0.csv` (walk-forward, the
  adopted opener-gap model's out-of-sample `pred_indep`). 5,057 games joined.
- Decision line ("opener"): CFBD opener when present (3,777 games), else PT's
  recorded opener (1,273 games, mostly 2019–20). This is what makes
  2019–2020 usable as dev for opener rules.
- Dev 2019–2023, holdout 2024–25. Flat −110. CLV = points beaten vs the
  consensus close. "Midweek" re-prices the same picks at PT's midweek line
  (roughly when PT's own numbers are public), "close" at the closing line.
- Selection was written into `scripts/research/pt_signals.py` before holdout
  was examined: over 9 rules × thresholds {2..7}, pick the single candidate with
  the highest dev ROI among those with dev n ≥ 150 and dev CLV > 0. The holdout
  was scored once. Frozen rule: `models/pt_rule.json`.

Features: `pt_fpi` (FPI line), `pt_sys_median` / `pt_sys_mean` / `pt_sys_std` /
`pt_sys_n` (consensus and disagreement across systems), FPI-vs-opener gap
(`opener − pt_fpi`), consensus-vs-opener gap, and our opener-gap model's
agreement or disagreement with the consensus.

## Each feature at its own dev-best threshold (n ≥ 150)

| Rule | Thr | Period | n | W-L-P | Win % | ROI | CLV | Midweek ROI | Close ROI |
|---|---|---|---|---|---|---|---|---|---|
| consensus_gap | ≥6 | dev | 212 | 115-96-1 | 54.50 | +4.05% | +2.48 | -10.54% | -10.05% |
| consensus_gap | ≥6 | holdout | 61 | 28-31-2 | 47.46 | -9.40% | +0.75 | -12.63% | -12.37% |
| consensus_gap, low disagreement (std ≤ dev median) | ≥4 | dev | 232 | 133-96-3 | 58.08 | +10.88% | +1.69 | +3.98% | +3.90% |
| | ≥4 | holdout | 69 | 33-33-3 | 50.00 | -4.55% | +0.65 | -8.82% | -4.55% |
| **fpi_gap (FROZEN)** | ≥7 | dev | 225 | 133-88-4 | 60.18 | +14.89% | +2.71 | +2.86% | +2.86% |
| **fpi_gap (FROZEN)** | ≥7 | holdout | 91 | 50-39-2 | 56.18 | +7.25% | +0.80 | +3.94% | +3.94% |
| fpi_gap, consensus agrees | ≥7 | dev | 208 | 120-84-4 | 58.82 | +12.30% | +2.74 | +0.18% | +0.18% |
| | ≥7 | holdout | 82 | 44-36-2 | 55.00 | +5.00% | +0.93 | +3.70% | +1.35% |
| massey_gap | ≥6 | dev | 705 | 365-329-11 | 52.59 | +0.41% | +0.49 | -1.91% | -2.46% |
| | ≥6 | holdout | 287 | 136-144-7 | 48.57 | -7.27% | +0.06 | -5.89% | -5.22% |
| sagarin_gap | ≥7 | dev | 310 | 162-144-4 | 52.94 | +1.07% | +1.68 | -7.40% | -6.11% |
| | ≥7 | holdout | 61 | 31-28-2 | 52.54 | +0.31% | +0.21 | -1.36% | +3.28% |
| opener-gap model (baseline) | ≥4 | dev | 1431 | 745-659-27 | 53.06 | +1.30% | +0.69 | -2.59% | -2.64% |
| | ≥4 | holdout | 483 | 251-221-11 | 53.18 | +1.52% | +0.52 | -0.34% | +0.10% |
| opener-gap, consensus agrees | ≥4 | dev | 1059 | 560-480-19 | 53.85 | +2.80% | +0.91 | -1.90% | -2.34% |
| | ≥4 | holdout | 378 | 197-172-9 | 53.39 | +1.92% | +0.56 | +0.32% | -0.93% |
| opener-gap, consensus disagrees | ≥5 | dev | 271 | 141-124-6 | 53.21 | +1.58% | +0.19 | -0.64% | -0.27% |
| | ≥5 | holdout | 58 | 31-26-1 | 54.39 | +3.83% | -0.01 | -1.25% | +12.50% |

Note: the opener-gap baseline here uses CFBD-or-PT openers and dev 2019–23, so
it differs from the earlier doc's dev 2021–23 numbers.

## FPI gap, every threshold

| Thr | Period | n | W-L-P | Win % | ROI | CLV | Midweek ROI |
|---|---|---|---|---|---|---|---|
| 2 | dev | 1950 | 997-920-33 | 52.01 | -0.71% | +0.68 | -4.25% |
| 3 | dev | 1332 | 684-626-22 | 52.21 | -0.32% | +0.85 | -4.69% |
| 4 | dev | 895 | 452-427-16 | 51.42 | -1.83% | +1.04 | -6.50% |
| 5 | dev | 581 | 297-275-9 | 51.92 | -0.87% | +1.39 | -6.56% |
| 6 | dev | 363 | 201-158-4 | 55.99 | +6.89% | +1.88 | -1.07% |
| 7 | dev | 225 | 133-88-4 | 60.18 | +14.89% | +2.71 | +2.86% |
| 2 | holdout | 833 | 415-401-17 | 50.86 | -2.91% | +0.35 | -4.66% |
| 3 | holdout | 552 | 283-258-11 | 52.31 | -0.13% | +0.40 | -2.44% |
| 4 | holdout | 368 | 199-162-7 | 55.12 | +5.24% | +0.44 | +2.06% |
| 5 | holdout | 227 | 123-99-5 | 55.41 | +5.77% | +0.64 | +3.19% |
| 6 | holdout | 140 | 79-58-3 | 57.66 | +10.09% | +0.66 | +6.52% |
| 7 | holdout | 91 | 50-39-2 | 56.18 | +7.25% | +0.80 | +3.94% |

**2026 to date (weeks 1–11, frozen rule, out of sample, not live):** 25 bets,
9-16-0, 36.0%, −31.27% ROI, CLV +0.72.

Binomial p-values vs the 52.38% break-even (one-sided): dev 133-88 p=0.012;
holdout 50-39 p=0.27; holdout + 2026 combined 59-55 p=0.59. Dev performance is
confined to the extreme tail (≥6), which is what a selection over ~54
candidates tends to find. CLV is positive at every threshold in every period,
including 2026: the market does move toward FPI after the open. ATS is not
established.

## MAE (spread, points)

| Predictor | Dev | Holdout |
|---|---|---|
| Close | 12.267 | 11.920 |
| PT computer-adjusted line | 12.308 | 11.956 |
| PT midweek line | 12.307 | 11.999 |
| Opener | 12.474 | 12.046 |
| System median | 12.586 | 12.257 |
| FPI | 12.589 | 12.312 |
| Sagarin | 12.772 | 12.236 |
| CFB-26 market-free (`pred_indep`) | 12.846 | 12.436 |
| Massey | 12.919 | 12.684 |
| FEI | 13.143 | 12.569 |

## Live wiring

- `scripts/weekly_pipeline.py` attaches the PT signal to every prediction run
  (`pt_*`, `pt_edge`, `flag_pt`) and lists flags under "SPREAD SIGNALS"
  (small stake / tracking — not CORE).
- The ledger records the PT columns and, the first time a game is flagged,
  `pt_bet_line` / `pt_bet_ts` / `pt_bet_edge`: the current consensus line and
  time the bet would be placed. Later re-records keep that first line. The
  summary grades `pt_bets` at the placed line, and `pt_bets_at_opener` at the
  opener for comparison.
- Timing caveat: PT posts the week's file early in the week (observed Tuesday
  morning PT), after the Sunday opener. Live bets are therefore graded at the
  line when the flag first appears, not at the opener the rule was frozen on.
  The midweek columns above are the realistic expectation: dev +2.86%,
  holdout +3.94% ROI.

Reproduce: `python scripts/fetch_prediction_tracker.py --all` then
`python scripts/research/pt_signals.py` (needs the lab file from
`scripts/research/spread_lab.py`).
