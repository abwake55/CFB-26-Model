# High-total under signal — November 24, 2026

Bet the under when the first total we see is at least 60. The decision line
is the captured opener if we have it (the week's line-snapshot `first` total,
else CFBD `over_under_open`), otherwise the current total. Implementation:
`src/high_total_under.py`. The flag is `flag_high_total_under`; the ledger
grades the bet at `high_total_under_bet_line` (frozen the first time the flag
appears) against the closing total.

This is a line-only rule: it does not use the totals model. Stake is 0.25
units until the promotion rule in `docs/betting_rule_2026-11-24.md` is met.
No new Odds API calls; snapshot openers come from files already on disk.

## Settings

The threshold of 60 was chosen on **2019–2023 only**. 2024–2025 is a held-out
score. Universe: FBS vs FBS, completed games, flat −110. ROI is
`(w − 1.1·l) / (1.1·(w+l))`. Under CLV is `bet line − close` (a higher
number is better for the under).

Cuts of 48, 52, 56, 60, 65 and the dev-mean total were compared on dev.
Under at a close ≥ 65 had the best dev ROI but only 49 holdout games and
went 23-26 (−10.39%). Under at a close ≥ 60 was the next-best dev rule
with a large sample and is the one that held.

## Results

At the **closing** total ≥ 60:

| Period | n | W-L-P | Win % | ROI |
|---|---|---|---|---|
| dev 2019–23 | 966 | 515-435-16 | 54.21 | +3.49% |
| holdout 2024–25 | 216 | 118-96-2 | 55.14 | +5.27% |

Fair **opener** test (select on the opening total ≥ 60, grade at the opener):

| Period | n | W-L-P | Win % | ROI | CLV | CLV>0 |
|---|---|---|---|---|---|---|
| dev 2019–23 | 133 | 76-56-1 | 57.58 | +9.92% | +0.54 | 55.6% |
| holdout 2024–25 | 209 | 118-90-1 | 56.73 | +8.30% | +0.70 | 56.5% |

Games that also failed the CORE gate (so this is incremental, not a
restatement of CORE) at a close ≥ 60:

| Period | n | W-L-P | Win % | ROI |
|---|---|---|---|---|
| dev 2019–23 | 830 | 442-373-15 | 54.23 | +3.54% |
| holdout 2024–25 | 209 | 113-94-2 | 54.59 | +4.22% |

College football totals overall are only mildly skewed (2019–25 mean 54.42,
median 54.0, skew +0.376) and unders versus the close are 50.44% across all
games. The edge is in the high posted totals, not in betting every under.

A 4% edge at −110 still needs a couple of thousand bets before the record
is decisive, so this stays at 0.25u with CLV as the early read. Promote or
demote with the same live-ledger rule as the other small-stake signals.
