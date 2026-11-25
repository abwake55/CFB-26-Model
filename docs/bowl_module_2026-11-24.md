# Bowl module, game-day update and edge ranges — November 24, 2026

Two of Nate Silver's points drove this work: update when the evidence
changes, and say how uncertain each number is. Three pieces were built:

1. A **bowl module**: bowl-specific adjustments for QB changes, transfer-portal
   exits and coaching changes, backtested on bowls only. **Status: paper.**
2. A **game-day update**: a scheduled QB check that re-projects the remaining
   games, saves timestamped projections and flags picks whose edge changed.
3. **Edge ranges**: each pick now shows what edges of its size actually
   delivered in the backtest, with an 80% range, in the outputs and the ledger.

No paid data was used. Everything comes from CFBD (Patreon key, in CI) and the
free Prediction Tracker CSVs. No new Odds API calls in the regular season;
during bowl season, at most one 3-credit call per daily run (see the schedule below).

## 1. Bowl module

### Data (CFBD, pulled in CI by `research_data_pull.yml`, `what=bowl,bowlplays`)

| File | What | Seasons |
|---|---|---|
| `data/research/cfbd_postseason_games.csv` | postseason games with bowl names | 2017–2025 (2026 not published yet) |
| `data/research/cfbd_coaches.csv` | head coaches by season (interim = 1-game season) | 2017–2026 |
| `data/research/cfbd_portal.csv` | transfer portal entries with dates | cycles after the 2020–2025 seasons |
| `data/research/cfbd_draft.csv` | NFL draft picks (post-hoc opt-out proxy, research only) | drafts 2018–2026 |
| `data/research/cfbd_bowl_first_passes.csv` | first pass plays per team per bowl (a check on who started) | 2017–2025 |
| `data/research/cfbd_passing_by_game.csv` | per-game passing (existing) | 2017–2026 |

Bowl lines: closing lines for every season 2017–2025. CFBD openers only
exist for 2021–2025 bowls; 2019–20 use Prediction Tracker's opener where
it has one. FBS-vs-FBS bowls with a closing line: 40, 39, 40, 26, 38, 42 (2017–22),
42, 46 (2023, 2024), 46 (2025).

### Features (`src/bowl_context.py`)

Per team, home minus away:
- `qb_change`: the season's most-started QB did not play in the bowl. A QB
  benched during the game still threw passes, so he **does not** count. That
  keeps bad in-game play from leaking into the feature. The first-pass plays
  agree with this definition in all 132 changes it found. They also show 19
  more games where the primary QB played but someone else threw first; those
  are not counted as changes.
- `qb_starter_share`: the regular-season attempt share of the QB who plays.
- `portal_out`, `portal_out_q` (3★+), `portal_qb_out`: the team's players
  who entered the portal before kickoff.
- `interim_coach`, `coach_change`: an interim coach runs the bowl, or the
  team had more than one head coach that season.
- `draft_r1_3`: round 1–3 draft picks the next spring. This is a **post-hoc**
  opt-out proxy. It's reported for research, never used to pick a rule, and
  never used live.

### Is the market slow to react? (slope of points per unit, home minus away)

| Feature | Period | n | Cover margin at close | Opener→close move |
|---|---|---|---|---|
| QB change | dev 2017–22 | 225 | −1.96 | −2.10 |
| QB change | holdout 2023–24 | 88 | −4.04 | −2.39 |
| QB change | 2025 | 46 | −3.55 | −2.12 |
| Interim coach | dev / holdout / 2025 | 225 / 88 / 46 | −4.19 / −6.96 / −2.99 | −1.89 / −3.34 / +0.11 |
| Portal (all) | dev / holdout / 2025 | 106 / 88 / 46 | −0.05 / +0.09 / −0.89 | ≈0 |
| Draft R1–3 (post-hoc) | dev / holdout / 2025 | 225 / 88 / 46 | +0.71 / +1.16 / −1.00 | ≈0 |

The opener moves toward QB and coaching news, but the close still doesn't
fully price it. Portal counts carry almost no signal. Few games had an interim
coach (19 dev, 7 holdout), so treat that slope with care. Full table:
`outputs/research/bowl_lab_features.csv`.

### Backtest (`scripts/research/bowl_lab.py`)

The adjustment is a ridge fit of the cover margin at the closing line on the
live features, using dev bowls (2017–22) only. 27 variants were tried: 4
feature sets, 3 ridge strengths and 3 thresholds (one set is research-only and
can't be selected). Dev picked one; holdout and 2025 were scored once. The
draft proxy could never be selected.

**Selected: all live features, λ=5, |adjustment| ≥ 2, bet at the closing
line (game-day, starters known).** Results at −110:

| Period | n | W-L-P | Win % | ROI |
|---|---|---|---|---|
| dev 2017–22 | 61 | 37-22-2 | 62.7% | +19.1% |
| **holdout 2023–24** | 48 | 29-18-1 | 61.7% | **+17.4%** |
| 2025 (extra unseen) | 21 | 12-9-0 | 57.1% | +9.1% |

- Holdout plus 2025: 41-27-1 (60.3%). That's one-sided p ≈ 0.057 vs a coin
  flip and p ≈ 0.12 vs the 52.4% break-even, before any correction for the
  27 variants tried.
- Almost every variant was positive on the holdout (`outputs/research/bowl_lab_results.csv`).
  That suggests a real effect, but samples are small: 40–46 bowls a season.
- **CLV:** the rule bets at the close, so its CLV is 0 by construction.
  - Bet at the **opener**, the same adjustment went 33-14-1 on the holdout
    (+33.3%, CLV +2.20 points) and 12-8-1 in 2025 (CLV +1.39). This is an upper
    bound: QB news usually isn't known when the line opens.
  - The part that doesn't depend on the QB (coach and portal), bet at the
    opener, went 14-10 on the holdout (+11.4%, CLV +1.24) and 8-5 in 2025 (CLV +0.54).

**The usual signals don't carry over to bowls:**

| Rule | dev 2017–22 | holdout 2023–24 | 2025 |
|---|---|---|---|
| Main model, all games, at close | 75-68-3 (+0.1%) | 46-40-1 (+2.1%) | 22-21-1 (−2.3%) |
| Opener-gap ≥5, at opener | 22-20 (0.0%, CLV +2.17) | 13-22 (−29.1%, CLV −0.70) | 4-2-2 |
| PT FPI ≥7, at opener | 2-0 | 1-5 (CLV −0.42) | 0 bets |
| Every bowl underdog, at close | 108-114-3 (−7.0%) | 43-44-1 (−5.6%) | 22-23-1 (−6.5%) |
| Underdogs of 7+, at close | 45-37 (+4.8%) | 12-19 (−26.1%) | 6-8 |

### Decision

- **Bowl adjustment: paper.** It shows as a flag on the bowl board
  (`outputs/picks/bowls_2026.json`) and in `outputs/predictions/ledger_2026_bowls.csv`.
  - Each flag records the first line it was seen at (the bet line) and the
    time. The row freezes at kickoff and is graded, with CLV against the last
    pre-kickoff line.
  - It doesn't clearly hold up: p ≈ 0.06 with 27 variants tried.
  - Graduation follows the betting rule (50+ live bets with positive CLV and
    ≥52.4% ATS). Bowls give about 40 games a year, so this takes several seasons.
- **Opener-gap and PT signals: paper only in bowls** (see the betting rule).

### How the 2026 bowls are handled

`scripts/bowl_board.py` runs from the game-day workflow every day in December
and January 1–20. It no-ops until CFBD publishes the 2026 postseason schedule.
Then, on each run, it does the following:
- **Schedule and lines:** gets the FBS bowls and CFP games and the CFBD lines,
  which are free.
- **Odds API:** makes at most one call, and only when a bowl kicks off within
  48 hours and the free quota check shows ≥150 credits left.
- **Likely QB per team** (`src/qb_watch.py`), strongest source first:
  1. a hand-entered override in `data/context/qb_overrides.csv` (confirmed news)
  2. the primary QB is in the transfer portal → next QB by attempts
  3. the QB who started the last game
- **Projections:** runs the regular model stack (main, opener-gap, PT,
  pricing, edge ranges) where it can, then adds the bowl adjustment.
- **Saves:**
  - a timestamped row per bowl, appended to `outputs/predictions/bowl_projections_2026.csv`
  - the bowl ledger
  - the board JSON

CFBD has no injury, opt-out or depth-chart feed, and ESPN doesn't publish
college depth charts. So "confirmed" only comes from overrides, and
everything else is labeled "likely".

## 2. Game-day update (`scripts/game_day_update.py`, `.github/workflows/game_day_update.yml`)

| When | What |
|---|---|
| Saturdays Aug–Nov, 9:00 AM PT | QB check and re-projection for the week's remaining games |
| Daily Dec 1–31 and Jan 1–20, 8:30 AM PT | the same on Saturdays, plus the bowl board once the schedule is out |
| Manual | `mode` = auto / regular / bowls; `odds` box (off by default) |

Regular season, for each ledger game not yet kicked off:
- **QB:** refreshes the season's per-game passing (only new weeks are
  fetched; cached in `data/context/qb_passing_2026.csv`) and works out the
  likely starter.
- **Re-projection:** the market-free (opener-gap) margin moves by the
  dev-fitted QB effect in `models/qb_adjust.json` when a team's likely
  starter isn't its most-started QB.
  - The main model is tied to the market line, and FPI is external, so
    neither gets a QB adjustment.
  - Every edge is re-priced at the newest snapshot line. The snapshot is
    already paid for, so this costs **zero Odds API credits**.
- **Flags a recorded pick (PT / opener-gap) when:**
  - its edge moved ≥1.5 points from the bet line,
  - its edge fell below the signal's threshold at the current line, or
  - a team in the game has a **new** starter (≤2 starts) per the box scores
    or an override.
- **History:** appends every row with a timestamp to
  `outputs/predictions/game_day_projections_2026.csv`. The ledger's recorded
  rows are never changed. The latest summary goes to `outputs/picks/gameday_2026_wWW.json`.

**Regular-season QB evidence is thin** (`scripts/research/qb_out_effect.py`).
"Primary out" means the season's most-started QB threw no pass:

| | Period | n games | QB-out games | vs close | vs market-free model |
|---|---|---|---|---|---|
| primary out (this game) | dev 2019–23 | 2,900 | 838 | +0.28 ± 0.54 | −0.68 ± 0.56 |
| | holdout 2024–25 | 1,294 | 332 | −0.01 ± 0.83 | −1.37 ± 0.86 |
| primary out last game | dev | 2,603 | 783 | +0.38 ± 0.55 | −0.25 ± 0.57 |
| | holdout | 1,168 | 292 | +0.85 ± 0.89 | −0.09 ± 0.91 |

In the regular season the closing line already prices QB absences, so the
game-day update is a **risk check**, not a new signal. The model adjustment is
small (−0.68 points per side). Bowls are different: there the close under-reacts.

## 3. Edge ranges (`src/edge_ranges.py`, `models/edge_ranges.json`)

For each spread signal, `scripts/research/fit_edge_ranges.py` regresses the
bet side's realized cover margin on the stated edge. A **season-cluster
bootstrap** gives the 80% range, so season-to-season drift is included and
not just game noise. It was fit on dev 2019–23 and checked on 2024–25. The
live parameters are refit on 2019–25.

Added to pipeline outputs, the ledger (`{spread,opener_gap,pt}_edge_exp/_lo80/_hi80`,
`pred_margin_lo80/hi80`) and the weekly text ("expect +x (80%: lo to hi)"):

| Signal, flagged bets | dev: predicted (80%) | dev: realized | holdout: predicted (80%), dev fit | holdout: realized |
|---|---|---|---|---|
| Main model edge ≥3 (n 962 / 163) | +0.40 (−0.37 to +1.16) | +0.31 ± 0.49 | +0.48 (−0.11 to +1.07) | −0.01 ± 1.17 |
| Opener-gap ≥5 (n 688 / 330) | +1.78 (+1.19 to +2.37) | +1.26 ± 0.60 | +1.56 (+1.07 to +2.05) | +0.51 ± 0.84 |
| PT FPI ≥7 (n 220 / 89) | +5.20 (+3.23 to +7.17) | +5.00 ± 1.22 | +4.50 (+2.80 to +6.21) | +1.59 ± 1.68 |

Values are the bet side's cover margin in points; ± is one standard error.

- **The main model's edge doesn't grow with its size:** the slope is about
  0, so a 6-point main-model edge "expects" about +0.2.
- **PT and opener-gap realized less on the holdout** than the dev-fit range
  said. This is the selection effect: both rules were picked on dev. The
  live fit includes 2024–25, which pulls it down, but treat the ranges as
  optimistic for these two signals.
- **Game-outcome range** (`pred_margin_lo80/hi80`): the 10th–90th
  percentile miss is about −20 to +19.5 points. It covered 79–82% of all
  games, but only 72–78% of flagged games.
- **Bowl flags** get their own range. It's fit only on the 2023–25 bowls the
  adjustment wasn't fit on, with a game-level bootstrap because three seasons
  are too few to cluster. Adjustments above 6.3 points (the 95th percentile)
  are capped.

## Limits

- Bowl samples are small: 40–46 games a season and 48 holdout bets. Even the
  best rule isn't significant once you allow for the variants tried.
- The bowl backtest knows who actually played. Live, it only has the
  "likely" starter unless an override is entered. A wrong guess turns a
  real edge into noise.
- Portal timing changed. The 2025–26 window opened in January, so very few
  entries came before the bowls, and portal features will be mostly zero in 2026.
- CFBD's coach records for the current season can lag. An interim bowl coach
  may not show up until CFBD updates. Use the override file's `note` column,
  or add coaching news there.
- Paid data: nothing essential. A paid real-time injury or depth-chart feed
  would replace "likely" starters with confirmed ones. I haven't priced one,
  and any purchase needs approval first.
