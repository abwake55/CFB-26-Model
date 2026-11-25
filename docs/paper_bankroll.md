# Paper bankroll — 2026 week 13 through the bowls and playoff

A $1,000 paper bankroll that logs every recommended bet at the line and
stake the betting rule calls for, then grades it after the game. Nothing
here places a real wager or touches a sportsbook account.

## Paths the Sunday health check should read

| What | Path |
|---|---|
| Bet-level ledger | `outputs/bankroll/paper_bankroll_2026.csv` |
| Summary (balance, record, ROI, CLV) | `outputs/bankroll/paper_bankroll_2026_summary.json` |
| Weekly ready-to-place slip | `outputs/picks/slip_2026_WNN.txt` (example: `outputs/picks/slip_2026_W13.txt`) |

The summary JSON is the one to open first. Fields:

- `current_balance` — running bankroll after settled bets, starting at 1000
- `open_risk_usd` — dollars currently outstanding
- `record` / `record_str` — wins-losses-pushes on staked bets
- `roi` — settled profit divided by settled dollars staked (`null` until a bet settles)
- `avg_clv_pts` / `clv_n` — mean closing-line value in points, bet side, on staked bets that have a close
- `avg_clv_sharp_pts` — same, versus the sharp book's last pre-kickoff line
- `by_signal` — the same numbers split by `core_under`, `pt_fpi`, `opener_gap`, `high_total_under`, `spread_model`
- `unit_usd` — 20
- `last_updated` — UTC timestamp of the last grade

`scripts/paper_bankroll.py update` rewrites the summary every Sunday (weekly
refresh) and on the Saturday / bowl-season game-day job.

## Unit size

**1u = $20, flat.** That is 2% of the $1,000 starting bankroll. It is not
recalculated as the balance moves.

A flat unit keeps the weekly slip's dollar stakes stable so they can be
copied onto a ticket without doing arithmetic each week. A 2% rolling
unit would change the stake on every bet after the first settled week,
which is more bookkeeping than this bankroll needs. Revisit only if the
balance halves or doubles.

## Stakes (from `docs/betting_rule_2026-11-24.md`)

| Signal | Regular season | Bowls / CFP |
|---|---|---|
| CORE unders | 1u ($20) | 1u (the rule did not demote this) |
| PT FPI-vs-opener ≥ 7 | 0.25u ($5), or 0.50u ($10) when \|edge\| ≥ 10 | paper, $0 |
| Opener-gap ≥ 5 | 0.25u ($5) | paper, $0 |
| High-total under (first total ≥ 60) | 0.25u ($5) | paper, $0 |
| Main-model spreads | paper, $0 | paper, $0 |
| Bowl adjustment | n/a | paper, $0 |

The 10-point cutoff for a half unit on the FPI signal is deliberate: 7 is
the flag threshold, so 10 is clearly above it. Week 13: North Texas
(edge 11.37) is 0.50u; Miami (7.31), Ohio State (8.41) and Hawai'i (9.77)
are 0.25u.

To avoid stacking two bets on the same market in one game, only one
spread signal and one totals signal is staked. Preference is CORE over
high-total under, and FPI over opener-gap over the main-model spread.
The lower-priority flag can still be logged at $0 so it is tracked.

## Pricing: DraftKings Predictions, with a sportsbook fallback

The user asked to price paper bets on DraftKings Predictions (the
prediction-market product), not at a default −110.

**What we found (24–25 Nov 2026, no login, no signup, no paid feed):**

- There is a public website, [predictions.draftkings.com](https://predictions.draftkings.com).
  Search engines index prices on it for NFL and college basketball. DraftKings's
  own availability page lists college football among the sports contracts.
- There is **no official public developer API**. DKeX (the exchange) has an
  API for approved market makers only. The Odds API, which this repo already
  uses on the free 500-credit plan, lists `draftkings` (the sportsbook) and
  several other exchanges (Kalshi, ProphetX, Polymarket) but **not**
  DraftKings Predictions.
- Third-party feeds that do carry those prices (Betstamp, OddsRelay, Apify
  scrapers) need an account and/or payment. We did not sign up.
- From this environment, and almost certainly from GitHub Actions, the
  public site is blocked by Akamai (HTTP 403). Unauthenticated GETs from
  `src/dk_predictions.py` therefore return nothing.
- When a public contract price *is* captured, the implied American odds
  include the published DKeX fee (flat $0.02 per contract for prices
  between $0.30 and $0.94; a sliding fee outside that band; no settlement
  fee). A winning contract pays $1.

**Fallback, used for every week 13 bet:** the sportsbook line from the
existing week snapshot / ledger, priced at −110, with
`price_source=sportsbook_fallback` and `fallback=true`. The slip says so.
If a public Predictions price later becomes readable without an account,
`update` will use it for newly logged bets only — already logged rows are
not rewritten.

We do not spend Odds API credits on this, and we do not invent a
Predictions price.

## Freeze rule

A bet is logged only if its kickoff is still in the future. Once kickoff
has passed, `frozen_at_kickoff` is set and the bet's game, signal, side,
line, price, stake and logged time are never edited. After the game,
`grade` fills only `closing_line`, `clv_pts`, `clv_sharp_pts`, `result`,
`profit_usd`, `status` and the derived `running_balance`.

## Week 13 opening slate

Logged on 24 Nov 2026 (America/Los_Angeles) at the lines sent to the
group, all before the first kickoff (Temple at Memphis, Friday 27 Nov
12:30 PM PT):

| Pick | Signal | Stake | Line logged |
|---|---|---|---|
| Miami −25.5 vs Boston College | FPI | $5 | −25.5 (kickoff Sat 9:00 AM PT) |
| Ohio State −10 vs Michigan | FPI | $5 | −10 (Sat 9:00 AM PT) |
| North Texas −3.5 vs UAB | FPI | $10 | −3.5 (Sat 1:00 PM PT) |
| Hawai'i −13.5 vs Sacramento State | FPI | $5 | −13.5 (Sat 8:00 PM PT) |
| Boise State/Colorado State under 75.5 | high-total under | $5 | first total 75.5 (Sat 3:00 PM PT) |
| West Virginia/Utah under 68.5 | high-total under | $5 | first total 68.5 (Fri 6:00 PM PT) |
| Kansas/Oklahoma State under 67.5 | high-total under | $5 | first total 67.5 (Sat 3:15 PM PT) |
| Temple/Memphis under 64.5 | high-total under | $5 | first total 64.5 (Fri 12:30 PM PT) |
| SMU/Stanford under 61 | high-total under | $5 | first total 61 (Sat 12:30 PM PT) |

Open risk: $50. Starting balance: $1,000.

The slip's "worst acceptable" number is where the edge would fall below
the signal's threshold (7 points for FPI, the logged total for a
high-total under — do not take a lower number). The worst acceptable
price on these rows is −110.

## How it is updated

```
python3 scripts/paper_bankroll.py update
```

That is what CI runs. It seeds week 13 if those rows are missing and the
games have not started, syncs new flags from
`outputs/predictions/ledger_2026.csv` for games that have not kicked off,
grades completed games from that ledger (no new API calls), rewrites the
summary, and writes this week's slip.

Also available: `seed-week13`, `sync`, `grade`, `slip [--week N]`.

Wired into `.github/workflows/weekly_refresh.yml` (after the ledger is
graded) and `.github/workflows/game_day_update.yml`. Both already use
`.github/scripts/commit_and_push.sh` (rebase-and-retry, concurrency
group, no overlapping cancels). The game-day commit path list includes
`outputs/bankroll`. The newsletter is never sent from this job.
