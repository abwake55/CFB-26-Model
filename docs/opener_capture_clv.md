# Opener capture, bet lines and CLV

## What runs when (Pacific)

| Workflow | Schedule | Odds API calls per run |
|---|---|---|
| Line Snapshots | daily 6 AM; Sun 1 PM; Sun 5 PM | 1 (3 credits), shared by snapshot + ledger record + decisions via `ODDS_API_CACHE` |
| Weekly Refresh | Sun 7 AM, Tue 12 AM | 1 (3 credits), shared across ledger / audit / decisions / newsletter |
| Saturday Re-deal | Sat 8 AM | 1 (3 credits) |

Before this change each Line Snapshots run made 3 separate calls (snapshot,
ledger record, record_decisions) = 9 credits. With the per-job cache it is 3.
The two Sunday runs are new. Expected CI spend per week ≈ 9 snapshot runs ×3 +
2 refresh ×3 + 1 re-deal ×3 = 36 credits (≈155/month), down from ≈ 7×9 + 2×(4
calls ×3) + 3 ≈ 90 credits/week. Region stays `us` (no Pinnacle, no extra
cost); the app's own live calls are separate and not counted here.

Every real call logs The Odds API quota headers (`x-requests-last`, `-used`,
`-remaining`) to `data/odds_api_usage/<YYYY-MM>_<workflow>.csv`, and each
workflow ends with `scripts/odds_usage.py`, a free `/v4/sports` quota check.

## Snapshots (`data/lines_snapshots/lines_<season>_W<week>.json`)

- `first` / `first_seen`: the first line captured — with the Sunday runs this
  is the opener (or within hours of it).
- `current`: the latest line, never updated after the game's kickoff, so it is
  the last pre-kickoff line.
- `history`: every distinct quote with its time.
- `sharp_spread` / `sharp_total` / `sharp_book`: LowVig or BetOnline when the
  Odds API returns them (us region).

## Ledger (`outputs/predictions/ledger_2026.csv`)

For each spread signal (`pt`, `opener_gap`, `spread`), the first time a game
is flagged the ledger stores `<sig>_bet_line` (current consensus spread),
`<sig>_bet_ts` and `<sig>_bet_edge`. Re-records keep that first value, so a
flag that later disappears is still graded as the bet that would have been
made. Grading adds:

- `snap_open_spread` / `snap_open_ts`: first captured line for the game.
- `closing_spread`: consensus close (CFBD, frozen at kickoff).
- `closing_sharp_spread` / `_book` / `_ts`: last pre-kickoff sharp line from
  the snapshot history, else from the raw quote archive
  (`data/lines_snapshots/quotes`).
- `<sig>_clv`, `<sig>_clv_sharp`: points beaten on the bet side.

`ledger_2026_summary.json` reports each `<sig>_bets` bucket with W-L-P, units,
`avg_clv_pts` (vs consensus) and `avg_clv_sharp_pts` (vs sharp, with its n).
Rows recorded before this change have no `<sig>_bet_line`; their existing
buckets (`spread_flags`, `opener_gap_flags`) are unchanged.
