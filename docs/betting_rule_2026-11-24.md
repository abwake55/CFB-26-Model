# Recommended betting rule — November 24, 2026

| Bucket | Rule | Stake | Status |
|---|---|---|---|
| CORE unders | `src/gates.py` (unchanged) | 1u | real plays |
| PT FPI-vs-opener | `opener − FPI line ≥ 7` (`models/pt_rule.json`) | 0.25–0.5u | small stake, live-tracked |
| Opener-gap | `|market-free margin + opener| ≥ 5` | paper or 0.25u | live-tracked |
| High-total under | first total ≥ 60 (`src/high_total_under.py`) | 0.25u | small stake, live-tracked |
| Main-model spreads, ML EV | — | none | paper only |
| Bowl adjustment (bowls only) | `|bowl_adj| ≥ 2` (`models/bowl_adjust.json`) | none | paper, bowl ledger |
| PT / opener-gap **in bowls** | same rules | none | paper only in bowls |

Bowls (added Nov 24, see `docs/bowl_module_2026-11-24.md`): the
regular-season signals didn't carry over to bowls. Opener-gap went 13-22 on
the 2023–24 holdout bowls and PT FPI went 1-5. They stay on the bowl board
as paper. The bowl adjustment (QB change, coaching change, portal) went
29-18-1 on the holdout (+17.4%) and 12-9 in 2025 (+9.1%). That isn't
significant once you allow for the 27 variants tried, so it's paper too.
The high-total under has not been tested on bowls; treat it as paper there.

Uncertainty: each signal now carries `expected cover (80% range)` from
`models/edge_ranges.json`. Use it to keep stakes small, not to pick bets.
On the 2024–25 holdout the PT and opener-gap ranges ran optimistic: PT
flagged bets realized +1.6 ± 1.7 points against a dev-fit range of 2.8–6.2.

Game day: before betting a flagged pick, check `outputs/picks/gameday_2026_wWW.json`
(Saturday 9 AM PT). If a team in the game has a new starting QB or the edge
fell below the threshold at the current line, pass or keep it at the minimum stake.

Execution:
- Bet a spread signal at the line when it first appears (Sunday opener
  snapshots, then daily). Take the best number across your books.
- Bet a high-total under at the first total we see: the captured opener
  if we have it, otherwise the current total. Threshold 60, set on 2019–23.
- Buy +0.5 only when the weekly text says "buy +0.5 if ≤−120" and the
  price is −120 or better. Do not pay −125 or worse.
- Do not filter or size by the pricer's EV: holdout EV buckets were not
  monotone and predicted win rates ran 2–8 points high.

Evidence and graduation:
- PT FPI ≥7: dev 133-88-4 (+14.89%, CLV +2.71); holdout 50-39-2 (+7.25%,
  CLV +0.80, p=0.27); 2026 weeks 1–11 out of sample 9-16 (−31.27%, CLV +0.72).
  Positive CLV everywhere, ATS not established.
- High-total under ≥ 60: dev close 515-435-16 (+3.49%); holdout 118-96-2
  (+5.27%). At the opener: holdout 118-90-1 (+8.30%, CLV +0.70). Mostly
  incremental to CORE. Details in `docs/high_total_under_2026-11-24.md`.
- Promote a signal to 1u only after 50+ live ledger bets with mean CLV > 0
  vs both consensus and sharp close and ATS ≥ 52.4%. Demote to paper if live
  CLV is ≤ 0 after 30 bets.
- Track in `outputs/predictions/ledger_2026_summary.json`: `pt_bets`,
  `opener_gap_bets`, `high_total_under_bets` (graded at the line placed,
  with `avg_clv_pts` and `avg_clv_sharp_pts`).
