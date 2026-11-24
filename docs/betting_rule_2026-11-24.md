# Recommended betting rule — November 24, 2026

| Bucket | Rule | Stake | Status |
|---|---|---|---|
| CORE unders | `src/gates.py` (unchanged) | 1u | real plays |
| PT FPI-vs-opener | `opener − FPI line ≥ 7` (`models/pt_rule.json`) | 0.25–0.5u | small stake, live-tracked |
| Opener-gap | `|market-free margin + opener| ≥ 5` | paper or 0.25u | live-tracked |
| Main-model spreads, ML EV | — | none | paper only |

Execution:
- Bet a spread signal at the line when it first appears (Sunday opener
  snapshots, then daily). Take the best number across your books.
- Buy +0.5 only when the weekly text says "buy +0.5 if ≤−120" and the
  price is −120 or better. Do not pay −125 or worse.
- Do not filter or size by the pricer's EV: holdout EV buckets were not
  monotone and predicted win rates ran 2–8 points high.

Evidence and graduation:
- PT FPI ≥7: dev 133-88-4 (+14.89%, CLV +2.71); holdout 50-39-2 (+7.25%,
  CLV +0.80, p=0.27); 2026 weeks 1–11 out of sample 9-16 (−31.27%, CLV +0.72).
  Positive CLV everywhere, ATS not established.
- Promote a signal to 1u only after 50+ live ledger bets with mean CLV > 0
  vs both consensus and sharp close and ATS ≥ 52.4%. Demote to paper if live
  CLV is ≤ 0 after 30 bets.
- Track in `outputs/predictions/ledger_2026_summary.json`: `pt_bets`,
  `opener_gap_bets` (graded at the line placed, with `avg_clv_pts` and
  `avg_clv_sharp_pts`).
