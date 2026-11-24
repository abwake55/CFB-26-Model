# Midseason all-market evaluation

The old live evaluator only inspected a filtered set of unders. This revision
adds an independent board for both spread sides, both moneylines, and both total
sides. Existing CORE approval rules remain unchanged. Research candidates are
not automatically promoted to real-money recommendations.

## Selection and calibration

Each sportsbook quote retains its side, line, odds, book and timestamps. Quotes
older than one hour and games already started are excluded. The engine takes the
best estimated EV for each market/side, rather than counting books as picks.
Spread and total probabilities use prior-season out-of-sample residuals with
explicit integer-score push mass. Moneyline probabilities use prior-season win
frequencies within a fixed +/-10 percentage-point forecast neighborhood. At least
300 unique calibration games are required. Current-season outcomes are excluded
from calibration until a suitable chronologically frozen evidence set is built.

Research A requires estimated EV >=2% and positive EV after a 95% Wilson sampling
uncertainty deduction. Research B requires estimated EV >=2% but does not clear
that deduction. These are fixed research thresholds, not independently validated
confidence labels. The bounds do not include model error or multiple-selection
bias. More candidates do not imply more profit. Missing availability, weather,
or model-data freshness remains a REVIEW blocker. No candidate carries units.

## Historical evidence

Run `python scripts/audit_markets.py`. The checked-in market_audit.json records
season-forward results by market and tier. Spreads/totals assume -110; moneylines
use historical odds where present. There is no historical decision-time quote
verification, so these results cannot establish executable ROI.

Re-run on 2026-11-24 after the SP+ leak fix (docs/leak_fix_2026-11-24.md):
Research A spreads 981 / 951 / 29 (50.78% excluding pushes); Research A totals
827 / 744 / 15 (52.64%). The 2025 fold for Research A spreads is 162 / 181 / 6;
totals are 176 / 166 / 2. The earlier 57.06% spread figure came from the leak
and is withdrawn. Neither market shows a validated edge at -110.

## Operations

- App and batch pipeline share the same evaluator, including QB-adjusted reruns.
- record_decisions.py archives immutable market decisions and exports
  outputs/predictions/current_market_board.json with a generation timestamp.
- grade_market_edges.py grades the first eligible PAPER candidate per game/market
  at the recorded price; REVIEW rows and repeated snapshots do not inflate records.
- Weekly and research workflows run the audit, record and grading steps.
- Training writes the latest included game date into model metadata. Old model
  files without this provenance are explicitly freshness-unverified.
- Removed setup-python's pip cache after the Nov 17 job failed with maximum call
  stack size exceeded in that step. This needs verification on GitHub Actions.
- A main-branch change to the evaluator or weekly workflow starts a full seasonal
  refresh; that push-triggered run skips the newsletter email step.

Before promoting any market, review prospective price-specific returns and
calibration, uncertainty, source freshness, and closing-price evidence separately.
The retraining implementation already incorporates completed current-season games;
a successful refresh is required to update the deployed model binaries. This
revision does not claim those binaries have already been retrained.
