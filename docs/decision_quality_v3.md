# Decision quality v3

## Behavior

All new price-specific probability estimates and challenger models are paper-only.
The old retrospectively selected CORE rule remains `core_candidate` for research;
`core_total` requires a reviewed, approved decision. v3 deliberately does not
approve any live stake yet. This is a change from the former automatic 1-unit
CORE label. No historical bets, picks or ledger rows are rewritten by this change.

A prospective total candidate requires a future kickoff, a forecast or confirmed
indoor venue collected within 24 hours, an explicit availability review for both
teams, and a book quote with both observation and provider update times within
60 minutes. Missing data and unresolved availability produce REVIEW and zero units.
Weather collected for a forecast is archived; retrospective observed weather is
not substituted. Neutral venues without a verified lookup remain unknown.

Quotes retain book, event, market, side, exact line, exact odds, provider update
and observation timestamps. Consensus remains a model input, never an executable
price. Each quote is evaluated jointly by line and odds. The output shows estimated
win/loss/push probabilities, expected profit per unit risk, minimum acceptable
total at that quote's odds and worst acceptable odds at that line. Threshold is
2% estimated EV, fixed for prospective research, not a demonstrated profitable
threshold. The minimum-total and maximum-price limits are conditional: they must
not be combined into a different untested quote. Quoted books are not verified
as accessible to the user; these are research comparisons.

Calibration uses only earlier seasons' out-of-sample errors, with integer score
rounding and explicit push mass. It is an empirical approximation, not a
validated conditional cover model. A minimum 300 calibration games is required.
Current-season outcomes never recalibrate current-season estimates. Saved OOS
files must identify `season_holdout_v2`; production model files are SHA-256
fingerprinted. Decisions and quotes use content-addressed, write-once JSON.
The legacy mutable ledger remains for compatibility; immutable decision records
are the source for prospective v3 evaluation.

The newsletter now shares the candidate/live gate. All informational spreads,
review totals, moneylines and other paper signals have zero units. App, CLI and
weekly predictions use the same decision module. User QB point sliders are
explicit scenarios and force a new review; they are not learned injury values.
Live EPA rolls and YTD now reset at the season boundary, matching training.

## Research models

- Possession model: ridge offense/defense effects fitted to scoring per drive,
  combined with a team/opponent regulation-drive pace model. Fixed garbage filter
  excludes overtime, half/game-end drives, >28-point third-quarter margins and
  >21-point fourth-quarter margins. This is our research definition, not FEI.
  Games must precede fitting by at least 12 hours. Historical evaluation freezes
  the model before each season; live research fits on available prior games.
  This preseason-frozen historical test is conservative but does not validate
  the within-season live update schedule. Special teams, field position and
  possession-end score data quality need further validation.
- Ensemble: constrained quarter-step weights over current model, possessions
  and market; selected on the preceding available evaluation season only.
- Matchup baseline: imputed/scaled ridge residual model using lagged EPA,
  passing/rushing matchup differences, havoc, tempo, talent, Elo and returning
  passing production. Alpha is chosen on a disjoint prior non-COVID season.
  No unlagged current-game EPA enters this model.
- Analyst comparison: records source, exact price, publication AND collection
  times. Later-collected articles cannot be inserted into earlier decisions.
  Each analyst/game/market is counted once. Agreement groups are observational;
  they do not establish incremental skill over the market.

Initial local matchup test, 2022–2025 (3,107 games): totals MAE 12.494 vs market
12.484; spread MAE 12.063 vs market 11.999. No promotion. Possession evaluation
requires downloading drives with the existing CFBD credential in GitHub Actions.

## Inputs and commands

```
python -m unittest discover -s tests -v
python scripts/refresh_drives.py --start 2018 --end 2026
python scripts/evaluate_challengers.py
python scripts/collect_context.py --weather
python scripts/collect_context.py --report /path/to/sourced-report.json
python scripts/decision_report.py
```

Availability report schema (replace placeholders with verified facts):

```json
{
  "game_id": 123,
  "kind": "availability",
  "source_url": "https://official-team-or-conference.example/report",
  "published_at": "2026-10-01T10:00:00Z",
  "payload": {
    "home_qb_status": "unknown", "away_qb_status": "unknown",
    "home_ol_status": "unknown", "away_ol_status": "unknown",
    "home_defense_status": "unknown", "away_defense_status": "unknown"
  }
}
```

Allowed states: available, out, limited, questionable, unknown. Unknown never
means healthy. No automated injury feed or paid analyst subscription has been
invented or purchased. Reports require sourced input. Replacement-QB quality
and numeric injury effects are NOT fitted yet: this requires historical,
decision-time player availability and performance records. The current feature
is an availability review gate, not a claim that missing player data is solved.

Analyst report: same outer fields with kind `analyst`; payload requires analyst,
market (`totals`, `spreads`, `h2h`), side (`Under`/`Over` or exact team name),
line (except h2h), American odds, book, rationale. Optional notes may contain
scheme/personnel observations; they do not automatically shift a prediction.

The research workflow collects drives, runs comparisons, archives forecasts and
new decisions, then commits reports. It sends no email. Weekly and daily workflows
also archive context before their prediction runs. No future job is guaranteed
unless its provider credentials/quota remain available.

## Prospective evaluation and promotion

Freeze this policy, retain all eligible decisions, grade the first priced decision
per game, and distinguish REVIEW from eligible PAPER records. Report realized
paper returns, drawdown, probability calibration and same-book pre-kickoff CLV.
Only quotes observed before kickoff and updated within the final hour qualify as
closing proxies. Point movement and price movement are reported separately;
price CLV is computed only at an identical total. No postgame CFBD line is
asserted to be a contemporaneous close.

Do not promote based on these reused 2019–2025 seasons, a small winning streak,
or agreement among correlated pundits. Review sufficient new-season evidence,
uncertainty, coverage and the ability to take the quoted prices before changing
live approval. No code automatically turns a winning report into permission to bet.

## Research and provider references

- FEI: https://bcftoys.com/2025-fei
- Ensemble ratings: https://thepowerrank.com/cfb-playoff-predict-method/
- Opponent-adjusted strength: https://vsin.com/college-football/detailing-how-college-football-strength-ratings-are-created/
- College football score probabilities: https://arxiv.org/abs/2212.08116
- Quote schema: https://the-odds-api.com/liveapi/guides/v4/
- Forecast schema: https://open-meteo.com/en/docs
- CFBD drive schema: https://github.com/CFBD/cfbd-python/blob/main/docs/Drive.md
