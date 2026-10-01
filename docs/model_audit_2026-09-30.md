# Model audit and corrections — September 30, 2026

The largest defect was future-result leakage into rolling EPA. Historical bowl
and playoff rows use week 1; sorting by week put December/January results before
September games in the same season. The corrected pipeline sorts by kickoff
timestamp before applying the one-game lag. Live recent-form windows use kickoff
timestamps too.

## Changes

- Hold the most recent prior season out of walk-forward training for blend and
  parameter selection. Calibration trains on earlier seasons and calibrates on
  later whole seasons using sigmoid calibration; no random future/past folds.
- Preserve selected LightGBM parameters and spread/classifier sample weighting
  during production refitting. Record the actual parameters in model metadata.
- Set a deterministic training seed and enable LightGBM row subsampling (the
  previous subsample values had no effect with subsample_freq=0).
- Share probability blending, preseason adjustment and edge calculations across
  the app, CLI, training evaluation and weekly pipeline. Include conference and
  available weather context in CLI/weekly predictions. The CLI uses the CORE
  gate; other signals carry no units.
- Deduplicate exact repeated game rows and reject conflicting duplicate game
  IDs. Fourteen duplicate FBS rows were removed, including 11 in 2025.
- Grade pushes as returned stakes in targets, walk-forward summaries and the
  app's backtester. Fix inverted team spread signs in weekly paper picks.
- Reject missing market totals in the CORE gate. Report CORE uncertainty and
  market prediction baselines instead of relying solely on hit rate.
- Mark rebuilt EPA matrices with kickoff_v1 and refuse stale matrices in
  training. This prevents old cached features silently reintroducing leakage.

## Reproducible evaluation

Seven folds, 2019–2025, 5,184 unique FBS games. For each fold, the most recent
prior non-COVID season selects the blend and calibration; only earlier seasons
train. COVID 2020 is evaluated but is never training/validation data. The 2019
fold has one training season and therefore no classifier calibration fold.

This run used the **fixed fallback**, without Optuna, under Python 3.12,
scikit-learn 1.8.0 and LightGBM 4.7.0. Tuning remains optional through
requirements-training.txt. Do not describe this run as Optuna-tuned.

| Metric | Corrected model | Comparison |
| --- | ---: | ---: |
| Spread MAE | 12.164 | Market: 12.196 |
| Totals MAE | 12.756 | Market: 12.668 |
| Home-win Brier score | 0.17283 | Previous stored predictions on identical games: 0.17806 |
| CORE record | 228–182–3 | 413 bets, 410 decisive outcomes |
| CORE decisive win rate | 55.61% | 95% Wilson interval: 50.77%–60.34% |
| CORE profit at −110 | +27.8 units | Risk 1.1 units to win 1 per bet |
| CORE ROI | +6.12% | 95% week-block bootstrap interval: −3.03% to +15.73% |

Spread accuracy is essentially market-level and totals accuracy is slightly
worse than the market. Lower Brier is useful evidence of better probability
quality on these historical games, but it does not by itself establish betting
value. These comparisons bundle several corrections and are not an ablation
isolating the contribution of any single change.

| Season | CORE W–L–P | Profit units |
| --- | ---: | ---: |
| 2019 | 59–40–0 | +15.0 |
| 2020 | 30–30–2 | −3.0 |
| 2021 | 48–45–0 | −1.5 |
| 2022 | 30–16–0 | +12.4 |
| 2023 | 40–36–1 | +0.4 |
| 2024 | 14–12–0 | +0.8 |
| 2025 | 7–3–0 | +3.7 |

CORE remains a **retrospectively selected strategy**, not a proven live edge.
The intervals do not correct for repeated strategy searches. Historic CFBD
lines lack decision-time timestamps, and historic weather is observed weather,
not archived forecasts. Much of the architecture was developed using these
seasons. Small samples in recent folds warrant prospective tracking.

The refreshed walk-forward results and CORE history/metrics in this branch
come from this corrected run. Saved production model binaries and the large
cached feature matrix are deliberately not replaced: activation requires a
fresh feature rebuild and production refit. Existing 2026 bets and ledger
snapshots retain their recorded values.

## Activate and verify

After merging, run the existing **Weekly Data Refresh & Retrain** GitHub Actions
workflow with refresh_mode=full. Its feature-rebuild step runs before training,
so it replaces the stale cache and writes newly trained production models.

For a local rebuild:

```bash
pip install -r requirements.txt
python src/features.py
python src/model.py
python scripts/walk_forward.py
python scripts/evaluation_report.py
python -m unittest discover -s tests -v
```

To enable Optuna, install requirements-training.txt instead. Rerun the whole
walk-forward evaluation with the same environment; changing the optimizer
changes the fitted model and invalidates direct reuse of the fallback figures.
CFB_MODEL_JOBS controls LightGBM workers (default 4).

Verification completed: 22 regression tests, compilation, seven full historical
folds, production training on 6,371 unique non-COVID games, reload of saved
production models, parameter preservation, and exact agreement between
training's 2025 evaluation and walk-forward predictions on all 784 games.
Production verification wrote to an isolated temporary directory.

## Next modeling work

1. Preserve per-book line and price snapshots before kickoff, with immutable
   prediction/model fingerprints, then evaluate realized returns and CLV.
2. Validate CORE prospectively on future games, with thresholds fixed in
   advance. The bootstrap interval currently includes negative returns.
3. Add decision-time QB/injury availability and archived weather forecasts.
   Test each addition against the same chronological baseline.
4. Align early-season EPA priors: serving currently uses prior-season tails,
   while training windows reset at the season boundary. Treat this remaining
   distribution difference as a separate experiment rather than tuning it on
   the reported historical sample.
