# Spread research — November 24, 2026

Walk-forward over 2019–2025. Each season is predicted by models trained only on
earlier seasons (2020 never trained on). All features are pre-kickoff.
Settings (thresholds, rating parameters) were **chosen on dev 2019–2023 only**.
**2024–2025 is a true holdout.** Openers exist only from 2021 (CFBD Bovada
opener for 2021–22, consensus opener for 2023+), so the dev period for any
"bet at opener" rule is 2021–2023. Bets are flat at −110. CLV is points
gained vs the consensus closing spread. Reproduce with `scripts/research/`
(`spread_lab.py`, `lab_eval.py`, `run_variants.py`, `segments.py`,
`movement_filter.py`, `sharp_vs_soft_2026.py`). Summary CSVs are in
`outputs/research/`.

The main model replica is `spread_lab.py` "base": the same recipe and
features as `scripts/walk_forward.py`, spread part only.

## 1. Opener-anchored models (bet at opener)

| Rule (thr chosen on dev) | Period | n | W-L-P | Win % | ROI | CLV pts | Same bets at close: ROI |
|---|---|---|---|---|---|---|---|
| Main model at close, edge≥2 | dev 19–23 | 1763 | 891-845-27 | 51.32 | −2.02% | — | — |
| | holdout | 473 | 237-228-8 | 50.97 | −2.70% | — | — |
| Main model at opener, edge≥4 | dev 21–23 | 321 | 173-143-5 | 54.75 | +4.52% | +0.045 | +4.13% |
| | holdout | 59 | 26-33-0 | 44.07 | −15.87% | −0.076 | −4.55% |
| Opener-anchored residual, edge≥2 (dev = 2023 only) | dev | 518 | 260-249-9 | 51.08 | −2.48% | +0.125 | −4.55% |
| | holdout | 717 | 347-355-15 | 49.43 | −5.63% | +0.130 | −5.91% |
| Market-free margin vs opener, gap≥11 | dev 21–23 | 115 | 63-47-5 | 57.27 | +9.34% | +2.083 | +2.27% |
| | holdout | 35 | 19-14-2 | 57.58 | +9.92% | +0.543 | +4.13% |
| Blend main + opener-anchored, edge≥3 | dev 2023 | 253 | 133-115-5 | 53.63 | +2.38% | +0.192 | +0.48% |
| | holdout | 253 | 124-124-5 | 50.00 | −4.55% | +0.150 | −3.39% |

Holdout MAE (games with an opener, n=1571):

| Predictor | MAE |
|---|---|
| Opener | 12.064 |
| Close | 11.952 |
| Opener + main residual | 12.156 |
| Opener + opener-anchored residual | 12.273 |

Shrinking either model toward the opener (k fitted on dev) gave holdout MAE
12.073–12.080, never better than the opener alone. The market-free model's
gap to the opener gives positive CLV at every threshold, in dev and holdout.
ATS results are mixed.

## 2. In-season opponent-adjusted ratings (tuned)

Grid of 18 settings: prior regression {0.4, 0.6, 0.8} × ridge pseudo-games
{2, 4, 8} × recency half-life {none, 35 days}. Returning production and portal
are already main-model features.

- **As residual features**, every setting worsened dev residual MAE. Base
  12.4966; the best setting (pr 0.4 / λ 8 / hl 35) scored 12.5156. With that
  setting, main@open edge≥4 went 200-150-9 (57.14%, +9.09%, CLV +0.081) on dev.
  On holdout it went 52-41 (55.91%, +6.74%), but CLV was −0.177, worse than
  base. Not adopted.
- **In the market-free margin model**, the best dev setting was pr 0.8 / λ 2 /
  no recency. Dev MAE was 12.877 vs 13.054 base; holdout 12.457 vs 12.637.
  Opener-gap≥5 (thr chosen on dev):

| Period | n | W-L-P | Win % | ROI | CLV | CLV>0 | At close ROI |
|---|---|---|---|---|---|---|---|
| dev 21–23 | 688 | 359-315-14 | 53.26 | +1.69% | +0.742 | 55.5% | −2.43% |
| holdout | 330 | 167-155-8 | 51.86 | −0.99% | +0.415 | 51.2% | −1.27% |

## 3. Starting-QB features (CFBD per-game passing, pulled in CI)

| Variant | Dev resid MAE | Holdout resid MAE | Main@open holdout | Opener-gap holdout |
|---|---|---|---|---|
| Base | 12.4966 | 12.0716 | 26-33, −15.87%, CLV −0.08 | 19-14-2, +9.92%, CLV +0.54 |
| Pre-kickoff QB stability (prev-game change, share, starts) | 12.5002 | 12.0856 | 30-31, −6.11%, CLV +0.08 | 107-102-6, −2.26%, CLV +0.45 |
| Confirmed starter (DIAGNOSTIC, needs lineup) | 12.4705 | 12.0237 | 12-8-1, +14.55%, CLV −0.19 | 26-18-1, +12.81%, CLV +1.29 |

The confirmed-starter variant is the only feature set that lowered holdout
error. It is not adoptable at the opener, because the starter isn't known
then. It is worth building as a game-day check (see open items).

## 4. Segments (fixed rule: main model edge≥3; dev chooses, holdout confirms)

Segments where dev ROI > 0 with n ≥ 100 and holdout ROI > 0:
- At close: weeks 5–9 (dev 192-150, +7.18%; holdout 38-28, +9.92%) and bet home
  (dev +0.41%; holdout +4.69%).
- At opener: P4 vs P4 (holdout 45-30, +14.55%), weeks 5–9, weeks 10+,
  conference games, and underdogs.

No holdout segment beats −110 break-even at p < 0.05 once you allow for the
~32 segment/rule tests. These are leads, not rules. G5 vs G5 was negative in
both periods for both rules. Full table: `outputs/research/segments.csv`.

## 5. Movement and sharp filters (bet at close)

- Opener→close agreement (2021–25): requiring the market to move toward the
  model did not hold up. Main edge≥5 + agree: holdout 4-2 (n too small).
  Main edge≥3 where the line moved **against** the model went dev 129-102
  (55.84%, +6.61%) and holdout 45-34 (56.96%, +8.75%). It held for the
  in-season variant too, but it was one of ~20 filters tried. Paper-track only.
- Sharp vs soft books: only measurable in the 2026 live archive (Oct 2 – Nov 24,
  335 completed games). LowVig/BetOnline vs the median of DraftKings,
  FanDuel, BetMGM and BetRivers, first pre-kickoff snapshot:
  - Bet the soft line in the sharp direction when |gap| ≥ 0.5: 58-48-2 (54.72%, +4.46%, p=0.35).
  - Same, but only when the model agrees: 23-13 (63.89%, +21.97%, p=0.11).
  - Not significant. Historical Pinnacle/sharp data needs a paid feed (The Odds API historical).

## Adopted

**Opener-gap signal** (`src/opener_gap.py`). Market-free margin model with
in-season ratings (pr 0.8, λ 2) vs the opener; flag when |gap| ≥ 5. It is the
only change that improved both holdout ROI at the opener and CLV over the main
model's dev-chosen opener rule: −15.87% → −0.99% ROI, CLV −0.08 → +0.42.

Holdout ROI is still below zero, so treat flags as paper or small stakes
until live results confirm them. Against the main model at edge≥3 (holdout
99-91, −0.53%, CLV +0.157), the gain is in CLV, not ROI. The flag is recorded
in the ledger (`flag_opener_gap`) and graded at the line used
(`opener_gap_line`): the opener, else the current spread.
