# CFB Betting Model — Master Plan

**Goal:** Build a full pipeline that goes from raw data → team power ratings → game predictions → line comparison → bet recommendations, covering spreads, totals, moneylines, and player props.

---

## 1. Recommended Tech Stack

**Use Python.** It's the industry standard for sports modeling, has the best libraries, and has the most community resources for exactly this type of project.

### Tools you'll need (all free):
| Tool | What it does |
|------|-------------|
| **Python 3.11+** | Core language |
| **Jupyter Notebooks** | Interactive data exploration (great for beginners) |
| **pandas** | Data manipulation (tables, filtering, aggregating) |
| **numpy** | Math and array operations |
| **scikit-learn** | Machine learning models |
| **xgboost** | Gradient boosting — the workhorse of sports prediction |
| **matplotlib / seaborn** | Visualizations and charting |
| **requests** | Pulling data from APIs |
| **SQLite (via sqlite3)** | Simple local database for storing game data |

### Setup steps:
1. Install Python from [python.org](https://python.org)
2. Install VS Code (free editor) from [code.visualstudio.com](https://code.visualstudio.com)
3. Install packages: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests jupyter`

---

## 2. Data Sources

### Primary (free, high quality):
- **collegefootballdata.com** — The backbone of any CFB model. Free API with historical game data, box scores, advanced stats (PPA/EPA), recruiting rankings, and more going back to 2000. Get a free API key at [collegefootballdata.com](https://collegefootballdata.com).
- **The Odds API** — Historical and live betting lines from major sportsbooks. Free tier gives 500 requests/month — enough for historical pulls. [the-odds-api.com](https://the-odds-api.com)
- **College Football Reference (Sports Reference)** — Good for historical cross-checking, player stats, and year-over-year trends. [sports-reference.com/cfb](https://www.sports-reference.com/cfb/)

### Secondary (useful add-ons):
- **ESPN FPI (Football Power Index)** — A public power rating you can compare your model against. Accessible via ESPN's unofficial API.
- **247Sports / On3** — Recruiting rankings (strong predictor of program trajectory over 3–4 year windows).
- **Weather data** — Affects totals significantly; use OpenWeather API (free tier) for outdoor stadiums.

### Data you'll collect:
- Game results (score, location, date, week)
- Team-level advanced stats: EPA/play, success rate, explosiveness, havoc rate
- Recruiting composite rankings (rolling 4-year averages)
- Historical point spreads and totals
- Player stat lines (for props)

---

## 3. How a Betting Model Works (Conceptual)

The core loop is:

```
Raw Data → Feature Engineering → Power Ratings → Game Simulation → Predicted Line
                                                                          ↓
                                               Compare to Vegas Line → Find Edge (EV+)
```

**Step-by-step:**

1. **Collect data** — Pull game-by-game stats for every team, every season.
2. **Build team ratings** — Estimate how good each team is on offense, defense, and special teams, adjusting for opponent strength.
3. **Build a game model** — Given two teams, predict the expected score (and distribution around it).
4. **Generate your own line** — Your model produces a spread (e.g., Team A by 7) and total (e.g., 52 points).
5. **Compare to the Vegas line** — If your model says -7 but Vegas has -3.5, there may be value on Team A.
6. **Filter for edge** — Only bet when the gap between your number and Vegas's number is large enough to overcome the vig (typically 3+ points for spreads).

---

## 4. Project Architecture (Folder Structure)

```
CFB-Betting-Model/
│
├── data/
│   ├── raw/              # Unprocessed API responses (JSON/CSV)
│   ├── processed/        # Cleaned, merged datasets
│   └── lines/            # Historical betting lines
│
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_power_ratings.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtesting.ipynb
│
├── src/                  # Reusable Python modules
│   ├── data_collection.py
│   ├── features.py
│   ├── ratings.py
│   ├── model.py
│   └── backtester.py
│
├── outputs/
│   ├── predictions/      # Weekly game predictions
│   ├── bets/             # Recommended bets
│   └── charts/           # Visualizations
│
├── CFB_Model_Master_Plan.md
└── requirements.txt
```

---

## 5. Phased Roadmap

### Phase 1 — Foundation (Weeks 1–2)
**Goal: Get your environment set up and historical data flowing.**

- [ ] Install Python, VS Code, Jupyter
- [ ] Get API key from collegefootballdata.com
- [ ] Pull 5 seasons of game data (2019–2024) into CSV files
- [ ] Pull historical betting lines from The Odds API
- [ ] Explore the data in a Jupyter notebook — understand what columns exist
- [ ] Merge game results with betting lines into one clean dataset

**Key milestone:** A single CSV/database with every game from 2019–2024 including final score, spread, and total.

---

### Phase 2 — Feature Engineering (Weeks 3–4)
**Goal: Turn raw box scores into meaningful predictors.**

Features to build per team, per game (rolling averages over last 3 games):
- **Offensive EPA/play** — Efficiency per offensive play (best single predictor)
- **Defensive EPA/play** — Efficiency of defense
- **Success rate** — % of plays that gained positive EPA
- **Explosiveness** — Average EPA on "explosive" plays (20+ yard gains)
- **Havoc rate** — TFLs + sacks + passes defended per play (defensive disruption)
- **Turnover margin** — Adjusted for luck (fumble recovery rate reverts to 50%)
- **SOS (Strength of Schedule)** — Rolling average of opponent quality
- **Recruiting rating** — 4-year composite average (program talent)
- **Home field advantage** — Fixed 2.5–3 point adjustment

**Key milestone:** A feature matrix where each row = one game, columns = team stats for both teams.

---

### Phase 3 — Team Ratings (Week 5)
**Goal: Build a single-number team strength rating.**

Start simple — two approaches work well:

**Option A: Regression-based ratings**
Run a linear regression predicting point differential, with team identifiers as features. Each team gets an offensive and defensive rating. Adjust weekly.

**Option B: Elo ratings**
A self-updating system (like chess Elo) that adjusts team ratings based on each game's result vs. expectation. Simpler to implement.

*Recommendation for beginners: Start with Elo, then layer in regression ratings later.*

**Key milestone:** A ratings table updated weekly — e.g., "Alabama: +18.5 vs. average team."

---

### Phase 4 — Game Prediction Model (Weeks 6–8)
**Goal: Predict the spread and total for any matchup.**

Model types to try (in order of complexity):
1. **Linear Regression** — Predicts point differential from team features. Best starting point.
2. **Ridge/Lasso Regression** — Adds regularization to prevent overfitting on small samples.
3. **XGBoost** — More powerful, captures non-linear relationships. Use once you have the basics working.

**What the model outputs:**
- Predicted point differential (→ your spread)
- Predicted total points (→ your total)
- Win probability (→ your moneyline)

**Key milestone:** Given any two teams, your model produces a spread and total within 4–5 points of Vegas on average (MAE < 5).

---

### Phase 5 — Backtesting (Week 9)
**Goal: See how your model would have performed historically.**

Simulate betting on every game from 2022–2024:
- Compare your model's line to the opening Vegas line
- Record every game where you had a "bet" (edge > threshold)
- Calculate ATS record, ROI, and Brier score

**What good looks like:**
- ATS win rate above 52.4% = profitable (breaks even at ~52.4% due to vig)
- ROI > 3% sustained = strong signal
- Sample size matters — need 500+ "bet" games before trusting results

**Key milestone:** Backtest report showing your model's historical ROI by bet type and edge size.

---

### Phase 6 — Player Props (Weeks 10–12)
**Goal: Extend to individual player stats.**

Props are often softer markets than spreads — there's more edge to be found. Focus on:
- Passing yards (most liquid, data-rich)
- Rushing yards
- Receiving yards / receptions

Data sources:
- Snap counts (proxies for opportunity)
- Target share / air yards
- Opponent pass defense rank

Simple starting model: predict each player's stat line based on their season average, adjusted for opponent and game environment (pace, implied total).

**Key milestone:** A weekly props sheet with projected stat lines vs. sportsbook over/unders.

---

### Phase 7 — Live Season Workflow (Ongoing)
**Goal: Automate weekly predictions before games.**

Weekly process:
1. Pull latest team stats from API (Monday after week's games)
2. Run feature engineering on updated data
3. Generate predictions for upcoming week's games
4. Compare to Vegas lines (pull Tuesday/Wednesday lines)
5. Flag games where your number differs by 3+ points
6. Log all bets and results in a tracking spreadsheet

---

## 6. Bankroll Management

The model is useless without proper bet sizing. Follow these rules:

**Kelly Criterion (simplified):**
- Only bet when your estimated edge is real (>3 point discrepancy minimum)
- Use **half-Kelly** to account for model uncertainty
- Formula: `Bet size = (edge / odds) × 0.5 × bankroll`
- A conservative starting point: bet 1–3% of bankroll per game

**Practical rules:**
- Start with a flat betting unit (e.g., 1 unit = 1% of bankroll)
- Never bet more than 3 units on a single game
- Track every bet in a spreadsheet: date, game, your line, Vegas line, bet, result, P&L
- Do NOT chase losses — the model is a long-run tool

**Expected variance:** Even a profitable model will have 5–10 game losing streaks regularly. Plan emotionally and financially for this.

---

## 7. Key Metrics to Track

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **ATS Win %** | Spread bet accuracy | > 52.4% |
| **MAE (spread)** | How close your line is to Vegas | < 5 points |
| **Brier Score** | Win probability calibration | < 0.24 |
| **ROI** | Return on investment | > 3% |
| **CLV (Closing Line Value)** | Are you beating the closing line? | Positive |

*Note: Closing Line Value (CLV) is arguably the best indicator of a real edge. If you consistently bet at better numbers than where the line closes, you have a genuine edge — even if your short-term record looks bad.*

---

## 8. What to Avoid

- **Overfitting** — Don't tune your model to look good on historical data. Use walk-forward validation (train on years 1–3, test on year 4).
- **Small samples** — A 60% ATS record over 20 games is noise. You need hundreds of bets.
- **Recency bias in the model** — Weight recent games more, but don't throw out older data entirely.
- **Ignoring market efficiency** — Vegas lines are very good. Your edge will be small — that's fine.
- **Betting every game** — The model should tell you NOT to bet most games. Discipline is the hardest part.

---

## 9. Recommended Learning Resources

- **"Mathletics" by Wayne Winston** — Excellent intro to sports modeling (includes football)
- **cfbfastR documentation** — R-based but the statistical concepts translate directly
- **Pinnacle's betting resources** — [pinnacle.com/en/betting-articles](https://www.pinnacle.com/en/betting-articles) — best free content on betting math
- **r/sportsbook and r/CFB** — Community discussion, but treat claims of model performance with skepticism

---

## Next Steps (Start Here)

1. **Today:** Install Python and VS Code
2. **This week:** Sign up for collegefootballdata.com API and run the starter data collection script (see `src/data_collection.py`)
3. **Next week:** Pull 5 seasons of data and load it into a Jupyter notebook
4. **Goal by end of June:** Have a clean dataset and first Elo ratings running

The best time to start is now — the 2025 season kicks off in late August, which gives you roughly 3 months to build, backtest, and validate before putting real money on it.
