# World Cup 2026 Match Predictor

ML model that predicts FIFA World Cup 2026 match outcomes (W/D/L), exact scores, and win probabilities using a Poisson-style approach with Dixon-Coles correction.

## Approach
Predict expected goals (`λ_home`, `λ_away`) for each match using a Poisson regression on engineered features (Elo ratings, recent form, squad market value, home advantage, tournament class). From those two expected-goals numbers we derive:
- W/D/L probabilities
- Exact-score probabilities (full distribution over a 21×21 grid)
- Tournament-level outcomes via Monte Carlo simulation

To prevent data leakage, all training data is frozen at **2026-06-10** (the day before WC 2026 kickoff). Games from the tournament itself are held out for validation.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Model: Predicting the World Cup

If you just want to generate predictions, run these in order (~5–10 minutes the first time, ~1 minute on subsequent runs).

### Step 1: Download source data (one-time, ~5 minutes)

These two commands download raw datasets to `data/raw/`. **Skip if already done** — the script is idempotent (it won't re-download what's already there).

```bash
python -m src.data.loader              # ~3 MB
python -m src.data.squad_value_loader  # ~80 MB
```

**Produces:**
- `data/raw/results.csv` — every international match since 1872 (49k+ rows)
- `data/raw/shootouts.csv` — penalty shootout outcomes
- `data/raw/former_names.csv` — historical country renames
- `data/raw/transfermarkt/players.csv` — 47k Transfermarkt player profiles
- `data/raw/transfermarkt/player_valuations.csv` — 500k historical valuations
- `data/raw/transfermarkt/national_teams.csv` — current team values
- `data/raw/wc_squads_fjelstul.csv` — historical WC squad rosters

### Step 2: Build features (one-time, ~30 seconds)

These commands transform the raw data into the tables the model trains on. **Run them in order** — each depends on the previous one's output. **Skip if `data/processed/` is already populated**.

```bash
python -m src.features.elo              # ~5 sec
python -m src.features.squad_values     # ~60 sec
python -m src.features.group_standings  # ~5 sec
python -m src.features.build            # ~5 sec
```

**Produces:**
- `data/processed/matches_with_elo.csv` — every match annotated with each team's Elo rating going into that match
- `data/processed/final_elo.csv` — each team's final Elo rating
- `data/processed/squad_values.csv` — team-year squad value snapshots (556 rows: 5 historical WCs × ~100 teams + 48 WC 2026 qualifiers)
- `data/processed/group_standings.csv` — for each historical WC group match, points-before and dead-rubber flag
- `data/processed/features.csv` — the final training-ready table (49k rows, 13 feature columns plus targets)

### Step 3: Train the model and predict every match (~15 seconds)

```bash
python -m src.prediction.wc2026
```

**What it does:**
1. Trains the Poisson regression on `features.csv` (everything 1990 → 2026-06-10). Takes ~10 seconds.
2. Predicts every WC 2026 match using features frozen at the cutoff date.
3. Prints all 72 match predictions in chronological order with W/D/L probabilities and most-likely scores.
4. Evaluates against any WC 2026 matches that have already been played (real-world validation).

**Produces:**
- `data/processed/wc2026_predictions.csv` — 72 rows, one per match. Columns: `date`, `home`, `away`, `actual_score` (if played), `expected_goals_home`, `expected_goals_away`, `prob_home_win`, `prob_draw`, `prob_away_win`, `most_likely_score`, `most_likely_score_prob`.

### Step 4: Run the Monte Carlo tournament simulation (~50 seconds)

```bash
python -m src.prediction.simulate_wc2026
```

**What it does:**
1. Re-trains the model (yes, training happens fresh again — see "How training works" below).
2. Pre-caches knockout match predictions for all 48 × 47 = 2,256 possible matchups.
3. Runs 20,000 Monte Carlo tournament simulations:
   - Samples scorelines for unplayed group matches from each match's predicted distribution.
   - Uses actual results for matches already played.
   - Computes group standings, determines top-2 advancers and 8 best 3rd-place qualifiers.
   - Runs 5 knockout rounds (R32 → R16 → QF → SF → Final), sampling outcomes.
4. Tallies per-team probabilities of advancing, reaching each round, and winning the WC.

**Produces:**
- `data/processed/wc2026_simulation.csv` — 48 rows, one per team. Columns: `team`, `p_advance`, `p_reach_r16`, `p_reach_qf`, `p_reach_sf`, `p_reach_final`, `p_win_wc`.

## How training works

**There is no saved model file.** Every time you run `wc2026.py` or `simulate_wc2026.py`, the model trains fresh on `features.csv`. The training takes ~10 seconds, which is fast enough that saving / loading would add complexity without saving time.

This means:
- **The cached files are the *data*, not the *model*.** Everything in `data/processed/` is the result of feature engineering. The model itself is rebuilt from `features.csv` on each prediction run.
- **If you change the data, predictions automatically use the updated features.** Re-running `python -m src.features.build` regenerates `features.csv`; the next prediction run trains a new model on it.
- **You can re-run prediction commands repeatedly** — they'll give very similar (but not bit-exact) results because the Monte Carlo simulation uses random sampling.

For larger models (neural networks, gradient-boosted trees on millions of rows), training takes hours and saving the model is essential. For our Poisson regression on ~30k matches, "always retrain" is the simpler and safer pattern.

---

## Other things you can run

### Backtest the model against past World Cups

```bash
python -m src.evaluation.backtest
```

Trains three models (one per past WC), evaluates each against the actual outcomes of WC 2014, 2018, and 2022. Reports log-loss, accuracy, RPS, confusion matrix, and naive-baseline comparison.

### Diagnose per-match prediction quality

```bash
python -m src.evaluation.diagnostics
```

For each past-WC backtest, lists the 10 worst-loss matches and produces a calibration table showing whether predicted probabilities match observed frequencies.

### Explore the data interactively

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

EDA notebook covering date density, missing values, the Poisson distribution check, home advantage measurement, scoring rate over time, country-name consistency, etc.

---

## Project layout

```
src/
  data/                  → data loading + Transfermarkt download
  features/              → Elo, recent form, squad values, group standings, tournament class
  models/                → Poisson regression + Dixon-Coles correction
  evaluation/            → backtests + per-match diagnostics
  prediction/            → WC 2026 per-match + Monte Carlo simulation
data/
  raw/                   → source CSVs (gitignored)
  processed/             → cleaned + feature-engineered tables (gitignored)
notebooks/               → EDA / exploration
docs/                    → 9-file "build it yourself" walkthrough
tests/                   → unit tests
README.md                → this file
issues.md                → engineering log: bugs, design decisions, limitations
```

---

## Data sources

| Source | Used for | Size |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | Match history 1872-present | ~3 MB |
| [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Player valuations over time + current national-team aggregates | ~80 MB |
| [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) | Historical WC squad rosters (1930-2022) | <1 MB |

All free, public, ODbL or similarly permissive licenses.

---

## Documentation

| File | What's in it |
|---|---|
| `README.md` | This file — quick-start + project overview |
| `docs/` | 9-file walkthrough that teaches you how to build this model from scratch. Start with `docs/00_overview.md`. |
| `issues.md` | Engineering log of bugs found, design decisions, and limitations. Reads like a debugging journal. |

For a beginner who wants to *understand* the model, read `docs/` in order. For an engineer maintaining or extending it, read `issues.md` to learn what we've already tried and why we made each choice.

---

## Headline result

After all the work, what does the model say about WC 2026?

| # | Team | P(win WC) |
|---|---|---|
| 1 | 🇪🇸 Spain | **18.8%** |
| 2 | 🇦🇷 Argentina | 14.6% |
| 3 | 🏴 England | 12.6% |
| 4 | 🇫🇷 France | 11.5% |
| 5 | 🇵🇹 Portugal | 4.7% |
| 6 | 🇩🇪 Germany | 4.2% |
| 7 | 🇳🇱 Netherlands | 4.2% |
| 8 | 🇧🇷 Brazil | 4.1% |
| 9 | 🇲🇦 Morocco | 3.4% |
| ... | ... | ... |

Top 4 teams cover ~57% of championship probability. Top 9 cover ~77%. About 23% probability mass is spread across the remaining 39 teams — meaningful upset potential.

---

## Status

- [x] Project setup + dependencies
- [x] Data acquisition (match history + Transfermarkt + Fjelstul)
- [x] Exploratory data analysis
- [x] Elo rating pipeline
- [x] Feature engineering (form, days_since_last, tournament_class)
- [x] Baseline Poisson model + Dixon-Coles correction
- [x] 3-iteration backtest against WC 2014, 2018, 2022
- [x] Squad market value feature (with calibrated citizenship-based fallback)
- [x] Dead-rubber detection (group standings)
- [x] WC 2026 per-match predictions (validated on 12 played matches: log-loss 1.00)
- [x] Monte Carlo tournament simulation (20k runs, championship probabilities)

**Performance:** log-loss 0.93-1.08 on WC backtests, 1.00 on real WC 2026 matches so far — in the published academic-model range.

---

## License

MIT for our code. Underlying data licenses per their respective sources (see Data sources table above).
