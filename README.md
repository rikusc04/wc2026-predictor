# World Cup 2026 Match Predictor

ML model that predicts FIFA World Cup 2026 match outcomes (W/D/L), exact scores, and win probabilities using a Poisson-style approach with Dixon-Coles correction.

## Versions

| Version | What's in it | Status |
|---|---|---|
| **v1** | Poisson regression + Dixon-Coles correction. Features: Elo, recent form, days-since-last, tournament class, squad market value (with citizenship-based fallback), dead-rubber flag, binary `neutral` venue. 20k-run Monte Carlo tournament sim. | shipped |
| **v2 — Phase 1, Item 1** | Graded multi-host advantage feature (`host_advantage_home/away` ∈ {0.0, 0.3, 0.7, 1.0}) replaces v1's binary `neutral`. Models WC 2026's three-host CONCACAF format and the CONMEBOL↔CONCACAF Americas adjacency. | shipped |
| **v2 — Phase 1, Item 3** | FIFA's official 48-team knockout bracket replaces the random-with-group-avoidance shortcut. Argentina overtakes Spain as the most-likely champion under fixed seeding. | shipped |
| **v2 — Phase 1, Item 2** | Native-altitude advantage feature (binary, fires when team is lifelong-resident at the venue's altitude). Native-advantage framing rather than naive "visitor penalty," accounting for FIFA's mandatory 2-week acclimation that equalizes visitor disadvantage. | shipped |
| **v2 — Refinements** | CAF↔AFC adjacency fixes the Qatar 2022 backtest regression. Per-venue knockout cache routes Mexico through correct host-advantage + altitude at Azteca knockouts (Mexico's P(win WC) +1.1pt). Plus a 47-check invariant suite (`tests/check_v2_invariants.py`) to catch silent-off-by-N regressions. | shipped |
| v2 — Phase 1 follow-ups | Per-host learned home-advantage coefficients (Mexico's Azteca ≠ Liechtenstein's home crowd). | deferred |
| **v2 — Phase 2.1** | Lineup-aware starting-XI market value (`lineup_value_home/away`) from StatsBomb open data (314 internationals: WC 2018/22, Euro 20/24, Copa 24, AFCON 23). WC 2022 backtest improved 1.1001 → 1.0914. WC 2026 forecast unchanged (no StatsBomb coverage). | shipped |
| **v2 — Phase 2.2a** | Modal-XI lineup predictor for WC 2026 — each qualifier's predicted starting XI = top-11 by appearance count over last 5 StatsBomb matches; citizenship-top-11 fallback for teams with no coverage. Unlocks the lineup_value feature for the WC 2026 forecast. | shipped |
| **v2 — Phase 2.2b** | Actual lineups for the 12 played WC 2026 matches (Wikipedia per-group pages) override predicted values. Diagnostic: modal-XI predictor matches ~35-50% of actual starters per side — informative for Phase 2.2c scoping. ALSO surfaced a bracket bug: Groups C and D were inverted since Phase 1 Item 3 (chronological inference vs FIFA seeding). Fix routes 8 teams through correct R32 slots; Spain and Argentina both rise ~1pt. | shipped |
| **v2 — Phase 2.2c** | Squad-filter on modal-XI: WC 2026 26-man squads scraped from Wikipedia restrict the predicted starting XI to currently-rostered players. Kills the "Neymar starts for Brazil although he wasn't called up" failure mode. Squad→StatsBomb name matcher handles Korean/Japanese surname-first swap (sorted-tokens) and Portuguese mononyms (token-subset). Overlap diagnostic also rewritten to compare at player_id level instead of last-name substring — exposes the true overlap and fixes Korea-style false negatives. Mean overlap **3.9 → 5.4 / 11** across the 21 modal_xi sides in the 12 played matches; Brazil 1→7, South Korea 0→6, Ecuador 1→4. | shipped |
| v2 — Phase 2.2d | Per-player ratings (FBRef/club-Elo) + position-weighted aggregation. | not started |
| v3 | Hypothetical follow-up (calibration, tournament scope expansion). | not defined |

See `issues.md` for the engineering log of both versions.

## Approach
Predict expected goals (`λ_home`, `λ_away`) for each match using a Poisson regression on engineered features (Elo ratings, recent form, squad market value, host advantage, tournament class). From those two expected-goals numbers we derive:
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

## Headline result (v2 through Phase 2.2b)

After all the work, what does the model say about WC 2026?

| # | Team | P(win WC) | Δ vs v1 |
|---|---|---|---|
| 1 | 🇪🇸 Spain | **17.9%** | −0.9 |
| 2 | 🇦🇷 Argentina | 17.0% | +2.4 |
| 3 | 🇫🇷 France | 9.9% | −1.6 |
| 4 | 🏴 England | 9.2% | −3.4 |
| 5 | 🇲🇽 Mexico | **4.7%** | +2.8 |
| 6 | 🇧🇷 Brazil | 4.5% | +0.4 |
| 7 | 🇵🇹 Portugal | 4.0% | −0.7 |
| 8 | 🇩🇪 Germany | 3.5% | −0.7 |
| 9 | 🇨🇴 Colombia | 3.3% | new top-11 |
| 10 | 🇳🇱 Netherlands | 3.2% | −1.0 |
| 11 | 🇲🇦 Morocco | 3.0% | −0.4 |

**Spain is #1, Argentina close behind, Mexico in the top 5.** The full v2 stack — graded host advantage, altitude, FIFA bracket (with the Group C/D fix from 2.2b), per-venue knockout cache, modal-XI lineup prediction with actual-lineup override for played matches — combines to shift probabilities meaningfully from where v1 left them.

Five effects compound across the v2 work:
1. **Item 1** (graded host advantage + CAF↔AFC refinement) — Americas teams gain across all WC 2026 matches.
2. **Item 2** (altitude native advantage) — Mexico's group games at Azteca/Zapopan and Colombia's two altitude games get a small boost.
3. **Item 3** (FIFA bracket) — bracket-path effects per team.
4. **Per-venue knockout cache** — Mexico's R32+R16 at Azteca get the altitude+host compound (issues #44).
5. **Phase 2.2a modal-XI lineup_value** — France's predicted XI €656M (Mbappé-heavy) lifts France; Argentina's €459M is more accurate than headline squad value (less peak-Messi); Spain's balanced XI rises to #1.

Top 4 cover ~52% of championship probability. See `issues.md` items #25–58 for the full design, backtest comparisons, and known limitations.

---

## Status

### v1 (shipped)
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

**v1 performance:** log-loss 0.93–1.06 on WC 2014/18/22 backtests, 1.00 on the first 12 real WC 2026 matches — in the published academic-model range.

### v2 — Phase 1
- [x] **Item 1 — Multi-host advantage** (graded `host_advantage_home/away`, alias-normalized country names, CONMEBOL↔CONCACAF Americas adjacency). Backtest: −0.031 on WC 2014, −0.040 on WC 2018, +0.055 on WC 2022 (Qatar regression — see issues #26).
- [x] **Item 2 — Native-altitude advantage** (`src/features/altitude.py` — binary feature: team is altitude-native if home elevation ≥ venue − 500m). Framed as native advantage rather than visitor penalty since FIFA's 2-week mandatory acclimation roughly equalizes the visitor disadvantage. Fires on 5 WC 2026 fixtures + ~900 historical training rows. See issues #36–41.
- [x] **Item 3 — Real FIFA 48-team knockout seeding** (`src/prediction/bracket.py` encodes FIFA's bracket tree + 3rd-place eligibility matrix; chronological-order group labeling derived from fixture data). Argentina now top of championship table at 17.3%; France/Germany drop notably on tougher bracket paths. See issues #30–35.
- [x] **Refinements** — CAF↔AFC adjacency in `confederations.py` fixes the Qatar 2022 backtest regression (1.1120 → 1.1001). Per-venue knockout cache (`KNOCKOUT_VENUES` in `bracket.py`, 4 caches in the simulator) routes Mexico through correct host + altitude at Azteca knockouts (P(win WC) 3.0% → 4.4%). Plus `tests/check_v2_invariants.py` — 47-check end-to-end smoke test. See issues #42–44.

### v2 — Phase 2
- [x] **Phase 2.1 — Lineup-aware starting-XI value** (`src/data/lineups_loader.py` + `src/features/lineup_values.py`). 314 international matches from StatsBomb open data → `lineup_value_home/away` features. WC 2022 backtest improved 1.1001 → 1.0914 (Cameroon-Brazil 2022 type misses softened). WC 2026 forecast unchanged (no StatsBomb coverage). See issues #46–52.
- [x] **Phase 2.2a — Modal-XI lineup predictor for WC 2026** (`src/features/lineup_predictor.py`). For each qualifier: predicted XI = top-11 by appearance count over last 5 StatsBomb matches; citizenship-top-11 fallback for teams without StatsBomb coverage (32 modal_xi / 16 citizenship / 0 NaN). First v2 feature to move the WC 2026 headline by 2+ percentage points. See issues #53–58.
- [x] **Phase 2.2b — Actual lineups + bracket bug fix** (`src/data/wc2026_actual_lineups.py`). 264 starter rows from Wikipedia per-group pages for the 12 played matches override predicted values. **Diagnostic finding:** modal-XI predictor matches ~35-50% of actual starters per side (informative for Phase 2.2c scoping). **Bracket bug surfaced:** Groups C and D were inverted since Phase 1 Item 3 (chronological vs FIFA seeding). Fix in `bracket.py:WC2026_FIFA_GROUPS`. See issues #59–62.
- [ ] Phase 2.2c — Expand training-data lineup coverage with fresher recent matches (NOT just StatsBomb expansion — modal-XI diagnostic shows freshness > volume).
- [ ] Phase 2.2d — Per-player ratings (replace market value with FBRef/club-Elo) + position-weighted aggregation.

### v2 — Phase 3 (polish)
- [ ] Joint MLE for Dixon-Coles ρ (replace two-stage fit)

---

## License

MIT for our code. Underlying data licenses per their respective sources (see Data sources table above).
