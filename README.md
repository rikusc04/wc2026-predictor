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

### Why Poisson?

Soccer goals are rare, roughly independent events scattered across a 90-minute window — exactly the setting the **Poisson distribution** describes. If a team scores at a rate of λ goals per match, then under Poisson, P(scoring exactly k goals) = (λ^k · e^-λ) / k!. This isn't just a convenient assumption — it's empirically validated. Count goals per team per match across 49k matches in the dataset and the histogram tracks Poisson almost perfectly (the EDA notebook verifies this).

So the modeling problem reduces to: *for each match, predict λ_home and λ_away.* Once we have those two numbers, every other quantity — W/D/L probabilities, exact-score probabilities, tournament outcomes — falls out by arithmetic.

### Poisson regression

The "ML model" is a **Poisson regression**: a linear model whose output is interpreted as log(λ). Given a feature vector x (Elo difference, recent form, host advantage, lineup value, …), it predicts λ = exp(β · x). The β coefficients are fit by maximum likelihood on ~32,000 historical matches.

Why not a neural network or gradient-boosted trees? With ~30k training matches and ~13 features, a Poisson GLM is the right complexity-fit. Bigger models would overfit; their main advantage — capturing non-linear interactions — isn't exercised here. The leverage in this problem is in **feature engineering** (host advantage, lineup quality, altitude, bracket structure), not in model complexity. That's why v2 is mostly about new features, not a new model.

### Dixon-Coles correction

Pure Poisson has a known weakness on soccer data: it **under-predicts the four most common low scores** — 0-0, 1-0, 0-1, and 1-1 — because in tight matches teams cluster around those scorelines more than two independent Poisson draws would suggest. Dixon & Coles (1997) proposed a clean fix: introduce a single parameter ρ that nudges those four probabilities while leaving the rest of the distribution alone. We fit ρ post-hoc after the Poisson fit (you see it printed as `Dixon-Coles ρ = -0.0535` during training).

### From λ to predictions

Given λ_home and λ_away for a match, three derived quantities matter:

- **W/D/L probabilities** — sum the joint score distribution over the regions i > j (home win), i = j (draw), i < j (away win).
- **Exact-score probabilities** — the full 21×21 grid (capped at 20 goals per side, which captures >99.99% of probability mass).
- **Tournament-level outcomes** — Monte Carlo: sample 20,000 hypothetical tournaments by drawing scorelines from each match's distribution, run them through the FIFA bracket, and tally how often each team advances at each stage.

### Preventing data leakage

To prevent data leakage, all training data is **frozen at 2026-06-10** (the day before WC 2026 kickoff). The model never sees a single match from the tournament it's predicting. WC 2026 matches are held out for validation, which is what makes the live evaluation on the 12 already-played matches meaningful — those predictions are out-of-sample by construction.

For deeper validation, the backtest script trains three *separate* models — each frozen before WC 2014, 2018, 2022 — and evaluates each against its respective tournament. See "Backtest" below.

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

If you just want to generate predictions, run these in order (~30 minutes the first time, ~1 minute on subsequent runs).

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

### Step 2: Build features (one-time, ~25 minutes)

These commands transform the raw data into the tables the model trains on. **Run them in order** — each depends on the previous one's output. **Skip if `data/processed/` is already populated**.

```bash
python -m src.features.elo              # ~5 sec
python -m src.features.squad_values     # ~23 min
python -m src.features.group_standings  # ~5 sec
python -m src.features.build            # ~5 sec
```

**What each feature captures:**

This is where most of the engineering effort lives. The Poisson model is simple; the features are what make it predictive.

- **Elo** (`elo.py`) — a relative-strength score that updates after every match (winners gain rating from losers, scaled by margin of victory and match importance). Elo encodes *long-run* team strength: form, history, and reputation collapsed into one number that updates online as matches are played.
- **Recent form** (built inside `build.py`) — rolling goals scored and conceded over the team's last few matches. Captures *short-run* condition that Elo is too slow to reflect (injuries, lineup churn, in-form windows).
- **Squad market value** (`squad_values.py`) — total Transfermarkt valuation of the squad. A proxy for player quality at a level Elo can't see; Elo treats every match the same regardless of *who's actually on the pitch*.
- **Lineup value** (loaded inside `build.py` from Phase 2.1's StatsBomb data + Phase 2.2a's modal-XI predictor + Phase 2.2c's squad filter) — refines squad value down to the *predicted starting XI* rather than the whole squad. Argentina's depth chart includes 50 internationals over a year; their starting XI is the meaningful subset.
- **Days since last match** — fatigue / freshness signal.
- **Tournament class** (friendly / continental / World Cup) — captures the well-documented fact that teams play differently in low-stakes vs high-stakes matches.
- **Group standings + dead-rubber flag** (`group_standings.py`) — for historical group-stage matches, identifies games where one or both teams had already secured advancement or elimination. Dead rubbers have *different goal distributions* (goalkeepers rested, B-team minutes, etc.) and the model needs to know so it doesn't treat them as full-effort data.
- **Host advantage** (`confederations.py`) — graded advantage for hosts and their continental neighbors (CONMEBOL teams get partial credit at a CONCACAF tournament; CAF and AFC teams get adjacency credit where appropriate).
- **Altitude** (`altitude.py`) — binary flag for teams lifelong-adapted to their venue's altitude (Mexico at Azteca, Bolivia in La Paz). Framed as *native advantage* rather than visitor penalty because FIFA's mandatory 2-week acclimation roughly cancels the visitor disadvantage at the venue side.

`build.py` is the final assembly step — it joins everything into `features.csv`, the single training-ready table the model consumes.

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

**Why Monte Carlo?**

Step 3 gives us *per-match* probabilities (Argentina vs Jordan: 84% Argentina). But the question we actually want answered is *per-tournament* (Argentina: 17% chance of winning the WC). There's no closed-form way to get from one to the other because every team's path depends on group standings, third-place qualification, and bracket pairings — all of which depend on the results of *other* matches.

The way out is sampling. Run 20,000 simulated tournaments, each time drawing random scorelines from the predicted distributions and playing the bracket through to the final. After 20k runs, the fraction of simulations in which each team won the WC is our estimate of their championship probability. The standard error on a 17% estimate after 20k draws is well under one percentage point — good enough that re-running the script produces nearly identical numbers.

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

This is the central ML question: *does the model generalize, or did it just memorize the training data?* You can't answer that with WC 2026 itself because the tournament isn't over. You can with past WCs.

The script trains three *separate* models — one frozen before each of WC 2014, 2018, 2022 — and evaluates each against its respective tournament. Each trained model never sees any data from the tournament it's being tested on. Five metrics:

- **Log-loss** — penalizes confident wrong predictions much more heavily than uncertain ones. Lower is better. A "65% confident, actual draw" hurts modestly; a "95% confident, actual loss" hurts a lot. Random-guessing baseline is ~1.10. The model scores 0.90–1.06 across the three past WCs, which lands in the published academic-model range.
- **Accuracy** — % of W/D/L picks correct. Less informative than log-loss because it treats "65% home win, actual draw" the same as "95% home win, actual draw" — both wrong, but very different *kinds* of wrong.
- **RPS (Ranked Probability Score)** — like log-loss but *ordinal-aware*: predicting "away win" when the result is "draw" is penalized less than predicting "away win" when the result is "home win." Makes sense for W/D/L because draw sits between the two wins on a natural ordering.
- **Confusion matrix** — predicted outcome × actual outcome. Reveals systematic biases: does the model over-predict draws? Under-predict upsets? Where is the residual error concentrated?
- **Naive-baseline comparison** — does the model beat trivial strategies like "always pick home team" or "always pick higher-Elo team"? If not, the whole stack of feature engineering is theater. (It does, but the check matters.)

### Diagnose per-match prediction quality

```bash
python -m src.evaluation.diagnostics
```

Two outputs, both finer-grained than the aggregate metrics from the backtest:

- **10 worst-loss matches per WC** — the predictions that hurt the score the most. Useful for spotting systematic blind spots. Example: Cameroon beating Brazil at WC 2022 was a worst-loss match — the model was ~80% Brazil, the result was 1-0 Cameroon. Investigating these losses is what motivated Phase 2.1 (lineup value, since Brazil rested most of its first-choice XI) and Phase 2.2 (squad-aware lineups). Looking at *why specific predictions failed* is how the next round of feature engineering gets prioritized.
- **Calibration table** — across all matches where the model said "60–70% home win," what fraction *actually* were home wins? If a "70% confident" model is right 50% of the time, it's *miscalibrated* (overconfident). A well-calibrated model has predicted % ≈ observed % in every bucket. Calibration is independent of accuracy — you can be accurate but miscalibrated, or calibrated but inaccurate. Calibration matters most for downstream uses: betting markets, decision-making under uncertainty, and the Monte Carlo simulation itself (which relies on the predicted probabilities being trustworthy).

### Explore the data interactively

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

EDA = Exploratory Data Analysis. Not part of the model itself — it's the kind of work you do *before* modeling to understand what you're dealing with. Covers:

- **Date density over time** — how many matches per year across the 1872–2026 history. Reveals the explosion of internationals post-1990, which is part of why training data is cut at 1990 (pre-1990 matches are sparse, slower-paced, and from a different competitive era).
- **Missing values** — which columns are sparse, where the gaps cluster, and what fallback strategies are required (e.g., the citizenship-based squad-value fallback exists because Fjelstul squad data only covers historical WCs, not friendlies).
- **Poisson distribution check** — does the per-team-per-match goal count actually look Poisson-distributed in the data? **This is the validation that justifies the entire Poisson-regression approach.** If this check failed, the model would have needed to be rebuilt from scratch with a different distributional assumption (negative binomial, zero-inflated Poisson, etc.).
- **Home advantage measurement** — empirical goal differential at home vs neutral, quantifying what "home advantage" actually amounts to in the data (~0.4 expected-goal advantage). Sets the baseline that the host-advantage feature has to beat.
- **Scoring rate over time** — has football gotten more or less goal-heavy across eras? (Spoiler: less goal-heavy until ~1990, then roughly flat.)
- **Country-name consistency** — checks that "USA," "United States," and "U.S.A." don't get split into three teams. Foreshadows the `former_names.csv` join that handles renames like Zaïre → DR Congo.

The Poisson distribution check is the most important thing here — it's the empirical evidence that the modeling approach is appropriate for the data, rather than a stylistic choice.

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

## Headline result (v2 through Phase 2.2c)

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
- [x] **Phase 2.2c — WC 2026 squad filter for modal-XI** (`src/data/wc2026_squads.py` + `src/features/squad_to_sb.py`). 26-man squads scraped from Wikipedia restrict the predicted XI to currently-rostered players, killing the "Neymar starts for Brazil although he wasn't called up" failure mode. Squad→SB name matcher handles Korean/Japanese surname-first swap (sorted-tokens) and Portuguese mononyms (token-subset). Overlap diagnostic rewritten to compare at SB player_id level; mean overlap **3.9 → 5.4 / 11** across 21 modal_xi sides (Brazil 1→7, South Korea 0→6, Ecuador 1→4). See issues #63+.
- [ ] Phase 2.2d — Per-player ratings (replace market value with FBRef/club-Elo) + position-weighted aggregation.

### v2 — Phase 3 (polish)
- [ ] Joint MLE for Dixon-Coles ρ (replace two-stage fit)

---

## License

MIT for our code. Underlying data licenses per their respective sources (see Data sources table above).
