# 00 — Overview

This guide walks through building a machine-learning model that predicts the outcomes (W/D/L), exact scores, and win probabilities of FIFA World Cup matches. The specific target is the **2026 World Cup** (USA / Canada / Mexico, 48-team format), but the approach generalizes to any tournament.

The model is a **Poisson regression with Dixon-Coles correction**, fed by **Elo ratings, recent form, and squad market values** as features. It's a well-studied statistical-modeling approach for soccer. It is not necessarily an AI, but rather, careful feature engineering on top of a classical generalized linear model.

You should be able to follow this guide end-to-end if you're comfortable in Python (pandas, numpy, scikit-learn vocabulary). You don't need ML background — ML concepts are introduced as they come up.

## What you'll build

By the end you'll have a Python project that can:

- Download and clean international match history going back to 1872
- Compute Elo ratings for every national team at every point in history
- Engineer features (recent form, days since last match, home advantage, tournament class, squad market value) without data leakage
- Train a Poisson regression that predicts the expected goal counts for both teams in any match
- Apply a Dixon-Coles correction to improve draw probabilities
- Backtest the model against past World Cups (2014, 2018, 2022) to measure quality
- Generate predictions for the 2026 World Cup with calibrated W/D/L and exact-score probabilities

The full source is in `src/` — a structured Python project with separate modules for data loading, features, models, evaluation, and prediction

## What kind of ML problem is this?

This is a **supervised regression problem with a count-valued target**.

- **Supervised**: we have labeled training data. Every historical match has known scores. We use these to teach the model the relationship between features and outcomes.
- **Regression**: we predict numbers (goal counts), not categories.
- **Count-valued**: the target is non-negative integers (0, 1, 2, ...). This is what makes Poisson the natural distribution choice; it's designed for counts.

It's not unsupervised (we have labels), not classification (we predict counts, not categories — though we *derive* class probabilities from the counts), not reinforcement learning (no agent making sequential decisions), not generative (we don't sample synthetic data).

The full taxonomy of ML problems is:

```
Machine learning
├── Supervised
│   ├── Classification     (predict a label: spam/ham, image class)
│   └── Regression         (predict a number: house price, goal count)  ← this project
├── Unsupervised
│   ├── Clustering         (group similar examples)
│   ├── Dimensionality     (compress high-dimensional data)
│   └── Density estimation (model the distribution)
├── Reinforcement learning (agent + environment + reward)
└── Generative             (sample new data: text, images, audio)
```

If "neural networks" is the word you most associate with ML, note that's only one tool inside this taxonomy — and not the right one for our problem (we have ~30,000 training matches and the relationships are mostly linear in log-space; a 100M-parameter network would massively overfit).

## Vocabulary you'll see often

Here's a quick-reference glossary for the terms that come up throughout this guide. Please refer back as needed.

- **Feature**: a number (or category) describing one training example. For us, features include team Elo ratings, recent goals scored, whether a match is at a neutral venue, etc. The model uses features to predict the target.
- **Target**: what we're trying to predict. For us: the home and away goal counts in each match.
- **Training data**: matches whose outcomes we know. The model learns from these.
- **Test (or eval) data**: matches the model has never seen during training. Used to measure how well the model would do on real new matches. For us: held-out past World Cups.
- **Hyperparameter**: a tunable setting on the model that controls how it learns. Examples: regularization strength, number of training iterations, the K-factor in Elo. Not learned from data — set by the human.
- **Regression**: a model that predicts a number (vs. classification which predicts a category). Predicting "Brazil scores 2.3 goals on average" is regression. Predicting "Brazil wins / draws / loses" is classification.
- **GLM (Generalized Linear Model)**: a family of models that look like linear regression but transformed to handle non-Gaussian targets. Poisson regression (for counts) and logistic regression (for probabilities) are both GLMs.
- **Poisson distribution**: a probability distribution for non-negative integer counts (0, 1, 2, ...). Has one parameter, λ ("lambda"), which is both its mean and variance. Goals in soccer match the Poisson shape closely.
- **λ ("lambda")**: the expected count in a Poisson distribution. For us, λ_home = expected home goals, λ_away = expected away goals.
- **Score matrix**: a grid where cell (i, j) gives the probability of the scoreline `i-j`. Built by multiplying P(home=i) × P(away=j) under independence assumption.
- **Dixon-Coles correction**: a small adjustment to the score matrix to fix Poisson's tendency to under-predict low-scoring draws (0-0, 1-1).
- **ρ ("rho")**: the single tunable parameter of the Dixon-Coles correction. Negative ρ means boost draws.
- **Elo rating**: a single number representing a team's strength. Updates after each match (winner gains points, loser loses points). Used as a feature in the model.
- **Log-loss**: the standard metric for evaluating probabilistic predictions. Lower is better. Penalizes confident-wrong predictions more than uncertain-wrong ones. Random predictor scores ln(3) ≈ 1.099 for 3-class outcomes.
- **Accuracy**: % of test predictions where the highest-probability class matched the actual outcome. Less informative than log-loss for probabilistic models, but more intuitive.
- **RPS (Ranked Probability Score)**: like log-loss but gives partial credit for being close. Useful when classes have a natural ordering (Win → Draw → Loss).
- **Calibration**: how well predicted probabilities match observed frequencies. "When the model says 70%, does it actually happen 70% of the time?" A separate property from accuracy.
- **Data leakage**: when future information accidentally gets into the training process. Almost always inflates apparent model quality without improving real-world performance.
- **Imputation**: filling in missing values so the model can train. We use median imputation — replace each NaN with the column's median.
- **Snapshot**: a recorded value of something at a specific moment in time. We snapshot squad market values at each World Cup's kickoff date, then forward-fill to in-between dates.
- **Forward-fill**: for any date without an exact snapshot, use the most recent prior snapshot. "What was Brazil worth on 2015-09-01?" → look up Brazil's value as of the 2014 snapshot.
- **Aggregate column vs. atomic row**: a *dataset* often provides both a summary statistic (e.g., total team value) and the individual rows that produce it (e.g., per-player values). The aggregate is convenient; the atomic rows let you re-compute the aggregate if it's missing.
- **Calibration factor**: a multiplier that converts a fallback measurement to the same scale as the original. If aggregate A and atomic-sum B disagree, calibration factor `c = A/B` lets you estimate A when only B is available.
- **Backtest**: simulating "if I had built this model in the past, how would it have performed?" by training on pre-event data and evaluating on the actual event. The gold standard for time-series ML evaluation.
- **Naive baseline**: a simple, "dumb" predictor whose performance you compare against. If your fancy model only matches the dumb baseline, you've added no signal.
- **Confusion matrix**: a grid showing predicted-class vs. actual-class counts. Reveals systematic biases (e.g., "never predicts draws").
- **Sentinel value**: a placeholder used to mean "no data," like `-1`, `"N/A"`, `"not applicable"`, or empty string. Easy to mishandle if you don't check for them explicitly.
- **Production model**: the final trained model with no held-out test set; trained on every available data point and used to generate the actual predictions you ship.
- **Static forecast**: every prediction uses features as of a single fixed date. Standard for "one-shot pre-tournament" reports.
- **Live forecast**: features update as new matches happen; each prediction has a different "as of" date. Used for ongoing operational forecasting during a tournament.
- **Monte Carlo simulation**: repeatedly sampling random outcomes from probability distributions, then counting how often each combined outcome happens. We use this to estimate tournament-level probabilities (e.g., P(team X wins WC)) from per-match probabilities.
- **Conditioning**: using known information to refine probability estimates. "Given that X happened, what's the probability of Y?" During a tournament, we condition the simulation on matches that have already been played rather than re-sampling them.

## Prerequisites

**Python knowledge** required:

- Standard library, virtual environments (`venv`)
- pandas DataFrames (filtering, groupby, merge, rolling)
- numpy arrays and basic vector operations
- scikit-learn pipelines and column transformers

**Math** that helps but isn't required:

- Basic probability (what a distribution is)
- Logistic / linear regression vocabulary (you'll learn Poisson regression here from scratch)

**Tools** you'll need:

- Python 3.10+ on macOS/Linux/Windows
- ~500 MB of disk for datasets
- Internet for the initial data downloads

## Project layout

```
ml_proj/
├── src/
│   ├── data/         data loading + cleaning
│   ├── features/     Elo, recent form, squad values, tournament class
│   ├── models/       Poisson regression + Dixon-Coles
│   ├── evaluation/   metrics, backtest
│   └── prediction/   apply trained model to future matches
├── data/
│   ├── raw/          source datasets (downloaded, never modified)
│   └── processed/    cleaned + feature-engineered tables (generated)
├── notebooks/        exploration, EDA, learning notes
├── tests/            unit tests
├── docs/             this guide
├── requirements.txt
└── README.md
```

A few conventions worth knowing about:

- **Raw vs. processed data is a hard separation.** Never overwrite raw data. `raw/` is what you downloaded; `processed/` is what your code generated. If a feature breaks, you can regenerate `processed/` from `raw/`. Both directories are gitignored because data files are big and shouldn't be in version control.

- **`src/` is split into 5 modules** corresponding to the ML pipeline phases. This pays off when you swap one piece (try a different model, add a new feature) without rewriting everything.

- **Notebooks are for exploration only.** Modeling logic lives in `src/`. Notebooks are great for poking at data and visualizing things, bad for production code (hard to test, hard to reuse).

## The path through this guide

The guide proceeds in the same order you should build the project:

1. **`01_data_acquisition.md`** — Find and download datasets. Establish leakage-prevention discipline at the source level.
2. **`02_eda.md`** — Exploratory data analysis. The 7 questions you should ask of any new dataset, and what each one tells you about modeling decisions.
3. **`03_features.md`** — Compute Elo ratings, recent form, and other features. The critical leakage-avoidance pattern (`shift(1)`) is introduced here.
4. **`04_modeling.md`** — Why Poisson, what's a GLM, how to fit two regressions for home/away goals, and the Dixon-Coles correction for draws.
5. **`05_evaluation.md`** — Train/validation/test splits in a time-series setting. Log-loss, accuracy, RPS, calibration. The 3-iteration backtest design.
6. **`06_advanced_features.md`** — Adding new data sources (squad market values). Multi-source data integration. Name-matching gotchas.
7. **`07_prediction.md`** — From backtest to production. Tournament simulation.
8. **`08_lessons_and_pitfalls.md`** — Dead ends generalized into "watch out for X." The bugs that consume the most time in any ML project.

The estimated total build time for a competent Python developer reading and implementing along is **15–25 hours**, with the EDA, modeling, and advanced-feature sections being the slowest.

## A note on what this guide *isn't*

This is not a state-of-the-art research recipe. Modern best-in-class soccer prediction systems use player-level modeling (lineup prediction, per-player ratings derived from club football), neural network ensembles, and bespoke loss functions. Those approaches add 10× the complexity for maybe 0.05 log-loss improvement.

This guide builds a **strong baseline** — the kind of model that academic papers and football journalists routinely cite, scores in the published-model range (log-loss 0.93–1.06 on WC matches), and is interpretable. If you want to extend to player-level modeling, you'll have a solid foundation to build on.

The guide is also opinionated about *understanding what you're doing*. Every modeling choice (why Poisson, why Dixon-Coles, why a 1990 cutoff, why backtest against multiple WCs) is justified, not just stated. ML projects are mostly debugging your own assumptions; the best defense is knowing why you made them.

Now to the actual work. Start with **`01_data_acquisition.md`**.
