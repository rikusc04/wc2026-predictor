# 04 — Modeling

This chapter covers the model itself: Poisson regression with a Dixon-Coles correction. By the end you'll understand why this is the standard approach for soccer prediction, how generalized linear models work, why we fit two regressions (one for home goals, one for away goals), and how to derive every quantity you care about (W/D/L probabilities, exact-score probabilities, expected goal margin) from a pair of fitted models.

### Vocabulary used in this chapter

If any of these are unfamiliar, the glossary at the start of `00_overview.md` has plain-language definitions:

- **Poisson distribution / regression** — model for count data
- **λ ("lambda")** — the expected count parameter
- **Score matrix** — grid of probabilities for each exact scoreline
- **GLM** — generalized linear model; family that includes Poisson and logistic regression
- **Dixon-Coles correction** — adjustment for soccer's draw-rate quirk
- **ρ ("rho")** — single parameter of the Dixon-Coles correction
- **Hyperparameter** — model setting set by humans, not learned from data
- **Regularization** (L2) — penalty added to the loss to discourage extreme coefficients

The math here is more involved than other chapters, but every formula has a paragraph of intuition before and after. If something doesn't click, skim past it and come back; the intuition is more important than the algebra.

## Why Poisson?

The Poisson distribution describes counts of events that happen at a roughly constant rate. Classic examples:

- Emails arriving in an hour.
- Typos per page.
- Customers entering a shop per minute.

**Goals in a soccer match** are an almost-canonical Poisson use case, because:

- The count is a non-negative integer (0, 1, 2, ...).
- Goal-scoring events happen roughly independently throughout the match (somewhat true; we'll revisit).
- Most matches have small counts (0, 1, 2, 3 common; 7+ rare).
- The distribution shape (right-skewed, integer-valued) matches what we observe.

A Poisson distribution is characterized by one parameter, **λ ("lambda")** — the expected count. Once you know λ, the entire probability distribution is fixed:

```
P(k goals | λ) = (λ^k × exp(-λ)) / k!
```

You don't need to memorize the formula. The key intuition is: **λ is both the mean and the variance of the count.** It's the only parameter you need.

For a typical international match, λ might be around 1.4 (teams score, on average, ~1.4 goals each). The corresponding Poisson distribution looks like:

```
goals → P
  0      24%
  1      35%
  2      24%
  3      12%
  4       4%
  5+      1%
```

If we know one team's λ_home (expected home goals) and the other's λ_away (expected away goals), and we *assume independence*, we can compute the probability of any scoreline:

```
P(home = i, away = j) = Poisson(i | λ_home) × Poisson(j | λ_away)
```

Stack these into a matrix, sum the relevant cells, and we have W/D/L probabilities and exact-score probabilities.

The whole modeling problem reduces to: **given features for a match, predict λ_home and λ_away.** That's what a Poisson regression does.

## From Poisson distribution to Poisson regression

A Poisson regression is a kind of **generalized linear model (GLM)**.

GLMs are a family of models that look like linear regression but with a twist. Standard linear regression assumes:

```
y = β₀ + β₁·x₁ + β₂·x₂ + ... + noise
```

The output `y` can be any real number. That's fine for predicting house prices or temperature, but bad for counts. Goal counts can't be negative.

The GLM trick is to apply a **link function** to the prediction. For Poisson regression, the link is `log`:

```
log(λ) = β₀ + β₁·x₁ + β₂·x₂ + ...
```

Equivalently:

```
λ = exp(β₀ + β₁·x₁ + β₂·x₂ + ...)
```

Because `exp(...)` is always positive, λ is always positive. The linear combination can be any real number, and `exp()` maps it into the valid range for λ.

This is the same pattern as **logistic regression**, which predicts a probability:

```
logit(p) = β₀ + β₁·x₁ + ...    →    p = 1 / (1 + exp(−...))
```

Both Poisson regression and logistic regression are GLMs. They differ in the link function (log vs. logit) and the noise distribution (Poisson vs. Bernoulli).

## Why two models?

Each match has two outcomes we care about: home goals and away goals. We fit them with **two separate Poisson regressions**:

- **Home model**: `log(λ_home) = β₀ + β₁·home_elo + β₂·away_elo + β₃·home_form_scored + ...`
- **Away model**: `log(λ_away) = γ₀ + γ₁·home_elo + γ₂·away_elo + γ₃·away_form_scored + ...`

Same feature columns, different learned coefficients. Why this works:

- The home model learns "what makes the home team score" — its own attacking quality, the opponent's defensive weakness, the home boost.
- The away model learns "what makes the away team score" — its attacking quality, the opponent's defensive weakness, no home boost.

The β and γ coefficients absorb the asymmetry. We don't need to design separate features for "home perspective" and "away perspective"; the same features can be used with different coefficients to capture different effects.

There are more sophisticated joint formulations (Karlis-Ntzoufras bivariate Poisson, hierarchical Bayesian models) that share information between the two equations. Two-independent regressions is the simpler and surprisingly competitive baseline.

## sklearn implementation

scikit-learn ships `PoissonRegressor`, which fits the GLM with L2 regularization:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "home_elo_pre", "away_elo_pre",
    "home_form_scored", "home_form_conceded",
    "away_form_scored", "away_form_conceded",
    "home_days_since_last", "away_days_since_last",
]
BOOL_FEATURES = ["neutral"]
CATEGORICAL_FEATURES = ["tournament_class"]
ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


def _make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bool", "passthrough", BOOL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("reg", PoissonRegressor(alpha=0.1, max_iter=500)),
    ])


def train(cutoff: pd.Timestamp) -> TrainedModels:
    df = load_training_frame(cutoff)

    home_model = _make_pipeline()
    home_model.fit(df[ALL_FEATURES], df["home_score"])

    away_model = _make_pipeline()
    away_model.fit(df[ALL_FEATURES], df["away_score"])

    return TrainedModels(home_model=home_model, away_model=away_model, ...)
```

A few hyperparameter choices to justify:

- **`alpha=0.1`** — L2 regularization strength. Some shrinkage prevents the model from over-fitting noisy team-pair effects, especially for teams with few matches. Too much shrinkage and the model can't fit; too little and it memorizes training data.
- **`max_iter=500`** — iterations for the optimizer. Default is often 100; bump up if you see convergence warnings.

`StandardScaler` on numerics: not strictly required for the model to work, but it makes regularization act fairly across features (without it, `home_days_since_last` measured in days would get penalized less than `home_elo_pre` measured in hundreds of points). Always scale numeric features for regularized models.

`OneHotEncoder` on categoricals: turns the `tournament_class` column into 5 binary columns (one per class, minus a reference). The model learns a coefficient per class, capturing how that tournament type shifts the expected goal count up or down.

For the full file, see `src/models/poisson.py`.

## Deriving everything from two λs

Given fitted models, prediction is a one-liner per match:

```python
lam_h = home_model.predict(X)   # expected home goals, shape (n_matches,)
lam_a = away_model.predict(X)   # expected away goals, shape (n_matches,)
```

But we want richer outputs than just expected goals. From the two λs, under the independence assumption, we can compute:

### W/D/L probabilities

Build a score matrix:

```python
from scipy.stats import poisson

def score_matrix(lam_h: float, lam_a: float, max_goals: int = 20) -> np.ndarray:
    """P(home=i, away=j) under independent Poisson assumption."""
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    return np.outer(ph, pa)
```

The matrix is `(max_goals+1) × (max_goals+1)`. The entry at `[i, j]` is the probability of the scoreline `i-j`.

Sum the appropriate cells to get outcome probabilities:

```python
def outcome_probs(lam_h: float, lam_a: float) -> tuple[float, float, float]:
    M = score_matrix(lam_h, lam_a)
    p_home = np.tril(M, -1).sum()   # rows > cols → home goals > away goals → home win
    p_draw = np.trace(M)            # diagonal → home goals == away goals → draw
    p_away = np.triu(M, 1).sum()    # rows < cols → home goals < away goals → away win
    return float(p_home), float(p_draw), float(p_away)
```

### Exact-score probabilities

The score matrix itself *is* the exact-score distribution. The most likely scoreline is its argmax:

```python
def most_likely_score(lam_h: float, lam_a: float) -> tuple[int, int, float]:
    M = score_matrix(lam_h, lam_a)
    idx = np.unravel_index(np.argmax(M), M.shape)
    return int(idx[0]), int(idx[1]), float(M[idx])
```

For most matches, the most likely scoreline has 10-15% probability (no single scoreline is dominant; soccer is high-variance). The exact-score distribution is more informative than the modal score.

### Why MAX_GOALS matters

The matrix is finite. We truncate at some maximum goal count for both teams. If MAX_GOALS is too small relative to the predicted λ, probability mass falls off the edge and your probabilities don't sum to 1.

For typical international matches (λ in the 0.5–3.0 range), `MAX_GOALS = 10` is fine. But the model can produce λ values up to 10+ for extreme mismatches (Brazil vs. San Marino). At λ=10.5, there's substantial probability mass at scorelines like 11-0, 12-0, 13-0 — enough that a `MAX_GOALS = 10` matrix loses ~50% of the total probability.

**Set `MAX_GOALS = 20`** for safety. The cost is a 441-cell matrix instead of 121, which is negligible computationally.

## The draw under-prediction problem

When you backtest this baseline model, you'll notice something strange: **the model almost never predicts a draw as the most likely outcome**, even though ~25% of soccer matches end drawn.

This is a known failure mode of *independent* Poisson. The independence assumption says home goals and away goals are statistically uncorrelated. In real football, they're slightly *positively* correlated at low scores:

- At 0-0, teams play more conservatively (don't want to concede the first goal).
- At 1-1, teams sometimes ease off the gas.
- At 1-0, the leading team protects the lead; the trailing team takes more risks (less likely to *also* score, because they're committing forward).

Independent Poisson can't capture these dynamics. It under-predicts low-scoring draws (0-0, 1-1) and over-predicts narrow wins (1-0, 0-1).

## Dixon-Coles correction

Dixon & Coles (1997) proposed a small correction that fixes this exact issue. They apply a multiplicative correction **τ** to only four cells of the score matrix:

| Scoreline | τ multiplier |
|---|---|
| 0–0 | 1 − λ_h · λ_a · ρ |
| 1–0 | 1 + λ_a · ρ |
| 0–1 | 1 + λ_h · ρ |
| 1–1 | 1 − ρ |
| All others | 1 (untouched) |

There's one new parameter: **ρ** ("rho"). When ρ is negative (and Dixon & Coles found it usually is, around −0.1 to −0.15 for English club football):

- τ(0,0) > 1 → boost 0-0 probability
- τ(1,1) > 1 → boost 1-1 probability
- τ(1,0) < 1 → dampen 1-0 probability
- τ(0,1) < 1 → dampen 0-1 probability

The net effect: mass moves from "narrow wins" toward "low-scoring draws," matching what real data shows.

### Implementation

```python
def score_matrix(lam_h: float, lam_a: float, max_goals: int = 20, rho: float = 0.0) -> np.ndarray:
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    M = np.outer(ph, pa)
    if rho != 0.0:
        M[0, 0] *= 1.0 - lam_h * lam_a * rho
        M[1, 0] *= 1.0 + lam_a * rho
        M[0, 1] *= 1.0 + lam_h * rho
        M[1, 1] *= 1.0 - rho
        M = np.maximum(M, 0.0)   # clip negatives — see below
        M = M / M.sum()          # renormalize
    return M
```

Two implementation gotchas:

1. **τ can drive cells negative** for extreme λ (e.g., λ_h × λ_a × ρ > 1). Probability can't be negative; clip to zero.
2. **The correction perturbs total mass slightly**, so renormalize so probabilities sum to 1.

### Fitting ρ via maximum likelihood

ρ is a parameter of the model; you fit it on training data. The cleanest approach is **joint maximum likelihood** — fit λ_home, λ_away, and ρ together to maximize the likelihood of observed scores.

A simpler **two-stage approach**:

1. Fit the two Poisson regressions assuming ρ=0 (so they produce λ_home, λ_away for each match).
2. Fix the λs and search for the ρ that maximizes the τ-weighted likelihood.

The two-stage approach is approximate (the optimal λs would change slightly under non-zero ρ) but ~99% as good and much simpler to code:

```python
from scipy.optimize import minimize_scalar

def fit_rho(home_model, away_model, train_df) -> float:
    X = train_df[ALL_FEATURES]
    lam_h = home_model.predict(X)
    lam_a = away_model.predict(X)
    h_scores = train_df["home_score"].astype(int).to_numpy()
    a_scores = train_df["away_score"].astype(int).to_numpy()

    # Only the four affected cells contribute non-trivially
    low_score_mask = (
        ((h_scores == 0) & (a_scores == 0)) |
        ((h_scores == 1) & (a_scores == 0)) |
        ((h_scores == 0) & (a_scores == 1)) |
        ((h_scores == 1) & (a_scores == 1))
    )
    h_low = h_scores[low_score_mask]
    a_low = a_scores[low_score_mask]
    lh_low = lam_h[low_score_mask]
    la_low = lam_a[low_score_mask]

    def neg_log_lik(rho: float) -> float:
        tau_vals = np.ones_like(lh_low)
        m00 = (h_low == 0) & (a_low == 0)
        m10 = (h_low == 1) & (a_low == 0)
        m01 = (h_low == 0) & (a_low == 1)
        m11 = (h_low == 1) & (a_low == 1)
        tau_vals[m00] = 1.0 - lh_low[m00] * la_low[m00] * rho
        tau_vals[m10] = 1.0 + la_low[m10] * rho
        tau_vals[m01] = 1.0 + lh_low[m01] * rho
        tau_vals[m11] = 1.0 - rho
        if (tau_vals <= 0).any():
            return 1e10   # infeasible
        return -np.log(tau_vals).sum()

    result = minimize_scalar(neg_log_lik, bounds=(-0.3, 0.3), method="bounded")
    return float(result.x)
```

This is a 1D optimization — fast, robust. The bounded interval `(-0.3, 0.3)` is a sanity guard; in practice ρ converges to roughly −0.05 to −0.15.

### Expected DC effect size

Dixon-Coles is a *correction*, not a structural change. Effects:

- **Log-loss** typically improves by 0.005–0.02 (small but measurable).
- **Confusion-matrix draws** may go up modestly. For some leagues (low-scoring European leagues) the argmax of some matches flips to "draw". For international football with many mismatches, the effect on argmax is smaller.

If you don't see *any* effect from DC, double-check:

- Is `rho` being passed to `score_matrix` at prediction time?
- Is `fit_rho` being called during training?
- Is the fitted ρ in a sensible range (−0.2 to 0)?

## Hyperparameter summary

The Poisson + DC model has these tunable settings:

| Parameter | Value | Why |
|---|---|---|
| `alpha` (PoissonRegressor) | 0.1 | L2 regularization. Mild shrinkage. |
| `max_iter` | 500 | Convergence safety. |
| `MAX_GOALS` | 20 | Score matrix size. Covers extreme λ. |
| `ρ` bound | (−0.3, 0.3) | Plausible range for soccer. |
| Form window | 10 matches | Recent-form rolling window. |
| Train start year | 1990 | From EDA — scoring rate stabilizes. |

Most of these can be tuned with a held-out validation set. In practice the model is fairly insensitive to the exact values (within reason).

## Two-stage estimation drift

Worth naming explicitly: our two-stage Dixon-Coles fit is approximate.

The "correct" approach is to fit λ_home, λ_away, and ρ *jointly* by maximum likelihood — meaning the λ predictions in the home and away models are slightly different (and slightly better) than what you get from the two-stage approach.

In practice, the joint approach is harder to code (you need to write the GLM yourself rather than use `sklearn.PoissonRegressor`), and the accuracy gain is small (<0.005 log-loss). The two-stage approach is the right v1 choice.

If you wanted to extend to the joint fit, the recipe is:

1. Write your own loss function combining the Poisson likelihoods for both teams plus the τ correction.
2. Use scipy or PyTorch's autograd to optimize all coefficients simultaneously.
3. Skip `sklearn.PoissonRegressor` entirely.

## Common modeling pitfalls

**Confusing "feature dim" with "team dim".** If you have 200 teams and add a one-hot encoding of "home team", you've just added 200 columns to your feature set. With ~30k training rows, the model can severely overfit team-specific effects. Use a single numeric feature (Elo) instead.

**Training without StandardScaler on regularized models.** Without scaling, regularization penalizes features measured in big units (Elo, ratings) much more than features measured in small units (boolean flags). The model then learns to ignore the high-magnitude features. Always scale before applying regularization.

**Treating regression output as classification probabilities.** PoissonRegressor's `.predict()` returns λ, not class probabilities. You have to derive W/D/L probabilities via the score matrix.

**Forgetting that `MAX_GOALS` is a hyperparameter.** A too-small `MAX_GOALS` silently drops probability mass. Always sanity-check that your W/D/L probabilities sum to ~1.0 for extreme matchups.

**Two independent Poissons ≠ joint Poisson.** Our model assumes independence between home and away goals. Real data has slight correlation. Dixon-Coles handles the low-score case; for richer correlation modeling you'd need a bivariate Poisson (Karlis-Ntzoufras 2003).

## What's next

You have a trained model. Now you need to measure how good it is. The right way to do this in time-series ML — and the meaning of "good" in this context — is `05_evaluation.md`.
