# 05 — Evaluation

You have a trained model. The next question is: how good is it?

### Vocabulary used in this chapter

If any of these are unfamiliar, the glossary at the start of `00_overview.md` has plain-language definitions:

- **Training / validation / test sets** — different slices of data used at different stages
- **Backtest** — train on past data, evaluate on a known past event
- **Log-loss** — primary metric for probabilistic predictions; lower is better
- **Accuracy** — % correct argmax predictions
- **RPS** — ranked probability score; partial credit for being close
- **Calibration** — whether predicted probabilities match observed frequencies
- **Naive baseline** — a simple "dumb" predictor to compare against
- **Confusion matrix** — predicted-vs-actual class breakdown



This chapter answers that question. We cover:

- The training / validation / test taxonomy and how it maps to our setup.
- Why "evaluation" in classical supervised ML is different from "evaluation" in LLM / RL territory.
- The four metrics worth computing (accuracy, log-loss, RPS, calibration) and what each one tells you.
- The naive baseline pattern — why every model evaluation must include a "dumb" comparison.
- Time-based backtesting — the right way to test a time-series model.
- Confusion matrices — what they reveal about systematic biases.

By the end, you'll know not just whether your model "works" but in what specific ways it does or doesn't, and where to look first if numbers seem off.

## A note on terminology: pre-training, post-training, eval

Modern ML discourse uses these terms loosely; the specific definitions vary by context:

- **Pre-training** (LLM context): training a large generic model on a giant corpus before specializing it. Doesn't apply to classical supervised learning like ours.
- **Post-training** (LLM context): alignment, RLHF, instruction tuning after pre-training. Also doesn't apply to us.
- **Eval(uation)** (everywhere): measuring how well your model performs on held-out data. **This applies to everyone**.

The cleaner taxonomy for our setup is:

- **Training**: fit model parameters to minimize training loss.
- **Validation**: tune hyperparameters (regularization strength, feature window sizes, ρ bounds) by holding out a subset of training data, training on the rest, evaluating, and picking the configuration that wins.
- **Test / eval**: measure final performance on a truly held-out dataset — data the model has never been touched by, in any way, including hyperparameter selection.

For time-series problems like ours, these splits must be **chronologically ordered**. You can't randomly shuffle and split, because:

- The test set must be in the future relative to training (otherwise it's leakage).
- Validation must be in the future relative to *its* training data (same reason, applied to the inner loop).

The right split looks like:

```
[ train       ] [ validation ] [ test       ]
   1990–2010      2010–2014      2014+
```

In our project, we mostly skip validation (we use reasonable defaults from EDA rather than searching over hyperparameters) and put effort into rigorous testing.

## Our test design: three-iteration backtest

The standard test pattern is "hold out the last X% of data." For a tournament prediction model, a more informative version is:

> **Train the model as if you were predicting an actual past tournament. Compare predictions to known results.**

This is called **backtesting** and it's how rigor is done in time-series ML.

For us, the design is three iterations:

| Backtest | Training data | Test set |
|---|---|---|
| WC 2014 model | 1990-01-01 → 2014-06-11 | 64 matches of WC 2014 |
| WC 2018 model | 1990-01-01 → 2018-06-13 | 64 matches of WC 2018 |
| WC 2022 model | 1990-01-01 → 2022-11-19 | 64 matches of WC 2022 |

Each model is trained from scratch on data through one day before its target tournament's kickoff. It then predicts the 64 matches of that tournament, and we compare predictions to actual results.

Three backtests give us 192 held-out matches across three real tournaments. That's a much stronger signal than a single 64-match snapshot.

**Why kickoff−1 day, not calendar-year boundaries?** Because the months before each WC are filled with friendlies, qualifiers, and Nations League matches that are critical pre-tournament data. Cutting at "end of previous year" would skip 6+ months of important matches.

**Why not predict more WCs?** Diminishing returns. WC 2010, 2006, 2002, etc. are progressively less indicative of modern football (different rules, different player pool, less reliable data). Three recent tournaments is enough to characterize a model.

**Why retrain for each test rather than using one model?** Because that's what you'd do in real life. The 2026 production model will train on all data available up to 2026. The 2022 backtest model trains on all data available up to 2022. We're simulating the *production process*, not just measuring transferability.

## Metric 1: Accuracy

The most intuitive metric: what percent of W/D/L outcomes did we predict correctly?

```python
preds = np.argmax(probs, axis=1)         # 0=H, 1=D, 2=A
correct = (preds == y_true).mean()
print(f"accuracy: {correct:.1%}")
```

For our test sets, expect 45–60%. A random predictor scores ~33% (1/3 for each class). Even strong WC models rarely exceed 60% because international soccer is genuinely high-variance.

**The catch:** accuracy is the *least informative* metric you can compute for a probabilistic model.

Why? Because it throws away your model's uncertainty. A model that says "Brazil 70% / Draw 20% / Away 10%" gets the same credit as a model that says "Brazil 51% / Draw 25% / Away 24%" if both argmax to "Brazil wins" and Brazil wins. They're different models in important ways, but accuracy can't tell them apart.

Accuracy is fine as a sanity check but should never be your primary metric for a probabilistic model.

## Metric 2: Log-loss

This is *the* standard metric for probabilistic prediction. The formula:

```
log_loss = -mean( log(probability_assigned_to_correct_class) )
```

For each test example, look at the probability your model assigned to the *actual* outcome. Take the log. Negate. Average across examples.

Concretely:

- If your model says "Argentina has 70% chance to win" and Argentina wins, the contribution is `-log(0.70) = 0.357`.
- If Argentina loses, the contribution is `-log(0.30) = 1.204`.

Confidently-wrong predictions are punished much more than uncertain-wrong predictions. **This is the right way to evaluate a probability model.**

Properties:

- **Lower is better.** Perfect predictions (probability 1.0 on the correct class) give log-loss 0.
- **Random uniform predictor**: log(3) ≈ 1.099 for 3-class problems.
- **Range of "good" models for WC matches**: 0.95–1.05.

```python
from sklearn.metrics import log_loss
ll = log_loss(y_true, probs, labels=[0, 1, 2])
```

**Why log-loss is the right metric:** it has the desirable property of being *strictly proper*. This means the only way to minimize expected log-loss in the long run is to report your true beliefs about probabilities. You can't game log-loss by over- or under-stating uncertainty. This is why it's used everywhere from forecasting tournaments to evaluating language models.

## Metric 3: Ranked Probability Score (RPS)

Log-loss treats the three classes (H, D, A) as unrelated. But soccer outcomes have a natural ordering: H is "closer to" D than it is to A. A model that predicts "60% home, 30% draw, 10% away" when the answer is "draw" is closer-to-right than a model that predicts "60% home, 10% draw, 30% away" — they're both wrong about the argmax, but the first one has draws as its second-most-likely outcome.

RPS captures this. The formula:

```
RPS = mean over matches of Σ_k (cumulative_pred_k − cumulative_actual_k)²
```

Where:

- For 3 classes ordered [H, D, A], `cumulative_pred = [p_H, p_H + p_D, 1.0]`.
- `cumulative_actual` for actual=H is `[1, 1, 1]`; for D it's `[0, 1, 1]`; for A it's `[0, 0, 1]`.

For each match, compute the squared error between the cumulative vectors. Average across matches. Lower is better.

```python
def rps(y_true: np.ndarray, probs: np.ndarray) -> float:
    cum_pred = np.cumsum(probs, axis=1)
    cum_true = np.zeros_like(probs)
    for i, y in enumerate(y_true):
        cum_true[i, y:] = 1.0
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1)))
```

**Random uniform**: ~0.44 (un-normalized).
**Good models**: 0.38–0.40 (un-normalized).

Some conventions divide by (N−1) where N is the number of classes, giving values in [0, 1] with random ~0.22 and good models ~0.18–0.20. Both conventions exist; report which you're using.

**When to use RPS over log-loss:** when the *closeness* of wrong predictions matters. For betting / portfolio applications, RPS is often more relevant than log-loss because partial credit is the goal. For pure forecasting comparisons, log-loss is more standard.

## Metric 4: Calibration

A well-calibrated model has the property: **when it says "60% chance," it should happen 60% of the time.**

Calibration is *separate from* accuracy and log-loss. A model can be very accurate but poorly calibrated (always says 90%+, gets most right but is over-confident on the wrong ones). A model can be poorly accurate but well-calibrated (says "55%" on a 50/50 game and 55% of the time it happens).

For probabilistic forecasting — sports betting, weather, election forecasting — calibration is often the most important property.

To measure it: bucket predictions by predicted probability and compute the actual rate of occurrence in each bucket.

```python
def calibration_table(probs_for_class: np.ndarray, actual_for_class: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions by predicted probability and compute actual rate per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(probs_for_class, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low": bins[b],
            "bin_high": bins[b + 1],
            "n": int(mask.sum()),
            "predicted_mean": float(probs_for_class[mask].mean()),
            "actual_mean": float(actual_for_class[mask].mean()),
        })
    return pd.DataFrame(rows)
```

A calibration plot is `predicted_mean` on the x-axis vs. `actual_mean` on the y-axis. Perfect calibration is the diagonal line. Above the diagonal: under-confident. Below: over-confident.

For small test sets (192 matches across our three backtests), calibration plots are noisy. They're more useful for production models trained on years of predictions.

## The naive baseline

A model "improving log-loss to 1.05" sounds impressive — until you discover that always predicting `[0.5, 0.25, 0.25]` regardless of the matchup also scores 1.05.

You haven't built a model; you've expensively reproduced a constant. The defense against this is to **always evaluate a naive baseline** alongside your model.

For soccer:

- **Naive 1**: always predict the historical mean class distribution (e.g., `[0.45, 0.25, 0.30]`).
- **Naive 2**: predict based on Elo only — use the Elo expected-score formula directly, with a fixed draw rate.

We use Naive 2:

```python
def naive_elo_probs(features: pd.DataFrame, draw_rate: float = 0.28) -> np.ndarray:
    elo_diff = features["elo_diff"].to_numpy()
    e_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = np.full_like(e_home, draw_rate)
    non_draw = 1.0 - p_draw
    p_home = non_draw * e_home
    p_away = non_draw * (1.0 - e_home)
    return np.column_stack([p_home, p_draw, p_away])
```

This baseline uses only the Elo difference and a fixed assumed draw rate (28% is typical for WC). No Poisson math, no form features, no tournament class, no neutrality.

Compare the model's log-loss to the baseline's. If they're equal (within 0.005), your fancy features and Poisson math added nothing — fix that first before claiming progress.

For our setup, the model beats naive Elo by ~0.005–0.010 log-loss. Small but consistent. This is informative: most of the signal is in Elo; the remaining features add modest lift.

## Confusion matrices

Numerical metrics summarize. Confusion matrices localize.

```
                 predicted
                 H    D    A
actual  H       26    0    3
        D       10    0    3
        A        9    0   13
```

Rows are actual outcomes; columns are model's argmax predictions. Cell (H, A) is "actual was home win, model predicted away win." A perfect classifier has zero off-diagonal.

For our model, you'll typically see:

- High home/home (the model correctly predicts most home wins).
- High away/away (mostly correct for away wins).
- **Zero D column** (the model never argmaxes to draw).

That zero D column is the canonical Poisson failure we discussed in chapter 4. The model still *assigns probability* to draws (often 22–28%), just rarely makes draw the most likely outcome. The Dixon-Coles correction shifts these probabilities up, but rarely past the argmax threshold.

This is why argmax-based accuracy and log-loss can move in opposite directions when you change the model. DC fixes the probabilities (improves log-loss) without flipping argmaxes (accuracy unchanged).

## Reading the results: what to look for

When you run a backtest, walk through this checklist:

**1. Did the model beat naive Elo on log-loss?** If yes (even by 0.005), the model is adding signal. If no, your features are redundant with Elo and you should simplify or add genuinely new information.

**2. Is the log-loss in the "real model" range (0.93–1.06)?** Below: suspect leakage (your model is too good, which is usually because it saw the answer). Above 1.10: model is worse than uniform random; something is structurally broken.

**3. Is performance consistent across the three backtests?** If 2014 is great and 2022 is bad, possibilities:
   - **Random variance** (64 matches is a small sample). Most likely.
   - **Specific tournament was upset-heavy** (WC 2022 was — Saudi over Argentina, Morocco semis, Japan beating Germany/Spain).
   - **Model is overfitting to recent eras** (less likely with a Poisson regression but possible).
   
   To distinguish: look at the per-match predictions and check whether the worst losses correspond to known surprising outcomes.

**4. What does the confusion matrix say?**
   - Zero predicted draws: known Poisson failure. Address with Dixon-Coles (small effect) or accept.
   - Lopsided home/away predictions: probably your home-advantage feature is over- or under-weighted.
   - Most errors on a specific tournament: look for systematic biases (e.g., African teams predicted too low → squad-value undercount for those teams).

**5. Is the model behaving calibrated for the matches it's most confident about?** Filter to predictions with p_max > 0.7 (i.e., the model was 70%+ sure). How often does it win? Should be ~70%. If it's 50%, the model is over-confident.

## Common evaluation pitfalls

**Reporting only the headline metric.** Accuracy alone hides over-confidence. Log-loss alone hides which outcome class you're failing on. Always report multiple metrics + the confusion matrix.

**Comparing against a different metric definition than the literature.** RPS in particular has multiple conventions (with and without normalization by N−1). Always state the formula.

**Using a single random split for evaluation.** Random splits don't work for time-series data — they leak future information into the training set. Always use chronologically-ordered splits.

**Tuning hyperparameters on the test set.** If you adjust hyperparameters because the test set looks bad, your test set is now indirectly part of training. The "true" test set is the *next* one you run after locking hyperparameters. This is why hyperparameter tuning should use a separate validation set.

**Forgetting to seed for reproducibility.** Even with the same features, sklearn's solver has small randomness. If you don't set `random_state`, your reported metrics may differ slightly across runs. Set `random_state=42` everywhere.

**Drawing strong conclusions from small samples.** 64 matches × 3 backtests = 192 examples. That's enough to distinguish good models from bad, but not enough to distinguish "model A is 0.005 better than model B" from noise. A difference needs to be consistent across all three backtests *and* not within the per-tournament noise band.

## What's next

You've measured the model and (hopefully) found it's marginally beating naive baselines. The next move is to add genuinely new information — features that aren't redundant with Elo. The headline candidate is **squad market value**, and the chapter on integrating multi-source data is `06_advanced_features.md`.
