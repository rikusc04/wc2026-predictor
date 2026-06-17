# 08 — Lessons and pitfalls

You finished the model. This chapter is for the next time — generalizing the bugs and dead ends you'll inevitably hit on the next ML project. Some of these are project-specific lessons that surface during this particular build; others are universal to ML work.

## The data work / model work ratio

For a project like this, expect:

- **70–80% data engineering** — finding sources, downloading, cleaning, validating, joining, normalizing.
- **10–15% modeling** — fitting the actual regression, tuning hyperparameters.
- **10–15% evaluation** — backtesting, computing metrics, interpreting results.

This ratio is *normal*. New ML practitioners often think "I'll spend a week on modeling and a day on data prep" and discover the reverse. Plan accordingly.

When you find yourself spending hours on a "small" data integration step (name matching, time-zone alignment, sentinel-value detection), you're not failing — you're doing the actual work.

## Lessons from this project

These are real dead ends, gotchas, and lessons learned during this specific build. They generalize.

### Pre-compiled datasets > scrapers

The original plan for getting historical squad market values was Wikipedia scraping or Transfermarkt scraping. Both are dead ends in practice:

- **Transfermarkt actively blocks** automated traffic. Captchas, 403s, IP blocks. Even if technically possible, it violates ToS.
- **Wikipedia WebFetch tools truncate** long pages. A WC squads page contains 32 teams × 23 players = ~700 entries; summarization-based fetchers see only the first ~150 names.

What worked was finding **pre-compiled GitHub datasets** that already did this work:

- [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) for historical WC rosters.
- [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) for player valuation time series.

**Lesson:** before writing a scraper, search aggressively for pre-compiled data. Someone has probably already done it. Saves days.

### `appearances` doesn't always mean what you think

We initially planned to extract WC rosters from a Transfermarkt-mirrored `appearances.csv` (player-by-match log of who played). 1.88 million rows; surely enough.

Turns out: 0 rows had `competition_id = "FIWC"` (FIFA World Cup). The appearances data covered club matches only, not international ones. The information existed in spirit but not in the column we needed.

**Lesson:** verify that a dataset contains what you think it contains *before* writing the integration code. A 30-second `grep` saves a 2-hour debugging session.

### Sentinel values masquerading as data

The Fjelstul dataset stores mononym players (Brazilian Neymar, Hulk, etc.) as:

```
given_name: "not applicable"
family_name: "Neymar"
```

A name matcher that concatenates these naively constructs `"not applicable Neymar"` and looks for it in Transfermarkt's name index. Result: zero match. Brazil's 2014 squad value comes out 50% lower than reality.

This is a class of bug that's:

- **Invisible at sample inspection.** Brazil rows in the matched table look normal.
- **Visible only at aggregate.** The team totals are wrong.
- **Hard to find without expecting it.** You'd never search for "why is Brazil 2014 not in the top 5?"

**Lesson:** scan for sentinel values explicitly. Common ones: `"N/A"`, `"not applicable"`, `"Unknown"`, empty strings, the literal string `"NaN"`, `-1`, `0` for fields that can't logically be zero. Each is a placeholder you need to handle.

### A sparse feature is barely a feature

If a feature you've added is NaN for 85%+ of training rows, the model is essentially learning from a small subset of the data — even if the imputer makes the pipeline run without errors. This shows up as: feature is "in the model" by every superficial check, but the backtest improvement is tiny and inconsistent.

The diagnostic is to compute the **non-null rate of every numeric feature** before training:

```python
for col in NUMERIC_FEATURES:
    pct = features[col].notna().mean() * 100
    print(f"  {col}: {pct:.1f}% non-null")
```

A feature at 12% non-null can technically inform the model, but you're really training a smaller model on 12% of the data with respect to that feature. The other 88% contribute nothing — they all see the same imputed median.

The fix usually isn't model-side; it's data-side. **Extend the feature's coverage** to more training rows. In our project, the squad-value feature initially covered only WC participants (32 teams × 5 years ≈ 160 entities). Extending to all TM-tracked nations (~105 teams × 5 years ≈ 525 entities) was the difference between "barely usable" and "actually informative."

The general principle: **if a feature is sparse, the model can't help. The fix lives upstream, in how you derive the feature, not in how you consume it.**

**A realistic expectation-setter.** Extending coverage rarely takes you all the way to dense. In our project, going from "WC participants only" to "all TM-tracked nations" doubled the informative-row count (12% → 24%) but didn't approach 100%. The remaining gaps are usually data-coverage gaps the source dataset cannot fill:

- Periods before the source started tracking (no historical snapshots exist).
- Entities the source doesn't track at all (small/regional federations).

These are honest gaps. Don't fabricate values to fill them — the imputer handles the rows. What matters is that the *non-null* rows you have are accurate and representative. Doubling from 6k to 12k informative rows is a real win, even if the headline missing-percentage only dropped from 88% to 76%.

### Don't abstract until you have 3 callers

When you write a piece of logic for the first time, write it inline. The second time, **consider** extracting it but it's often fine to duplicate. The third time, refactor into a shared helper.

This rule comes up constantly in ML projects because:

- **Exploratory code grows in bursts.** You write a feature, decide it's not the right approach, throw it away. If you'd prematurely abstracted, you wasted refactoring effort.
- **The first two callers often have subtly different needs.** Your "shared" abstraction has to accommodate both, growing parameters and edge cases. With three callers, the right shape becomes clear.
- **Bug-hunting on duplicated logic is rarely the slow path.** It feels wrong, but fixing the same bug in 2 places is faster than designing a clean abstraction prematurely.

A concrete example from this project: the function that identifies WC group memberships from match data was written twice — once for historical dead-rubber detection, once for the tournament simulator. Two callers, slight differences in return shape. We left it duplicated through the build because the cost wasn't high.

When a third caller was about to be needed (a third place that wanted group structure), we refactored: extracted `identify_groups()` to a shared module, updated all call sites. The refactor took ~30 minutes and removed ~50 lines of duplication. Doing it earlier would have meant designing an abstraction that fit only two needs perfectly.

**The discipline:** if a piece of logic has 1 or 2 callers and is short, leave it duplicated. Note its existence (`# duplicated in src/foo/bar.py:42`) if you think a third caller is coming. When the third caller arrives, refactor. Not before.

### Iterate over the target set, not the source table

A common bug shape: you write a loop over a source dataset (e.g., `national_teams.csv`) and assume it covers every entity you care about. It doesn't. Some entities you need to process are missing from the source entirely — not just missing a column value, but missing the *row*.

The fix is structural: **separate the "target set" from the "source table" and iterate over the target.** For each target entity, look it up in the source; if missing, run a fallback that doesn't depend on the source's existence.

A concrete example from this project: WC 2026 has 48 qualifiers, but Transfermarkt's `national_teams.csv` only tracks 118 nations total, and 5 of the 48 qualifiers (Cape Verde, Curaçao, DR Congo, Haiti, Ivory Coast) aren't in it. Looping over `national_teams.csv` and filtering to "is this a WC 2026 team" would silently miss those 5. Looping over the 48 qualifiers and looking each up in `national_teams.csv` (with a fallback for misses) catches the gap.

This generalizes: any time your "source" is the union of multiple tables, iterate over the *intersection* you care about, not over any single table.

### Calibration when filling in missing aggregates

When you fall back to a different way of measuring the same quantity, the two ways probably don't agree numerically. A **calibration factor** is the multiplier that converts the fallback measurement to the same scale as the original. Without one, your fallback values are systematically biased — usually too high or too low — and the model sees a feature whose scale shifts depending on whether the value came from the original method or the fallback.

If a dataset has both an aggregate column and the atomic rows it's derived from, and the aggregate is missing for some entities, you can fill it in by computing from the atomic rows. The trap is **calibration**: the atomic-row computation may have different *units* than the original aggregate.

A real example from this project: when `total_market_value` was missing for England/France/Spain in the national-teams data, we fell back to summing top-26 player values filtered by `country_of_citizenship`. The numbers were too high — 1.5× the typical official aggregate — because citizenship includes "eligible but not currently called up" players, while the official aggregate is closer to the actual squad.

The fix is a calibration factor: learn (TM aggregate) / (atomic sum) from entities that have *both* signals, then apply to entities with only the atomic signal.

But there's a second subtlety: **how you compute the calibration factor matters**. Using the median ratio across all entities gives the wrong answer when the typical entity has ratio ≈ 1.0 but the entities you're trying to fill in are the unusual ones with a "calling-up haircut" applied. The median ratio gets dominated by the easy cases (where the gap doesn't exist), while you need the factor for the hard cases (where it does).

**The pattern that works:** restrict the calibration computation to entities most like the ones you're trying to fill in. For us, that meant top-half by citizenship pool size — the big football nations where the haircut applies. Magnitude-weighted ratio (sum / sum) over that subset gives ~0.78. Applied to England/France/Spain, the citizenship-based sums become comparable to other teams' TM aggregates.

**Generalizing:** when you fill in missing aggregates with a fallback computation, always ask: "what's the typical ratio between fallback and aggregate, computed on entities that are most similar to the ones I'm filling in?" If you don't think about this carefully, your fallback estimates will have systematic bias the model can't undo.

### Aggregate columns missing for important rows

The `national_teams.csv` from a public Transfermarkt mirror had a `total_market_value` column populated for ~115 of 118 teams. The three missing rows: England, France, Spain — among the most-tracked national teams in the world.

The reason: probably an upstream scrape failure for those specific pages on the dataset's last refresh. Unknowable, fixable.

The lesson is broader than this specific case. Datasets often provide **both an aggregate column (sum) and the atomic rows it's derived from** (here: total team value and individual player values). The aggregate is convenient but partial; the atomic rows are exhaustive but unstructured.

The robust pattern: **use the aggregate column when available, fall back to summing atomic rows when not.** Code it so the model never sees NaN unless the data genuinely doesn't exist.

For us:

```python
def get_team_value(team_id, national_teams, players):
    aggregate = national_teams.loc[team_id, "total_market_value"]
    if pd.isna(aggregate):
        return sum_top_n_player_values(team_id, players)
    return aggregate
```

This pattern surfaces in many domains:

- Population statistics with missing country aggregates (sum from regional rows).
- Financial reports with missing quarter totals (sum from monthly figures).
- Sports stats with missing season totals (sum from per-match records).

If your dataset has both views, the redundancy is your friend. Use it.

**The diagnostic discipline:** before integrating any aggregate column, do a coverage check — what fraction of your target entities have the aggregate populated? If under 100%, ask why, and have a fallback ready.

### The leakage that almost shipped

When working on the time-based form features, the first version of the code was:

```python
df["form_scored"] = df.groupby("team")["goals_for"].transform(
    lambda s: s.rolling(5).mean()
)
```

This *includes the current match's goal count* in the current match's feature. Training on this would let the model "see" the answer and report fantasy accuracy.

The fix is `.shift(1)` before `.rolling()`. One character, massive impact.

**Lesson:** for every feature you compute, ask: "what information, exactly, was used? Was any of it from the current row or future rows?" If yes, fix it before training.

The general principle is **at-prediction-time discipline**: every feature value must be computable from information available at the moment of prediction. Apply this discipline at every layer (data cutoff, rolling windows, joins).

### Probability matrices that don't sum to 1

When predicting Brazil vs. San Marino, the model produced λ_home ≈ 10.6. With `MAX_GOALS = 10`, the resulting probability matrix had `P(Brazil wins) = 51%`, `P(draw) = 0%`, `P(San Marino wins) = 0%`. Total: 51%.

The missing 49% was probability mass at scorelines like 11-0, 12-0, etc. — beyond our matrix size.

The fix was trivial (`MAX_GOALS = 20`). The interesting part is *how the bug surfaced*: not as an exception, not as obviously-wrong numbers, but as a quiet probability-summing-to-the-wrong-number. The model was internally consistent; the surrounding code had an unstated assumption.

**Lesson:** for any probabilistic output, **always verify** that the probabilities sum to 1.0 (within numerical tolerance). One assertion catches this class of bug.

### The "model never predicts X" failure

When you look at a confusion matrix and the model never argmaxes to a particular class, it's almost always one of three things:

1. **The class is rare in training data** (model didn't see enough examples to learn it).
2. **The model architecture under-represents that outcome** (independent Poisson under-predicts draws).
3. **The feature set doesn't distinguish that case from neighbors** (model has no way to identify "this is a likely draw" matchup).

For us, "model never predicts draws" was case (2) — independent Poisson's known failure mode. The Dixon-Coles correction is the specific remedy.

**Lesson:** always look at the confusion matrix, not just headline metrics. A class that's never predicted means the model is structurally missing something.

### Marginal improvement is still real

After three backtests, our model beats naive Elo by 0.005–0.010 log-loss. That's a small number. Was the work worth it?

In a well-trodden domain (academic WC prediction), the published-model range is 0.95–1.00 log-loss. Naive Elo is around 1.04. Our model is around 1.00 — *real, measurable, in the published-model range*.

The reason it *feels* small is that the dynamic range of log-loss in this regime is also small. The difference between a random predictor (1.099) and a perfect predictor (0.0) is one digit, but the practical "useful model" zone spans only 0.95–1.05. Improvements of 0.005 in that zone are meaningful.

**Lesson:** calibrate your expectations to the domain. "Tiny" improvements may be all that's available — and they may be all you need.

## Universal ML pitfalls

Beyond this project, these are traps that ML practitioners hit repeatedly across problems.

### Confusing in-sample with out-of-sample

Your model trained on the data. You predicted the same data. The model looks amazing.

This is the easiest mistake in ML and surprisingly persistent. The fix is unconditional: **always evaluate on data the model has never seen**. For time-series problems, this means a strict chronological split with the test set in the future relative to training.

### Tuning hyperparameters on the test set

You evaluate on the test set, log-loss is 1.10. You bump `alpha` from 0.1 to 0.5; log-loss drops to 1.05. You report 1.05.

What you actually did: you used the test set as a validation set, which means it's no longer a test set. Your reported metric is optimistic — the *true* test would be on data not yet seen.

The fix is to **lock hyperparameters before touching the test set**. Use a separate validation slice for tuning, or set hyperparameters from prior knowledge / defaults and accept whatever the test set says.

### Random shuffling for time-series splits

You import `train_test_split`, split 80/20, train, evaluate. Log-loss is great.

For time-series data, this is leakage. Random splits put rows from the *same period* into both training and test. The model can effectively "learn" patterns by seeing some matches from a tournament in training and others from the same tournament in test.

The fix is **chronological splits only** for time-series. Either use a cutoff date or do walk-forward validation (predict each "next month" using only past data).

### Confidence intervals you don't have

The model predicts "Brazil wins with 70% probability." Where does the 70% come from? It's a single number from a deterministic computation. There's no uncertainty around it.

But in reality, the 70% itself has uncertainty — if the training data had been slightly different, the model might output 65% or 75%. This is called **epistemic uncertainty**, and most simple models (including ours) ignore it.

For most use cases, point estimates are fine. For high-stakes decisions (betting large sums, deploying medical models), you'd want Bayesian methods or bootstrap aggregation to get confidence intervals on the predictions themselves.

### Comparing models with different evaluation sets

You read a paper claiming log-loss 0.85 on WC matches. You compare to your 1.00 and feel inadequate.

Then you read the paper's methodology: they evaluated on a single recent tournament, in-sample, with hand-picked matches. Your number is on three full tournaments, out-of-sample, with every match included.

These aren't comparable. Always check the exact evaluation set before comparing metrics.

### Premature deployment

Your model works on the backtest. You run a full pipeline, ship it, and the tournament starts. After 16 matches, your accuracy is 50% — well below your backtest's 56%.

Is the model broken? Probably not. 16 matches is a small sample; the variance band of accuracy is 35–75% (95% confidence interval). The headline "50%" is just unlucky.

**The variance of small-sample metrics is huge.** Don't conclude anything from less than ~50 evaluations.

### Dataset drift after deployment

The model trained on data through 2026. The tournament starts. New patterns emerge that weren't in training (new tactical trends, COVID-style stadium changes, etc.).

This is **dataset drift**, and it's a constant problem in deployed ML. The model trained on past data, but the present may not match the past.

Defenses:

- **Monitor performance metrics** in production.
- **Retrain frequently** when fresh data arrives.
- **Build in calibration checks** (do my probabilities still match observed frequencies?).

For a one-shot tournament prediction, this is less of an issue. For ongoing production deployment, it's central.

## Things to do differently next time

Reflecting on this project, here's what'd I'd do differently from the start:

**1. Allocate more upfront time for data sourcing.** The 70/15/15 rule means days of data work. Estimate higher than you think.

**2. Find pre-compiled datasets first.** GitHub's data community has done a lot of this work already. Search before building.

**3. Build the leakage-prevention layer before any features.** Make the cutoff date a hard constraint baked into the data loader. Make `.shift(1)` patterns into reusable utilities. Make it harder to leak than not to.

**4. Verify schema and content before integration.** A 30-second inspection of column names and row counts prevents hours of debugging mismatched joins.

**5. Establish baselines early.** Build the naive predictor in the first hour. Every subsequent measurement is "how much better than naive." Without a baseline, you can't tell if you're improving.

**6. Prefer interpretable models for sports.** Soccer / basketball / etc. are explainable domains; statistical models (Poisson, hierarchical Bayesian) outperform black-box deep learning at small data scales (< 100k samples). Reach for sklearn before TensorFlow.

**7. Time your iterations.** Each backtest iteration takes ~30s to train + predict. Make sure your re-test loop is fast; if it takes 10 minutes, you'll iterate less and miss bugs.

**8. Document decisions, not just code.** Why is the cutoff 1990? Why is `alpha=0.1`? Why two models, not one? These choices accumulate; without a written record, future-you (or future-someone) will re-litigate them.

## Reading list

If you want to go deeper than this guide:

**On Poisson modeling of soccer:**

- Maher, M.J. (1982) "Modelling association football scores" — the foundational paper.
- Dixon & Coles (1997) "Modelling association football scores and inefficiencies in the football betting market" — introduces the τ correction.
- Karlis & Ntzoufras (2003) "Analysis of sports data by using bivariate Poisson models" — the modern joint-Poisson formulation.

**On ML evaluation:**

- Gneiting & Raftery (2007) "Strictly Proper Scoring Rules, Prediction, and Estimation" — why log-loss matters, mathematically.
- Niculescu-Mizil & Caruana (2005) "Predicting good probabilities with supervised learning" — calibration measurement and correction.

**On general ML practice:**

- The scikit-learn documentation, especially the `Model evaluation` and `Cross-validation` chapters.
- "Designing Data-Intensive Applications" by Martin Kleppmann — not ML-specific but the gold standard for thinking about data pipelines.

## Final thought

Sports prediction models are humbling. The data is noisy, the variance is high, and even the best models miss obvious results regularly. If your model says "France 60% to win the final" and France loses, that's not necessarily a model failure. It's a 40% event happening, and 40% events happen 40% of the time.

The goal isn't perfect prediction. It's calibrated prediction — outputs you can trust to mean what they say. A 60% win probability that actually corresponds to 60% wins over the long run is a useful model. A 95% win probability that corresponds to 60% wins is a broken one. Build models that earn their numbers.
