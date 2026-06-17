# 02 — Exploratory data analysis

Once you have data loaded, **look at it before doing anything else.** This is exploratory data analysis (EDA), and it's the most underrated step in ML. Every modeling choice you'll make later depends on what the data actually looks like — skipping EDA is the #1 way to ship a quietly broken model.

This chapter introduces a 7-question framework for EDA on any new dataset, walks through each question with code, and explains what the answers tell you about modeling decisions.

## Why EDA matters

There's a temptation to skip EDA because it "doesn't make progress" — you don't end the session with a trained model. But EDA prevents three specific failure modes that destroy ML projects:

1. **Modeling assumptions that don't hold.** Every model has assumptions (Poisson assumes counts are independent, neural networks assume i.i.d. data, etc.). If your data violates them, the model is wrong in a way you won't detect without specifically checking.

2. **Hidden data quality issues.** Missing values, duplicate rows, unit mismatches, encoding bugs. These corrupt models in subtle ways: the model "trains" fine but produces nonsense.

3. **Wasted time on the wrong features.** You spend a week building a feature that turns out to be useless because, e.g., the underlying signal is dominated by noise. EDA reveals signal-to-noise upfront.

EDA is best done in a Jupyter notebook. Notebooks are ideal for it: you can mix prose and code, see plots inline, and iterate quickly. **Modeling code goes in `src/`; exploration goes in `notebooks/`.** That separation matters because notebooks are hard to test, hard to version-control well, and hard to use as building blocks.

For this project, the EDA notebook is `notebooks/01_data_exploration.ipynb`.

## The 7 EDA questions

Regardless of the dataset, ask these questions in order:

1. **What's the date coverage and density?** (For time-series data.)
2. **Are there missing values? Where?**
3. **What's the breakdown by relevant categorical groupings?**
4. **What's the distribution of the target variable?**
5. **What asymmetries exist in the data?** (Home advantage, group effects, etc.)
6. **Is there temporal drift?** (Has the data changed over time?)
7. **Are identifiers consistent?** (Team names, player IDs, etc.)

Each question has a code pattern and a "what to do with the answer" follow-up. Let's walk through each.

## Question 1: Date coverage and density

**Goal:** know exactly when your data starts, when it ends, and whether the density is uniform.

```python
import pandas as pd
import matplotlib.pyplot as plt

df["year"] = pd.to_datetime(df["date"]).dt.year
matches_per_year = df.groupby("year").size()

fig, ax = plt.subplots(figsize=(11, 4))
matches_per_year.plot(ax=ax)
ax.set_title("International matches per year")
ax.set_ylabel("matches")
ax.set_xlabel("year")
plt.show()

print(f"earliest year: {matches_per_year.index.min()}")
print(f"latest year:   {matches_per_year.index.max()}")
print(f"median matches/year (pre-1950):  {int(matches_per_year[matches_per_year.index < 1950].median())}")
print(f"median matches/year (post-2000): {int(matches_per_year[matches_per_year.index >= 2000].median())}")
```

**Expected output pattern:** a curve that's sparse early, then ramps up rapidly. For international soccer, you'll see something like:

- Pre-1950: ~24 matches/year (sparse — different rules, fewer FIFA members, no qualifiers)
- Post-2000: ~970 matches/year (dense — modern professional era)

**What to do with this:**

If pre-1950 data is much sparser and structurally different (different rules, different competition formats), it may *hurt* model quality if included as if it were comparable to modern matches. The model's view of "what a normal match looks like" gets pulled toward an outdated regime.

The practical implication: consider a **start-year cutoff** for training. We end up using 1990 onward in this project. The rationale comes from Question 6 (temporal drift in scoring rates), but Question 1 is where you first notice the issue.

## Question 2: Missing values

**Goal:** confirm your training data is clean.

```python
print("nulls per column:")
print(df.isna().sum())
print()
print(f"score dtype: home={df['home_score'].dtype}, away={df['away_score'].dtype}")
print(f"min home_score: {df['home_score'].min()}, max: {df['home_score'].max()}")
print(f"min away_score: {df['away_score'].min()}, max: {df['away_score'].max()}")
```

**Expected output:** zero nulls in the training slice. If there *are* nulls, you need a strategy:

- **Drop** the affected rows if they're rare (<5%) and the missingness isn't informative.
- **Impute** with a sensible default (median for numeric, mode for categorical) if they're more common.
- **Carry forward** for time-series data with sparse measurements (e.g., quarterly squad values).

**Subtle thing to notice:** if scores are `float` instead of `int`, that's a clue that some rows had `NaN` originally and pandas auto-promoted the type. Even if you've now filtered those out, double-check.

In our case, scores being `float` reflects that the dataset includes upcoming, unplayed matches (WC 2026 fixtures with `NaN` scores). After our cutoff filter, the training slice has zero nulls — but the `float` type is a leftover from before filtering. Harmless, but worth understanding.

**Extra check: are scores plausible?** Max home_score of 31 sounds wrong until you remember Australia 31–0 American Samoa (2001). It's real, it's just an outlier. Worth noting because that single result will pull averages around at the team-pair level.

## Question 3: Categorical breakdown

**Goal:** understand the composition of your data along relevant axes.

For international soccer, the natural axis is **tournament type**:

```python
tournament_counts = df["tournament"].value_counts()
print(f"distinct tournament names: {len(tournament_counts)}")
print(tournament_counts.head(15))
```

You'll find ~200 distinct tournament names — too many to model individually. The fix is to **bucket** them into a manageable number of classes:

```python
def classify_tournament(name: str) -> str:
    """Bucket a tournament name into a coarse class.

    Order matters: "qualif" is checked before "world cup" because
    "FIFA World Cup qualification" contains both substrings.
    """
    name = name.lower()
    if "friendly" in name:
        return "friendly"
    if "qualif" in name:
        return "qualifier"
    if "world cup" in name or "fifa" in name:
        return "world_cup"
    if any(k in name for k in (
        "euro", "copa", "africa", "asian", "concacaf",
        "uefa", "afcon", "gold cup", "nations league",
    )):
        return "continental"
    return "other"

df["tournament_class"] = df["tournament"].apply(classify_tournament)
df["tournament_class"].value_counts()
```

**Watch out for the order of substring checks.** A naive ordering would bucket "FIFA World Cup qualification" as `world_cup` because the check matches "world cup" first. The fix is to check the more-specific pattern (`qualif`) before the less-specific one. This is the kind of subtle bug that's invisible in summary stats but corrupts your features.

**Why bucket?** Different tournament types may have different scoring patterns (friendlies tend to be lighter on stakes, knockout matches tend to be defensive). Bucketing lets the model condition on this.

## Question 4: Target distribution

**Goal:** understand what you're predicting at the most basic level.

For us, the target is goals per team per match. The question is whether the **Poisson distribution** is a reasonable model for it.

**Quick ML refresher on Poisson:** the Poisson distribution describes the count of events that happen at a roughly constant rate (emails per hour, typos per page, goals per match). It has one parameter, λ ("lambda"), which is *both the mean and the variance*. This is its defining property — and the property we want to verify in our data.

```python
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

recent = df[pd.to_datetime(df["date"]) >= "2000-01-01"].copy()
goals = pd.concat([recent["home_score"], recent["away_score"]])

mean_g = goals.mean()
var_g = goals.var()
print(f"goals per team-match (2000+): mean = {mean_g:.3f}, variance = {var_g:.3f}")
print(f"variance / mean ratio: {var_g / mean_g:.3f}  (Poisson would be ~1.0)")

# Overlay theoretical Poisson with the same mean
xs = np.arange(0, 12)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(xs - 0.2, np.bincount(goals.dropna().astype(int), minlength=12)[:12] / len(goals.dropna()),
       width=0.4, label="empirical")
ax.bar(xs + 0.2, poisson.pmf(xs, mean_g), width=0.4, label=f"Poisson(λ={mean_g:.2f})")
ax.set_xlabel("goals"); ax.set_ylabel("probability")
ax.legend(); plt.show()
```

**Expected output:** roughly 1.4 mean, ~2.4 variance, ratio ~1.7.

The ratio being 1.7 (above 1.0) is called **overdispersion** — the data is more spread out than a pure Poisson would predict. This is a warning, not a death sentence. Two responses:

1. **Stay with Poisson but use the Dixon-Coles correction.** DC specifically addresses Poisson's tendency to mis-predict low-scoring draws, which is most of the overdispersion in soccer.
2. **Switch to negative-binomial regression.** Allows mean ≠ variance. More flexibility but more complexity.

In practice, the overdispersion is partly an artifact of pooling matches with very different λ values. Once we condition on team strength (a 5-1 from Brazil vs. Macao isn't a "random surprise"), the residual variance shrinks. So we start with Poisson + Dixon-Coles and verify it works.

## Question 5: Asymmetries

**Goal:** find structural differences in the data that need to be modeled.

For soccer, the big asymmetry is **home advantage**. Playing at home boosts expected goal count.

```python
recent["margin"] = recent["home_score"] - recent["away_score"]
home_games = recent[~recent["neutral"]]
neutral_games = recent[recent["neutral"]]

print(f"home games (2000+): {len(home_games):,}")
print(f"  avg margin:    {home_games['margin'].mean():+.3f}")
print(f"  home win rate: {(home_games['margin'] > 0).mean():.1%}")
print()
print(f"neutral games (2000+): {len(neutral_games):,}")
print(f"  avg margin:    {neutral_games['margin'].mean():+.3f}")
print(f"  home win rate: {(neutral_games['margin'] > 0).mean():.1%}")
```

**Expected output:**

```
home games: margin ≈ +0.67 goals, win rate ≈ 51%
neutral games: margin ≈ +0.17 goals, win rate ≈ 41%
```

The differential — about **0.5 goals per match** — is the empirical home advantage. This sets your prior for what the model should "rediscover" when fit.

**What the +0.17 on neutral games means:** even on neutral venues, the "home team" (which is sometimes a designation, e.g., higher-seeded team in a tournament) tends to win slightly more often. This isn't a bug — it's a real measurement detail. Tournament organizers often nominally designate a "home" team in knockouts. Not all "neutral" games are perfectly neutral.

**Implication for modeling:** include `is_neutral` as a feature so the home boost only applies when `neutral=False`.

## Question 6: Temporal drift

**Goal:** check whether the data-generating process has changed over time.

```python
df["decade"] = (df["year"] // 10) * 10
goals_by_decade = df.groupby("decade").apply(
    lambda g: (g["home_score"].mean() + g["away_score"].mean()),
    include_groups=False,
)
print(goals_by_decade.tail(10).round(2))
```

**Expected output for our data:**

```
decade
1930    4.32
1940    4.34
1950    4.00
1960    3.48
1970    2.97
1980    2.53     ← lowest
1990    2.78
2000    2.80
2010    2.74
2020    2.71
```

Total goals per match dropped from ~4.3 (1930s) to ~2.7 (modern). Stable from 1990 onward.

**This is the rationale for our 1990 cutoff** in training. Pre-1990 data was generated under a different "scoring regime" — different tactics, different rules, smaller player pool. Including it would pull the model's estimate of "normal scoring rate" toward a number that doesn't match modern football.

**Two ways to handle temporal drift:**

1. **Hard cutoff** — train only on matches from 1990 onward. Simple, what we use.
2. **Recency weighting** — every match contributes, but recent matches count more (e.g., exponential decay). More flexible but adds a hyperparameter.

We start with the hard cutoff because it's simpler. If we wanted to squeeze out more accuracy later, recency weighting is the natural next step.

## Question 7: Identifier consistency

**Goal:** make sure entity names match across time and tables.

For international soccer, this means **country names**. Did "West Germany" become "Germany" in your data? Are "Czech Republic" and "Czechoslovakia" listed separately even though one inherits the other? Is "Turkey" sometimes "Türkiye"?

A naive check:

```python
former = load_former_names()  # historical renames from the data source
teams_in_results = set(df["home_team"]).union(df["away_team"])
former_names_listed = set(former["former"])

# Names that look like "former" entries in former_names.csv
suspicious = sorted(t for t in teams_in_results if t in former_names_listed)
print(f"teams matching a 'former' entry: {len(suspicious)}")
```

But this check often returns empty because **the dataset maintainer may have pre-canonicalized renames**. (E.g., our dataset uses "Germany" continuously since 1909, having pre-merged West Germany.) Empty result doesn't mean the problem doesn't exist — it means the maintainer has handled it.

A better check is to look at **defunct teams** (teams that stop appearing):

```python
last_match = pd.concat([
    df[["home_team", "date"]].rename(columns={"home_team": "team"}),
    df[["away_team", "date"]].rename(columns={"away_team": "team"}),
]).groupby("team")["date"].max()
defunct = last_match[last_match < pd.Timestamp("2000-01-01").date()].sort_values()
print(f"  found {len(defunct)} defunct teams. Notable ones:")
for name in ["German DR", "Czechoslovakia", "Yugoslavia", "Soviet Union"]:
    if name in defunct.index:
        print(f"  - {name}: last played {defunct[name]}")
```

You'll typically find:

- **East Germany** (German DR): a real separate team that ceased to exist in 1990. Don't rename — it really was different from West Germany.
- **Czechoslovakia → Czech Republic + Slovakia** in 1993: a split. No clean successor. Leave the historical name as-is; both successors start fresh.
- **Yugoslavia → Serbia + 4 others** in early 1990s: same as Czechoslovakia.

**The general principle:** renames (one-to-one) need a mapping; splits and merges (one-to-many or many-to-one) usually leave the historical entity as defunct. Don't force a split successor to "inherit" the parent's ratings — they really did start as new teams.

For our project, no canonicalization is needed because the maintainer already did the obvious renames and the splits naturally die off. But you have to *check* this; don't assume.

## Synthesizing: what EDA tells you about modeling

A good EDA produces a **decision list** for your modeling work. From the questions above, ours would be:

1. **Start year for training:** 1990 (from Questions 1 and 6).
2. **Tournament class as a feature:** yes, with the 5-bucket scheme above (from Question 3).
3. **Distribution:** Poisson + Dixon-Coles correction; revisit if needed (from Question 4).
4. **Home advantage feature:** include `is_neutral` (from Question 5).
5. **Country renaming:** none needed; defunct teams die naturally (from Question 7).
6. **Missing value handling:** drop rows with null scores; otherwise no nulls in training slice (from Question 2).

Every one of these is documented so that downstream modeling code can refer back to "why is this here?" without you having to remember.

## Pitfalls in EDA

A few traps:

**Confirmation bias.** It's easy to find what you expect and stop looking. The Poisson histogram looks "close enough" so you skip the variance check. The team names *seem* consistent so you skip the renames check. EDA is a place to be especially skeptical.

**Plot quality matters.** Mismatched bar widths, log-vs-linear scale, truncated axes — these can hide important patterns. Use consistent, plain styling (no truncation, linear unless log is justified).

**Don't generalize from EDA snapshots.** Patterns visible in a single year might not hold across years. If you're slicing by year for any reason, also do the unsliced view.

**Distinguish data issues from model issues.** A pre-1950 outlier in scoring rate is a *data* fact (genuinely different era), not a *model* bug. Don't try to "fix" data that's faithfully reporting reality.

## What's next

EDA produces decisions; the next step is encoding those decisions as features. That's `03_features.md` — Elo, recent form, and the leakage-avoidance discipline that's the spine of any time-series ML project.
