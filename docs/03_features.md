# 03 — Feature engineering

Once EDA has told you what's in the data, feature engineering is where you turn it into something a model can learn from. For our project this means:

- **Elo ratings**, capturing team strength
- **Recent form**, capturing short-term scoring/conceding rates
- **Categorical encodings** for tournament class and home/neutral
- **Time-based features** like days since last match

The hardest part of feature engineering isn't writing the code. It's avoiding **data leakage** at the row level — every feature value for match M must use only information available *strictly before* match M kicks off. This chapter introduces the leakage-avoidance discipline (`shift(1)`, time-aware joins) that's the spine of any time-series ML project.

## What's a feature?

A feature is a number (or category) describing a training example that the model uses to predict the target. For us, the target is goal counts; the features are everything else.

In supervised learning, the training loop is:

```
for (features, target) in training_data:
    predicted = model(features)
    loss = compare(predicted, target)
    update_model_to_reduce_loss
```

The model only sees what's in `features`. So the quality of your features sets the ceiling on what the model can learn. If "home team's recent attacking form" matters and you don't include a feature for it, the model can't learn it — no matter how sophisticated the model is.

Good features are:

- **Predictive** — correlated with the target.
- **Available at prediction time** — you can compute them for new matches.
- **Not leaking the answer** — they don't encode future information.

The third one is where projects die. Let's start there.

## The leakage-avoidance discipline

A feature leaks if it uses any information not available at the moment of prediction. The classic leak in time-series ML is the **rolling-stat leak**:

You want a "team's average goals in last 5 matches" feature. Naively, pandas makes this easy:

```python
# WRONG — leaks the current match's own outcome into the feature
df["form_scored"] = (
    df.groupby("team")["goals_for"]
      .transform(lambda s: s.rolling(5).mean())
)
```

`s.rolling(5).mean()` at row N includes rows N-4 through N — *including the current match*. So the feature value for match M is partly determined by match M's outcome. In training, this means the model gets the answer baked into its inputs. The model "learns" with apparent ease but fails completely at prediction time, when the current match's outcome isn't known.

**The fix is `shift(1)`** — offset the rolling window by one row, so the rolling stat at row N uses rows N-5 through N-1, never row N:

```python
# CORRECT — the rolling stat at row N uses only rows < N
df["form_scored"] = (
    df.groupby("team")["goals_for"]
      .transform(lambda s: s.shift(1).rolling(5).mean())
)
```

This pattern is the workhorse of time-series feature engineering. Whenever you compute a feature from past observations of the same entity, `shift(1)` before the rolling.

A second leakage pattern is **time-based joins**. If you have a "team strength" rating that updates daily, and you join it to a match table by team name only:

```python
# WRONG — joins to whatever team strength is currently in the rating table,
# which may have been updated using the very match we're trying to predict
features = matches.merge(team_strength, on="team")
```

The fix is to also join on date, with explicit "use the most recent rating before this match" logic:

```python
# CORRECT — use the rating as of (or before) the match date
features = pd.merge_asof(
    matches.sort_values("date"),
    team_strength.sort_values("date"),
    on="date", by="team", direction="backward",
)
```

`pd.merge_asof` with `direction="backward"` finds the closest match where the right table's date is ≤ the left table's date. This is the pattern for any rating, valuation, or rolling metric.

## Elo ratings — the spine of the feature set

Elo is a single number that represents a team's strength. Originally invented for chess (Arpad Elo, 1960s), now used in nearly every competitive ranking system: chess, tennis, video games, NBA, NFL, FIFA's own "World Football Elo Ratings" at eloratings.net.

**Conceptually:**

- Every team has a rating (typically 800–2200; "average" is ~1500).
- After every match, both teams' ratings update based on the result.
- The difference between two ratings encodes the probability one beats the other.

**Why we use Elo and not (for example) FIFA rankings:**

- FIFA rankings have known issues (undervalue recent form, overvalue match volume).
- Elo is computed from match results, so it adapts continuously and rigorously.
- Elo is interpretable — a 100-point rating gap corresponds to a specific expected score.

**Why we compute Elo as a feature rather than using it directly as the model:**

- Elo is one-dimensional. Two teams with the same Elo can play very different football (defensive Italy vs. attacking Netherlands). To predict *score*, you need richer features.
- The Poisson regression learns the *function* mapping (home_elo, away_elo, other features) → expected goals, which can capture nuance Elo alone can't.

### The Elo update math

Three pieces:

**1. Expected score from rating difference.**

If team A has rating R_A and team B has R_B, the expected score for A is:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

This is a sigmoid-like function with values in [0, 1]. Some intuition:

- Equal ratings: E_A = 0.5 (50/50 game).
- A is 400 points higher: E_A ≈ 0.91 (A wins ~91% of the time in chess; for soccer, "expected score" is interpreted as a soft predicted result where win=1, draw=0.5, loss=0).
- A is 800 points higher: E_A ≈ 0.99.

The 400 scaling factor is conventional — it makes a 400-point gap correspond to a 10:1 ratio in expected score.

**2. Actual score for the match.**

```
actual_score(home_goals, away_goals):
    if home > away:  return (1.0, 0.0)   # home win
    if home < away:  return (0.0, 1.0)   # away win
    return (0.5, 0.5)                    # draw
```

**3. Rating update.**

```
R_A_new = R_A + K × (actual_A − expected_A)
```

K is a hyperparameter — how violently ratings update per match. Chess uses K=20-32; soccer Elo typically uses K=30 with goal-margin scaling.

For soccer, we multiply K by a **goal-margin multiplier** (a 5-0 win should move ratings more than 1-0) and a **tournament-importance multiplier** (friendlies should move ratings less than World Cup matches):

```python
TOURNAMENT_WEIGHT = {
    "friendly":    0.5,
    "qualifier":   1.0,
    "continental": 1.25,
    "world_cup":   2.0,
    "other":       0.75,
}

def margin_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:  return 1.0
    if g == 2:  return 1.5
    if g == 3:  return 1.75
    return (11 + g) / 8.0   # eloratings.net formula

k = K_BASE * margin_multiplier(hg - ag) * TOURNAMENT_WEIGHT.get(t_class, 1.0)
```

### Computing Elo over all history

Critical implementation detail: **iterate through matches chronologically** and update ratings in place. For each match, record the "pre-match" rating of both teams — that's what goes into our feature table:

```python
def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = matches.sort_values("date").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    pre_home, pre_away = [], []
    post_home, post_away = [], []

    for i, row in enumerate(df.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        hg, ag = row.home_score, row.away_score
        t_class = classify_tournament(row.tournament)

        r_h = ratings.get(home, INITIAL_RATING)
        r_a = ratings.get(away, INITIAL_RATING)
        pre_home.append(r_h)
        pre_away.append(r_a)

        e_h = expected_score(r_h, r_a)
        s_h, s_a = actual_score(hg, ag)

        k = K_BASE * margin_multiplier(int(hg - ag)) * TOURNAMENT_WEIGHT.get(t_class, 1.0)

        new_r_h = r_h + k * (s_h - e_h)
        new_r_a = r_a + k * (s_a - (1 - e_h))

        ratings[home] = new_r_h
        ratings[away] = new_r_a
        post_home.append(new_r_h); post_away.append(new_r_a)

    df["home_elo_pre"] = pre_home
    df["away_elo_pre"] = pre_away
    df["home_elo_post"] = post_home
    df["away_elo_post"] = post_away
    return df, ratings
```

The output has four new columns per match:

- `home_elo_pre`, `away_elo_pre` — ratings just before kickoff. **These are what feed into the model as features.**
- `home_elo_post`, `away_elo_post` — ratings after the match. Used for the next iteration's lookup, sometimes useful for visualization.

**Why "pre" not "post":** the pre-match ratings represent information available *at prediction time*. Post-match ratings include the match's own result — using them as features would be leakage.

For the full implementation, see `src/features/elo.py`.

### Elo warm-up

There's a subtle question about *when* to start computing Elo.

If you start in 1990 (your training cutoff), every team begins at the default rating (e.g., 1500) and takes years of matches to converge to a realistic value. Brazil's "true" rating of ~2050 isn't reached until ~1995. So your 1990-1995 features are biased.

The fix: **compute Elo over the full match history (1872+)**, even though you only *train the prediction model* on 1990+ matches. The pre-1990 matches don't go into training, but they do update Elo ratings so that by 1990, every team's rating reflects decades of history.

This is sometimes called the "warm-up" period. It's a small detail with measurable impact on feature quality.

## Recent form

Elo is a single number per team. To capture attacking vs. defensive style separately, add:

- **Average goals scored in the last 10 matches** (attacking form)
- **Average goals conceded in the last 10 matches** (defensive form)

Implementation uses the `shift(1).rolling().mean()` pattern from the start of this chapter, applied per team:

```python
def _team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, date) — one row per match contributes two rows."""
    home = matches[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]

    away = matches[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]

    return pd.concat([home, away], ignore_index=True)


def _recent_form_table(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    tv = _team_perspective(matches).sort_values(["team", "date"]).reset_index(drop=True)
    tv["date"] = pd.to_datetime(tv["date"])

    grouped = tv.groupby("team", group_keys=False)
    tv["form_scored"] = grouped["goals_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    tv["form_conceded"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    tv["days_since_last"] = grouped["date"].transform(
        lambda s: (s - s.shift(1)).dt.days
    )

    return tv[["team", "date", "form_scored", "form_conceded", "days_since_last"]]
```

Then merge back to the main match table, joining home-team form by `(home_team, date)` and away-team form by `(away_team, date)`.

Note the `min_periods=1` — for a team's very first matches, the rolling stat has fewer than 10 observations; we still compute it from whatever's available rather than returning NaN.

### A subtle duplicate-row gotcha

The merge here has a trap. If the same team plays two matches on the same date (rare but exists in our data — older tournaments with quick turnaround), the form table has *two* rows for that `(team, date)`. The merge to the matches table then *duplicates* the match row.

The fix: deduplicate after computing the rolling stats:

```python
form = tv[["team", "date", "form_scored", "form_conceded", "days_since_last"]]
form = form.drop_duplicates(subset=["team", "date"], keep="first").reset_index(drop=True)
```

We keep "first" because the rolling stat at the first row uses only data from before that date, which is the correct "state going into today" semantics.

This is the kind of bug that's invisible from sample inspection (the data *looks* fine) but visible from row counts. **Always check that your output row count matches your expected row count.**

## Categorical and boolean encodings

Two more features:

- `tournament_class` (categorical: friendly / qualifier / world_cup / continental / other) — already computed in EDA.
- `is_neutral` (boolean) — already in the source data.

For sklearn pipelines, encode these properly:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "home_elo_pre", "away_elo_pre",
    "home_form_scored", "home_form_conceded",
    "away_form_scored", "away_form_conceded",
    "home_days_since_last", "away_days_since_last",
]
BOOL_FEATURES = ["neutral"]
CATEGORICAL_FEATURES = ["tournament_class"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("bool", "passthrough", BOOL_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])
```

A few choices to justify:

- **`StandardScaler` for numerics**: scales each feature to mean=0, std=1. This isn't strictly required for Poisson regression (the algorithm doesn't care about units) but helps the optimizer converge faster and makes regularization act fairly across features.
- **`handle_unknown="ignore"`** on the one-hot encoder: at prediction time, you might see a tournament name not in the training set; this tells the encoder to silently drop it rather than crash.
- **`sparse_output=False`**: one-hot encoders default to sparse matrices; we explicitly request dense, which simpler downstream code expects.

The `preprocessor` is wrapped in a `Pipeline` along with the regressor — see `04_modeling.md`.

## Putting it together: the feature pipeline

Each feature module produces a self-contained artifact:

1. **Loader** (`src/data/loader.py`) → `data/raw/results.csv` (raw match data).
2. **Elo** (`src/features/elo.py`) → `data/processed/matches_with_elo.csv` (annotated with Elo pre/post columns).
3. **Build features** (`src/features/build.py`) → `data/processed/features.csv` (with `elo_diff`, `tournament_class`, recent form, days_since_last).
4. **Training** consumes `features.csv` (see `04_modeling.md`).

This pipeline is run by executing each module:

```bash
python -m src.data.loader            # downloads raw data
python -m src.features.elo           # computes Elo
python -m src.features.build         # builds full features
```

Each module is idempotent (cached, safe to re-run) and self-contained (you can re-run later modules without redoing earlier ones, as long as their outputs exist).

## A leakage audit checklist

Before training, walk through every feature and ask: **what data, exactly, was used to compute this value?**

- Elo pre-match: matches strictly before this match's date — ✓ no leak.
- Recent form: `shift(1).rolling(10).mean()` — ✓ no leak.
- Days since last match: `shift(1)` difference — ✓ no leak.
- Tournament class: a function of the tournament name, which is known at scheduling time — ✓ no leak.
- `is_neutral`: known at scheduling time — ✓ no leak.

If any feature draws from the future, your model's evaluation is meaningless. Spend the 30 minutes on the audit; it's the cheapest insurance you can buy.

## Common feature-engineering pitfalls

**Forgetting to sort before computing rolling stats.** `rolling` operates over rows as they appear; if your data isn't sorted by date, you'll get garbage. Always `.sort_values("date")` first.

**Confusing `.transform` and `.apply` in groupby.** `transform` returns an output aligned with the input index (good for assigning back to a column). `apply` returns whatever the function returns (often unaligned). For per-group rolling stats, use `transform`.

**Joining on team name without considering team-name changes over time.** If your data uses "Germany" throughout but a contemporary source uses "West Germany", joining breaks silently. Always check the merge result counts.

**Treating boolean columns as floats.** Pandas may load booleans as `True`/`False`, as `1`/`0`, as `"true"`/`"false"`. Cast explicitly. The one-hot encoder + `passthrough` will silently mishandle string booleans.

**Generating features for matches whose target is unknown.** The training data should exclude any row where the target (`home_score`, `away_score`) is NaN. If your data includes scheduled-but-unplayed matches, drop them from the training pool.

## What's next

Features are built. The next step is fitting the model that consumes them — Poisson regression, two-model setup, Dixon-Coles correction. That's `04_modeling.md`.
