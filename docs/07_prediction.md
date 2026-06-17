# 07 — Production prediction

You have a trained, evaluated model. The final step is using it to predict the target event — for us, the 2026 World Cup. This chapter covers:

### Vocabulary used in this chapter

If any of these are unfamiliar, the glossary at the start of `00_overview.md` has plain-language definitions:

- **Production model** — the final trained model with no held-out test set; trains on every available data point
- **Snapshot** (in this context) — a frozen feature state at a specific date
- **Static forecast** — every prediction uses features as of a single fixed date
- **Live forecast** — features update as new matches happen
- **Monte Carlo simulation** — repeatedly sampling random outcomes from probability distributions, then counting how often each combined-outcome happens
- **Conditioning** — using known information to refine probability estimates ("given that X happened, what's the probability of Y?")


- Switching from "backtest mode" to "production mode."
- Generating per-match predictions for the tournament.
- Simulating the bracket via Monte Carlo to estimate tournament-level probabilities (who's most likely to win it all).
- Handling tournament-specific details (multiple hosts, altitude, expanded format).
- Communicating the predictions in a way that's useful and honest.

## Backtest mode vs. production mode

Throughout development you've been in **backtest mode**: training on data up to a *historical* cutoff, then evaluating on a known past tournament. The evaluation tells you how well the model would have done if you'd deployed it back then.

**Production mode** is similar but uses the most recent cutoff:

| Mode | Training data | Predict |
|---|---|---|
| Backtest (2022) | 1990 → 2022-11-19 | WC 2022 (known outcomes) |
| Production (2026) | 1990 → 2026-06-10 | WC 2026 (unknown outcomes) |

Same code, same pipeline. Only the cutoff date differs:

```python
PRODUCTION_CUTOFF = pd.Timestamp("2026-06-11")  # day before WC 2026 kickoff
models = train(PRODUCTION_CUTOFF)
```

The production model has access to all data the backtests had, plus the additional 3.5+ years between WC 2022 and WC 2026 kickoff. More data → marginally better estimates → marginally more reliable predictions.

## The team-state snapshot pattern

To predict a future match, the feature row needs values for every column the model was trained on: Elo, recent form, squad value, etc. But the future match hasn't happened yet, so we can't just read those features from a pre-built table — we need to **compute each team's feature state as of the prediction cutoff**.

The pattern: walk the team's match history up to the cutoff date, derive the team's "current" state, store it in a lookup dict, then build feature rows by joining the predicted matches against the dict.

```python
def compute_team_state_at_cutoff() -> dict[str, dict]:
    """For each team, compute their feature state as of cutoff."""
    results, _ = load_results(apply_cutoff=True)
    results = results.sort_values("date").reset_index(drop=True)

    team_recent = {}     # team → list of (date, goals_for, goals_against)
    team_last_date = {}

    for _, row in results.iterrows():
        for team, gf, ga in [
            (row["home_team"], row["home_score"], row["away_score"]),
            (row["away_team"], row["away_score"], row["home_score"]),
        ]:
            team_recent.setdefault(team, []).append((row["date"], gf, ga))
            if len(team_recent[team]) > 10:
                team_recent[team] = team_recent[team][-10:]
            team_last_date[team] = row["date"]

    state = {}
    for team, matches in team_recent.items():
        gfs = [m[1] for m in matches if pd.notna(m[1])]
        gas = [m[2] for m in matches if pd.notna(m[2])]
        state[team] = {
            "form_scored": float(np.mean(gfs)) if gfs else np.nan,
            "form_conceded": float(np.mean(gas)) if gas else np.nan,
            "last_match_date": team_last_date.get(team),
        }
    # Layer in Elo (final_elo.csv), squad value (squad_values.csv year=2026)
    ...
    return state
```

Then for each match to predict:

```python
def build_match_features(home, away, match_date, neutral, state):
    h = state.get(home, {})
    a = state.get(away, {})
    return {
        "home_elo_pre": h.get("elo"),
        "home_form_scored": h.get("form_scored"),
        "home_days_since_last": (match_date - h.get("last_match_date")).days,
        ...
    }
```

This gives you per-match feature rows you can hand to `model.predict()`.

## Static vs. live forecasts

The above is a **static forecast** — every prediction uses the cutoff-date snapshot. A live forecast would update each team's state after every played match (so a team's predicted strength rises if they win and falls if they lose during the tournament).

| Mode | When to use | Cost |
|---|---|---|
| **Static** | One-shot pre-tournament report — "what would you have said the day before kickoff?" | Simple, predictions don't shift |
| **Live** | Operational forecasting during the tournament | More complex, each prediction has a different "as of" date |

For a one-shot WC 2026 forecast that goes into a writeup or comparison, **static is the right answer**. The implementation is in `src/prediction/wc2026.py`.

## Filtering to actual WC 2026 participants

A subtle but important step: your training data and features cover *all* international teams. For WC 2026 predictions, you only care about the 48 qualified teams.

Don't take the "top 48 by Elo" or "top 48 by squad value" — neither is the actual qualifier list. Italy didn't qualify for WC 2018 or 2022 despite being top-10 by squad value. Norway has high value (Haaland, Ødegaard) but no recent WC appearances.

The right source for the qualifier list is the **tournament fixture data itself**. Your match-history dataset includes the WC 2026 fixtures (with NaN scores for unplayed games) — extract participants from there:

```python
def wc_2026_participants(results_df: pd.DataFrame) -> set[str]:
    wc_mask = (
        (pd.to_datetime(results_df["date"]).dt.year == 2026)
        & (results_df["tournament"] == "FIFA World Cup")
    )
    wc_2026 = results_df[wc_mask]
    return set(wc_2026["home_team"]).union(set(wc_2026["away_team"]))
```

That should return 48 team names. Use this set as the universe of teams in all downstream prediction code.

## Predicting individual matches

For each scheduled WC 2026 match:

```python
def predict_match(home_team: str, away_team: str, kickoff_date: date,
                  is_neutral: bool, models: TrainedModels,
                  features_df: pd.DataFrame) -> dict:
    # Look up the latest feature row for each team
    row = build_match_feature_row(home_team, away_team, kickoff_date, is_neutral, features_df)
    lam_h, lam_a = models.predict(pd.DataFrame([row]))
    p_h, p_d, p_a = outcome_probs(lam_h[0], lam_a[0], rho=models.rho)
    score_dist = score_matrix(lam_h[0], lam_a[0], rho=models.rho)
    return {
        "lambda_home": float(lam_h[0]),
        "lambda_away": float(lam_a[0]),
        "p_home_win": p_h,
        "p_draw": p_d,
        "p_away_win": p_a,
        "score_matrix": score_dist,
    }
```

The `build_match_feature_row` helper assembles the same features the model was trained on — Elo (from latest available values), recent form (rolling stats up to kickoff date), tournament class (`"world_cup"`), `is_neutral` (true for almost all WC matches except host-team games), squad value (current snapshot).

For the full file, see `src/prediction/wc2026.py` (or build it analogously to the backtest predictor).

## Conditioning the simulation on already-played results

If the tournament is in progress, some matches have already happened. You **don't want to re-sample those** — you know exactly what occurred. The right approach:

```python
for match in group_matches:
    if match["played"]:
        h, a = match["actual_home_goals"], match["actual_away_goals"]
    else:
        h, a = sample_score(match["expected_goals_home"],
                            match["expected_goals_away"], rho, rng)
    # ... update standings with (h, a) ...
```

**Why this matters:** the simulation's headline output (P(team X wins WC)) changes over time as more matches are played, *even with the same trained model*. Running the simulation on day 1 and again on day 10 gives different numbers — not because the model improved, but because we're conditioning on more observed information.

For a team that's been mathematically eliminated, their `P(win WC)` drops to 0%. For a team that's locked up a group, their `P(advance)` jumps to 100%. This is just probability respecting facts.

## Group-stage predictions

For each of the 12 groups, predict every match. Use the predictions to estimate group-stage outcomes:

```python
def simulate_group_stage(group_matches: list[dict], n_sims: int = 10_000) -> dict:
    """Monte Carlo simulate each group's matches and return advancement probabilities."""
    points = defaultdict(lambda: defaultdict(int))   # team → sim_idx → points
    goal_diff = defaultdict(lambda: defaultdict(int))

    for sim in range(n_sims):
        for match in group_matches:
            # Sample a scoreline from the predicted distribution
            score_dist = match["score_matrix"]
            flat_idx = np.random.choice(score_dist.size, p=score_dist.flatten())
            home_goals, away_goals = np.unravel_index(flat_idx, score_dist.shape)

            home, away = match["home_team"], match["away_team"]
            if home_goals > away_goals:
                points[home][sim] += 3
            elif home_goals < away_goals:
                points[away][sim] += 3
            else:
                points[home][sim] += 1
                points[away][sim] += 1

            goal_diff[home][sim] += home_goals - away_goals
            goal_diff[away][sim] += away_goals - home_goals

    # ... rank teams within each group, count how often each finishes 1st/2nd, etc.
```

The Monte Carlo simulation samples 10,000 versions of each group, computes the table for each, and reports the **probability that each team advances to the knockout round**.

This is more informative than reporting per-match probabilities because it accounts for the *full path* — a team might be a slight favorite in each of its three matches but still likely to fail to advance if all three are tight.

## Performance: pre-caching predictions

Naive Monte Carlo on a tournament is slow because every simulation needs predictions for ~31 knockout matches (R32: 16, R16: 8, QF: 4, SF: 2, F: 1 = 31). With 20,000 simulations, that's **620,000 individual model.predict() calls**. Each `predict()` call on a single row is dominated by sklearn's per-call overhead — it would take ~25 minutes.

The fix is to **pre-compute predictions for every possible matchup once** and cache them in a dict.

```python
def precompute_knockout_lambdas(teams: list[str], state: dict, models) -> dict:
    """For every (home, away) pair of WC participants, compute (λ_h, λ_a)
    assuming neutral venue. Returns dict (home, away) → (λ_h, λ_a).
    """
    rows = []
    pairs = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            h_st = state.get(h, {})
            a_st = state.get(a, {})
            rows.append({
                "home_elo_pre": h_st.get("elo", 1500),
                "away_elo_pre": a_st.get("elo", 1500),
                # ... etc, knockout features (all neutral)
            })
            pairs.append((h, a))
    feats_df = pd.DataFrame(rows)
    lam_h, lam_a = models.predict(feats_df)   # ONE big batch predict
    return {pair: (float(lh), float(la)) for pair, lh, la in zip(pairs, lam_h, lam_a)}
```

For 48 WC participants, that's 48 × 47 = **2,256 predictions** computed in a single batch (~1 second). During the sim loop, each "predict this knockout match" call becomes a dict lookup: O(1).

The total speedup is roughly **30×**: from ~25 minutes to ~50 seconds for 20k simulations.

**This pattern generalizes.** Any Monte Carlo where the underlying predictions are functions of *fixed* features (here: team identity + neutral venue) can benefit from pre-computing the prediction surface once. The simulation loop just samples from the pre-computed outputs.

## Knockout-stage simulation

In the knockouts, draws aren't allowed. After 90 minutes (or 120 with extra time), there's a winner. If the predicted score is tied, you need a rule for assigning a winner.

Two options:

**Option 1: just use the W/D/L probabilities and treat draws as a 50/50 shootout.**

```python
def simulate_knockout_match(home, away, p_h, p_d, p_a):
    outcome = np.random.choice(["H", "D", "A"], p=[p_h, p_d, p_a])
    if outcome == "H":
        return home
    if outcome == "A":
        return away
    return home if np.random.random() < 0.5 else away   # shootout coin flip
```

**Option 2: use historical shootout outcomes to model the shootout slightly better.**

Penalty shootouts are nearly random — research consistently shows team strength barely predicts shootout outcomes — so option 1 is fine. Don't over-engineer this.

## Bracket-level probabilities

Putting it all together:

```python
def simulate_full_tournament(n_sims: int = 100_000) -> dict[str, dict[str, float]]:
    """Simulate the entire tournament N times. Return:
       team → {advance_from_group, reach_round_of_16, ..., win_tournament} percentages.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for sim in range(n_sims):
        group_winners = simulate_group_stage(GROUPS)
        ro16_winners = simulate_round(group_winners, ROUND_OF_16_MATCHES)
        qf_winners = simulate_round(ro16_winners, QUARTERFINAL_MATCHES)
        sf_winners = simulate_round(qf_winners, SEMIFINAL_MATCHES)
        champion = simulate_match(sf_winners)

        for team in group_winners:
            counts[team]["advance_from_group"] += 1
        for team in ro16_winners:
            counts[team]["reach_quarterfinal"] += 1
        # ... etc

    return {team: {k: v / n_sims for k, v in stats.items()}
            for team, stats in counts.items()}
```

The result is a per-team table:

```
team        advance  ro16    qf     sf    final  win
Argentina   95%      82%     63%    44%   28%    14%
Brazil      94%      80%     61%    42%   26%    12%
...
```

This is the headline output of the model — "who's most likely to win the World Cup, and by how much."

## WC 2026 specifics

The 2026 World Cup has structural differences from 2014, 2018, 2022:

**1. 48-team format (vs. 32 previously).**

Group stage: 12 groups of 4 (group matches: 12 × 6 = 72 total).
Knockout: round of 32 (top 2 from each group + 8 best third-place) → round of 16 → QF → SF → 3rd-place → final.

Your group-stage simulation code needs to handle "12 groups, top 2 plus 8 best thirds advance."

**2. Three hosts (USA, Canada, Mexico).**

All three host teams play at home. Apply the home advantage feature accordingly:

```python
HOST_TEAMS = {"United States", "Canada", "Mexico"}

def is_match_neutral(home_team: str, away_team: str, venue_country: str) -> bool:
    if home_team in HOST_TEAMS and venue_country == home_team:
        return False
    if away_team in HOST_TEAMS and venue_country == away_team:
        return False
    return True
```

Note that even host teams play some away games when they're in another host country (USA playing in Mexico isn't a home game for the USA).

**3. Altitude.**

Some matches are in Mexico City (2,240m elevation) and Guadalajara (1,560m). Altitude affects player performance — research shows ~5–10% drop in aerobic output. If you want to model this:

```python
ALTITUDE_VENUES = {
    "Mexico City": 2240,
    "Guadalajara": 1560,
    # ... rest are sea level
}

def altitude_adjustment(venue_city: str, team_country: str) -> float:
    """Return a multiplier to home/away λ for altitude effects."""
    elevation = ALTITUDE_VENUES.get(venue_city, 0)
    if elevation < 1000:
        return 1.0
    # Teams from high-altitude countries (Bolivia, Ecuador, Mexico, Colombia)
    # are less affected. Apply a smaller penalty.
    high_altitude_countries = {"Bolivia", "Ecuador", "Mexico", "Colombia", "Peru"}
    if team_country in high_altitude_countries:
        return 0.95
    return 0.85
```

This adjusts predicted goal counts for matches at altitude. The multiplier values are rough — a research-grade analysis would calibrate them from data.

For a v1, you may skip altitude entirely. The effect is small relative to other factors.

## Output formats

Predictions are most useful when presented in multiple complementary views:

**1. Per-match probabilities table.**

```
date         home         away          p_H    p_D   p_A   λ_h   λ_a   top score
2026-06-11   Mexico       Poland        58%    23%   19%   1.61  0.84  1-0 (15%)
2026-06-12   Argentina    Saudi Arabia  82%    13%   5%    2.40  0.45  2-0 (16%)
...
```

**2. Bracket advancement probabilities.**

```
team         group_pos_1   group_pos_2   reach_ro16   reach_qf   reach_sf   final   win
Argentina    72%           20%           92%          71%        50%        32%     17%
Brazil       68%           22%           90%          68%        47%        29%     14%
France       65%           24%           89%          65%        43%        26%     12%
...
```

**3. Most likely bracket.**

Run the simulation once with each match decided by argmax (deterministically) and report the bracket that emerges. This is the "median forecast" — interesting but should not be over-interpreted (the simulation distribution is much more informative).

## Honesty about uncertainty

The simulation gives precise-looking numbers. Be careful about how you present them.

A few honest framings:

- **The most likely champion has ~10–15% win probability.** Even the strongest model can't beat the inherent variance of single-elimination tournaments.
- **The "top 5 contenders" together have ~50% combined win probability.** The other 50% is spread across the remaining 43 teams.
- **Surprises happen and aren't model failures.** Saudi Arabia 2-1 Argentina in 2022 was a ~3% probability event under any reasonable model. It happened anyway. Such events are *expected* in tournaments of 64+ matches — they're not bugs in your model.

Reporting "Argentina is the favorite at 14%" is honest. Reporting "Argentina will win the World Cup" is wrong.

## When to retrain

If your model is deployed and the tournament is in progress, you have a choice:

- **Static**: predict the entire tournament from your pre-kickoff model. Don't update as results come in.
- **Rolling**: after each round, re-train (or just re-compute Elo + recent form) using actual results so far, then re-predict the remaining rounds.

Rolling is more accurate but harder to communicate. For a one-shot pre-tournament forecast, static is the natural choice.

## Common prediction pitfalls

**Treating model output as ground truth.** Probabilities are estimates, not facts. A "70% win probability" predicted by your model is still a 30% loss probability — and losses happen 30% of the time at that level if your model is well-calibrated. (If they don't, your model is mis-calibrated, which is your problem, not the universe's.)

**Reporting only point estimates.** "Argentina will reach the semifinal" — based on what? 50%? 70%? Always pair predictions with probabilities.

**Overstating precision.** Reporting probabilities to 4 decimal places ("14.327%") implies a precision the model doesn't have. Round to whole percentages or 1 decimal at most.

**Ignoring tournament-specific factors.** WC 2026 has 48 teams; treating it as a 32-team tournament will produce wrong bracket simulations. WC 2026 has three hosts; treating it as a single-host event misapplies home advantage.

**Treating squads as static.** A player's injury or club performance shift can move team strength meaningfully. If your model uses pre-tournament squad values, late-breaking injury news won't be reflected. State this limitation explicitly.

## A note on "predicting scores"

Beginners often ask: "can your model predict the actual score of each match?" The answer is **yes — and we already do**. The Poisson model's fundamental output is the *expected goals* for each team (λ_home, λ_away). From those two numbers we derive everything:

- The full **score matrix** (a grid of P(home=i, away=j) for every possible (i, j))
- **W/D/L probabilities** (sums of regions of the matrix)
- **Most likely scoreline** (argmax of the matrix)
- **Expected total goals** (λ_h + λ_a)
- **Over/under probabilities** (P(total > 2.5), etc.)

For our WC 2026 output, we expose `expected_goals_home`, `expected_goals_away`, `most_likely_score`, and `most_likely_score_prob` as columns. All come from the same single prediction.

### Why exact-score accuracy is "only" 12–15%

A natural follow-up: "how often does the most-likely-score match the actual result?" Answer: **roughly 12–15% of the time** for a typical soccer match.

This sounds low until you realize the same range applies to published academic models, sports betting books (Bet365, Pinnacle, Vegas), and any other forecaster: there are simply too many plausible scorelines (2-0, 3-0, 4-1, 3-2, etc.) for any single one to dominate. Soccer is genuinely high-variance.

A few useful framings:

- A model that correctly identifies "Spain wins 2-0" once out of every 7 such matches is doing extremely well, not poorly.
- The *probability of the most-likely-score* (`most_likely_score_prob`) is itself an output worth showing — it tells you how confident the model is in the scoreline. For mismatches it's ~15% (Germany-Curaçao); for evenly-matched games it's ~10-12% (Brazil-Morocco type).
- If you care about scorelines at all, also expose the **full distribution** (top-5 most likely scorelines + their probabilities) rather than just the argmax. The argmax can be misleading when several scorelines are nearly tied.

### When to NOT focus on exact-score predictions

For most use cases, exact-score predictions are the wrong thing to optimize for. Instead:

- **Betting:** W/D/L probabilities + over/under goal lines.
- **Tournament forecasts:** advance / reach round / win probabilities.
- **Match analysis:** expected goals (λ values) as inputs to other analyses.
- **Fantasy / news:** highlighting upset candidates, not predicting individual scores.

The model already computes everything you need for these. Build whichever summary statistic the use-case actually wants.

## What's next

You've shipped a working model. The last chapter is reflection — what didn't work, what you learned, what to do differently next time. `08_lessons_and_pitfalls.md`.
