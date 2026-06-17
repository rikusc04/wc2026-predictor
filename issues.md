# Issues & Known Limitations

Running log of bugs found, design choices, and limitations encountered while building v1 of the World Cup 2026 predictor.

## How to read this file

This is the engineering log — a record of every non-trivial bug, design choice, or limitation we hit while building the model. Each item follows a roughly consistent structure:

- **Symptom** — what was visibly wrong (a number that didn't match expectations, a missing row, etc.)
- **Cause / diagnosis** — the underlying reason, usually only obvious after investigation
- **Fix** (where applicable) — what code or data change resolved it
- **Why we accepted it** (for known limitations) — when we chose not to fix, and the reasoning

If a term is unfamiliar (e.g., "leakage," "log-loss," "imputation," "snapshot"), see the glossary at the top of `docs/00_overview.md` for plain-language definitions. The relevant chapter of `docs/` always has more depth than what this log records.

The numbering reflects rough chronological order of discovery. Items get added but rarely removed — even resolved bugs are useful to future-you, who will hit a similar shape and want to recognize it.



## Bugs found and fixed

### 1. Tournament classifier ordering
**Symptom:** 8,771 WC qualifier matches were bucketed as `world_cup`.
**Cause:** `"FIFA World Cup qualification"` contains both `"world cup"` and `"qualif"`. My original `classify_tournament()` checked `world cup` first, so qualifiers got mislabeled.
**Fix:** Reorder checks so `qualif` is tested before `world cup`. See `src/features/tournaments.py`.

### 2. Feature row inflation (49,747 vs 49,405 input rows)
**Symptom:** `features.csv` had ~342 more rows than `matches_with_elo.csv`.
**Cause:** Some teams play multiple matches on the same date (rare but real in older tournaments). `_recent_form_table` was creating multiple `(team, date)` rows for those cases, which then duplicated rows during the merge.
**Fix:** `drop_duplicates(subset=["team", "date"], keep="first")` after computing rolling stats. See `src/features/build.py`.

### 3. `MAX_GOALS=10` truncation
**Symptom:** Brazil vs. San Marino test showed P(Brazil wins) = 51% instead of ~99%; W/D/L sum to only 51%.
**Cause:** When λ_home=10.59, significant Poisson probability mass lives at scorelines ≥11. Our 11×11 matrix cut that off entirely.
**Fix:** Bumped `MAX_GOALS` to 20 in `src/models/poisson.py`. Total mass now drifts by <0.5% even for extreme λ.

### 4. Country-name "must map" check returned empty (red herring, not actually a bug)
**Symptom:** Section 7 of EDA notebook showed no teams needing renaming, which looked too clean.
**Cause:** My check only flagged teams appearing in `former_names.csv`'s `former` column. Defunct national teams in our data (German DR, Czechoslovakia, Yugoslavia) actually live in the `current` column or aren't in `former_names.csv` at all.
**Resolution:** Investigated; confirmed dataset uses "Germany" continuously since 1909 (already merged West Germany). Defunct teams (Czechoslovakia, Yugoslavia, German DR) naturally stop appearing — no canonicalization needed. EDA notebook updated to show this more useful view.

### 5. Squad-value name matching missed mononym players (Brazilians)
**Symptom:** Brazil 2014 wasn't in the top 10 by squad value; Brazil 2022 had only 10/26 players matched. Should have been top 5 in both.
**Cause:** Fjelstul WC database stores mononym players (Neymar, Hulk, Fred, Fernandinho, Marcelo, Paulinho, etc.) as `family_name="Neymar", given_name="not applicable"`. My matcher was constructing the string `"not applicable Neymar"` and looking it up in Transfermarkt's name index — never hits.
**Fix:** `_match_player()` in `src/features/squad_values.py` now detects the `"not applicable"`/empty sentinel and tries `family_name` alone (and `given_name` alone) before the fuzzy fallback.
**Verification:** post-fix output: Brazil 2014 = #3 at €455M (23/23 matched), Brazil 2022 = €1003M (24/26 matched). Spain 2014 jumped from €543M (19/23) to €583M (23/23). Portugal 2014: €238M (16/23) → €289M (22/23). The mononym handler correctly handles Brazilian, Portuguese, and similar naming conventions.

### 6. `appearances.csv` from dcaribou is club-only
**Symptom:** First attempt at building historical WC squad values returned 0 rows even though `games.csv` contained 5 WCs of matches (FIWC competition_id, 64 matches each for 2006/2010/2014/2018/2022).
**Cause:** The dcaribou Transfermarkt mirror's `appearances.csv` only tracks club-football appearances (Bundesliga, Premier League, etc.) — international matches aren't covered.
**Resolution:** Pivoted to the Fjelstul WC database on GitHub, which has clean roster data for every WC 1930–2022. Used `(year, team, player_name)` from Fjelstul + Transfermarkt `player_valuations.csv` to compute per-WC squad values.

### 7. Transfermarkt itself blocks WebFetch; Wikipedia squad pages truncated mid-roster
**Context:** Attempted direct scraping as a fallback before discovering the Fjelstul dataset.
**Findings:**
  - Transfermarkt: WebFetch returns "unable to fetch" — they aggressively block automated traffic.
  - Wikipedia: pages like `2014_FIFA_World_Cup_squads` are long enough that the WebFetch summarizer only sees the first ~8 of 32 teams, no matter how the prompt is structured. Asking for "Groups C–D" returns "I can only see Groups A–B from the provided content."
**Lesson:** For multi-team pages, per-team Wikipedia URLs would work (`Brazil_at_the_2014_FIFA_World_Cup` etc.) but would require 160+ fetches. Pre-compiled GitHub datasets (Fjelstul) are the right answer when they exist.

## Known model limitations (accepted in v1)

### 5. Model never argmax-predicts draws
**Symptom:** Confusion matrices across all three WC backtests show **zero** predicted draws.
**Cause:** Independent Poisson under-predicts low-scoring draws. P(draw) is rarely the argmax even when it's the right answer (~25% of WC matches end in a draw).
**Mitigation tried:** Added Dixon-Coles correction (see #6 below).
**Status:** Partial. The DC boost is too small to flip the argmax in WC matches.

### 6. Dixon-Coles correction had marginal effect
**Symptom:** Fitted ρ ≈ −0.06 (vs. classical ≈ −0.15 for English football). Log-loss got slightly **worse** by ~0.003 across all three backtests; argmax behavior unchanged.
**Cause:** Two contributing factors:
  - International football includes many friendlies/mismatches that dilute the optimizer's preferred ρ.
  - DC redistributes mass toward draws — for WC matches with large Elo gaps, that hurts more than it helps.
**Status:** Code is in place (`src/models/poisson.py` — `fit_rho`, `_tau`, score-matrix support). Could revisit with joint MLE or WC-restricted fit.

### 7. Two-stage estimation drift
**Description:** We fit λ_home and λ_away assuming ρ=0, then estimate ρ on top. Strictly, the optimal λs depend on ρ. A joint MLE (fit all three together) would be slightly more correct.
**Cost of fix:** Moderate code change; replace `sklearn.PoissonRegressor` with a hand-rolled optimizer over the combined Dixon-Coles likelihood.
**Status:** Not done. v1 uses two-stage; impact is small.

### 8. Model only marginally beats the naive Elo baseline
**Symptom:** Log-loss delta vs. naive Elo is ~0.005–0.010 across the three backtests. Real but small.
**Diagnosis of feature redundancy:**
  - `form_scored` / `form_conceded`: highly correlated with Elo (Elo already updates from every match)
  - `tournament_class`: has zero variance within a WC backtest (all matches are `world_cup`)
  - `days_since_last`: weak signal at international level (most teams have ~similar inter-match spacing)
  - `is_neutral`: the one feature meaningfully helping
**Implication:** Adding genuinely new information (squad value, lineups) likely helps more than tuning the existing math.

### 9. Performance degradation over time
**Observation:** Log-loss 2014 (0.93) → 2018 (0.97) → 2022 (1.06). Accuracy 60.9% → 56.2% → 48.4%.
**Likely cause:** WC 2022 was historically upset-heavy. Saudi Arabia beat Argentina (pre-tournament ~3000:1 longshot), Morocco reached the semifinals, Japan beat Germany AND Spain. Single-tournament noise is high (n=64). Not necessarily a model trend.
**Status:** Acknowledged. We'd need more backtests (or non-WC competitive matches) to distinguish noise from genuine drift.

### 10. Aggressive extrapolation beyond training Elo distribution
**Symptom:** Brazil (Elo 2029) vs. San Marino (Elo 952) gives λ_home=10.59, unrealistic.
**Cause:** Largest Elo gaps in training are ~800. The Poisson regression is linear in log-space; asked about a 1077 gap, it extrapolates linearly.
**Impact on WC predictions:** None. All 48 qualified teams sit within the training distribution. Largest WC 2026 Elo gap will be ~500.
**Status:** Documented in the smoke test as an "out-of-distribution check"; not a bug to fix.

### 11. No injury / squad / lineup information
**Description:** Elo treats teams as stable identities. The classic example: Brazil 1-7 Germany 2014. Brazil was missing Neymar (injured) and Thiago Silva (suspended). Elo can't see this, so it punishes Brazil's identity for what was partly a circumstance.
**Mitigation:** Law of large numbers (~50 Brazil matches since absorb the shock). Bounded K-factor caps single-match impact at ~100 Elo points.
**Real fix:** Player-level / lineup-aware modeling. Deferred to v2.

## Known data limitations (squad value coverage)

### 12. No squad value for WC 1990, 1994, 1998, 2002
**Description:** Transfermarkt player valuations begin around 2000. Fjelstul's roster data goes back further, but without TM valuations we can't price the rosters. Training matches in those years will get NaN squad value, which the downstream pipeline imputes with the median.
**Impact:** Reduces effective signal for the v1/v2 backtest models on pre-2002 matches, which is most of model v1's pre-WC-2006 training data.

### 13. Squad value undercount for non-Western naming conventions
**Description:** Some player names — particularly African and Middle Eastern players from older WCs — don't have exact TM matches even after the mononym fix. Fuzzy threshold (0.85) skips ambiguous cases.
**Impact:** Squad values for some teams (e.g., Algeria, Ghana, Ivory Coast historical squads) are likely 10–20% undercount. Less of a problem for relative ranking than absolute values, and the downstream model uses both via standardization.

### 14. WC 2026 squad values use current snapshot (post-tournament-start)
**Description:** WC 2026 is in progress as of model build time. Our snapshot is from the dataset's most recent update (~mid-2025). Player valuations may have shifted slightly during the tournament.
**Impact:** Minor. Squad values change on quarterly TM revaluation cycles; intra-tournament drift is small.

### 15. `national_teams.csv` includes teams that didn't qualify for WC 2026
**Symptom:** Italy appears in the top-10 squad values for "WC 2026" despite not qualifying for the tournament.
**Cause:** `national_teams.csv` lists every nation TM tracks, not just WC 2026 participants.
**Fix:** `build_current_2026()` in `src/features/squad_values.py` now reads the actual qualifier list from the WC 2026 fixture rows in `results.csv` (date ≥ 2026-06-11, tournament == "FIFA World Cup") and filters out anyone else.

### 16. Multiple WC 2026 qualifiers missing from initial 2026 squad-values output
**Symptom:** After filtering to WC 2026 qualifiers, only 39 of the 48 teams had a squad value; 9 were missing.
**Diagnosis:** three distinct failure modes:
  - **Name-spelling mismatch (1 team)** — TM uses `Bosnia-Herzegovina`, results.csv uses `Bosnia and Herzegovina`. Fixable via `TM_NAME_FIXUPS`.
  - **In TM but no aggregate value (3 teams)** — England, France, Spain all have entries in `national_teams.csv` but their `total_market_value` field is empty. This is a data-quality gap in the public mirror, surprising given they are among the most-tracked national teams in the world.
  - **Not in TM at all (5 teams)** — Cape Verde, Curaçao, DR Congo, Haiti, Ivory Coast. These smaller federations don't have entries in `national_teams.csv` at all, and their `current_national_team_id` is unassigned for players in `players.csv`. No team_id to filter on.
**Fix v1 (didn't work):** initial attempt summed top-26 player valuations via `current_national_team_id`. Doesn't work for England/France/Spain — 0 players have those team_ids set in `players.csv` (the data-quality gap goes deeper than just the aggregate column).
**Fix v2 (works):** use `country_of_citizenship` instead — populated for ~all players. Sum top-26 by citizenship gives a "talent pool" estimate. Then calibrate.
**Fix v3 (5 small federations):** even with v2 working, the 5 federations not in `national_teams.csv` at all (Cape Verde, Curaçao, DR Congo, Haiti, Ivory Coast) weren't being attempted. Added a third tier: for any WC 2026 qualifier missing from the national_teams loop, query players.csv directly by citizenship. Required a `RESULTS_TO_CITIZENSHIP` mapping for spelling differences (`Curaçao → Curacao`, `Ivory Coast → Cote d'Ivoire`).
**Coverage post-fix:** 48/48 — all WC 2026 qualifiers covered. The new entries are tagged `source="citizenship_only"` for traceability. Values: Cape Verde ~€51M, DR Congo ~€122M, Haiti ~€45M, Curaçao ~€25M, Ivory Coast ~€380M.

### 17b. Squad-value feature too sparse for the model to learn from
**Symptom:** After the initial squad-value integration, `home_squad_value` and `away_squad_value` were NaN in **87.8% / 88.9%** of training rows. Backtest improvements were modest (−0.007 / −0.006 on WC 2014/2018, +0.015 regression on WC 2022).
**Diagnosis:** the historical snapshots only covered WC participants (32 teams per year via Fjelstul rosters). The vast majority of training matches involve non-WC teams (e.g., Estonia vs. Latvia friendlies, Andorra in Euro qualifiers) — these had NaN squad values throughout, leaving the median imputer to fill them. Net result: the model could only *learn from* the 12% of rows where the feature was non-null.
**Fix (Path B):** extended `build_historical()` to do a two-pass computation per snapshot year:
  1. **Pass 1 — Fjelstul rosters for WC participants** (accurate, ~32 teams).
  2. **Pass 2 — citizenship-based estimate for all other TM-tracked nations** (~70 additional teams per year, calibrated with the 0.78 factor).
This covers roughly 100 teams per snapshot year instead of 32. The non-participant rows are tagged `source="citizenship_calibrated_historical"` for auditability.
**Actual impact:** missing rate dropped from **87.8% → 76.2%** (home) and **88.9% → 77.6%** (away). Less dramatic than the initially-hoped 30-50% because two structural gaps remain:
  - **Pre-2006 training matches** (~1990-2005, roughly a third of training data) have no snapshot to forward-fill from. Permanent gap given the dataset's history.
  - **Tiny federations** (San Marino, Liechtenstein, Bhutan, Andorra, etc.) aren't in TM's 118-team `national_teams.csv` at all, so even in post-2006 matches involving these teams the lookup fails.
The doubling of informative rows (~6k → ~12k non-null) is still material — the model now has ~2× more rows from which to learn the feature's effect on goal counts. Whether that translates to backtest log-loss improvement is the next thing to verify.

### 17c. Asymmetry: historical snapshots cover all teams, WC 2026 snapshot covers only qualifiers
**Description (not a bug, but a design decision worth recording):** after Path B, the historical snapshots (2006, 2010, 2014, 2018, 2022) intentionally include all ~100 TM-tracked national teams, while the WC 2026 snapshot is filtered to the 48 actual qualifiers.
**Rationale:**
  - Historical snapshots are used as **training data** for matches over many years. Training data includes non-WC matches involving non-WC teams (e.g., Italy in Euro qualifiers and Nations League). For those rows to have a `squad_value` feature, every team needs to be snapshotted.
  - The WC 2026 snapshot is used for **predictions** of WC 2026 matches, which only involve the 48 qualifiers. Snapshotting Italy 2026 wouldn't be used anywhere (training is cut off at 2026-06-10 by leakage policy; post-cutoff matches don't exist in our world).
**How to apply:** if you ever extend the project to predict different tournaments, remember the asymmetry — "future" snapshots scope to participants of that tournament; "historical" snapshots scope as wide as possible so the feature is informative across all training matches.

### 18. Static vs. live forecast tradeoff in production prediction
**Description (design choice, not a bug):** the WC 2026 prediction pipeline uses a **static forecast** — all 72 group-stage matches are predicted using features frozen at 2026-06-10 (the day before kickoff). Earlier WC 2026 results do not update Elo or recent-form for later matches in the same simulation.
**Why static:** matches the standard "pre-tournament forecast" framing. Reports "this is what we would have said the day before kickoff" — clean comparison across teams, no leakage between matches we're predicting.
**Why not live:** a live forecast would re-compute features after each played match. More accurate forecasting day-to-day, but harder to communicate (each prediction has a different "as of" date) and the implementation is more involved.
**How to apply:** if you ever need a live forecast (e.g., for in-progress betting), the data loader supports it — just call `load_results(apply_cutoff=False)` and re-run the Elo/form pipeline through the match date. Static is right for the headline pre-tournament report; live is right for ongoing operational use.

### 19. Conditioning the simulation on already-played match results
**Description (design choice):** the Monte Carlo tournament simulator (`src/prediction/simulate_wc2026.py`) does NOT re-sample matches whose actual result is already known. It uses the known scoreline as the "drawn sample" for those matches and only samples unplayed matches from the predicted score distribution.
**Rationale:** the goal is "given everything we now know, what's each team's probability of advancing/winning?" Using known results is more informative than ignoring them. As more matches are played, advancement probabilities update toward 0% or 1% for teams already mathematically eliminated/qualified.
**Important consequence:** the simulation's output changes over time *even with the same trained model* — running it on day 1 vs. day 10 of the tournament gives different probabilities because more matches are observed by day 10. The model coefficients are the same; only the conditioning information differs.

### 20. WC 2026 production model: real-world validation early results
**Result on 12 group-stage matches already played (as of build date):** log-loss 1.00, accuracy 50%. Right in the published-model range (0.95–1.05) and consistent with our backtest performance on past WCs.
**No catastrophic predictions yet:** max single-match log-loss is ~1.7. No Brazil-Cameroon-style 4.0 surprises so far.
**Most misses are draws:** the Poisson model's known weak point (under-predicts draw as argmax). Several actual draws (Canada-Bosnia, Qatar-Switzerland, Brazil-Morocco, Netherlands-Japan) had predicted favorite probabilities that lost log-loss when results came back as draws. This is the same pattern we saw in backtests; no surprise here.
**Implication:** the model is performing at its expected level on out-of-sample new data. Squad value, Elo, and form features generalize correctly. The headline tournament-winner probabilities can be trusted at the level we trust the backtest results.

### 17. Calibration factor when using fallback aggregation
**Symptom:** Top-26 by `country_of_citizenship` overstates actual squad value because it includes eligible-but-not-called-up players. For Germany: TM aggregate €773M vs. citizenship top-26 €1166M — a 1.5× overstatement.
**Naive fix that doesn't work:** median ratio across all teams. Came back at ~1.0 because most countries (~85% of TM's 105 tracked teams) have small citizenship pools where top-26 IS basically the squad. The big-country bias gets washed out by the median.
**Fix:** in `_calibration_factor()`, restrict the calibration to the **top half of teams sorted by citizenship_top26** — these are the deep-talent-pool countries where the "calling-up haircut" actually applies, and they're the teams the fallback applies to. Use total-over-total (magnitude-weighted) rather than median of ratios, so big teams contribute proportionally.
**Result:** calibration factor ~0.78, which puts England (€1780M citizenship × 0.78 = €1389M), France (€1360M), Spain (€976M) at the top of WC 2026 — leading the field but not 2× ahead of Portugal/Brazil/Germany.

## Deferred to v2

### 16. Goalscorer / player-level modeling
Would let us model team strength as a function of *who is on the roster*, not just team identity. Adds significant data complexity (player IDs, lineup prediction model, club performance integration). Deferred entirely to v2.

### 17. WC 2026-specific handling (task #9)
- **Multi-host advantage**: US, Canada, Mexico all hosts. Three teams get home boosts.
- **Altitude effect**: Some matches in Mexico City (elev. 2,240m). Worth a feature.
- **48-team format**: First WC with this structure; bracket simulation differs from 2014/2018/2022 backtests.

Not yet implemented.

## Task #9: tournament simulation design choices

### 21. Random pairing in knockout bracket (vs. official FIFA seeding)
**Description (simplification):** when the simulator builds the round-of-32 bracket from the 32 advancing teams, it pairs them up **randomly** with a same-group-avoidance constraint, rather than implementing FIFA's official seeding rules for the 48-team format.
**Why:** FIFA's seeding for the 48-team WC 2026 format is documented but complex (which group winner faces which 3rd-place qualifier depends on which groups produced the qualifying 3rd-placers, etc.). Implementing it correctly is ~200 lines of bracket-construction logic.
**Cost:** the random-pairing approach causes slight over-mixing — Spain might face Argentina in R32 rather than meeting in the final as the seeded bracket intends. Aggregated across 20k simulations, the impact on individual team P(win WC) is roughly ±1-2% (estimated; we haven't done a sensitivity test).
**How to apply:** for the published-headline-quality probabilities, fix this. For "directional best team ordering," the current approach is fine. The relevant function is `pair_with_group_avoidance()` in `src/prediction/simulate_wc2026.py`.

### 22. Penalty shootouts modeled as 50/50 coin flip
**Description (research-backed simplification):** when a knockout match is sampled as a draw, the simulator does NOT extend to extra time + shootout in detail. Instead, it picks a winner via random coin flip.
**Why:** decades of academic research consistently show shootout outcomes are essentially uncorrelated with team strength — the team that "wins" a shootout is largely random. (Dixon-Coles 1997, Apesteguia & Palacios-Huerta 2010, and others.) A 50/50 coin flip captures this well.
**Cost:** ~zero. Real shootouts may have a very small skill component (better goalkeepers, more disciplined kickers) but it doesn't meaningfully change tournament-winner probabilities.

### 23. Pre-cached knockout lambdas for performance
**Description (production pattern):** the simulator predicts knockout matches using a pre-built cache of (home, away) → (λ_h, λ_a) pairs computed once at the start. With 48 participants, that's 48 × 47 = 2,256 directed pairs cached up front.
**Why:** without caching, 20k simulations × 31 knockout matches each = 620k individual `model.predict()` calls. Each per-row predict is dominated by overhead (sklearn's batching is unfavorable on size-1 calls). With caching: one big batch predict (2,256 rows) at the start, then O(1) dict lookups during the sim loop.
**Speedup:** ~30x. Without caching: ~25 minutes for 20k sims. With caching: ~50 seconds.
**Generalizes to:** any Monte Carlo simulation where the underlying predictions are functions of (fixed) features. Pre-compute the predictions, cache them, then sample from the cached output.

### 24. Simulation-result headlines (for posterity)
**Spain 18.6% to win WC 2026.** Argentina 14.6%, England 13.0%, France 11.1% — the top 4 cover 57% of championship probability.
**Germany 4.5% despite 99.8% to advance** — model has them advancing from groups but exiting in early knockouts (only 35.6% to reach QF). Ecuador as a surprisingly strong group-mate.
**Morocco 3.5% as the highest non-traditional contender** — reflects their 2022 WC semifinal run + €456M squad value.
**Brazil only 4.2%** — below their historical range due to (a) Morocco in their group and (b) mid-pack squad value relative to top European teams.
**~43% probability of a non-top-4 winner** — the model is appropriately humble about variance. Roughly 2-in-5 simulations end with a winner outside Spain/Argentina/England/France.

## Big-picture summary

- **Feature-limited, not model-limited.** More tuning of the Poisson / Dixon-Coles math won't help much. Adding genuinely new information (squad value, lineups) is the highest-leverage move.
- **Beating naive Elo by ~0.005 log-loss.** Real but modest. Published academic WC models tend to score 0.95–1.00 log-loss. We're at 0.93–1.06 — within range but not state of the art.
- **Most "obvious" bugs were data-shaped, not algorithm-shaped.** The tournament classifier, the merge duplication, the country-name check — all were correctness issues in how we prepared data, not in the modeling math.
