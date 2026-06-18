# Issues & Known Limitations

Running log of bugs found, design choices, and limitations encountered while building the World Cup 2026 predictor.

## How to read this file

This is the engineering log — a record of every non-trivial bug, design choice, or limitation we hit while building the model. Each item follows a roughly consistent structure:

- **Symptom** — what was visibly wrong (a number that didn't match expectations, a missing row, etc.)
- **Cause / diagnosis** — the underlying reason, usually only obvious after investigation
- **Fix** (where applicable) — what code or data change resolved it
- **Why we accepted it** (for known limitations) — when we chose not to fix, and the reasoning

If a term is unfamiliar (e.g., "leakage," "log-loss," "imputation," "snapshot"), see the glossary at the top of `docs/00_overview.md` for plain-language definitions. The relevant chapter of `docs/` always has more depth than what this log records.

The numbering reflects rough chronological order of discovery. Items get added but rarely removed — even resolved bugs are useful to future-you, who will hit a similar shape and want to recognize it.

The file is split into two top-level sections corresponding to the model versions: **v1** (baseline Poisson + Dixon-Coles, items #1–24) and **v2** (multi-host advantage on top of v1, items #25+). Each section ends with its own big-picture summary.

---

# v1 — Baseline Poisson + Dixon-Coles

The original model: per-match Poisson regression for home/away goals on a fixed feature set (Elo, recent form, days-since-last, tournament class, squad market value, binary `neutral` venue, dead-rubber flag), with a Dixon-Coles correction layered on the score matrix and a 20k-run Monte Carlo tournament simulator on top.

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

## Big-picture summary (v1)

- **Feature-limited, not model-limited.** More tuning of the Poisson / Dixon-Coles math won't help much. Adding genuinely new information (squad value, lineups) is the highest-leverage move.
- **Beating naive Elo by ~0.005 log-loss.** Real but modest. Published academic WC models tend to score 0.95–1.00 log-loss. We're at 0.93–1.06 — within range but not state of the art.
- **Most "obvious" bugs were data-shaped, not algorithm-shaped.** The tournament classifier, the merge duplication, the country-name check — all were correctness issues in how we prepared data, not in the modeling math.

---

# v2 — Multi-host advantage

Phase 1, item 1 of the v2 roadmap. The motivating observation: WC 2026 is the first World Cup hosted across three countries (USA, Canada, Mexico — all CONCACAF). v1 used a single binary `neutral` flag, which (a) gave Mexico's home advantage at Azteca the same weight as Liechtenstein's home advantage in Vaduz, and (b) gave Brazil playing in "neutral" East Rutherford the same treatment as Brazil playing in "neutral" Tokyo — even though the first venue is full of Brazilian fans and adjacent to CONMEBOL.

### 25. Graded `host_advantage_{home,away}` feature replaces binary `neutral`
**Design:** for each side of a match, compute a value in {0.0, 0.3, 0.7, 1.0}:
- **1.0** — team's country == match country (true home, after alias normalization for renames like Russia/Soviet Union, Eswatini/Swaziland, DR Congo/Zaïre, …)
- **0.7** — team's confederation == match country's confederation (intra-confederation proximity, e.g. Curaçao playing in the US — both CONCACAF)
- **0.3** — Americas adjacency only: CONMEBOL ↔ CONCACAF. The one explicit cross-confederation bonus, modeling that CONMEBOL teams playing in North America are not as neutral as Asian/African teams there.
- **0.0** — elsewhere.

Implementation: `src/features/confederations.py` holds the country→confederation table (covers all 336 distinct teams and 269 match countries in the data, with a `HOME_ALIASES` dict handling 632 of the 968 historical-rename edge cases). `src/features/build.py` adds `host_advantage_home` and `host_advantage_away` columns to `features.csv`. `src/models/poisson.py` swaps `neutral` out of `BOOL_FEATURES` and adds the two new columns to `NUMERIC_FEATURES_REQUIRED`.

**Why graded, why 4 levels:** users (er, I) chose 4 levels over 3 specifically because the 0.3-level "Americas adjacency" captures something real about CONMEBOL teams' relationship to a US/Canada/Mexico-hosted tournament that intra-confederation alone (0.7) and pure-neutral (0.0) cannot. For Russia 2018 and Qatar 2022 backtests the 0.3 level never fires; the scheme effectively collapses to 0.0/0.7/1.0 there.

### 26. Backtest results: helps 2014 and 2018, hurts 2022
**Log-loss comparison vs. v1 (lower better):**

| WC | v1 | v2 | Δ |
|---|---|---|---|
| 2014 (Brazil, CONMEBOL host) | 0.93 | **0.899** | −0.031 |
| 2018 (Russia, UEFA host) | 0.97 | **0.930** | −0.040 |
| 2022 (Qatar, AFC host) | 1.06 | **1.115** | +0.055 |

**Net change:** ~−0.005 averaged across the three. So this is a marginal positive for log-loss on average, masking real per-tournament asymmetry.

**Diagnosis of the 2022 regression:**
- Qatar gets a 1.0 host_advantage boost despite being a *weak* host team (eliminated in groups with 3 losses). The model now over-predicts Qatar.
- Morocco — a CAF team that ran to the semifinals — gets 0.0 in Qatar. The graded scheme has no CAF↔AFC adjacency (no Middle East / North Africa proximity bonus). v1's `neutral=True` gave Morocco's group-stage opponents no boost either, but the 0.7 boost v2 now gives AFC opponents (Japan, Saudi Arabia, South Korea) actually compounds with the Qatar-host effect to push the model further from reality.

**Why 2014/2018 improve:** in both cases the host's confederation has many qualified teams (CONMEBOL had 6 + Brazil at 1.0 in 2014; UEFA had 14 + Russia at 1.0 in 2018), so the 0.7 same-conf boost lifts a coherent block of matches that v1's binary flag was missing entirely.

**Implication:** the graded scheme works best when the host's confederation is well-represented in the tournament. For 2026 (CONCACAF host, only 6 CONCACAF qualifiers, but 10 CONMEBOL qualifiers get the 0.3 Americas bonus), the effect should be visible but less pronounced than in 2014/2018.

### 27. Live WC 2026 evaluation: slight regression on 12 played matches
**Result on the same 12 played matches v1 was evaluated on (issues.md #20):** log-loss 1.099, accuracy 41.7%. v1 numbers were 1.00 and 50%.

**Caveat:** n=12 is statistical noise (one strong upset moves log-loss by ~0.05). The 192-match backtest (3 WCs × 64 matches) is much more reliable, and there the net change is mildly positive.

**Specific misses v2 worsens vs. v1:**
- Sweden 5-1 Tunisia: v2 had Sweden at 34.5% (close call), but the actual blowout favored Sweden anyway. v1 would have been similar.
- Brazil 1-1 Morocco: v2 says Brazil 35.7% (slight Morocco favorite from the away 0.3 Americas adjustment giving Brazil less of a boost). v1 had Brazil higher. The actual draw was a moderate hit either way.
- Netherlands 2-2 Japan: same draw pattern — Poisson under-predicts draws as argmax, and v2's coefficient adjustments slightly shifted the W/D/L mass without flipping the model's persistent "never predicts draws" weakness (#5).

### 28. WC 2026 forecast shift: Americas teams gain at Europe's expense
**Spain 17.6%** (was 18.8%) | **Argentina 16.9%** (was 14.6%) | **England 11.6%** | **France 10.7%** | **Brazil 4.9%** (up from 4.1%). Colombia breaks into the top 10 at 3.0%. Mexico rises notably to 2.9%. Top-4 still covers ~57% of championship mass.

**Direction matches the design intent:** the new feature redistributes probability toward teams that are nominally at home (US/Canada/Mexico) and same-region (CONMEBOL — Argentina, Brazil, Colombia, Uruguay, Ecuador, Paraguay). UEFA and CAF/AFC favorites lose a small fraction of probability each, going to the closer-to-home teams.

**Argentina almost catches Spain:** a 2.3-point gain narrowed the gap from 4.2 points to 0.7. In v2, the simulator's "treat all knockouts as if in the US" assumption gives Argentina a 0.3 boost in every late-round match, which compounds across 5 knockout rounds.

### 29. Known limitations of v2 host advantage (carried into v3)
- **No CAF↔AFC adjacency.** Morocco, Tunisia, Egypt, Senegal in a Middle East host get 0.0 — likely too low given geographic proximity. Worth adding for future AFC/CAF-region World Cups (e.g., a hypothetical Morocco 2030).
- **No per-host magnitude differentiation.** Mexico's home at Azteca (altitude + intimidation) and US's home at Inglewood (large neutral-ish crowds) get the same 1.0 weight. Per-host learned coefficients are a candidate v2-item-1b extension.
- **All knockouts treated as if in US.** Bracket geography in WC 2026 actually distributes knockout matches across all three host countries; Mexico in a Mexico-City quarterfinal gets only 0.7 in our cache when the real value is 1.0. The knockout cache would need to expand from 2,256 to 6,768 entries to be venue-specific.
- **No travel-distance / time-zone-delta features.** A Japan team flying into Phoenix with a 16-hour time shift has a real disadvantage that "0.0 host_advantage" only crudely captures. A continuous-distance feature would be more principled but requires city-coordinate data.
- **Sub-state alias coverage is incomplete.** 132 historical (team, country) pairs still don't normalize to a 1.0 self-match (down from 968 pre-alias). Mostly obscure (Sápmi/Norway+Finland+Russia spans, Crimea pre/post-2014, etc.). Tail risk is small.

---

## v2 Phase 1 — Item 3: Real FIFA knockout seeding

Replaces the random-with-group-avoidance shortcut (issues.md #21) with FIFA's published 48-team bracket. The simulator's per-round random shuffles are gone; once group standings resolve, every knockout matchup is determined by the fixed bracket tree.

### 30. Bracket tree encoded from FIFA's published WC 2026 schedule
**Source:** Wikipedia *2026 FIFA World Cup knockout stage* page, cross-checked structurally (eligibility matrix internally consistent, total slot count matches 32 R32 + 8 R16 + 4 QF + 2 SF + 1 Final = 47 knockout matches).
**Encoding:** `src/prediction/bracket.py` holds three pieces:
1. **`R32_MATCHUPS`** (16 entries) — each entry is `(match_num, slot_a, slot_b)`. Slots are tagged `("W", group)` for group winner, `("RU", group)` for runner-up, or `("3RD", match_num)` for one of the 8 reserved 3rd-place slots.
2. **`THIRD_PLACE_ELIGIBILITY`** — for each of the 8 reserved slots (matches 74, 77, 79, 80, 81, 82, 85, 87), the 5-letter set of source groups whose 3rd-placer can fill it. Excludes the slot's paired-winner group plus 6 others determined by FIFA's bracket-half rules.
3. **`R16_FEEDERS`, `QF_FEEDERS`, `SF_FEEDERS`, `FINAL_FEEDERS`** — fixed feeder mappings, e.g., R16 match 89 = winners of R32 matches 74 and 77.

### 31. 3rd-place assignment via bipartite matching, not hand-coded table
**Problem:** FIFA's official documentation enumerates 495 = C(12,8) scenarios (which 8 of 12 groups produced the qualifying 3rd-placers), with one canonical slot assignment per scenario. Manually transcribing the full 495-row table is error-prone.
**Approach:** for each scenario, solve a bipartite matching over the eligibility constraints. Each of the 8 reserved slots accepts exactly 5 possible source groups; backtracking finds a valid assignment that uses each qualifier group exactly once. For the great majority of the 495 scenarios the matching is unique or near-unique because rare-eligibility groups like K (only Match 80 accepts a 3K) force the rest. When multiple matchings exist, the backtracker returns the first one found in deterministic sorted order — close enough to FIFA's canonical for Monte Carlo aggregation.
**Verification:** 3 hand-checked test cases in `bracket.py` `__main__` (including the Wikipedia-quoted example {E,F,G,H,I,J,K,L}) all produce valid matchings.

### 32. FIFA group labels (A–L) recovered from chronological match order
**Problem:** `identify_groups()` in `src/features/group_standings.py` discovers the 12 groups via BFS over the "teams played each other" graph and labels them alphabetically by traversal order — which has no relation to FIFA's official A/B/C/.../L. Plugging FIFA's bracket (which says "Group A winner faces Group F's 3rd-place qualifier" etc.) into mislabeled groups would scramble the entire bracket.
**Fix:** `derive_fifa_group_labels()` in `simulate_wc2026.py` traverses the WC 2026 fixtures in chronological-then-original-row order (`kind="stable"` sort). The team listed as `home_team` in each first-of-its-group match seeds that group's FIFA letter, in order A→L. Result on the actual WC 2026 data:
- Group A: Mexico, South Africa, South Korea, Czech Republic (host country in Group A — matches FIFA tradition)
- Group B: Canada, Bosnia and Herzegovina, Qatar, Switzerland (host country #2 in Group B)
- Group C: United States, Paraguay, Australia, Turkey (host country #3 in Group C)
- Groups D–L: Brazil/Haiti/Morocco/Scotland; Germany/Curaçao/Ecuador/Ivory Coast; Netherlands/Japan/Sweden/Tunisia; etc.
**Subtle bug caught en route:** initial implementation used `sort_values("date")` which defaults to quicksort (unstable). For groups whose first matches fell on the same day, sort would scramble their order non-deterministically. Switching to `kind="stable"` preserves CSV row order within ties, which is how FIFA schedules.

### 33. Per-team P(win WC) shifts substantially when the bracket is fixed
**Headline shift:** Argentina overtakes Spain as the most-likely champion for the first time. The fixed FIFA bracket reveals team-specific path effects that random pairing averaged out:

| Team | v1 (random) | v2 Item 1 only | v2 Item 1 + 3 (FIFA bracket) | Δ from Item 1 |
|---|---|---|---|---|
| Argentina | 14.6% | 16.9% | **17.3%** | +0.4 (now #1) |
| Spain | 18.8% | 17.6% | 16.0% | −1.6 |
| England | 12.6% | 11.6% | 12.8% | +1.2 |
| France | 11.5% | 10.7% | 8.9% | −1.8 (tough draw) |
| Brazil | 4.1% | 4.9% | 5.8% | +0.9 |
| Portugal | 4.7% | 4.2% | 5.4% | +1.2 |
| Germany | 4.2% | 4.0% | 2.8% | −1.2 |
| Netherlands | 4.2% | 3.5% | 3.3% | −0.2 |

**Interpretation:** under random pairing every team gets an "average bracket." Under fixed seeding some teams (Argentina, Brazil, Portugal) get easier paths than average; others (France, Germany) get harder ones. This is the signature of going from a draw-blind sim to a draw-aware one. The shifts are 1–2 percentage points per team in either direction, which is meaningful for headline numbers (Argentina vs Spain order changes) but doesn't flip the overall "top 4 cover ~57%" conclusion.

### 34. Bracket assumes our group letters match FIFA's
**Risk:** the simulator's "Group A → Group L" labels are derived from chronological scheduling order in `results.csv` (issue #32). If a future data refresh shifts the order of matches on a tied date — or if FIFA's actual published labeling differs from what's inferable from kickoff order — the bracket gets applied to mis-labeled groups, which would scramble all later-round matchups.
**Mitigation:** the `simulate_full_tournament` function prints the derived group assignments before running. If the print shows Mexico in Group B (or any other obvious violation of the "host country in Group A" convention), that's a flag that the labeling went wrong.
**Long-term:** if FIFA publishes a machine-readable group labeling we can pin to, encode it directly instead of inferring.

### 35. Knockout cache size unchanged
**Note:** the precomputed `lam_cache` (issue #23 in v1) still holds 2,256 = 48 × 47 directed pairs. The bracket-driven sim makes more deterministic use of those pairs (each (team_a, team_b) pair is queried in a specific bracket slot rather than uniformly at random), but the cache itself doesn't need to grow. Sim runtime is unchanged at ~50 seconds for 20k runs.

## v2 Phase 1 — Item 2: Altitude native advantage

The naive framing — "visitors play worse at altitude" — turns out to be the wrong one once you account for FIFA's mandatory ~2-week acclimation period at base camp before WC matches. The literature (notably McSharry 2007 BMJ, on CONMEBOL qualifiers) shows that with a proper acclimation camp the residual visitor disadvantage is roughly half the "fly in cold" figure, AND it's essentially equal across all visitors (since they all acclimate). What's NOT equalized is the lifelong cardiopulmonary adaptation that altitude-native teams keep — that's the feature worth modeling.

### 36. Native-advantage framing instead of visitor-penalty framing
**Why not just feature `altitude_delta` (visitor's home altitude minus venue altitude)?** Because after FIFA's mandated 2-week pre-tournament camp, every visitor in Mexico City has approximately the same residual disadvantage. A `delta` feature would mostly just shift the intercept rather than discriminate between teams. What does discriminate is whether the team has *lifelong* altitude exposure — that's binary, can't be replicated by an away camp, and only Mexico (Estadio Azteca / Aztecas at 2240m), Bolivia (La Paz at 3640m), Ecuador (Quito at 2850m), and Colombia (Bogotá at 2640m) qualify among current CONMEBOL/CONCACAF national teams; plus Ethiopia, Eritrea, Yemen, Afghanistan among CAF/AFC. So the feature is binary: `altitude_native_{home,away}` ∈ {0.0, 1.0} based on whether team altitude ≥ venue altitude − 500m.

**Threshold choice (1500m).** Below this, altitude effects on aerobic performance are negligible in the literature. Above this, blood oxygen saturation begins dropping enough to matter. Mexico City (2240m), Zapopan/Guadalajara (1560m), La Paz (3640m), Quito (2850m), Bogotá (2640m), Addis Ababa (2355m), Asmara (2325m) all clear the bar. Madrid (660m), Tehran (1190m), Salta/Argentina (1187m) don't — the model isn't asked to learn an effect that's mostly noise at those altitudes.

### 37. Coverage: which historical and WC 2026 matches the feature lights up
**Training data (49,405 rows):** 876 rows (1.8%) with `altitude_native_home == 1.0`, 103 rows (0.2%) with `altitude_native_away == 1.0`. Combined: 923 rows where at least one side gets the boost. Dominated by Bolivia in La Paz (130 home matches), Mexico in Mexico City (245), Colombia in Bogotá (89), Ecuador in Quito (108), Addis Ababa (237). Sparse but consistent — the model has enough signal to fit a coefficient.

**WC 2026 fixtures (72 matches):** exactly 5 matches activated, all in the group stage —
- Mexico vs South Africa (Mexico City)
- Mexico vs South Korea (Zapopan, 1560m — clears threshold)
- Mexico vs Czech Republic (Mexico City)
- Uzbekistan vs Colombia (Mexico City — Colombia gets `altitude_away=1.0` as a *visitor* benefiting from their native altitude exposure)
- Colombia vs DR Congo (Zapopan — Colombia gets `altitude_home=1.0`)

The 67 other group-stage matches are all at sub-threshold US/Canada venues plus Monterrey-area Mexican cities. Knockouts go through the precomputed `lam_cache` (issues #23, #35) which defaults to a US venue — so no altitude effect in knockouts for any team.

### 38. Learned coefficients and direction sanity check
After re-training the production model with altitude features added:

| Feature | Home-goals coef | Away-goals coef (sign) |
|---|---|---|
| `host_advantage_home` | +0.040 | (mirrors as − on away model) |
| `host_advantage_away` | −0.034 | |
| `altitude_native_home` | **+0.028** | mirrors as − |
| `altitude_native_away` | **−0.012** | |

Direction is right (home altitude-native = more home goals; away altitude-native = fewer home goals). Magnitudes are smaller than host_advantage because the feature only fires on ~2% of training rows. In log-space terms, an altitude-native home team gets about a 2.8% λ boost in expected goals — meaningful but smaller than the 4% boost from host advantage. The model also learns a slightly asymmetric effect (+0.028 vs −0.012), suggesting the *home-team* altitude benefit is more clearly identified than the *away-team* boost (which is rarer in the training data).

### 39. WC 2026 forecast shift from adding altitude
| Team | v2 Items 1+3 only | v2 Items 1+2+3 (with altitude) | Δ |
|---|---|---|---|
| Argentina | 17.3% | 17.1% | −0.2 |
| Spain | 16.0% | 16.7% | +0.7 |
| England | 12.8% | 13.3% | +0.5 |
| France | 8.9% | 9.0% | +0.1 |
| Portugal | 5.4% | 5.5% | +0.1 |
| Brazil | 5.8% | 5.4% | −0.4 |
| Mexico | 3.3% | 3.0% | −0.3 |
| Germany | 2.8% | 2.6% | −0.2 |

**Net effect is small but not noise.** Mexico's small drop is counterintuitive — they were supposed to *gain* from altitude. The likely explanation:
1. Mexico's altitude boost only fires on 2 of their 3 group matches (Mexico City + Zapopan; not Monterrey). And it's small (~2.8% λ boost).
2. In knockouts, Mexico gets *no* altitude boost (cache assumes US venue).
3. Meanwhile the regression slightly re-weighted host_advantage and other coefficients when altitude got added, partially offsetting Mexico's group-stage gain.

Colombia's number is essentially unchanged (3.1% → 3.1%) despite being altitude-native at 2 group matches in Mexico — same compounding-effect story. The headline shift is more visible in Spain/England rising (which displaces some Argentina/Brazil mass): probably because with altitude making Mexico/Colombia slightly stronger in their group, the European teams' knockout opponents are slightly tougher on average, but the European teams themselves get no altitude penalty since they're acclimated.

### 40. Backtest impact: essentially zero (as expected)
Past WC venues — Brazil 2014 (sea level coastal cities), Russia 2018 (Moscow ~150m, St. Petersburg sea level), Qatar 2022 (Doha ~5m) — are all sub-threshold. So `altitude_native_*` is always 0 for those matches, and the backtest numbers are essentially identical to v2 Items 1+3 alone:

| WC | v2 Items 1+3 | v2 Items 1+2+3 | Δ |
|---|---|---|---|
| 2014 | 0.8991 | 0.8991 | 0 |
| 2018 | 0.9292 | 0.9292 | 0 |
| 2022 | 1.1120 | 1.1120 | 0 |

Tiny variations would come from regularization re-weighting other coefficients, but in practice it rounds to zero. This is the cleanest indication that the altitude feature is doing what we want — adding signal at altitude venues without disturbing the model elsewhere.

### 41. Known limitations of v2 Item 2 (carried into v3)
- **Knockout cache is venue-blind.** All knockouts in the simulator use a sea-level US venue. If Mexico's R32/R16 actually gets scheduled in Mexico City, our model underestimates Mexico's knockout chance by a small amount (~2.8% λ boost they'd actually get). Fixing it means doubling the lam_cache (one entry per (team_a, team_b, venue_altitude_band)) — small effort, very small payoff.
- **Binary native flag misses partial exposure.** Argentinian teams that play some matches at Salta (1187m) or Jujuy (1240m) get exactly 0, even though Northern Argentinian players have some altitude exposure. Realistically partial exposure is too small an effect to matter; a graded version would just add tuning noise.
- **Hand-coded city → elevation table.** Could be derived from a geocoding service or GeoNames database for completeness, but the current table covers every match in `results.csv` where altitude ≥ 1500m. Lower-altitude misses default to 0, which is the correct answer.
- **No interaction with humidity / heat.** Mexico City is high-altitude but cool/dry. Brasília is moderate-altitude (1170m, sub-threshold here) but tropical. The model lumps both into a single altitude_native term. For the 2030 WC bid that includes Saudi Arabia (low altitude, extreme heat), this gap will matter.
- **Doesn't catch altitude-native teams that didn't qualify.** Peru (sometimes plays in Cusco / Arequipa) and Chile (occasional Andean matches) have minor altitude exposure but don't have the lifelong-adaptation profile, so they're correctly out. But if a hypothetical Bolivia / Ecuador qualifier path produced a CONMEBOL altitude-native team meeting Mexico at Azteca, the model would see a clean signal.

---

## v2 Phase 1 Refinements

Targeted polish on items 1 and 3 of Phase 1 after the initial implementation surfaced specific gaps. Two refinements shipped; a third (per-host learned home-advantage coefficients) is deferred as it's a larger feature project.

### 42. Tiny invariant checker (`tests/check_v2_invariants.py`)
**Motivation:** after the v2 work spread across six modules and 20k-sim Monte Carlo output, "the numbers might be silently off" became a real worry. Writing full pytest infrastructure was overkill; eyeballing every metric is unreliable.
**What it does:** 44–47 explicit assertions (depending on `--backtest` flag) checking the end-to-end invariants that would actually reveal a broken model:
- Per-match W/D/L probabilities sum to 1.0 ± 1e-3
- Per-team tournament probabilities monotone non-increasing across rounds
- Σ p_win_wc across 48 teams ≈ 1.0 (exactly one champion per sim)
- Σ p_advance ≈ 32 (exactly 32 teams advance from groups)
- Top 5 includes Spain and Argentina (any "completely off" run drops them)
- Mexico in FIFA Group A, Canada in B, US in C (catches the inverted Group E/F bug shape recurring)
- Backtest WC 2014/18/22 log-loss in [0.80, 1.20] (rough sanity bounds)
- All four feature lookups (host_advantage, altitude_native_advantage, 3rd-place bipartite matching, bracket eligibility) match hand-verified canonical answers

**Caught one bug pre-shipping:** the checker found my CAF↔AFC adjacency change hadn't actually landed (a path typo in the first edit attempt). Without the checker, the change would've silently no-op'd.

**Wall time:** ~2 seconds on saved CSVs; +~15 seconds with `--backtest` to re-run the 3-WC suite.

### 43. CAF ↔ AFC adjacency added (0.3 cross-conf bonus)
**Why:** the Qatar 2022 backtest regressed under v2 Phase 1 Item 1 alone (log-loss 1.06 → 1.115). Diagnosis was that Morocco, Tunisia, Egypt — CAF teams within a few hours' flight of Qatar — were scored as fully neutral (`host_advantage = 0.0`) in the AFC-hosted tournament. The graded host advantage was missing the Middle East / North Africa proximity that v1 implicitly captured via `neutral=False` only for Qatar's home matches.
**Implementation:** `confederations.py` now has a `_CROSS_CONF_ADJACENCIES` tuple of frozensets (CONMEBOL↔CONCACAF, CAF↔AFC) instead of just the Americas adjacency. One line of structural change.
**Backtest impact:** Qatar 2022 log-loss improved from 1.1120 → 1.1001 (−0.012). Still worse than v1's 1.06, but moving in the right direction. WC 2014 unchanged (Brazil host, no CAF/AFC matches relevant). WC 2018 unchanged for the same reason.
**WC 2026 impact:** none directly — CONCACAF host, no CAF/AFC adjacency fires. Numerical shifts in the headline are < 0.5pt and traceable to coefficient redistribution from the extended training-data feature space.

### 44. Per-venue knockout lambda cache (4 caches instead of 1)
**Why:** v2 Item 2 (altitude native advantage) added a real feature, but the simulator's `precompute_knockout_lambdas` cached everything as if at a US sea-level venue. So Mexico's potential knockout matches at Azteca got *zero* altitude boost in the simulation — exactly the case where the new feature should matter most.
**FIFA-published venue assignments (Wikipedia knockout-stage page):** matches 79 (R32) and 92 (R16) are at Mexico City. Matches 83, 85, 96 are in Canada (Toronto/Vancouver). Match 75 is at Guadalupe (Monterrey area, Mexico, sea level). All other knockouts including the Final (Match 104) are at US venues.
**Implementation:** new `KNOCKOUT_VENUES` dict in `bracket.py` maps each FIFA match number to (city, country). The simulator's `precompute_knockout_caches()` now builds one cache per distinct venue *configuration* — 4 for WC 2026:
- `("United States", None)` — sea-level US, modal venue
- `("Canada", None)` — Canada (host_advantage 1.0 for Canada, 0.7 for US/Mexico)
- `("Mexico", None)` — Guadalupe (host_advantage 1.0 for Mexico, no altitude)
- `("Mexico", "Mexico City")` — Azteca (host_advantage 1.0 for Mexico AND altitude_native 1.0)

`sample_knockout_winner` now takes a `match_num` and routes through `_cache_key_for_match(match_num)` to pick the right cache. Cache total: 9,024 entries (still < 1 MB), built once at sim start. Runtime impact: ~+2 sec startup, ~50 sec → ~70 sec for 20k full sims.

**WC 2026 impact: this is the biggest single-feature change in the v2 refinement pass.** Mexico's P(win WC) jumped from 3.3% → **4.4%** (+1.1pt). Their p_reach_qf went from 30.9% → 44.3% (+13.4pt) — biggest of any team. This is the model finally recognizing that Mexico's R32 and R16 matches are at Azteca, where they get both 1.0 host advantage and 1.0 altitude native. Compounded across two rounds.

| Team | Pre-refinements (Items 1+2+3) | Post-refinements (CAF↔AFC + per-venue) | Δ |
|---|---|---|---|
| Argentina | 17.1% | 17.2% | +0.1 |
| Spain | 16.7% | 16.8% | +0.1 |
| England | 13.3% | 11.5% | −1.8 |
| France | 9.0% | 9.3% | +0.3 |
| Brazil | 5.4% | 5.8% | +0.4 |
| Portugal | 5.5% | 4.9% | −0.6 |
| **Mexico** | **3.0%** | **4.4%** | **+1.4** |
| Morocco | 3.2% | 3.5% | +0.3 |

England's −1.8pt is partly Monte Carlo noise, partly the cumulative effect of Mexico's R32/R16 strength (England is in Group L, on the same bracket half as Mexico via match 92's downstream feeder paths). The model now treats Mexico as a credible R16 winner more often, which means more often it's an England knockout opponent in QF.

### 45. Per-host learned home-advantage coefficients — deferred
**Motivation noted:** issues #29 originally listed per-host magnitude differentiation (Mexico's Azteca vs. US's neutral-ish home crowd) as a v2 follow-up. Currently `host_advantage` is a single feature with one learned coefficient — Mexico at home and Liechtenstein at home get the same magnitude.
**Why deferred:** the cleanest implementations (categorical-interaction feature, precomputed per-country home strength) are real feature engineering, not refinements. Combined with the design choice of *which* approach to use, it's a session-sized piece of work that overlaps in spirit with the v2 Phase 2 (player-aware) modeling. The current model is conservative — it underestimates strong-home magnitudes — but in a way that's biased toward humility (doesn't over-claim).
**Status:** parked as a v2 Phase 1.5 candidate or absorbed into Phase 2's broader feature push.

---

# v2 — Phase 2.1: Lineup-aware starting-XI value

Thin-slice opening of the broader player-/lineup-aware modeling phase. The motivation comes from the diagnostics' worst-loss matches: the Cameroon-Brazil 2022 upset (Brazil 1-0 by Cameroon in a dead rubber where Brazil rested most starters) had `prob_home_win = 1.6%`, contributing a log-loss of 4.11 — the single most damaging match in the entire WC 2022 backtest. The model had no way to see that Brazil was fielding a B-team.

### 46. StatsBomb open-data lineups loader (`src/data/lineups_loader.py`)
**Source:** `github.com/statsbomb/open-data` — free, public, with curated lineup + event data. Audit of their `competitions.json` revealed coverage for these men's international tournaments:
- FIFA World Cup 2018, 2022 (full 64 matches each)
- UEFA Euro 2020 (played 2021), Euro 2024
- Copa America 2024
- Africa Cup of Nations 2023 (played early 2024)

**Notable gap:** WC 2014 isn't in StatsBomb's open data. So our WC 2014 backtest will see NaN lineup_value for all 64 matches and rely entirely on the imputer.

**Fetched 314 matches total**, 6,908 starter rows (every match has exactly 11 starters per side, per StatsBomb's `start_reason: "Starting XI"` indicator). One CSV at `data/raw/statsbomb_lineups.csv`.

### 47. Lineup → market-value pipeline (`src/features/lineup_values.py`)
**Approach:** for each starter, match StatsBomb's `player_name` (and `player_nickname`) against Transfermarkt's `players.csv` via normalized-name index, then fuzzy-match fallback. Re-uses the helpers built for `squad_values.py` (`_normalize`, `_build_tm_name_index`, `_player_value_at`).

**Matching coverage:** 11/11 starters matched on 263 sides (42%), 10/11 on 124 sides (20%), 9/11 on 76 (12%). Combined ≥80% match rate on 463 of 628 sides (74%). Failures concentrate around very-recent youth call-ups not yet in the Transfermarkt mirror, and rare names where neither direct nor fuzzy match clears the 0.85 threshold.

**Output:** `data/processed/lineup_values.csv` with one row per (match_date, home_team, away_team, side) → `lineup_value_eur`, `n_starters_matched`, `n_starters_total`. 628 rows total (314 matches × 2 sides).

**Performance note:** the fuzzy-matching loop is slow (~4 minutes wall-time per build) because `SequenceMatcher.ratio()` is O(N*M) against ~47k TM player candidates. A future refactor could use `rapidfuzz` or a name-length-bucketed index to cut this to seconds. Not a blocker — only runs when re-deriving the feature.

### 48. Cameroon-Brazil 2022 — the proof case for the feature
The diagnostic value table directly shows what we wanted to capture:

| Team | Full squad value (`squad_value`) | Starting XI value (`lineup_value`) | XI/Squad ratio |
|---|---|---|---|
| Brazil | €1,003M | €425M | **42%** |
| Cameroon | €108M | €57M | 53% |

**Brazil rested 58% of their squad value for the dead-rubber match.** With only `squad_value` available, the v1 model gave Brazil 93.2% to win — the biggest single-match miss in any backtest. With `lineup_value` available, the model can in principle see Brazil's reduced lineup strength.

### 49. New model coefficients (after retraining on the expanded feature set)
The two new features are added to `NUMERIC_FEATURES_IMPUTED` (sparse — only ~250 rows have non-null lineup_value among 32,000 training rows). Their learned coefficients on the home-goals model:

| Feature | Coefficient (standardized) |
|---|---|
| `host_advantage_home` | +0.040 |
| `altitude_native_home` | +0.028 |
| `home_squad_value` | +0.014 |
| **`lineup_value_home`** | **+0.0041** |
| `lineup_value_away` | +0.0008 |

The lineup_value coefficient is small but positive in the right direction. It's small because:
1. Only ~0.5% of training rows have non-null lineup_value (StatsBomb covers ~250 of 49k internationals).
2. Highly correlated with `squad_value` — both capture team strength, the regression's L2 regularization splits weight between them.
3. The away coefficient is essentially zero — even less signal in the away direction, because StatsBomb-covered matches over-represent strong home teams (Euros + Copa hosts).

### 50. Backtest impact
| WC | v2 Phase 1 + refinements | v2 Phase 2.1 (with lineup_value) | Δ |
|---|---|---|---|
| 2014 | 0.8992 | 0.8992 | 0 (no StatsBomb coverage) |
| 2018 | 0.9284 | 0.9284 | 0 (StatsBomb only has WC 2018 itself, which is the held-out test set) |
| 2022 | 1.1001 | **1.0914** | **−0.0087** |

**The WC 2022 improvement is real and traceable.** StatsBomb's WC 2018 + Euro 2020 lineups are in the WC 2022 training fold, so the model fits a small `lineup_value` coefficient. At test time, the model uses each WC 2022 match's actual lineup_value to refine predictions — particularly for matches where lineup strength diverges meaningfully from full squad strength (like the Brazil dead rubber).

### 51. WC 2026 forecast: essentially unchanged
| Team | Pre-Phase 2 | Post-Phase 2.1 | Δ |
|---|---|---|---|
| Argentina | 17.2% | 17.6% | +0.4 |
| Spain | 16.8% | 16.3% | −0.5 |
| England | 11.5% | 11.3% | −0.2 |
| France | 9.3% | 8.7% | −0.6 |
| Mexico | 4.4% | 4.6% | +0.2 |

**All shifts are within Monte Carlo noise (~0.5pt at 20k sims).** This is expected: we don't have StatsBomb lineups for WC 2026 (not in their public coverage). Every WC 2026 match gets `lineup_value_home/away = NaN`, the model's `SimpleImputer` fills with the training-set median, the `StandardScaler` standardizes that to 0, so `lineup_value` contributes 0 to every WC 2026 prediction.

**This is the central limitation of the thin slice:** we've added a feature that improves the backtest but doesn't (yet) improve the WC 2026 forecast. To unlock the WC 2026 benefit we need lineup data for the WC 2026 matches — either by scraping the 12 played matches' published lineups, or by predicting lineups for unplayed matches.

### 52. Phase 2.1 limitations and Phase 2.2 directions
- **Coverage gap is the biggest weakness.** StatsBomb's open data covers ~300 of 49k training matches (~0.6%). To make the lineup feature meaningfully improve training-time fit, we'd want 10x more coverage — qualifiers, friendlies, U-21 internationals. Sources: Wikipedia per-match pages, FBRef international section, scrapy on national team sites.
- **No WC 2026 lineup data.** Without it, the feature is a backtest-only improvement. Phase 2.2 should add a lineup-prediction step (modal starters from each team's last N matches) so WC 2026 predictions benefit.
- **Aggregation is simplistic.** Just summing 11 player market values treats all positions as fungible. A goalkeeper's market value impacts goals-conceded more than goals-scored. Position-weighted aggregation is a natural Phase 2.2 enhancement.
- **No per-player ratings.** Market value is a noisy proxy for player skill (it lags performance, biases toward young/Western players). Pulling FBRef per-player ratings or computing club-football-based Elo-per-player is the headline Phase 2.2 work.
- **Fuzzy-matching performance.** ~4 min wall time isn't an issue at v2.1 scale but will balloon if we expand coverage 10x. Switch to `rapidfuzz` or a length-bucketed index when that happens.

## Big-picture summary (v2 Phase 2.1 — thin slice)

- The lineup-aware feature **provably helps WC 2022 backtest** (1.0914 vs. 1.1001 baseline, −0.009), and the proof case is exactly the Cameroon-Brazil 2022 match we used as motivation.
- The feature **doesn't yet help WC 2026** because we have no lineup data for those matches. The expected v2.2 work — modal-starter prediction + maybe partial scraping for played matches — would unlock it.
- The coefficient magnitude (+0.004) is much smaller than `host_advantage` (+0.04) or `altitude_native` (+0.028), reflecting sparse coverage. More data → larger coefficient → bigger improvement.
- **No other backtest changed** because StatsBomb data doesn't reach the WC 2014 or WC 2018 training/test folds. Cleanly isolated effect — the feature isn't disturbing anything where it shouldn't.

## Big-picture summary (v2 Phase 1 complete — Items 1 + 2 + 3)

- **Item 1 (multi-host advantage)** added a graded `host_advantage` feature replacing v1's binary `neutral`. The biggest behavioral change of v2: shifts probability mass toward Americas teams in a US/Canada/Mexico-hosted WC.
- **Item 2 (altitude native advantage)** added an `altitude_native` binary flag for teams with lifelong altitude exposure facing a venue at altitude. Theoretically the right framing (validated by the McSharry 2007 BMJ study showing FIFA's 2-week acclimation rule equalizes the visitor disadvantage), but only fires on 5 of 72 WC 2026 fixtures so the magnitude is small.
- **Item 3 (FIFA bracket)** replaced the random-pairing knockout shortcut with FIFA's published bracket tree + 3rd-place eligibility matrix. The biggest *headline* change: Argentina overtakes Spain as #1 because their FIFA bracket path turns out to be more favorable than random pairing assumed; France/Germany fall on tougher paths.
- The three items compound in directions that match domain intuition: Americas hosts → Americas teams stronger → Argentina up; FIFA bracket → individual teams routed through specific paths → Argentina up further, France/Germany down; altitude → marginal Mexico/Colombia boost in select group games.
- **Backtest summary for full v2 Phase 1:** WC 2014 0.93 → 0.899 (−0.031), WC 2018 0.97 → 0.929 (−0.041), WC 2022 1.06 → 1.112 (+0.052). Net is mildly positive (≈ −0.007 averaged). The Qatar 2022 regression is the single ugly number, traceable to the missing CAF↔AFC adjacency in Item 1's graded host scheme.
- **Live WC 2026 eval (n=12 played matches):** 1.099 log-loss vs v1's 1.00. Within noise band, but a real warning if it doesn't recover after more matches are played. Possible explanation: v2's coefficient changes make the model less confident on top European favorites (which is good for upset-heavy WC) but doesn't capture upsets enough to compensate.
- **What's left for v2 Phase 2:** player-/lineup-level modeling. The largest remaining opportunity, and where issues.md #8 says the genuine information lives.
