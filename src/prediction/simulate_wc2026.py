"""Monte Carlo tournament simulation for WC 2026.

For each WC 2026 match we already have predicted (lambda_home, lambda_away)
from src.prediction.wc2026. To estimate tournament-level outcomes (who tops
the group, who advances, who wins it all), we:

  1. Build the score matrix for each match — P(home goals = i, away goals = j).
  2. Sample a scoreline for each match by drawing from that matrix.
  3. Compute group standings from the sampled scorelines.
  4. Determine who advances (top 2 per group + 8 best 3rd-place teams).
  5. (Eventually: simulate knockout bracket.)
  6. Repeat 10,000+ times for stable probabilities.

This v1 focuses on group-stage advancement probabilities. Knockout bracket
simulation is a follow-up.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT, load_results
from src.features.group_standings import apply_match_to_standings, identify_groups
from src.models.poisson import score_matrix
from src.prediction.bracket import (
    FINAL_FEEDERS,
    GroupStandings,
    KNOCKOUT_VENUES,
    QF_FEEDERS,
    R16_FEEDERS,
    R32_MATCHUPS,
    SF_FEEDERS,
    build_r32_slot_to_team,
    r32_matchups_resolved,
)
from src.prediction.wc2026 import (
    CUTOFF,
    build_match_features,
    compute_team_state_at_cutoff,
    predict_wc_2026,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIM_OUTPUT_PATH = PROCESSED_DIR / "wc2026_simulation.csv"

N_SIMS_DEFAULT = 20_000

# 48-team format: top 2 per group + 8 best 3rd-place teams advance to R32
N_GROUPS = 12
THIRD_PLACE_QUALIFIERS = 8


def sample_score(lam_h: float, lam_a: float, rho: float, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a (home_goals, away_goals) tuple from the predicted distribution."""
    M = score_matrix(lam_h, lam_a, rho=rho)
    M = M / M.sum()  # renormalize just in case
    flat_idx = rng.choice(M.size, p=M.flatten())
    h, a = np.unravel_index(int(flat_idx), M.shape)
    return int(h), int(a)


def simulate_group(
    group_teams: set[str],
    group_matches: pd.DataFrame,
    rho: float,
    rng: np.random.Generator,
) -> list[tuple[str, int, int, int]]:
    """Simulate one group's 6 matches. Returns ranked teams as
    [(team, points, goal_diff, goals_for), ...] in finishing order.
    """
    standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group_teams}

    for _, m in group_matches.iterrows():
        h, a = sample_score(m["expected_goals_home"], m["expected_goals_away"], rho, rng)
        apply_match_to_standings(standings, m["home"], m["away"], h, a)

    # Rank by points, then goal difference, then goals scored, then random tiebreak
    ranked = sorted(
        standings.items(),
        key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"], rng.random()),
    )
    return [(t, s["pts"], s["gd"], s["gf"]) for t, s in ranked]


def simulate_tournament(
    predictions_df: pd.DataFrame,
    rho: float,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the Monte Carlo simulation. Returns a per-team table with
    probabilities of various outcomes.
    """
    rng = np.random.default_rng(seed)

    # Use the played matches column directly: where actual_score is set,
    # treat the simulated result as the actual result (so we're conditioning
    # on what's already known).
    pred = predictions_df.copy()
    pred["played"] = pred["actual_score"] != ""
    pred["actual_home_goals"] = np.nan
    pred["actual_away_goals"] = np.nan
    for i, row in pred[pred["played"]].iterrows():
        h, a = row["actual_score"].split("-")
        pred.loc[i, "actual_home_goals"] = int(h)
        pred.loc[i, "actual_away_goals"] = int(a)

    results_df, _ = load_results(apply_cutoff=False)
    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26_results = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= CUTOFF)
    ].copy()

    groups = identify_groups(wc26_results)
    print(f"identified {len(groups)} groups:")
    for label, teams in groups.items():
        print(f"  Group {label}: {', '.join(sorted(teams))}")

    # Map every WC 2026 match to its group
    match_to_group = {}
    for label, teams in groups.items():
        group_matches = pred[
            pred["home"].isin(teams) & pred["away"].isin(teams)
        ]
        for idx in group_matches.index:
            match_to_group[idx] = label

    # Per-team counters
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sim in range(n_sims):
        # For each group, simulate matches (or use actual results where played)
        all_third_places: list[tuple[str, int, int, int]] = []
        per_group_top2: dict[str, list[str]] = {}

        for label, teams in groups.items():
            group_matches = pred[
                pred["home"].isin(teams) & pred["away"].isin(teams)
            ]

            # For played matches, use the actual score; for unplayed, sample
            standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
            for _, m in group_matches.iterrows():
                if m["played"]:
                    h, a = int(m["actual_home_goals"]), int(m["actual_away_goals"])
                else:
                    h, a = sample_score(
                        m["expected_goals_home"], m["expected_goals_away"], rho, rng
                    )
                apply_match_to_standings(standings, m["home"], m["away"], h, a)

            ranked = sorted(
                standings.items(),
                key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"], rng.random()),
            )

            for pos, (team, s) in enumerate(ranked):
                if pos == 0:
                    counts[team]["group_1st"] += 1
                    counts[team]["advance"] += 1
                elif pos == 1:
                    counts[team]["group_2nd"] += 1
                    counts[team]["advance"] += 1
                elif pos == 2:
                    counts[team]["group_3rd"] += 1
                    all_third_places.append((team, s["pts"], s["gd"], s["gf"]))
                else:
                    counts[team]["group_4th"] += 1

            per_group_top2[label] = [ranked[0][0], ranked[1][0]]

        # Rank the 12 3rd-place teams; top 8 advance
        all_third_places.sort(key=lambda x: (-x[1], -x[2], -x[3], rng.random()))
        for t, _, _, _ in all_third_places[:THIRD_PLACE_QUALIFIERS]:
            counts[t]["advance"] += 1   # they were already counted in group_3rd
            counts[t]["third_place_advance"] += 1

    # Build the report DataFrame
    teams = set().union(*groups.values())
    rows = []
    for team in sorted(teams):
        rows.append({
            "team": team,
            "p_group_1st": counts[team]["group_1st"] / n_sims,
            "p_group_2nd": counts[team]["group_2nd"] / n_sims,
            "p_group_3rd": counts[team]["group_3rd"] / n_sims,
            "p_advance": counts[team]["advance"] / n_sims,
            "p_eliminated_in_groups": 1 - counts[team]["advance"] / n_sims,
        })
    return pd.DataFrame(rows)


def precompute_knockout_lambdas_for_venue(
    teams: list[str],
    state: dict,
    models,
    venue_country: str,
    venue_city: str | None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Predict (λ_home, λ_away) for every directed (home, away) pair at one venue.

    Returns dict keyed by (home, away). Used by `precompute_knockout_caches`
    to build one cache per distinct (venue_country, venue_city) tuple that
    appears in FIFA's knockout schedule.
    """
    rows = []
    pairs = []
    dummy_date = pd.Timestamp("2026-07-01")
    for h in teams:
        for a in teams:
            if h == a:
                continue
            rows.append(build_match_features(
                home_team=h, away_team=a,
                match_date=dummy_date,
                match_country=venue_country,
                match_city=venue_city,
                state=state,
                days_since_override=4.0,
            ))
            pairs.append((h, a))
    feats_df = pd.DataFrame(rows)
    lam_h, lam_a = models.predict(feats_df)
    return {(h, a): (float(lh), float(la)) for (h, a), lh, la in zip(pairs, lam_h, lam_a)}


def precompute_knockout_caches(
    teams: list[str], state: dict, models,
) -> dict[tuple[str, str | None], dict[tuple[str, str], tuple[float, float]]]:
    """Build one (home, away)→(λh, λa) cache per distinct knockout venue.

    v2 refinement: replaces the single "all knockouts assumed at US sea-level"
    cache with one cache per distinct (country, city-or-None) pair in
    `KNOCKOUT_VENUES`. WC 2026 has 4 distinct venue configurations:
        (United States, None)   — most R32/R16/QF/SF + Final, sea level
        (Canada, None)          — matches 83, 85, 96, sea level
        (Mexico, None)          — match 75 (Guadalupe area), sea level
        (Mexico, Mexico City)   — matches 79, 92, altitude (~2240m)

    The cache key uses None for sub-altitude-threshold cities to keep the
    key space small; only "Mexico City" actually matters for the altitude
    feature. (Zapopan at 1560m is borderline — a group-stage venue, not a
    knockout venue, so it isn't relevant here.)
    """
    distinct_venues: set[tuple[str, str | None]] = set()
    for _, (city, country) in KNOCKOUT_VENUES.items():
        # Only cities that trigger the altitude feature matter for keying;
        # everything else collapses to (country, None).
        from src.features.altitude import HIGH_ALTITUDE_CITIES, ALTITUDE_THRESHOLD
        if HIGH_ALTITUDE_CITIES.get(city, 0) >= ALTITUDE_THRESHOLD:
            distinct_venues.add((country, city))
        else:
            distinct_venues.add((country, None))

    caches = {}
    for country, city in sorted(distinct_venues, key=lambda x: (x[0], x[1] or "")):
        caches[(country, city)] = precompute_knockout_lambdas_for_venue(
            teams, state, models, country, city,
        )
    return caches


def _cache_key_for_match(match_num: int) -> tuple[str, str | None]:
    """Return the cache key (country, city-or-None) for a knockout match."""
    city, country = KNOCKOUT_VENUES[match_num]
    from src.features.altitude import HIGH_ALTITUDE_CITIES, ALTITUDE_THRESHOLD
    if HIGH_ALTITUDE_CITIES.get(city, 0) >= ALTITUDE_THRESHOLD:
        return (country, city)
    return (country, None)


def sample_knockout_winner(
    match_num: int,
    home: str,
    away: str,
    lam_caches: dict[tuple[str, str | None], dict],
    rho: float,
    rng,
) -> str:
    """Sample one knockout match's winner using the right per-venue cache.

    `lam_caches` maps (country, city|None) → {(h, a): (λh, λa)}. We look up
    the venue for this specific FIFA match number and use the appropriate
    cache. Draws go to penalty shootout (50/50, per issues.md #22).
    """
    key = _cache_key_for_match(match_num)
    lh, la = lam_caches[key][(home, away)]
    h, a = sample_score(lh, la, rho, rng)
    if h > a:
        return home
    if a > h:
        return away
    # Penalty shootout: research shows shootouts are essentially random
    return home if rng.random() < 0.5 else away


def derive_fifa_group_labels(wc26_matches: pd.DataFrame) -> dict[str, str]:
    """Map identify_groups()'s alphabetical auto-letters to FIFA's official A→L
    sequence by chronological order of each group's first match.

    `identify_groups` labels groups A, B, C, ... by alphabetical BFS-traversal
    of team names (whichever team is alphabetically first overall ends up in
    its group being relabeled "A"). FIFA's official labels follow tournament
    scheduling: Group A plays the first match (host country, traditionally),
    Group B plays the second, and so on through Group L.

    Returns: {auto_letter: fifa_letter}, e.g., {"E": "A", "A": "B", ...}.
    """
    auto_groups = identify_groups(wc26_matches)
    team_to_auto = {t: letter for letter, teams in auto_groups.items() for t in teams}

    # kind="stable" preserves original CSV row order within same-date ties.
    # Within a day, the row order in results.csv reflects FIFA's match-number
    # schedule, which is exactly what we want to use for A→L labeling.
    sorted_matches = wc26_matches.sort_values("date", kind="stable").reset_index(drop=True)
    fifa_order: list[str] = []
    for _, m in sorted_matches.iterrows():
        auto_letter = team_to_auto.get(m["home_team"])
        if auto_letter is not None and auto_letter not in fifa_order:
            fifa_order.append(auto_letter)
        if len(fifa_order) == 12:
            break

    fifa_letters = list("ABCDEFGHIJKL")
    return dict(zip(fifa_order, fifa_letters))


def rank_third_place_qualifiers(
    third_place_pool: list[tuple[str, int, int, int, str]],
    n_qualifiers: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Pick the top-N 3rd-placed teams across all groups (FIFA tiebreakers).

    Each item in the pool is (team, pts, gd, gf, fifa_group_letter).
    Returns: list of (team, group_letter) for the qualifiers in rank order.
    """
    sorted_pool = sorted(
        third_place_pool,
        key=lambda x: (-x[1], -x[2], -x[3], rng.random()),
    )
    return [(t, g) for t, _, _, _, g in sorted_pool[:n_qualifiers]]


def simulate_full_tournament(
    predictions_df: pd.DataFrame,
    models,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """Full Monte Carlo: group stage + 5 knockout rounds.

    Returns a per-team DataFrame with p_advance, p_reach_r16, p_reach_qf,
    p_reach_sf, p_reach_final, p_win_wc.
    """
    rng = np.random.default_rng(seed)
    rho = models.rho

    # Identify groups and relabel A-L to FIFA's official sequence (host
    # nation in Group A, then by chronological order of opening matches).
    results_df, _ = load_results(apply_cutoff=False)
    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26_results = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= CUTOFF)
    ].copy()
    auto_groups = identify_groups(wc26_results)
    auto_to_fifa = derive_fifa_group_labels(wc26_results)
    groups = {auto_to_fifa[auto]: teams for auto, teams in auto_groups.items()}
    print("  FIFA group assignments:")
    for fifa_letter in "ABCDEFGHIJKL":
        teams = groups[fifa_letter]
        print(f"    Group {fifa_letter}: {', '.join(sorted(teams))}")
    team_to_group = {team: label for label, teams in groups.items() for team in teams}
    all_teams = sorted(set().union(*groups.values()))

    # Pre-cache knockout match predictions PER VENUE. WC 2026 has 4 distinct
    # knockout venue configurations (US sea-level, Canada sea-level, Mexico
    # sea-level, Mexico City altitude); each gets its own 2,256-entry cache.
    print("  pre-caching knockout lambdas per venue...")
    state = compute_team_state_at_cutoff()
    lam_caches = precompute_knockout_caches(all_teams, state, models)
    print(f"  built {len(lam_caches)} venue caches × {len(next(iter(lam_caches.values()))):,} pairs each:")
    for (country, city), c in lam_caches.items():
        venue_desc = f"{city}, {country}" if city else f"{country} (sea level)"
        print(f"    {venue_desc}: {len(c):,} entries")

    # Annotate played matches
    pred = predictions_df.copy()
    pred["played"] = pred["actual_score"] != ""
    pred["actual_home_goals"] = np.nan
    pred["actual_away_goals"] = np.nan
    for i, row in pred[pred["played"]].iterrows():
        h, a = row["actual_score"].split("-")
        pred.loc[i, "actual_home_goals"] = int(h)
        pred.loc[i, "actual_away_goals"] = int(a)

    # Per-team counters
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sim in range(n_sims):
        if (sim + 1) % 2000 == 0:
            print(f"  simulation {sim + 1:,} / {n_sims:,}")

        # 1. Group stage
        group_standings_map: dict[str, GroupStandings] = {}
        third_place_pool: list[tuple[str, int, int, int, str]] = []

        for label, teams in groups.items():
            group_matches = pred[pred["home"].isin(teams) & pred["away"].isin(teams)]
            standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
            for _, m in group_matches.iterrows():
                if m["played"]:
                    h, a = int(m["actual_home_goals"]), int(m["actual_away_goals"])
                else:
                    h, a = sample_score(
                        m["expected_goals_home"], m["expected_goals_away"], rho, rng
                    )
                apply_match_to_standings(standings, m["home"], m["away"], h, a)
            ranked = sorted(
                standings.items(),
                key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"], rng.random()),
            )
            ranked_tuples = [(t, s["pts"], s["gd"], s["gf"]) for t, s in ranked]
            group_standings_map[label] = GroupStandings(label=label, ranked=ranked_tuples)
            for pos, (t, p, gd, gf) in enumerate(ranked_tuples):
                if pos == 0:
                    counts[t]["group_1st"] += 1
                    counts[t]["advance"] += 1
                elif pos == 1:
                    counts[t]["group_2nd"] += 1
                    counts[t]["advance"] += 1
                elif pos == 2:
                    counts[t]["group_3rd"] += 1
                    third_place_pool.append((t, p, gd, gf, label))
                else:
                    counts[t]["group_4th"] += 1

        # 2. Best-3rd-place tiebreaker → 8 qualify
        qualifying_thirds = rank_third_place_qualifiers(
            third_place_pool, THIRD_PLACE_QUALIFIERS, rng,
        )
        qualifier_groups = [g for _, g in qualifying_thirds]
        for t, _ in qualifying_thirds:
            counts[t]["advance"] += 1

        # 3. R32 — build the FIFA bracket
        slot_to_team = build_r32_slot_to_team(group_standings_map, qualifier_groups)
        r32_resolved = r32_matchups_resolved(slot_to_team)
        for _, ta, tb in r32_resolved:
            counts[ta]["reach_r32"] += 1
            counts[tb]["reach_r32"] += 1

        # match_winners is keyed by FIFA match number (73-104). Each round
        # we read feeders from the bracket and write winners.
        match_winners: dict[int, str] = {}
        for match_num, ta, tb in r32_resolved:
            match_winners[match_num] = sample_knockout_winner(match_num, ta, tb, lam_caches, rho, rng)
            counts[match_winners[match_num]]["reach_r16"] += 1

        # 4. R16 → QF (matches 89-96)
        for match_num, (feed_a, feed_b) in R16_FEEDERS.items():
            ta, tb = match_winners[feed_a], match_winners[feed_b]
            match_winners[match_num] = sample_knockout_winner(match_num, ta, tb, lam_caches, rho, rng)
            counts[match_winners[match_num]]["reach_qf"] += 1

        # 5. QF → SF (matches 97-100)
        for match_num, (feed_a, feed_b) in QF_FEEDERS.items():
            ta, tb = match_winners[feed_a], match_winners[feed_b]
            match_winners[match_num] = sample_knockout_winner(match_num, ta, tb, lam_caches, rho, rng)
            counts[match_winners[match_num]]["reach_sf"] += 1

        # 6. SF → Final (matches 101-102)
        for match_num, (feed_a, feed_b) in SF_FEEDERS.items():
            ta, tb = match_winners[feed_a], match_winners[feed_b]
            match_winners[match_num] = sample_knockout_winner(match_num, ta, tb, lam_caches, rho, rng)
            counts[match_winners[match_num]]["reach_final"] += 1

        # 7. Final → Champion (match 104)
        final_a, final_b = FINAL_FEEDERS
        champion = sample_knockout_winner(
            104, match_winners[final_a], match_winners[final_b], lam_caches, rho, rng,
        )
        counts[champion]["win_wc"] += 1

    rows = []
    for team in all_teams:
        rows.append({
            "team": team,
            "p_advance": counts[team]["advance"] / n_sims,
            "p_reach_r16": counts[team]["reach_r16"] / n_sims,
            "p_reach_qf": counts[team]["reach_qf"] / n_sims,
            "p_reach_sf": counts[team]["reach_sf"] / n_sims,
            "p_reach_final": counts[team]["reach_final"] / n_sims,
            "p_win_wc": counts[team]["win_wc"] / n_sims,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("step 1: training model and predicting matches...")
    predictions_df, models = predict_wc_2026()

    print()
    print(f"step 2: running full tournament simulation ({N_SIMS_DEFAULT:,} runs)...")
    sim = simulate_full_tournament(predictions_df, models, n_sims=N_SIMS_DEFAULT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sim.to_csv(SIM_OUTPUT_PATH, index=False)
    print()
    print(f"wrote {SIM_OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(sim)} teams)")

    disp = sim.copy()
    for col in ["p_advance", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_win_wc"]:
        disp[col] = disp[col].apply(lambda x: f"{x:.1%}")

    print()
    print("=== tournament probabilities (sorted by P(win WC)) ===")
    sorted_view = sim.sort_values("p_win_wc", ascending=False)
    disp_sorted = sorted_view.copy()
    for col in ["p_advance", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_win_wc"]:
        disp_sorted[col] = disp_sorted[col].apply(lambda x: f"{x:.1%}")
    print(disp_sorted.to_string(index=False))
