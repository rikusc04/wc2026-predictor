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


def precompute_knockout_lambdas(teams: list[str], state: dict, models) -> dict:
    """Predict (lambda_home, lambda_away) for every possible neutral-venue
    knockout match between WC 2026 participants. Returns dict (home, away) -> (lh, la).

    Knockout matches are all neutral (except theoretically host country at
    home, but we treat all as neutral for simplicity). Features:
      - elo, form, squad value: from team state at cutoff
      - days_since_last: 4 (typical knockout spacing)
      - neutral: True
    """
    rows = []
    pairs = []
    # match_date is unused when days_since_override is set, but we still need
    # a Timestamp to satisfy the signature.
    dummy_date = pd.Timestamp("2026-07-01")
    for h in teams:
        for a in teams:
            if h == a:
                continue
            rows.append(build_match_features(
                home_team=h, away_team=a,
                match_date=dummy_date, neutral=True,
                state=state, days_since_override=4.0,
            ))
            pairs.append((h, a))
    feats_df = pd.DataFrame(rows)
    lam_h, lam_a = models.predict(feats_df)
    cache = {}
    for (h, a), lh, la in zip(pairs, lam_h, lam_a):
        cache[(h, a)] = (float(lh), float(la))
    return cache


def sample_knockout_winner(home: str, away: str, lam_cache: dict, rho: float, rng) -> str:
    """Sample one knockout match's winner. Draws go to penalty shootout (50/50)."""
    lh, la = lam_cache[(home, away)]
    h, a = sample_score(lh, la, rho, rng)
    if h > a:
        return home
    if a > h:
        return away
    # Penalty shootout: research shows shootouts are essentially random
    return home if rng.random() < 0.5 else away


def pair_with_group_avoidance(teams: list[str], team_to_group: dict[str, str], rng) -> list[tuple[str, str]]:
    """Pair 32 teams into 16 matches, trying to avoid same-group meetings in R32.

    Pure constraint satisfaction isn't always possible randomly; we retry
    a few times then fall back to whatever we have.
    """
    for _ in range(20):
        shuffled = teams.copy()
        rng.shuffle(shuffled)
        pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
        same_group = any(team_to_group[h] == team_to_group[a] for h, a in pairs)
        if not same_group:
            return pairs
    # Couldn't avoid all same-group pairings; return last attempt
    return pairs


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

    # Identify groups
    results_df, _ = load_results(apply_cutoff=False)
    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26_results = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= CUTOFF)
    ].copy()
    groups = identify_groups(wc26_results)
    team_to_group = {team: label for label, teams in groups.items() for team in teams}
    all_teams = sorted(set().union(*groups.values()))

    # Pre-cache knockout match predictions (neutral venue)
    print("  pre-caching knockout lambdas (~2,256 matchups, batch predict)...")
    state = compute_team_state_at_cutoff()
    lam_cache = precompute_knockout_lambdas(all_teams, state, models)
    print(f"  cached {len(lam_cache)} (home, away) → (λh, λa) pairs")

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
        group_ranked: dict[str, list[tuple[str, int, int, int]]] = {}
        third_place_pool: list[tuple[str, int, int, int]] = []

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
            group_ranked[label] = ranked_tuples
            for pos, (t, p, gd, gf) in enumerate(ranked_tuples):
                if pos == 0:
                    counts[t]["group_1st"] += 1
                    counts[t]["advance"] += 1
                elif pos == 1:
                    counts[t]["group_2nd"] += 1
                    counts[t]["advance"] += 1
                elif pos == 2:
                    counts[t]["group_3rd"] += 1
                    third_place_pool.append((t, p, gd, gf))
                else:
                    counts[t]["group_4th"] += 1

        # 2. Best-3rd-place tiebreaker → 8 qualify
        third_place_pool.sort(key=lambda x: (-x[1], -x[2], -x[3], rng.random()))
        qualifying_thirds = [t for t, _, _, _ in third_place_pool[:THIRD_PLACE_QUALIFIERS]]
        for t in qualifying_thirds:
            counts[t]["advance"] += 1

        # 3. R32 — build the 32-team bracket
        r32_teams = []
        for ranked in group_ranked.values():
            r32_teams.append(ranked[0][0])
            r32_teams.append(ranked[1][0])
        r32_teams.extend(qualifying_thirds)
        for t in r32_teams:
            counts[t]["reach_r32"] += 1

        # 4. R32 → R16
        r32_pairs = pair_with_group_avoidance(r32_teams, team_to_group, rng)
        r16_teams = [
            sample_knockout_winner(h, a, lam_cache, rho, rng) for h, a in r32_pairs
        ]
        for t in r16_teams:
            counts[t]["reach_r16"] += 1

        # 5. R16 → QF
        rng.shuffle(r16_teams)
        r16_pairs = [(r16_teams[i], r16_teams[i + 1]) for i in range(0, 16, 2)]
        qf_teams = [sample_knockout_winner(h, a, lam_cache, rho, rng) for h, a in r16_pairs]
        for t in qf_teams:
            counts[t]["reach_qf"] += 1

        # 6. QF → SF
        rng.shuffle(qf_teams)
        qf_pairs = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, 8, 2)]
        sf_teams = [sample_knockout_winner(h, a, lam_cache, rho, rng) for h, a in qf_pairs]
        for t in sf_teams:
            counts[t]["reach_sf"] += 1

        # 7. SF → Final
        rng.shuffle(sf_teams)
        sf_pairs = [(sf_teams[i], sf_teams[i + 1]) for i in range(0, 4, 2)]
        final_teams = [sample_knockout_winner(h, a, lam_cache, rho, rng) for h, a in sf_pairs]
        for t in final_teams:
            counts[t]["reach_final"] += 1

        # 8. Final → Champion
        champion = sample_knockout_winner(final_teams[0], final_teams[1], lam_cache, rho, rng)
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
