"""Elo ratings for national teams.

We compute Elo over the FULL match history (1872+), even though we only
train the prediction model on 1990+ matches. The reason is warm-up: Elo
ratings need years of results to converge to realistic values. By the
time we hit 1990, every relevant team has a rating that reflects decades
of actual football.

The output that downstream code cares about is `elo_pre` for each match:
the rating each team had *just before* that match kicked off. Those are
the values fed into the Poisson model as features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_results, PROJECT_ROOT
from src.features.tournaments import classify_tournament


INITIAL_RATING = 1500.0

# Base K-factor — how much one match can move a rating.
# Multiplied by goal-margin and tournament weight below.
# K=30 is the eloratings.net convention for international football.
K_BASE = 30.0

# Importance multiplier by tournament type.
# Friendlies count less because squads are rotated and effort varies;
# World Cup matches count more because they're the highest-stakes games.
TOURNAMENT_WEIGHT = {
    "friendly": 0.5,
    "qualifier": 1.0,
    "continental": 1.25,
    "world_cup": 2.0,
    "other": 0.75,
}

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability-like 'expected score' for team A vs team B.

    Returns a value in [0, 1]. 0.5 means perfectly even.
    The 400 scaling factor is conventional — it makes a 400-point
    rating gap correspond to a 10:1 expected-score ratio.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_goals: float, away_goals: float) -> tuple[float, float]:
    """(home_score, away_score) where win=1, draw=0.5, loss=0."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def margin_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-margin multiplier.

    1-goal win → 1.0
    2-goal win → 1.5
    3-goal win → 1.75
    N-goal win (N≥4) → (11 + N) / 8, e.g. 4-goal → 1.875, 5 → 2.0, ...
    """
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    if g == 3:
        return 1.75
    return (11 + g) / 8.0


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk match history chronologically and assign pre/post Elo to every row.

    Returns:
        annotated_matches: copy of `matches` with four new columns:
            home_elo_pre, away_elo_pre  -- rating at the start of the match
            home_elo_post, away_elo_post -- rating after the match's update
        final_ratings: dict mapping team name to its final Elo rating.

    `matches` must contain: date, home_team, away_team, home_score,
    away_score, tournament.
    """
    df = matches.sort_values("date").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    pre_home = [0.0] * len(df)
    pre_away = [0.0] * len(df)
    post_home = [0.0] * len(df)
    post_away = [0.0] * len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        hg, ag = row.home_score, row.away_score
        t_class = classify_tournament(row.tournament)

        r_h = ratings.get(home, INITIAL_RATING)
        r_a = ratings.get(away, INITIAL_RATING)
        pre_home[i] = r_h
        pre_away[i] = r_a

        e_h = expected_score(r_h, r_a)
        e_a = 1.0 - e_h
        s_h, s_a = actual_score(hg, ag)

        k = K_BASE * margin_multiplier(int(hg - ag)) * TOURNAMENT_WEIGHT.get(t_class, 1.0)

        new_r_h = r_h + k * (s_h - e_h)
        new_r_a = r_a + k * (s_a - e_a)

        ratings[home] = new_r_h
        ratings[away] = new_r_a
        post_home[i] = new_r_h
        post_away[i] = new_r_a

    df["home_elo_pre"] = pre_home
    df["away_elo_pre"] = pre_away
    df["home_elo_post"] = post_home
    df["away_elo_post"] = post_away

    return df, ratings


def save_elo_features(df: pd.DataFrame, ratings: dict[str, float]) -> tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "matches_with_elo.csv"
    ratings_path = PROCESSED_DIR / "final_elo.csv"

    df.to_csv(features_path, index=False)

    rating_df = (
        pd.DataFrame({"team": list(ratings.keys()), "elo": list(ratings.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    rating_df.to_csv(ratings_path, index=False)

    return features_path, ratings_path


if __name__ == "__main__":
    # Compute over the full history (no cutoff) so Elo has a proper warm-up.
    # The 1990+ filter applies only when *training* the model, not when
    # computing Elo features.
    full, _ = load_results(apply_cutoff=True)  # still drops WC 2026 + future
    annotated, final_ratings = compute_elo(full)
    features_path, ratings_path = save_elo_features(annotated, final_ratings)

    print(f"processed {len(annotated):,} matches")
    print(f"tracked {len(final_ratings):,} teams")
    print(f"wrote {features_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {ratings_path.relative_to(PROJECT_ROOT)}")
    print()

    top = (
        pd.DataFrame({"team": final_ratings.keys(), "elo": final_ratings.values()})
        .sort_values("elo", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )
    print("=== top 25 by current Elo ===")
    print(top.to_string(index=False))
