"""Compute group-stage standings and detect dead-rubber matches.

A "dead rubber" is a group-stage match where the favored team has already
mathematically qualified for the knockout round — typically the third
(final) group match for a team that won its first two games. In those
matches, favorites routinely rest starters and play with less intensity,
making them much more upset-prone than the model would otherwise predict.

This module produces, per WC year:
  - For every group-stage match: round (1/2/3), points each team had
    BEFORE this match, and a boolean `is_dead_rubber` flag.
  - For knockout and non-WC matches: defaults (False/NaN).

The output gets joined into features.csv by date+team in build.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT, load_results


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "group_standings.csv"

DEAD_RUBBER_POINTS_THRESHOLD = 6  # if the leader has >= this many pts, qualified


def identify_groups(wc_matches: pd.DataFrame) -> dict[str, set[str]]:
    """Discover the groups for a single WC.

    Algorithm: a team's group-mates are the opponents in its first 3 WC matches
    (those are guaranteed to be group-stage matches). Build connected components
    of the "played in group stage" graph — each component is a group.

    Returns a dict mapping group label ("A", "B", ...) to the set of teams.
    """
    teams = set(wc_matches["home_team"]).union(set(wc_matches["away_team"]))

    # For each team, find their first 3 WC matches in chronological order
    adj: dict[str, set[str]] = {t: set() for t in teams}
    for team in teams:
        team_matches = wc_matches[
            (wc_matches["home_team"] == team) | (wc_matches["away_team"] == team)
        ].sort_values("date").head(3)
        for _, m in team_matches.iterrows():
            other = m["away_team"] if m["home_team"] == team else m["home_team"]
            adj[team].add(other)
            adj[other].add(team)

    # Connected components via BFS, labeled A, B, C, ...
    groups: dict[str, set[str]] = {}
    seen: set[str] = set()
    group_idx = 0
    for team in sorted(teams):
        if team in seen:
            continue
        group: set[str] = set()
        queue = [team]
        while queue:
            t = queue.pop()
            if t in group:
                continue
            group.add(t)
            queue.extend(adj[t] - group)
        groups[chr(ord("A") + group_idx)] = group
        group_idx += 1
        seen.update(group)

    return groups


def apply_match_to_standings(
    standings: dict[str, dict[str, int]],
    home: str,
    away: str,
    home_goals: int | float,
    away_goals: int | float,
) -> None:
    """Mutate `standings` to reflect one match's result.

    `standings` is the dict-of-dicts shape: {team: {"pts": int, "gd": int, "gf": int}}.
    Skips the update if either score is NaN (unplayed match).
    """
    if pd.isna(home_goals) or pd.isna(away_goals):
        return
    h, a = int(home_goals), int(away_goals)
    if h > a:
        standings[home]["pts"] += 3
    elif h < a:
        standings[away]["pts"] += 3
    else:
        standings[home]["pts"] += 1
        standings[away]["pts"] += 1
    standings[home]["gd"] += h - a
    standings[away]["gd"] += a - h
    standings[home]["gf"] += h
    standings[away]["gf"] += a


def _walk_group_standings(group_matches: pd.DataFrame, group_teams: set[str]) -> list[dict]:
    """For one group's 6 matches, walk chronologically and emit per-match rows
    with points-before and dead-rubber flag.

    Uses the shared `apply_match_to_standings` helper to keep standings logic
    consistent with the prediction-side simulator.
    """
    standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group_teams}
    team_played = {t: 0 for t in group_teams}

    rows = []
    for _, m in group_matches.sort_values("date").iterrows():
        home, away = m["home_team"], m["away_team"]
        hg, ag = m["home_score"], m["away_score"]

        round_num = max(team_played[home], team_played[away]) + 1

        is_dead_rubber = False
        if round_num == 3:
            higher_pts = max(standings[home]["pts"], standings[away]["pts"])
            if higher_pts >= DEAD_RUBBER_POINTS_THRESHOLD:
                is_dead_rubber = True

        rows.append({
            "date": m["date"],
            "home_team": home,
            "away_team": away,
            "group_round": round_num,
            "home_pts_before": standings[home]["pts"],
            "away_pts_before": standings[away]["pts"],
            "is_dead_rubber": is_dead_rubber,
        })

        apply_match_to_standings(standings, home, away, hg, ag)
        team_played[home] += 1
        team_played[away] += 1

    return rows


def compute_group_standings(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute group-round, points-before, and dead-rubber flag for every WC match.

    Output columns:
        date, home_team, away_team, group_round, home_pts_before,
        away_pts_before, is_dead_rubber

    Non-WC matches are not in the output; they'll get defaults during the
    feature join.
    """
    wc_all = results_df[results_df["tournament"] == "FIFA World Cup"].copy()
    wc_all["date"] = pd.to_datetime(wc_all["date"])
    wc_all["year"] = wc_all["date"].dt.year

    all_rows = []
    for year in sorted(wc_all["year"].unique()):
        year_matches = wc_all[wc_all["year"] == year].copy()

        groups = identify_groups(year_matches)
        n_teams = len(set().union(*groups.values()))
        n_group_matches = 6 * len(groups)  # 4 teams × 3 games / 2 per group
        print(f"  WC {year}: {n_teams} teams, {len(groups)} groups, "
              f"{n_group_matches} group matches")

        for group in groups.values():
            group_matches = year_matches[
                year_matches["home_team"].isin(group) &
                year_matches["away_team"].isin(group)
            ]
            all_rows.extend(_walk_group_standings(group_matches, group))

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    print("loading match history...")
    results, _ = load_results(apply_cutoff=True)
    print(f"  {len(results):,} matches loaded")
    print()

    print("computing group standings + dead-rubber flags per WC...")
    standings = compute_group_standings(results)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    standings.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(standings):,} rows)")

    # Sanity check: list the dead rubbers per WC
    print()
    print("=== dead rubbers found ===")
    dr = standings[standings["is_dead_rubber"]].copy()
    dr["date"] = pd.to_datetime(dr["date"]).dt.date
    dr["matchup"] = dr["home_team"] + " vs " + dr["away_team"]
    print(dr[["date", "matchup", "home_pts_before", "away_pts_before"]].to_string(index=False))
