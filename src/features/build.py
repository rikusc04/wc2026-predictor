"""Build the final training-ready feature table.

Takes matches_with_elo.csv (one row per match, Elo annotated) and produces
features.csv with:
- elo_diff
- is_neutral
- tournament_class (categorical)
- recent form per team (goals scored/conceded over last N matches)
- days since each team's last match

All rolling stats use `.shift(1)` to ensure we never see the current
match's outcome when computing its features. Leakage = death.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.tournaments import classify_tournament


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ELO_PATH = PROCESSED_DIR / "matches_with_elo.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
SQUAD_VALUES_PATH = PROCESSED_DIR / "squad_values.csv"
GROUP_STANDINGS_PATH = PROCESSED_DIR / "group_standings.csv"

FORM_WINDOW = 10  # matches to look back for recent form

# Map WC year to the kickoff date — the snapshot is "valid" from kickoff onward.
# For each match, we look up the most recent snapshot whose date <= match date.
SQUAD_SNAPSHOT_DATES = {
    2006: pd.Timestamp("2006-06-09"),
    2010: pd.Timestamp("2010-06-11"),
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
    2026: pd.Timestamp("2026-06-11"),
}


def _team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches so each row is one team's perspective on one match.

    Each match contributes two rows: one from the home team's POV
    (goals_for=home_score, goals_against=away_score) and one from the
    away team's POV. This lets us compute "per-team" rolling stats.
    """
    home = matches[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]
    home["is_home_match"] = True

    away = matches[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]
    away["is_home_match"] = False

    return pd.concat([home, away], ignore_index=True)


def _recent_form_table(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute per-(team, date) recent form features.

    Returns a DataFrame with one row per (team, date) and columns:
        form_scored, form_conceded, days_since_last

    For teams that played multiple matches on the same date (rare but
    real), we keep the first-of-day row. That row's `shift(1)`-based
    rolling stat correctly excludes anything from that same date — i.e.
    "state going into today" — which is what the model should see.
    """
    tv = _team_perspective(matches).sort_values(["team", "date"]).reset_index(drop=True)
    tv["date"] = pd.to_datetime(tv["date"])

    # shift(1) excludes the current match from its own rolling stat
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

    form = tv[["team", "date", "form_scored", "form_conceded", "days_since_last"]]
    return form.drop_duplicates(subset=["team", "date"], keep="first").reset_index(drop=True)


def _join_squad_values(features: pd.DataFrame) -> pd.DataFrame:
    """Add home_squad_value and away_squad_value via time-aware merge.

    For each match, looks up the most recent squad-value snapshot ≤ match
    date for each team. Matches before the earliest snapshot (pre-2006)
    get NaN; the downstream imputer handles it.
    """
    if not SQUAD_VALUES_PATH.exists():
        print("  (squad_values.csv not found — skipping squad-value join)")
        return features

    sv = pd.read_csv(SQUAD_VALUES_PATH)
    sv["snapshot_date"] = sv["year"].map(SQUAD_SNAPSHOT_DATES)
    sv = sv[sv["snapshot_date"].notna()].copy()
    sv = sv.sort_values("snapshot_date")

    features = features.sort_values("date").reset_index(drop=True)

    # Home-team join: merge_asof finds the most recent snapshot ≤ match date
    # for each (home_team, date) pair.
    home_sv = (
        sv[["snapshot_date", "team_name", "squad_value_eur"]]
        .rename(columns={"team_name": "home_team", "squad_value_eur": "home_squad_value"})
        .sort_values(["home_team", "snapshot_date"])
    )
    features = pd.merge_asof(
        features.sort_values("date"),
        home_sv.sort_values("snapshot_date"),
        left_on="date", right_on="snapshot_date",
        by="home_team",
        direction="backward",
    ).drop(columns=["snapshot_date"])

    away_sv = (
        sv[["snapshot_date", "team_name", "squad_value_eur"]]
        .rename(columns={"team_name": "away_team", "squad_value_eur": "away_squad_value"})
        .sort_values(["away_team", "snapshot_date"])
    )
    features = pd.merge_asof(
        features.sort_values("date"),
        away_sv.sort_values("snapshot_date"),
        left_on="date", right_on="snapshot_date",
        by="away_team",
        direction="backward",
    ).drop(columns=["snapshot_date"])

    return features


def build_features(matches_with_elo: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Produce the final feature table.

    Input: matches_with_elo.csv contents (every match + pre/post Elo)
    Output: same matches with feature columns added.
    """
    df = matches_with_elo.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["tournament_class"] = df["tournament"].apply(classify_tournament)

    form = _recent_form_table(df, window)

    home_form = form.rename(columns={
        "team": "home_team",
        "form_scored": "home_form_scored",
        "form_conceded": "home_form_conceded",
        "days_since_last": "home_days_since_last",
    })
    away_form = form.rename(columns={
        "team": "away_team",
        "form_scored": "away_form_scored",
        "form_conceded": "away_form_conceded",
        "days_since_last": "away_days_since_last",
    })

    df = df.merge(home_form, on=["home_team", "date"], how="left")
    df = df.merge(away_form, on=["away_team", "date"], how="left")

    df = _join_squad_values(df)
    df = _join_group_standings(df)

    return df


def _join_group_standings(features: pd.DataFrame) -> pd.DataFrame:
    """Add group_round, points-before, and is_dead_rubber columns.

    For non-WC matches and knockouts, the columns default to neutral values:
    is_dead_rubber=False, group_round=0, points-before columns NaN (imputed downstream).
    """
    if not GROUP_STANDINGS_PATH.exists():
        print("  (group_standings.csv not found — skipping dead-rubber join)")
        return features

    gs = pd.read_csv(GROUP_STANDINGS_PATH, parse_dates=["date"])

    features = features.merge(
        gs,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Defaults for non-WC matches and WC knockouts
    features["is_dead_rubber"] = features["is_dead_rubber"].fillna(False).astype(bool)
    features["group_round"] = features["group_round"].fillna(0).astype(int)
    # home_pts_before / away_pts_before stay NaN — the imputer handles them

    return features


def save_features(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    return FEATURES_PATH


if __name__ == "__main__":
    matches = pd.read_csv(ELO_PATH, parse_dates=["date"])
    features = build_features(matches)
    out = save_features(features)
    print(f"wrote {out.relative_to(PROJECT_ROOT)} ({len(features):,} rows)")
    print()

    feature_cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "is_neutral", "tournament_class", "elo_diff",
        "home_elo_pre", "away_elo_pre",
        "home_form_scored", "home_form_conceded", "home_days_since_last",
        "away_form_scored", "away_form_conceded", "away_days_since_last",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]
    # rename for display since results.csv uses "neutral", not "is_neutral"
    if "neutral" in features.columns and "is_neutral" not in features.columns:
        feature_cols = [c if c != "is_neutral" else "neutral" for c in feature_cols]

    print("=== sample feature rows (post-2020) ===")
    recent = features[features["date"] >= "2020-01-01"][feature_cols].head(10)
    print(recent.to_string(index=False))
    print()

    print("=== missing-value summary ===")
    cols_to_check = [
        "home_form_scored", "home_form_conceded", "home_days_since_last",
        "away_form_scored", "away_form_conceded", "away_days_since_last",
        "home_squad_value", "away_squad_value",
        "home_pts_before", "away_pts_before",
    ]
    for c in cols_to_check:
        if c not in features.columns:
            continue
        n_missing = features[c].isna().sum()
        print(f"  {c}: {n_missing:,} missing ({n_missing / len(features):.1%})")

    if "is_dead_rubber" in features.columns:
        n_dr = int(features["is_dead_rubber"].sum())
        print(f"\n  is_dead_rubber=True: {n_dr} matches "
              f"({n_dr / len(features):.2%} of all rows)")
