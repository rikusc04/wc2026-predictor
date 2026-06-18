"""Predict starting XI + lineup value for WC 2026 qualifiers.

v2 Phase 2.2a. Unlocks the lineup_value feature for WC 2026 predictions —
without this, every WC 2026 match got lineup_value=NaN (imputer median →
zero contribution to predictions, per issues #51).

Two paths per team:

1. **Modal-XI path** (used for 37 of 48 WC 2026 qualifiers): pull the team's
   most recent LAST_K matches from StatsBomb data (statsbomb_lineups.csv).
   Count appearances per player across those matches. Top 11 by appearance
   count = predicted starting XI. Sum their Transfermarkt valuations as of
   the WC 2026 kickoff date.

2. **Citizenship-fallback path** (used for ~11 teams with no StatsBomb
   coverage: Curaçao, Cape Verde, Haiti, Iraq, Jordan, New Zealand, Norway,
   Uzbekistan, Bosnia, plus some others when name-mapping fails): sum top-11
   Transfermarkt valuations among players whose `country_of_citizenship`
   matches the qualifier. Same fallback shape as `squad_values._citizenship_top26_at`,
   just at n=11.

The StatsBomb → results.csv team-name normalization handles the 3 known
mismatches (Côte d'Ivoire ↔ Ivory Coast, Cape Verde Islands ↔ Cape Verde,
Congo DR ↔ DR Congo).

Output: data/processed/wc2026_predicted_lineup_values.csv with one row per
qualifier and columns (team, snapshot_date, lineup_value_eur, n_starters,
source). The `source` field is "modal_xi" or "citizenship_top11" for audit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.squad_values import (
    _build_player_value_lookup,
    _build_tm_name_index,
    _citizenship_top26_at,
    _player_value_at,
)
from src.features.lineup_values import _match_starter


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
SB_TO_TM_PATH = PROJECT_ROOT / "data" / "processed" / "sb_player_to_tm.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "wc2026_predicted_lineup_values.csv"

WC_2026_KICKOFF = date(2026, 6, 11)
LAST_K_MATCHES = 5
MIN_STARTERS_FROM_SB = 8  # need at least this many matched to TM to trust modal

# StatsBomb uses different spellings than results.csv for a few teams.
# Apply this map when reading SB data so team names align with results.csv.
SB_TO_RESULTS: dict[str, str] = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

# results.csv (or other inputs) → Transfermarkt `country_of_citizenship`
# spelling. Reused from squad_values.py for the small federations.
RESULTS_TO_CITIZENSHIP: dict[str, str] = {
    "Curaçao": "Curacao",
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "South Korea": "Korea, South",
}


def _build_sb_to_tm_cache(
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    rebuild: bool = False,
) -> dict[int, int | None]:
    """Build (or load from cache) the StatsBomb → Transfermarkt player_id map.

    The fuzzy match is slow (~4 min for 1,800 unique starters). We cache the
    result so subsequent runs reuse it.
    """
    if SB_TO_TM_PATH.exists() and not rebuild:
        cached = pd.read_csv(SB_TO_TM_PATH)
        return dict(zip(cached["sb_player_id"], cached["tm_player_id"]))

    print(f"  building SB→TM cache (one-time, ~3-4 min)...")
    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())

    mapping: dict[int, int | None] = {}
    for sb_id, group in lineups.groupby("player_id"):
        first = group.iloc[0]
        tm_id = _match_starter(
            first["player_name"], first["player_nickname"] or "",
            tm_index, fuzzy_candidates,
        )
        mapping[int(sb_id)] = tm_id

    cache_df = pd.DataFrame([
        {"sb_player_id": k, "tm_player_id": v} for k, v in mapping.items()
    ])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(SB_TO_TM_PATH, index=False)
    print(f"  wrote {SB_TO_TM_PATH.relative_to(PROJECT_ROOT)}")
    return mapping


def predict_starting_xi(
    team: str,
    snapshot_date: date,
    lineups: pd.DataFrame,
) -> list[int] | None:
    """Return predicted starting XI as list of StatsBomb player_ids.

    None if the team has fewer than 3 StatsBomb matches before snapshot_date.
    """
    sb_team = next((k for k, v in SB_TO_RESULTS.items() if v == team), team)
    team_lineups = lineups[lineups["team"] == sb_team]
    snapshot_ts = pd.Timestamp(snapshot_date)
    team_lineups = team_lineups[pd.to_datetime(team_lineups["match_date"]) < snapshot_ts]

    n_matches = team_lineups["match_id"].nunique()
    if n_matches < 3:
        return None

    # Take the LAST_K most recent matches
    recent_match_ids = (
        team_lineups.drop_duplicates("match_id")
        .sort_values("match_date")
        .tail(LAST_K_MATCHES)["match_id"]
    )
    recent = team_lineups[team_lineups["match_id"].isin(recent_match_ids)]

    counts = recent.groupby("player_id").size().reset_index(name="n_starts")
    counts = counts.sort_values(["n_starts", "player_id"], ascending=[False, True])
    return counts.head(11)["player_id"].astype(int).tolist()


def predict_lineup_value(
    team: str,
    snapshot_date: date,
    lineups: pd.DataFrame,
    sb_to_tm: dict[int, int | None],
    valuations_by_player: dict[int, pd.DataFrame],
    players: pd.DataFrame,
) -> tuple[float | None, int, str]:
    """Return (lineup_value_eur, n_matched_starters, source).

    source is "modal_xi" or "citizenship_top11".
    """
    sb_xi = predict_starting_xi(team, snapshot_date, lineups)

    if sb_xi is not None:
        # Modal-XI path
        values: list[float] = []
        for sb_pid in sb_xi:
            tm_pid = sb_to_tm.get(int(sb_pid))
            if tm_pid is None or pd.isna(tm_pid):
                continue
            v = _player_value_at(int(tm_pid), snapshot_date, valuations_by_player)
            if v is not None:
                values.append(v)

        if len(values) >= MIN_STARTERS_FROM_SB:
            return float(sum(values)), len(values), "modal_xi"

    # Citizenship fallback (top-11 instead of top-26)
    citizenship_name = RESULTS_TO_CITIZENSHIP.get(team, team)
    val = _citizenship_top26_at(
        citizenship_name, players, valuations_by_player, snapshot_date, top_n=11,
    )
    return (float(val) if val is not None else None), 11, "citizenship_top11"


def build_wc2026_predictions(snapshot_date: date = WC_2026_KICKOFF) -> pd.DataFrame:
    """Compute predicted lineup_value for every WC 2026 qualifier."""
    # Load inputs
    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])

    players = pd.read_csv(TM_DIR / "players.csv")
    valuations = pd.read_csv(TM_DIR / "player_valuations.csv")
    valuations_by_player = _build_player_value_lookup(valuations)

    sb_to_tm = _build_sb_to_tm_cache(lineups, players)

    # Get WC 2026 qualifiers from results.csv
    results = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "results.csv")
    results["date"] = pd.to_datetime(results["date"])
    wc26 = results[(results["date"] >= "2026-06-11") & (results["tournament"] == "FIFA World Cup")]
    qualifiers = sorted(set(wc26["home_team"]) | set(wc26["away_team"]))

    rows = []
    for team in qualifiers:
        v, n, source = predict_lineup_value(
            team, snapshot_date, lineups, sb_to_tm, valuations_by_player, players,
        )
        rows.append({
            "team": team,
            "snapshot_date": snapshot_date.isoformat(),
            "lineup_value_eur": v,
            "n_matched_starters": n,
            "source": source,
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_wc2026_predictions()
    print(f"\nwrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} qualifiers)")
    print()
    print("=== sources ===")
    print(df["source"].value_counts().to_string())
    print()
    print("=== teams with no value computed (will stay NaN) ===")
    print(df[df["lineup_value_eur"].isna()][["team", "source"]].to_string(index=False))
    print()
    print("=== top 10 by predicted lineup value ===")
    top10 = df.dropna(subset=["lineup_value_eur"]).sort_values("lineup_value_eur", ascending=False).head(10)
    top10 = top10.copy()
    top10["lineup_value_eur"] = top10["lineup_value_eur"].apply(lambda v: f"€{v:,.0f}")
    print(top10.to_string(index=False))
    print()
    print("=== bottom 10 ===")
    bot10 = df.dropna(subset=["lineup_value_eur"]).sort_values("lineup_value_eur").head(10)
    bot10 = bot10.copy()
    bot10["lineup_value_eur"] = bot10["lineup_value_eur"].apply(lambda v: f"€{v:,.0f}")
    print(bot10.to_string(index=False))
