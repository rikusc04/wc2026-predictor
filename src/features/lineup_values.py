"""Per-match starting-XI market-value feature.

v2 Phase 2.1: cross-reference StatsBomb lineups with Transfermarkt
valuations to compute "lineup market value" per side per match. More
specific than the v1 squad_value (full 23-26 player roster) — captures
when a team's strongest players aren't on the field (Brazil B-team in
a dead rubber, Argentina missing Messi, etc.).

Pipeline:
  1. Read data/raw/statsbomb_lineups.csv (one row per starter).
  2. For each starter, match the StatsBomb player_name to a TM player_id.
     Try both player_name and player_nickname; fall back to fuzzy match.
  3. For each matched player, look up their valuation as-of-or-before
     the match date.
  4. Sum per (match_date, home_team, away_team, side) → starting XI value.
  5. Output a per-side row to data/processed/lineup_values.csv.

Output schema:
    match_date, home_team, away_team, side, lineup_value_eur,
    n_starters_matched, n_starters_total

Subsequent join in features.csv is on (match_date, home_team, away_team)
with the side dimension pivoted into home/away columns. Matches not in
StatsBomb's coverage (WC 2014, all qualifiers/friendlies before 2018) will
have NaN values which the model's SimpleImputer handles.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.squad_values import (
    _build_player_value_lookup,
    _build_tm_name_index,
    _fuzzy_lookup,
    _normalize,
    _player_value_at,
)


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "lineup_values.csv"


def _match_starter(
    player_name: str,
    player_nickname: str,
    tm_index: dict[str, list[int]],
    fuzzy_candidates: list[str],
) -> int | None:
    """Find a TM player_id for a StatsBomb starter. None if no confident match.

    Tries normalized player_name first, then nickname, then fuzzy match.
    Different shape from squad_values._match_player because StatsBomb gives
    a single full name string + an optional nickname, not separate
    given/family fields.
    """
    candidates = []
    if isinstance(player_name, str) and player_name.strip():
        candidates.append(player_name.strip())
    if isinstance(player_nickname, str) and player_nickname.strip():
        candidates.append(player_nickname.strip())

    # Direct hits first
    for cand in candidates:
        norm = _normalize(cand)
        if norm and norm in tm_index:
            return tm_index[norm][0]

    # Fuzzy fallback on the most-likely candidate (nickname usually closer)
    primary = candidates[-1] if candidates else ""
    if not primary:
        return None
    best = _fuzzy_lookup(_normalize(primary), fuzzy_candidates)
    if best is not None:
        return tm_index[best][0]
    return None


def build_lineup_values() -> pd.DataFrame:
    """Compute per-(match, side) starting-XI value, write CSV, return df."""
    if not LINEUPS_PATH.exists():
        raise FileNotFoundError(
            f"{LINEUPS_PATH} not found. "
            f"Run `python -m src.data.lineups_loader` first."
        )

    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"]).dt.date

    print(f"loaded {len(lineups):,} starter rows from "
          f"{lineups['match_id'].nunique()} matches in "
          f"{lineups['competition'].nunique()} competitions")

    print("loading Transfermarkt players + valuations...")
    players = pd.read_csv(TM_DIR / "players.csv")
    valuations = pd.read_csv(TM_DIR / "player_valuations.csv")
    print(f"  {len(players):,} TM players, {len(valuations):,} valuations")

    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())
    valuations_by_player = _build_player_value_lookup(valuations)

    print(f"matching {lineups['player_id'].nunique():,} unique starters to TM...")

    # Build a per-StatsBomb-player-id cache so we only try to match each player once
    sb_to_tm: dict[int, int | None] = {}
    for sb_id, group in lineups.groupby("player_id"):
        first = group.iloc[0]
        tm_id = _match_starter(
            first["player_name"], first["player_nickname"],
            tm_index, fuzzy_candidates,
        )
        sb_to_tm[int(sb_id)] = tm_id

    n_matched = sum(1 for v in sb_to_tm.values() if v is not None)
    print(f"  matched {n_matched} / {len(sb_to_tm)} ({n_matched/len(sb_to_tm):.1%})")

    # Now compute per-(match, side) value
    lineups["tm_player_id"] = lineups["player_id"].map(sb_to_tm)

    rows: list[dict] = []
    for (match_date, home, away, side), grp in lineups.groupby(
        ["match_date", "home_team", "away_team", "side"]
    ):
        values: list[float] = []
        n_total = len(grp)
        n_matched_side = 0
        for _, r in grp.iterrows():
            pid = r["tm_player_id"]
            if pd.isna(pid):
                continue
            v = _player_value_at(int(pid), match_date, valuations_by_player)
            if v is not None:
                values.append(v)
                n_matched_side += 1
        rows.append({
            "match_date": match_date,
            "home_team": home,
            "away_team": away,
            "side": side,
            "lineup_value_eur": sum(values) if values else None,
            "n_starters_matched": n_matched_side,
            "n_starters_total": n_total,
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_lineup_values()
    print(f"\nwrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")
    print()
    print("=== summary ===")
    print(f"matches with both sides: {df.groupby(['match_date','home_team','away_team']).size().eq(2).sum()}")
    print(f"mean lineup_value_eur: €{df['lineup_value_eur'].mean():,.0f}")
    print(f"median: €{df['lineup_value_eur'].median():,.0f}")
    print()
    print("=== sample (10 highest lineup values) ===")
    sample = df.dropna(subset=["lineup_value_eur"]).sort_values(
        "lineup_value_eur", ascending=False
    ).head(10)
    sample = sample.copy()
    sample["lineup_value_eur"] = sample["lineup_value_eur"].apply(lambda v: f"€{v:,.0f}")
    print(sample.to_string(index=False))
    print()
    print("=== match-rate distribution ===")
    print(df.groupby("n_starters_matched").size().to_string())
