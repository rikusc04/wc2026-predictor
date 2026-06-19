"""Per-match starting-XI club-Elo feature (v2 Phase 2.2d).

Upgrade over Phase 2.1's lineup_value_eur: the old feature summed each
starter's Transfermarkt market value, which treats all 11 positions the
same and lags during a player's prime. This module replaces it with a
position-weighted average of each starter's *club* Elo on the match date.

Why club Elo: a player's club is a sharp proxy for the level of
competition they face week-to-week. Vinicius at Real Madrid (clubelo
~2000) is reasonably distinguished from Vinicius at, say, a mid-table
La Liga side (~1500). The TM market value collapses both of those into
"a number that mostly tracks his age."

Why position-weighted: forwards drive goal output more than goalkeepers
in a Poisson model. Fixed weights — GK 0.8, DEF 1.0, MID 1.1, FWD 1.2 —
rather than learned, to avoid an extra fitting step and the leakage
risk that comes with it. Sensitivity to these constants is small
because the position mix is roughly constant across XIs.

Pipeline:
  1. Read data/raw/statsbomb_lineups.csv (one row per starter).
  2. Match each starter to a TM player_id (reuses Phase 2.1's SB→TM
     cache at data/processed/sb_player_to_tm.csv).
  3. For each starter, look up player_club_id via the most recent TM
     appearance ≤ match_date (src.features.club_lookup).
  4. Map club_id → clubelo shortname (data/processed/tm_club_to_clubelo.csv)
     → Elo at match_date (data/raw/clubelo/<Shortname>.csv).
  5. Aggregate per (match_date, home_team, away_team, side) as a
     position-weighted average.

Output schema:
    match_date, home_team, away_team, side, lineup_elo_weighted,
    n_starters_with_elo, n_starters_total

Coverage notes:
  - clubelo is European-only. Starters from MLS, Liga MX, Saudi, J-League,
    K-League etc. get None Elo. Most WC qualifiers from CONCACAF/AFC will
    have several such starters; the weighted-average aggregation handles
    that gracefully (just averages over what's available).
  - When fewer than MIN_STARTERS_WITH_ELO have any Elo, the side's
    lineup_elo_weighted is None — better than a noisy average over 2
    starters. The downstream imputer handles NaN.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.club_lookup import attach_club_to_lineups
from src.features.lineup_values import _match_starter
from src.features.squad_values import _build_tm_name_index


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "lineup_elo.csv"
SB_TO_TM_PATH = PROCESSED_DIR / "sb_player_to_tm.csv"
TM_TO_CLUBELO_PATH = PROCESSED_DIR / "tm_club_to_clubelo.csv"
CLUBELO_DIR = PROJECT_ROOT / "data" / "raw" / "clubelo"

POSITION_WEIGHTS = {"GK": 0.8, "DEF": 1.0, "MID": 1.1, "FWD": 1.2}

# Below this, the side's average mostly reflects the few European-club elites
# while ignoring most of the XI — sides with 5 of 11 starters with Elo can
# average ~2000 even when 6 of the starters are AFC/CONCACAF non-clubelo
# players. Emitting None (→ imputer median 1744) for thinly-covered sides is
# more honest than letting the noise bubble up.
MIN_STARTERS_WITH_ELO = 7


def position_group(position_name: str) -> str:
    """Bucket a StatsBomb position name into GK/DEF/MID/FWD.

    StatsBomb positions seen in our data: Goalkeeper, *Center Back*, *Back*,
    *Wing Back*, *Midfield*, *Wing*, *Forward*, Secondary Striker. The order
    of checks here matters — "Wing Back" contains "Back" so DEF wins, while
    "Left Wing" doesn't contain "Back" so it falls through to FWD.
    """
    if not isinstance(position_name, str):
        return "MID"  # safe default — midfielders get the median weight
    p = position_name.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    return "FWD"


def load_clubelo_for(shortname: str) -> pd.DataFrame | None:
    """Load the cached per-club history. Returns None if not on disk."""
    path = CLUBELO_DIR / f"{shortname}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["From", "To"])


class EloLookup:
    """Cached club_id → (date → Elo) lookup over the processed clubelo files.

    Holds one DataFrame per club in memory; the WC pipeline touches at most
    a few hundred unique clubs so memory is fine. Each lookup is an O(log n)
    bisect into the period table.
    """

    def __init__(self):
        if not TM_TO_CLUBELO_PATH.exists():
            raise FileNotFoundError(
                f"{TM_TO_CLUBELO_PATH} not found. "
                f"Run `python -m src.data.clubelo_loader` first."
            )
        mapping = pd.read_csv(TM_TO_CLUBELO_PATH)
        mapping = mapping.dropna(subset=["clubelo_shortname"])
        self._club_id_to_short: dict[int, str] = dict(
            zip(mapping["tm_club_id"].astype(int), mapping["clubelo_shortname"])
        )
        self._history_cache: dict[str, pd.DataFrame | None] = {}

    def elo_at(self, tm_club_id: int, target_date: pd.Timestamp) -> float | None:
        shortname = self._club_id_to_short.get(int(tm_club_id))
        if shortname is None:
            return None
        if shortname not in self._history_cache:
            self._history_cache[shortname] = load_clubelo_for(shortname)
        history = self._history_cache[shortname]
        if history is None or history.empty:
            return None

        # clubelo period rows are [From, To] inclusive of From, exclusive of To.
        # For dates before the first row, return None (no signal). For dates
        # past the last row's To, return the most recent Elo — clubs go on
        # break between seasons and the last period's Elo is still the right
        # snapshot.
        ts = pd.Timestamp(target_date)
        match = history[(history["From"] <= ts) & (history["To"] > ts)]
        if not match.empty:
            return float(match.iloc[-1]["Elo"])
        # Fallback: last period whose From ≤ target
        prior = history[history["From"] <= ts]
        if not prior.empty:
            return float(prior.iloc[-1]["Elo"])
        return None


def weighted_lineup_elo(
    starters_with_elo_and_position: list[tuple[float, str]],
) -> float | None:
    """Position-weighted average. Returns None below MIN_STARTERS_WITH_ELO."""
    if len(starters_with_elo_and_position) < MIN_STARTERS_WITH_ELO:
        return None
    num = 0.0
    den = 0.0
    for elo, pos_group in starters_with_elo_and_position:
        w = POSITION_WEIGHTS.get(pos_group, 1.0)
        num += w * elo
        den += w
    return num / den if den > 0 else None


def build_lineup_elo() -> pd.DataFrame:
    """Compute per-(match, side) starting-XI weighted Elo, write CSV, return df."""
    if not LINEUPS_PATH.exists():
        raise FileNotFoundError(
            f"{LINEUPS_PATH} not found. "
            f"Run `python -m src.data.lineups_loader` first."
        )

    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])

    print(f"loaded {len(lineups):,} starter rows from "
          f"{lineups['match_id'].nunique()} matches in "
          f"{lineups['competition'].nunique()} competitions")

    if SB_TO_TM_PATH.exists():
        cache = pd.read_csv(SB_TO_TM_PATH)
        sb_to_tm = dict(zip(cache["sb_player_id"], cache["tm_player_id"]))
        print(f"  using cached SB→TM map ({len(sb_to_tm):,} entries)")
    else:
        # Build it ourselves (slow path). Mirrors lineup_values.build_lineup_values.
        print("  SB→TM cache missing — building (~3-4 min)...")
        players = pd.read_csv(TM_DIR / "players.csv")
        tm_index = _build_tm_name_index(players)
        fuzzy_candidates = list(tm_index.keys())
        sb_to_tm: dict[int, int | None] = {}
        for sb_id, group in lineups.groupby("player_id"):
            first = group.iloc[0]
            tm_id = _match_starter(
                first["player_name"], first["player_nickname"],
                tm_index, fuzzy_candidates,
            )
            sb_to_tm[int(sb_id)] = tm_id
        pd.DataFrame([
            {"sb_player_id": k, "tm_player_id": v} for k, v in sb_to_tm.items()
        ]).to_csv(SB_TO_TM_PATH, index=False)

    lineups["tm_player_id"] = lineups["player_id"].map(sb_to_tm)
    print(f"  {lineups['tm_player_id'].notna().sum():,}/{len(lineups):,} starters matched to TM")

    print("attaching player→club from TM appearances...")
    lineups = attach_club_to_lineups(lineups)
    n_with_club = lineups["player_club_id"].notna().sum()
    print(f"  {n_with_club:,}/{len(lineups):,} starters have a TM club at match_date")

    print("looking up club Elo at match dates...")
    elo = EloLookup()

    rows: list[dict] = []
    for (match_date, home, away, side), grp in lineups.groupby(
        ["match_date", "home_team", "away_team", "side"]
    ):
        starters_data: list[tuple[float, str]] = []
        n_total = len(grp)
        for _, r in grp.iterrows():
            club_id = r["player_club_id"]
            if pd.isna(club_id):
                continue
            club_elo = elo.elo_at(int(club_id), pd.Timestamp(match_date))
            if club_elo is None:
                continue
            starters_data.append((club_elo, position_group(r["position"])))
        rows.append({
            "match_date": match_date.date() if hasattr(match_date, "date") else match_date,
            "home_team": home,
            "away_team": away,
            "side": side,
            "lineup_elo_weighted": weighted_lineup_elo(starters_data),
            "n_starters_with_elo": len(starters_data),
            "n_starters_total": n_total,
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_lineup_elo()
    print(f"\nwrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")
    print()
    print("=== summary ===")
    matched = df.dropna(subset=["lineup_elo_weighted"])
    print(f"sides with weighted Elo: {len(matched):,}/{len(df):,}")
    print(f"mean lineup_elo_weighted: {matched['lineup_elo_weighted'].mean():.0f}")
    print(f"median: {matched['lineup_elo_weighted'].median():.0f}")
    print(f"p10/p90: {matched['lineup_elo_weighted'].quantile(0.10):.0f} / "
          f"{matched['lineup_elo_weighted'].quantile(0.90):.0f}")
    print()
    print("=== top 10 sides by weighted Elo ===")
    print(matched.sort_values("lineup_elo_weighted", ascending=False).head(10).to_string(index=False))
    print()
    print("=== Elo-coverage distribution (n_starters_with_elo) ===")
    print(df.groupby("n_starters_with_elo").size().to_string())
